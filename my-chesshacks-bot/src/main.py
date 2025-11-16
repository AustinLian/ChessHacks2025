from .utils import chess_manager, GameContext
from chess import Move
import chess
import random
import time
from pathlib import Path
from collections import deque
import math

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    r"F:/VS Code Storage/ChessHacks2025/training\whiteNoise/checkpoints_delta_resnet20\best.pt"
)

NUM_PLANES = 18
NUM_PROMOS = 5   # [None, Q, R, B, N]
POLICY_DIM = 64 * 64 * NUM_PROMOS
CP_SCALE = 200.0

MATE_SCORE = 100000.0
TIME_LIMIT_SECONDS = 0.25   # per-move budget
MAX_SEARCH_DEPTH = 27        # root + 1 ply search

RESULT_TO_CP_SCALE = 400.0  # how much game-result head influences eval

ENGINE = None   # set at import time


# ============================================================
# BOARD → PLANES (must match training)
# ============================================================

PIECE_PLANES = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Same encoding as your training: (18, 8, 8).
    """
    P = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        p_idx = PIECE_PLANES[(piece.piece_type, piece.color)]
        r = 7 - chess.square_rank(sq)
        f = chess.square_file(sq)
        P[p_idx, r, f] = 1.0

    # side to move
    if board.turn == chess.WHITE:
        P[12, :, :] = 1.0
    else:
        P[12, :, :] = 0.0

    # castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        P[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        P[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        P[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        P[16, :, :] = 1.0

    # en-passant file
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        P[17, :, file] = 1.0

    return P


def move_to_index(move: chess.Move) -> int:
    """
    Same policy index scheme as training:
      (from * 64 + to) * 5 + promo_idx
    """
    from_sq = move.from_square
    to_sq = move.to_square
    promo = move.promotion if move.promotion is not None else None
    promo_idx = PROMO_PIECES.index(promo)  # 0..4
    return (from_sq * 64 + to_sq) * NUM_PROMOS + promo_idx


# ============================================================
# MODEL DEFINITION (must match train.py)
# ============================================================

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = h + self.short(x)
        return self.relu(h)


class ResNet20_PolicyDelta(nn.Module):
    def __init__(self, planes, policy_dim, cp_scale):
        super().__init__()
        self.cp_scale = cp_scale

        def make_layer(in_c, out_c, blocks):
            layers = [BasicBlock(in_c, out_c)]
            for _ in range(blocks - 1):
                layers.append(BasicBlock(out_c, out_c))
            return nn.Sequential(*layers)

        self.l1 = make_layer(planes, 128, 3)
        self.l2 = make_layer(128, 256, 3)
        self.l3 = make_layer(256, 256, 3)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 256)

        # Heads: must match trainer
        self.policy_head = nn.Linear(256, policy_dim)
        self.delta_head = nn.Linear(256, 1)
        self.cp_head = nn.Linear(256, 1)
        self.result_head = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc(x))

        policy_logits = self.policy_head(x)

        delta_scaled = self.delta_head(x).squeeze(-1)
        cp_scaled = self.cp_head(x).squeeze(-1)

        delta_real = delta_scaled * self.cp_scale
        cp_real = cp_scaled * self.cp_scale

        result_raw = self.result_head(x).squeeze(-1)
        result = torch.tanh(result_raw)  # [-1,1]

        return policy_logits, delta_real, delta_scaled, cp_real, cp_scaled, result


# ============================================================
# NN-GUIDED SEARCH ENGINE
# ============================================================

class NNEvalEngine:
    """
    Alpha-beta search guided by:
      - cp_head + result_head for evaluation
      - policy logits for move ordering at the root

    At the root, we blend:
      - NN policy distribution (over legal moves)
      - shallow search evals over legal moves (with anti-derp + phase-aware weights)
    """

    def __init__(self):
        print(f"[ENGINE] Initializing NN engine from {MODEL_PATH!r}")
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet20_PolicyDelta(
            planes=NUM_PLANES,
            policy_dim=POLICY_DIM,
            cp_scale=CP_SCALE,
        ).to(self.device)

        state = torch.load(MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.time_limit = TIME_LIMIT_SECONDS
        self.max_depth = MAX_SEARCH_DEPTH
        self.time_up = False
        self.tt = {}
        # cache: fen -> (policy_logits, eval_cp, cp_real, delta_real, result)
        self.nn_cache = {}

        # for anti-derp repetition
        self.recent_fens = deque(maxlen=16)

        print(f"[ENGINE] NN engine ready on {self.device}")

    def note_position(self, board: chess.Board):
        """Track recent FENs for anti-repetition bias."""
        self.recent_fens.append(board.fen())

    @torch.no_grad()
    def _policy_and_value(self, board: chess.Board):
        """
        Returns cached:
          - policy_logits: (POLICY_DIM,)
          - eval_cp: float (cp + result contribution)
          - cp_real: float (raw centipawn eval of side to move)
          - delta_real: float (best-move delta estimate)
          - result: float in [-1, 1]
        """
        fen = board.fen()
        cached = self.nn_cache.get(fen)
        if cached is not None:
            return cached

        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)

        (
            policy_logits,
            delta_real,
            _delta_scaled,
            cp_real,
            _cp_scaled,
            result,
        ) = self.model(x)

        policy_logits = policy_logits[0].detach().cpu()
        cp_real = float(cp_real[0].item())
        delta_real = float(delta_real[0].item())
        res_eval = float(result[0].item())  # [-1, 1]

        eval_cp = cp_real + res_eval * RESULT_TO_CP_SCALE

        out = (policy_logits, eval_cp, cp_real, delta_real, res_eval)
        self.nn_cache[fen] = out
        return out

    def _evaluate(self, board: chess.Board) -> float:
        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_SCORE
            return 0.0
        _, eval_cp, _, _, _ = self._policy_and_value(board)
        return max(-MATE_SCORE, min(MATE_SCORE, eval_cp))

    def _policy_scores_for_moves(self, board: chess.Board, legal_moves):
        """
        Used only where we explicitly want policy-based move ordering.
        Thanks to the cache, this does not create extra NN calls per node.
        """
        policy_logits, _, _, _, _ = self._policy_and_value(board)
        scores = []
        for mv in legal_moves:
            idx = move_to_index(mv)
            s = float(policy_logits[idx].item()) if 0 <= idx < POLICY_DIM else 0.0
            scores.append(s)
        return scores

    def _time_exceeded(self, start_time: float) -> bool:
        if self.time_limit <= 0:
            return False
        if time.perf_counter() - start_time >= self.time_limit:
            self.time_up = True
            return True
        return False

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, start_time: float) -> float:
        """
        Negamax alpha-beta search.

        Note on 1.1 ("use network once per node, not twice"):
        - Each unique FEN shares a single NN forward via self.nn_cache.
        - _evaluate and _policy_scores_for_moves both hit the same cached result.
        """
        if self._time_exceeded(start_time):
            return 0.0

        key = (board.fen(), depth, board.turn)
        if key in self.tt:
            return self.tt[key]

        if depth == 0 or board.is_game_over():
            val = self._evaluate(board)
            self.tt[key] = val
            return val

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            val = self._evaluate(board)
            self.tt[key] = val
            return val

        # Non-root nodes: we can skip policy-guided ordering if we want to be lighter.
        # Use a simple heuristic: captures first, then others.
        captures = []
        quiets = []
        for mv in legal_moves:
            if board.is_capture(mv):
                captures.append(mv)
            else:
                quiets.append(mv)
        ordered_moves = captures + quiets

        best_val = -float("inf")

        for mv in ordered_moves:
            if self.time_up:
                break
            board.push(mv)
            val = -self._search(board, depth - 1, -beta, -alpha, start_time)
            board.pop()

            if self.time_up:
                break

            if val > best_val:
                best_val = val
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break

        self.tt[key] = best_val
        return best_val

    def _material_score(self, board: chess.Board) -> int:
        """Rough material count to detect game phase."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }
        total = 0
        for piece in board.piece_map().values():
            if piece.piece_type in values:
                total += values[piece.piece_type]
        return total

    def select_move(self, board: chess.Board):
        """
        Root move selection:

          1. If only one legal move, just play it.
          2. Get NN policy over legal moves (prior_probs).
          3. Compute policy entropy → confidence.
          4. Compute material → phase (opening/midgame vs endgame).
          5. Set Wp/Ws (policy/search weights) and search_depth using both.
          6. For each move:
                - run shallow search
                - apply anti-repetition penalty
          7. Convert search_vals to distribution (search_probs).
          8. Blend prior_probs and search_probs with Wp/Ws.
          9. Choose argmax of blended probs (deterministic) & return probs.
        """
        start_time = time.perf_counter()
        self.time_up = False
        self.tt.clear()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, {}

        # 1. Forced move shortcut
        if len(legal_moves) == 1:
            only = legal_moves[0]
            return only, {only: 1.0}

        # 2. NN policy over legal moves (prior)
        policy_logits, _eval_cp, _cp_real, _delta_real, _res = self._policy_and_value(board)

        policy_scores = []
        for mv in legal_moves:
            idx = move_to_index(mv)
            if 0 <= idx < POLICY_DIM:
                policy_scores.append(float(policy_logits[idx].item()))
            else:
                policy_scores.append(-1e9)

        max_p = max(policy_scores)
        exp_p = [math.exp(s - max_p) for s in policy_scores]
        Zp = sum(exp_p)
        if Zp <= 0:
            prior_probs = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            prior_probs = [x / Zp for x in exp_p]

        # 3. Policy entropy → confidence
        H = 0.0
        for p in prior_probs:
            if p > 0.0:
                H -= p * math.log(p + 1e-12)
        H_max = math.log(len(prior_probs)) if len(prior_probs) > 1 else 0.0
        if H_max > 0.0:
            confidence = 1.0 - (H / H_max)  # 0 flat → 1 sharp
        else:
            confidence = 0.0

        # 4. Material → phase
        material = self._material_score(board)
        endgameish = material <= 16

        # 5. Set Wp/Ws and search_depth (1.2 + 1.5)
        # base policy weight depends on confidence
        Wp = 0.4 + 0.4 * confidence  # 0.4..0.8

        if endgameish:
            # in endgames, lean more on search
            Wp -= 0.1

        Wp = max(0.3, min(0.85, Wp))
        Ws = 1.0 - Wp

        # search depth: deeper in endgames / checks
        if endgameish:
            search_depth = self.max_depth
        else:
            search_depth = max(8, self.max_depth - 1)

        if board.is_check():
            # in check, search as deep as allowed
            search_depth = self.max_depth

        # 6. Shallow search evals + anti-derp repetition bias (1.4, 1.6-ish)
        search_vals = []
        cp_temp = 120.0  # centipawn temperature

        for mv in legal_moves:
            if self._time_exceeded(start_time):
                break

            board.push(mv)
            new_fen = board.fen()

            # run shallow search
            val = -self._search(
                board,
                depth=search_depth,
                alpha=-float("inf"),
                beta=float("inf"),
                start_time=start_time,
            )

            # basic anti-repetition bias:
            # if this move returns to a recently seen FEN and eval is near equal,
            # nudge it down slightly to avoid pointless shuffling
            if new_fen in self.recent_fens and abs(val) < 50.0:
                val -= 15.0  # ~0.15 pawn penalty

            board.pop()
            search_vals.append(val)

        # If we didn't manage to search all moves in time, fall back to pure policy
        if len(search_vals) < len(legal_moves):
            best_idx = int(np.argmax(policy_scores))
            best_move = legal_moves[best_idx]
            probs_dict = {mv: prior_probs[i] for i, mv in enumerate(legal_moves)}
            return best_move, probs_dict

        # 7. Convert search vals → distribution
        max_s = max(search_vals)
        exp_s = [math.exp((v - max_s) / cp_temp) for v in search_vals]
        Zs = sum(exp_s)
        if Zs <= 0:
            search_probs = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            search_probs = [x / Zs for x in exp_s]

        # 8. Blend policy + search with dynamic Wp/Ws
        combined = [
            Wp * prior_probs[i] + Ws * search_probs[i]
            for i in range(len(legal_moves))
        ]

        Zc = sum(combined)
        if Zc <= 0:
            final_probs = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            final_probs = [x / Zc for x in combined]

        # 9. Final move = argmax of blended probs (deterministic)
        best_idx = int(np.argmax(final_probs))
        best_move = legal_moves[best_idx]

        probs_dict = {mv: final_probs[i] for i, mv in enumerate(legal_moves)}
        return best_move, probs_dict

    def reset(self):
        self.tt.clear()
        self.nn_cache.clear()
        self.recent_fens.clear()
        self.time_up = False


# ============================================================
# ENGINE INIT
# ============================================================

try:
    ENGINE = NNEvalEngine()
except Exception as e:
    ENGINE = None
    print(f"[ENGINE] Failed to initialize NN engine, falling back to random. Error: {e}")


# ============================================================
# Submission platform hooks
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.01)

    board = ctx.board
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Fallback: random policy if ENGINE is None
    if ENGINE is None:
        print("NO ENGINE, using random.")
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {
            move: weight / total_weight
            for move, weight in zip(legal_moves, move_weights)
        }
        ctx.logProbabilities(move_probs)
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    # tell the engine about the current position for anti-derp repetition logic
    ENGINE.note_position(board)

    best_move, probs = ENGINE.select_move(board)
    if best_move is None:
        print("ENGINE returned no move, using random fallback.")
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {
            move: weight / total_weight
            for move, weight in zip(legal_moves, move_weights)
        }
        ctx.logProbabilities(move_probs)
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    # Ensure probabilities are defined over exactly current legal moves
    probs_filtered = {mv: probs.get(mv, 0.0) for mv in legal_moves}
    Z = sum(probs_filtered.values())
    if Z > 0:
        probs_filtered = {mv: p / Z for mv, p in probs_filtered.items()}
    else:
        p = 1.0 / len(legal_moves)
        probs_filtered = {mv: p for mv in legal_moves}

    ctx.logProbabilities(probs_filtered)
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    if ENGINE is not None:
        ENGINE.reset()
