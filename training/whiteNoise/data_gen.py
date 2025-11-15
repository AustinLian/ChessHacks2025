#!/usr/bin/env python3
"""
SELF-PLAY DATA GENERATOR (STOCKFISH-LABELED)
============================================

Produces MILLIONS of positions with:
  - planes: (18, 8, 8)
  - y_policy_best
  - cp_before
  - cp_after_best
  - delta_cp
  - game_result (POV of side to move)

Every position is produced by:

  1) Get SF best move + eval
  2) Store labels
  3) Choose *noisy move* for self-play:
       - temperature
       - Dirichlet root noise

This prevents positional collapse and generates diverse training data.

Outputs:
  shard_00001.npz
  shard_00002.npz
  ...

Each shard contains ~50,000 positions
"""

import chess
import chess.engine
import chess.pgn
import numpy as np
from pathlib import Path
import random
import time

# =====================================================================
# CONFIG
# =====================================================================

class Config:
    ENGINE_PATH = r"C:\Users\ethan\Downloads\ChessHacks\ChessHacks2025\training\whiteNoise\stockfish-windows-x86-64-avx2.exe"

    DEPTH = 12                   # use fixed depth for reproducibility
    TIME_LIMIT = None           # set to 0.03 if using time instead
    MAX_MOVES_PER_GAME = 200

    SHARD_SIZE = 50000          # NPZ entries per shard
    OUTPUT_DIR = "training\whiteNoise\processed"

    DIRICHLET_ALPHA = 0.3
    TEMPERATURE = 1.2           # >1 = more randomness


cfg = Config()


# =====================================================================
# BOARD → PLANES (same format as your supervised pipeline)
# =====================================================================

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
NUM_PLANES = 18

PROMO_PIECES = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
NUM_PROMOS = len(PROMO_PIECES)
POLICY_DIM = 64 * 64 * NUM_PROMOS


def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    promo = move.promotion if move.promotion is not None else None
    promo_idx = PROMO_PIECES.index(promo)
    return (from_sq * 64 + to_sq) * NUM_PROMOS + promo_idx


def fen_to_planes(board: chess.Board) -> np.ndarray:
    P = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        p_idx = PIECE_PLANES[(piece.piece_type, piece.color)]
        r = 7 - chess.square_rank(sq)
        f = chess.square_file(sq)
        P[p_idx, r, f] = 1

    # side to move
    P[12, :, :] = 1 if board.turn == chess.WHITE else 0

    # castling
    if board.has_kingside_castling_rights(chess.WHITE):  P[13,:,:] = 1
    if board.has_queenside_castling_rights(chess.WHITE): P[14,:,:] = 1
    if board.has_kingside_castling_rights(chess.BLACK):  P[15,:,:] = 1
    if board.has_queenside_castling_rights(chess.BLACK): P[16,:,:] = 1

    # en passant
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        P[17, :, file] = 1

    return P


# =====================================================================
# STOCKFISH WRAPPER
# =====================================================================

class StockfishEvaluator:
    def __init__(self, path, depth=None, time_limit=None):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.depth = depth
        self.time_limit = time_limit

    def limit(self):
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        return chess.engine.Limit(time=self.time_limit)

    def eval_and_best(self, board):
        info = self.engine.analyse(board, self.limit())
        score = info["score"].pov(board.turn)
        if score.is_mate():
            mate = score.mate()
            cp = 10000 * (1 if mate > 0 else -1)
        else:
            cp = score.score(mate_score=10000)

        pv = info.get("pv", None)
        best_move = pv[0] if pv else None
        return int(cp), best_move

    def close(self):
        self.engine.quit()


# =====================================================================
# SELF-PLAY EPISODE
# =====================================================================

def choose_noisy_move(board, best_move):
    """
    Returns a random legal move biased toward Stockfish best.
    """

    legal = list(board.legal_moves)

    # Pure deterministic = best move
    if cfg.TEMPERATURE <= 1e-6 and cfg.DIRICHLET_ALPHA <= 1e-6:
        return best_move

    # Make a simple softmax: best move gets big weight
    logits = np.zeros(len(legal), dtype=np.float32)
    for i, mv in enumerate(legal):
        logits[i] = 1.0 if mv == best_move else 0.0

    # Add Dirichlet noise
    noise = np.random.dirichlet([cfg.DIRICHLET_ALPHA] * len(legal))
    logits = logits + noise

    # Apply temperature
    logits = logits ** (1.0 / cfg.TEMPERATURE)

    # Normalize
    logits /= logits.sum()

    return random.choices(legal, weights=logits, k=1)[0]


def play_selfplay_game(sf):
    """
    Returns a list of labeled positions.
    """

    board = chess.Board()
    labels = []

    while not board.is_game_over() and len(labels) < cfg.MAX_MOVES_PER_GAME:

        planes = fen_to_planes(board)
        cp_before, best_move = sf.eval_and_best(board)
        if best_move is None:
            break

        # Eval after SF best move
        b2 = board.copy()
        b2.push(best_move)
        cp_after_opp_pov, _ = sf.eval_and_best(b2)   # eval from opponent POV
        cp_after = -cp_after_opp_pov                 # convert back to original POV

        delta = cp_after - cp_before
        policy_idx = move_to_index(best_move)

        # Store sample
        labels.append((planes, policy_idx, cp_before, cp_after, delta))

        # SELF-PLAY MOVE (NOISY)
        move = choose_noisy_move(board, best_move)
        board.push(move)

    # Convert game result (POV)
    res = board.result()
    if res == "1-0":
        g = 1
    elif res == "0-1":
        g = -1
    else:
        g = 0

    return labels, g


# =====================================================================
# MAIN LOOP — WRITES SHARDED NPZ FILES
# =====================================================================

def main():
    out = Path(cfg.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    sf = StockfishEvaluator(cfg.ENGINE_PATH, depth=cfg.DEPTH, time_limit=cfg.TIME_LIMIT)

    shard_id = 1
    buffer = []

    print("Starting self-play...")

    while True:
        labels, g = play_selfplay_game(sf)

        for (planes, pol, cp_before, cp_after, delta) in labels:
            buffer.append((planes, pol, cp_before, cp_after, delta, g))

        if len(buffer) >= cfg.SHARD_SIZE:
            X = np.stack([b[0] for b in buffer])
            policy = np.array([b[1] for b in buffer], np.int64)
            cp_before = np.array([b[2] for b in buffer], np.float32)
            cp_after = np.array([b[3] for b in buffer], np.float32)
            delta_cp = np.array([b[4] for b in buffer], np.float32)
            game_result = np.array([b[5] for b in buffer], np.float32)

            save_path = out / f"shard_{shard_id:05d}.npz"
            np.savez_compressed(save_path,
                                X=X,
                                y_policy_best=policy,
                                cp_before=cp_before,
                                cp_after_best=cp_after,
                                delta_cp=delta_cp,
                                game_result=game_result)

            print(f"[+] Saved shard {shard_id} with {len(buffer)} samples → {save_path}")

            buffer = []
            shard_id += 1

        # Print heartbeat
        if shard_id % 10 == 0:
            print(f"Still running… shards={shard_id-1}")


if __name__ == "__main__":
    main()
