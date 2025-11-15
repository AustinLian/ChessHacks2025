
import sys
from utils import chess_manager, GameContext
from chess import Move
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()  # go up 2 levels from src
sys.path.append(str(PROJECT_ROOT))

from training.models.resnet_policy_value import create_model
import random

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = Path("F:\VS Code Storage\ChessHacks2025\checkpoints/best_model.pt")
NUM_PLANES = 18
POLICY_DIM = 64*64*5  # 20480 moves
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL ONCE
# -----------------------------
model = create_model({'num_planes': NUM_PLANES, 'policy_dim': POLICY_DIM})
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()


# -----------------------------
# BOARD -> 18-PLANE CONVERSION
# -----------------------------
def board_to_planes(board):
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        plane_index = None
        if piece.color:  # white
            plane_index = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5}[piece.symbol().upper()]
        else:  # black
            plane_index = {'P':6,'N':7,'B':8,'R':9,'Q':10,'K':11}[piece.symbol().upper()]
        planes[plane_index, row, col] = 1.0
    # remaining planes 12-17 left as zeros (or fill as in training)
    planes = planes * 2.0 - 1.0  # normalize if training did
    return torch.tensor(planes, dtype=torch.float32).unsqueeze(0).to(device)


# -----------------------------
# MOVE -> POLICY INDEX
# -----------------------------
def move_uci_to_index(move: Move):
    """
    TODO: implement your 20480 move encoding exactly as in training dataset
    """
    return 0  # placeholder for now


# -----------------------------
# CHESSHACKS ENTRYPOINT
# -----------------------------
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Convert board
    input_planes = board_to_planes(ctx.board)

    # Forward pass
    with torch.no_grad():
        policy_logits, _ = model(input_planes)
        policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy().flatten()

    # Filter to legal moves
    move_to_idx = {m: move_uci_to_index(m) for m in legal_moves}
    move_weights = [policy_probs[move_to_idx[m]] for m in legal_moves]
    total_weight = sum(move_weights)
    move_weights = [w / total_weight for w in move_weights]

    # Log probabilities for ChessHacks
    ctx.logProbabilities({m: w for m, w in zip(legal_moves, move_weights)})

    # Return sampled move
    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # Clear any cached state if needed
    pass
