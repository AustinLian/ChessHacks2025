import torch
from chess import Move
from .utils import chess_manager, GameContext  # adjust if needed
import time

# -------------------------------
import sys

# Add the models folder directly to sys.path
sys.path.append(r"F:/VS Code Storage/ChessHacks2025/training/models")

# Now you can import your module
from resnet_policy_value import ResNetPolicyValue
# Model imports
# -------------------------------


# -------------------------------
# Model config
# -------------------------------
MODEL_PATH = "F:/VS Code Storage/ChessHacks2025/checkpoints/best_model.pt"
#MODEL_PATH =    "F:/VS Code Storage/ChessHacks2025/weights/trained_weights.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetPolicyValue(
    num_blocks=10,
    channels=128,
    input_planes=18,
    policy_channels=32,
    value_channels=32,
    policy_size=20480
).to(DEVICE)

# Load weights
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# -------------------------------
# Board -> Tensor (18 planes)
# -------------------------------
piece_to_plane = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}

def board_to_tensor(board):
    planes = torch.zeros((1, 18, 8, 8), dtype=torch.float32, device=DEVICE)

    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            plane = piece_to_plane[piece.symbol()]
            rank = 7 - (square // 8)
            file = square % 8
            planes[0, plane, rank, file] = 1.0

    planes[0, 12, :, :] = 1.0 if board.has_kingside_castling_rights(True) else 0
    planes[0, 13, :, :] = 1.0 if board.has_queenside_castling_rights(True) else 0
    planes[0, 14, :, :] = 1.0 if board.has_kingside_castling_rights(False) else 0
    planes[0, 15, :, :] = 1.0 if board.has_queenside_castling_rights(False) else 0

    planes[0, 16, :, :] = 1.0 if board.turn else 0
    planes[0, 17, :, :] = 0.0  # extra plane

    return planes


# -------------------------------
# Move to index mapping
# -------------------------------
# Placeholder: must match training encoding exactly
def move_to_index(move):
    # Simple 64*64 example mapping
    return move.from_square * 64 + move.to_square


# -------------------------------
# ChessHacks Entrypoint
# -------------------------------
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available!")

    # Convert board to tensor
    board_tensor = board_to_tensor(ctx.board)

    # Get policy & value
    with torch.no_grad():
        policy_logits, value = model.predict(board_tensor)
        policy_logits = policy_logits.cpu().numpy().flatten()

    # Map legal moves to probabilities
    move_probs = {}
    for move in legal_moves:
        idx = move_to_index(move)
        if idx < len(policy_logits):
            move_probs[move] = float(policy_logits[idx])
        else:
            move_probs[move] = 0.0

    # Normalize probabilities
    total = sum(move_probs.values())
    if total > 0:
        for move in move_probs:
            move_probs[move] /= total
    else:
        # fallback: uniform if something went wrong
        uniform_prob = 1.0 / len(legal_moves)
        for move in legal_moves:
            move_probs[move] = uniform_prob

    ctx.logProbabilities(move_probs)

    # Choose move according to probabilities
    import random
    weights = [move_probs[m] for m in legal_moves]
    chosen_move = random.choices(legal_moves, weights=weights, k=1)[0]

    return chosen_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # Reset model caches if needed
    pass
