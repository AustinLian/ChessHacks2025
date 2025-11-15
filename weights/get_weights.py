import torch
from pathlib import Path

# Path to your saved model
MODEL_PATH = Path(r"F:/VS Code Storage/ChessHacks2025/checkpoints/best_model.pt")
SAVE_WEIGHTS_PATH = Path(r"F:/VS Code Storage/ChessHacks2025/weights/trained_weights.pt")

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# If checkpoint contains 'model_state_dict', extract it
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint  # assume checkpoint is already a state dict

# Save only the weights
torch.save(state_dict, SAVE_WEIGHTS_PATH)

print(f"Saved weights to {SAVE_WEIGHTS_PATH}")
