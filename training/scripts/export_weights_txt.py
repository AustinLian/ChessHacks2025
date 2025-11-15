import sys
from pathlib import Path
import numpy as np
import torch

# --- make sure repo root is on sys.path ---
ROOT = Path(__file__).resolve().parents[2]  # .../ChessHacks2025
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.models.resnet_policy_value import ResNetPolicyValue


# Adjust these if you ever change the input planes or policy dim
INT_IN_CHANNELS = 18
INT_POLICY_SIZE = 64 * 64 * 5   # from your move encoding (64*64*NUM_PROMOS)

CKPT_PATH = Path("weights/sf_supervised.pt")   # same as --out you used when training
OUT_PATH  = Path("weights/engine_weights.txt")


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    print(f"Loading checkpoint from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    # your train script either saved {"model_state": state_dict} or a raw state_dict
    state = ckpt.get("model_state", ckpt)

    model = ResNetPolicyValue(in_channels=INT_IN_CHANNELS,
                     policy_size=INT_POLICY_SIZE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Flatten all parameters in a fixed, deterministic order
    flat_chunks = []
    for name, param in model.state_dict().items():
        w = param.detach().cpu().numpy().ravel()
        flat_chunks.append(w)
        print(f"{name:30s} -> {w.shape[0]} params")

    flat = np.concatenate(flat_chunks)
    print("Total params:", flat.size)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        # first line: number of floats, then one float per line
        f.write(str(flat.size) + "\n")
        for v in flat:
            f.write(f"{v:.8f}\n")

    print(f"Wrote weights to {OUT_PATH} ({flat.size} floats)")


if __name__ == "__main__":
    main()
