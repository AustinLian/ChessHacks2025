# training/inspect_sf_npz.py
import numpy as np
from pathlib import Path

NPZ_PATH = Path("training/data/processed/sf_supervised_dataset1519.npz")

def main():
    data = np.load(NPZ_PATH)
    print("Keys:", data.files)
    for k in data.files:
        arr = data[k]
        print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")

if __name__ == "__main__":
    main()
pu