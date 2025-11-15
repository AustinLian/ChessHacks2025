import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

POLICY_DIM = 64 * 64 * 5  # 20,480

class SFNPZDataset(Dataset):
    """
    Dataset for Stockfish-labeled NPZ.

    Trains on:
      - policy_idx : best-move index (int)
      - value_before      : scaled cp_before
      - value_after_best  : scaled cp_after_best
      - delta_target      : scaled delta_cp
    """

    def __init__(
        self,
        npz_path: str,
        cp_scale: float = 400.0,
        delta_scale: float = 200.0,
    ):
        data = np.load(npz_path)

        # Required fields from your generator
        self.X = data["X"]                       # (N, 18, 8, 8)
        self.y_policy_best = data["y_policy_best"]  # (N,)
        self.cp_before = data["cp_before"]       # (N,)
        self.cp_after_best = data["cp_after_best"]  # (N,)
        self.delta_cp = data["delta_cp"]         # (N,)

        # Precompute regression targets, scaled into [-1, 1]
        self.value_before = np.tanh(self.cp_before / cp_scale).astype(np.float32)
        self.value_after_best = np.tanh(self.cp_after_best / cp_scale).astype(np.float32)
        self.delta_target = np.tanh(self.delta_cp / delta_scale).astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        planes = torch.from_numpy(self.X[idx]).float()  # (18, 8, 8)
        policy_idx = torch.tensor(int(self.y_policy_best[idx]), dtype=torch.long)

        value_before = torch.tensor(self.value_before[idx], dtype=torch.float32)
        value_after_best = torch.tensor(self.value_after_best[idx], dtype=torch.float32)
        delta_target = torch.tensor(self.delta_target[idx], dtype=torch.float32)

        return {
            "planes": planes,
            "policy_idx": policy_idx,
            "value_before": value_before,
            "value_after_best": value_after_best,
            "delta_target": delta_target,
        }


def collate_sf(batch):
    """
    Collate dict -> dict of batched tensors.
    """
    out = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
