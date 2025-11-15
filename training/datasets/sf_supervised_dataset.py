# training/datasets/sf_supervised_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset

class SfSupervisedDataset(Dataset):
    """
    Supervised dataset backed by the Stockfish-labelled .npz file you generated.
    """

    def __init__(
        self,
        path: str,
        boards_key: str = "X",
        policy_key: str = "y_policy_best",
        value_key: str = "game_result",
    ):
        data = np.load(path)

        self.boards = torch.from_numpy(data[boards_key]).float()      # (N, 18, 8, 8)
        self.policy = torch.from_numpy(data[policy_key])              # (N,)
        self.value  = torch.from_numpy(data[value_key]).float()       # (N,)

        # optional: keep some extras around if you want them later
        self._raw = data

    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        # shape: (C, 8, 8), scalar policy index, scalar value in [-1,1]
        return self.boards[idx], self.policy[idx], self.value[idx]
