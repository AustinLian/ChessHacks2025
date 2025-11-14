import torch
from torch.utils.data import Dataset
import numpy as np


class SelfPlayDataset(Dataset):
    """
    Dataset for training from self-play games.
    
    Self-play positions include search policy (π) from MCTS/search,
    not just the played move.
    """
    
    def __init__(self, npz_path, transform=None):
        """
        Args:
            npz_path: Path to self-play buffer NPZ file
            transform: Optional transforms
        """
        data = np.load(npz_path)
        self.planes = data['planes']  # (N, 27, 8, 8)
        self.policies = data['policies']  # (N, 4672) - MCTS policy
        self.values = data['values']  # (N,) - game outcome
        self.transform = transform
        
    def __len__(self):
        return len(self.planes)
    
    def __getitem__(self, idx):
        planes = torch.from_numpy(self.planes[idx]).float()
        policy = torch.from_numpy(self.policies[idx]).float()
        value = torch.tensor(self.values[idx], dtype=torch.float32)
        
        if self.transform:
            planes = self.transform(planes)
        
        return planes, policy, value


class RollingBuffer:
    """
    Rolling buffer for self-play positions.
    
    Maintains a fixed-size buffer of recent self-play games,
    evicting oldest positions when full.
    """
    
    def __init__(self, max_size=500000, save_path=None):
        """
        Args:
            max_size: Maximum number of positions to store
            save_path: Path to save buffer periodically
        """
        self.max_size = max_size
        self.save_path = save_path
        self.planes = []
        self.policies = []
        self.values = []
        
    def add_game(self, game_positions, game_policies, game_values):
        """
        Add a complete game to the buffer.
        
        Args:
            game_positions: List of board planes (27, 8, 8)
            game_policies: List of search policies (4672,)
            game_values: List of game outcomes from each position's perspective
        """
        for pos, pol, val in zip(game_positions, game_policies, game_values):
            self.planes.append(pos)
            self.policies.append(pol)
            self.values.append(val)
        
        # Evict oldest if over capacity
        if len(self.planes) > self.max_size:
            overflow = len(self.planes) - self.max_size
            self.planes = self.planes[overflow:]
            self.policies = self.policies[overflow:]
            self.values = self.values[overflow:]
    
    def save(self, path=None):
        """Save buffer to NPZ file."""
        path = path or self.save_path
        if path is None:
            raise ValueError("No save path provided")
        
        np.savez_compressed(
            path,
            planes=np.array(self.planes),
            policies=np.array(self.policies),
            values=np.array(self.values)
        )
    
    def load(self, path):
        """Load buffer from NPZ file."""
        data = np.load(path)
        self.planes = list(data['planes'])
        self.policies = list(data['policies'])
        self.values = list(data['values'])
    
    def __len__(self):
        return len(self.planes)
    
    def is_empty(self):
        return len(self.planes) == 0


def collate_fn(batch):
    """Custom collate function for self-play batching."""
    planes, policies, values = zip(*batch)
    
    planes = torch.stack(planes, dim=0)
    policies = torch.stack(policies, dim=0)
    values = torch.stack(values, dim=0)
    
    return planes, policies, values
