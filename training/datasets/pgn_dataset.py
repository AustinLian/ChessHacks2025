import torch
from torch.utils.data import Dataset
import numpy as np
import chess
import chess.pgn
import io


class PGNDataset(Dataset):
    """
    Dataset for training from PGN files.
    
    Converts chess positions from PGN games into (planes, move_index, result) tuples.
    """
    
    def __init__(self, npz_path, transform=None):
        """
        Args:
            npz_path: Path to preprocessed NPZ file
            transform: Optional transforms to apply
        """
        data = np.load(npz_path)
        self.planes = data['planes']  # (N, 27, 8, 8)
        self.move_indices = data['move_indices']  # (N,)
        self.results = data['results']  # (N,)
        self.transform = transform
        
    def __len__(self):
        return len(self.planes)
    
    def __getitem__(self, idx):
        planes = torch.from_numpy(self.planes[idx]).float()
        move_idx = torch.tensor(self.move_indices[idx], dtype=torch.long)
        result = torch.tensor(self.results[idx], dtype=torch.float32)
        
        if self.transform:
            planes = self.transform(planes)
        
        return planes, move_idx, result


class StreamingPGNDataset(Dataset):
    """
    Dataset that streams PGN data without loading everything into memory.
    
    Useful for very large PGN collections.
    """
    
    def __init__(self, pgn_paths, max_games=None):
        """
        Args:
            pgn_paths: List of paths to PGN files
            max_games: Maximum number of games to load (None = all)
        """
        self.pgn_paths = pgn_paths if isinstance(pgn_paths, list) else [pgn_paths]
        self.max_games = max_games
        self.positions = []
        self._load_positions()
        
    def _load_positions(self):
        """Load positions from PGN files."""
        # TODO: Implement streaming PGN parsing
        # For now, placeholder
        pass
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        # TODO: Return (planes, move_idx, result)
        raise NotImplementedError()


def collate_fn(batch):
    """Custom collate function for batching."""
    planes, move_indices, results = zip(*batch)
    
    planes = torch.stack(planes, dim=0)
    move_indices = torch.stack(move_indices, dim=0)
    results = torch.stack(results, dim=0)
    
    return planes, move_indices, results
