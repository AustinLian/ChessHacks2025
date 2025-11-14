import torch
import os
import shutil
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, metrics, filepath, is_best=False):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    
    # Save best model separately
    if is_best:
        best_path = Path(filepath).parent / 'best_model.pt'
        shutil.copyfile(filepath, best_path)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load checkpoint from file.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model
        optimizer: Optional PyTorch optimizer
        
    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
    }


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5, pattern='checkpoint_epoch_*.pt'):
    """
    Remove old checkpoints, keeping only the most recent N.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        pattern: Glob pattern for checkpoint files
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob(pattern))
    
    # Remove older checkpoints
    if len(checkpoints) > keep_last_n:
        for ckpt in checkpoints[:-keep_last_n]:
            os.remove(ckpt)


def save_model_for_inference(model, filepath):
    """
    Save model for inference (model only, no optimizer).
    
    Args:
        model: PyTorch model
        filepath: Path to save model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model_for_inference(model, filepath):
    """
    Load model for inference.
    
    Args:
        model: PyTorch model
        filepath: Path to model file
    """
    state_dict = torch.load(filepath, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
