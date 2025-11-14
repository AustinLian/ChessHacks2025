import logging
import sys
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(logger, metrics, epoch=None, step=None):
    """
    Log training metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric name -> value
        epoch: Optional epoch number
        step: Optional step number
    """
    prefix = ""
    if epoch is not None:
        prefix += f"Epoch {epoch}"
    if step is not None:
        prefix += f" Step {step}"
    
    if prefix:
        prefix += " - "
    
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{prefix}{metrics_str}")


def format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class AverageMeter:
    """Compute and store average and current value."""
    
    def __init__(self, name=None):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        if self.name:
            return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"
        return f"{self.val:.4f} (avg: {self.avg:.4f})"
