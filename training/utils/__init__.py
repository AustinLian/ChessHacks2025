"""Utility functions for training."""

from .fen_encoding import fen_to_planes, planes_to_fen, augment_planes
from .move_encoding import (
    move_to_index,
    index_to_move,
    create_policy_mask,
    moves_to_policy_vector,
    policy_vector_to_moves
)
from .logging_utils import setup_logger, log_metrics, format_time, AverageMeter
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
    save_model_for_inference,
    load_model_for_inference
)
from .config import (
    load_config,
    save_config,
    merge_configs,
    get_nested,
    set_nested
)

__all__ = [
    # FEN encoding
    'fen_to_planes',
    'planes_to_fen',
    'augment_planes',
    # Move encoding
    'move_to_index',
    'index_to_move',
    'create_policy_mask',
    'moves_to_policy_vector',
    'policy_vector_to_moves',
    # Logging
    'setup_logger',
    'log_metrics',
    'format_time',
    'AverageMeter',
    # Checkpointing
    'save_checkpoint',
    'load_checkpoint',
    'cleanup_old_checkpoints',
    'save_model_for_inference',
    'load_model_for_inference',
    # Config
    'load_config',
    'save_config',
    'merge_configs',
    'get_nested',
    'set_nested',
]
