"""Neural network models for chess engine."""

from .resnet_policy_value import ResNetPolicyValue, create_model
from .tiny_value_net import TinyValueNet, create_tiny_model

__all__ = [
    'ResNetPolicyValue',
    'TinyValueNet',
    'create_model',
    'create_tiny_model',
]
