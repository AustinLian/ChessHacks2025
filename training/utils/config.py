import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(*configs):
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dictionaries
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    for config in configs:
        result = deep_update(result, config)
    return result


def deep_update(base_dict, update_dict):
    """
    Recursively update nested dictionary.
    
    Args:
        base_dict: Base dictionary
        update_dict: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    result = base_dict.copy()
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def get_nested(config, key_path, default=None):
    """
    Get value from nested config using dot notation.
    
    Example:
        get_nested(config, 'training.supervised.batch_size')
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        default: Default value if key not found
        
    Returns:
        Value at key path or default
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_nested(config, key_path, value):
    """
    Set value in nested config using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
