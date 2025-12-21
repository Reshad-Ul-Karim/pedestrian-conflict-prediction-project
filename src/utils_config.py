"""
Configuration loader and manager
Handles loading and merging of YAML configuration files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class Config:
    """Configuration manager"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize config from dictionary
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        
    def __getattr__(self, name: str) -> Any:
        """Get config value by attribute access"""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Get config value by dict access"""
        value = self._config[key]
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return copy.deepcopy(self._config)
    
    def update(self, other: Dict[str, Any]):
        """Update configuration"""
        self._config.update(other)
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def load_configs(*config_paths: str) -> Config:
    """
    Load and merge multiple configuration files
    
    Args:
        *config_paths: Paths to YAML config files
        
    Returns:
        Merged Config object
    """
    merged_config = {}
    
    for path in config_paths:
        config = load_config(path)
        merged_config.update(config.to_dict())
    
    return Config(merged_config)


def save_config(config: Config, output_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Config object
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

