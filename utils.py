#!/usr/bin/env python3
"""
Utility functions for VAE training.
Contains filename generation, file management, and other helper functions.
"""

import os
from datetime import datetime
from typing import Optional


class FilenameManager:
    """Centralized filename generation and management for VAE training."""
    
    def __init__(self, config_name: str = 'unified'):
        """
        Initialize filename manager.
        
        Args:
            config_name: Configuration name used as prefix for all files
        """
        self.config_name = config_name
    
    def get_filename(self, file_type: str, epoch: Optional[int] = None, 
                    suffix: str = "", extension: str = "png", 
                    model_save_path: str = "model_weights") -> str:
        """
        Generate standardized filenames for all training files.
        
        Args:
            file_type: Type of file ('checkpoint', 'best_model', 'periodic_checkpoint', 
                      'generated_samples', 'reconstruction_samples', 'final_samples', 'log_dir')
            epoch: Epoch number (for epoch-specific files)
            suffix: Additional suffix (e.g., '_mid_epoch')
            extension: File extension ('png', 'pth')
            model_save_path: Directory for model files
            
        Returns:
            Complete file path
            
        Raises:
            ValueError: If file_type is not recognized
        """
        if file_type == 'checkpoint':
            return os.path.join("checkpoints", f"{self.config_name}_training_checkpoint.pth")
        elif file_type == 'best_model':
            return os.path.join(model_save_path, f"{self.config_name}_vae_best_model.pth")
        elif file_type == 'periodic_checkpoint':
            if epoch is None:
                raise ValueError("epoch is required for periodic_checkpoint")
            return os.path.join("checkpoints", f"{self.config_name}_checkpoint_epoch_{epoch:03d}.pth")
        elif file_type == 'generated_samples':
            if epoch is None:
                raise ValueError("epoch is required for generated_samples")
            return os.path.join("sample_images", f"{self.config_name}_generated_epoch_{epoch:03d}{suffix}.png")
        elif file_type == 'reconstruction_samples':
            if epoch is None:
                raise ValueError("epoch is required for reconstruction_samples")
            return os.path.join("sample_images", f"{self.config_name}_reconstruction_epoch_{epoch:03d}{suffix}.png")
        elif file_type == 'final_samples':
            return f"{self.config_name}_final_samples.png"
        elif file_type == 'log_dir':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"runs/{self.config_name}_{timestamp}"
        else:
            raise ValueError(f"Unknown file_type: {file_type}. "
                           f"Valid types: checkpoint, best_model, periodic_checkpoint, "
                           f"generated_samples, reconstruction_samples, final_samples, log_dir")
    
    def get_config_prefix(self) -> str:
        """Get the config prefix for file matching."""
        return f"{self.config_name}_"
    
    def find_matching_files(self, directories: list, file_extension: str = "png") -> list:
        """
        Find all files matching the current config prefix in specified directories.
        
        Args:
            directories: List of directories to search
            file_extension: File extension to match (e.g., 'png', 'pth')
            
        Returns:
            List of full file paths that match the config prefix
        """
        matching_files = []
        config_prefix = self.get_config_prefix()
        
        for directory in directories:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    if (file.endswith(f'.{file_extension}') and 
                        file.startswith(config_prefix)):
                        matching_files.append(os.path.join(directory, file))
        
        return matching_files
    
    def update_config_name(self, new_config_name: str):
        """Update the config name for filename generation."""
        self.config_name = new_config_name


def create_directories(directories: list):
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def safe_remove_file(file_path: str) -> bool:
    """
    Safely remove a file with error handling.
    
    Args:
        file_path: Path to file to remove
        
    Returns:
        True if file was removed successfully, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except OSError as e:
        print(f"  ⚠️  Could not delete {os.path.basename(file_path)}: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_file_info(file_path: str) -> dict:
    """
    Get file information including size and modification time.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    if not os.path.exists(file_path):
        return {"exists": False}
    
    stat = os.stat(file_path)
    return {
        "exists": True,
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "modified_str": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    }


def validate_config_structure(config: dict, required_keys: list) -> list:
    """
    Validate that a configuration dictionary contains all required keys.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key names
        
    Returns:
        List of missing keys (empty if all present)
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    return missing_keys


def deep_merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Deep merge two dictionaries, with dict2 values taking precedence.
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge in (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def get_nested_value(data: dict, key_path: str, default=None, separator: str = '.'):
    """
    Get a value from a nested dictionary using dot notation.
    
    Args:
        data: Dictionary to search in
        key_path: Dot-separated path to the key (e.g., 'loss_config.mse_weight')
        default: Default value if key not found
        separator: Separator character for key path
        
    Returns:
        Value at the specified path or default
    """
    keys = key_path.split(separator)
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(data: dict, key_path: str, value, separator: str = '.'):
    """
    Set a value in a nested dictionary using dot notation.
    
    Args:
        data: Dictionary to modify
        key_path: Dot-separated path to the key (e.g., 'loss_config.mse_weight')
        value: Value to set
        separator: Separator character for key path
    """
    keys = key_path.split(separator)
    current = data
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def print_config_summary(config: dict, title: str = "Configuration Summary") -> None:
    """
    Print a formatted summary of a configuration dictionary.
    
    Args:
        config: Configuration dictionary to summarize
        title: Title for the summary
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    def print_dict(data, indent=0):
        for key, value in data.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    print_dict(config)