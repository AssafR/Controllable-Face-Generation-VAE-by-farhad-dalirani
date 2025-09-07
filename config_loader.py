#!/usr/bin/env python3
"""
Unified configuration loader for VAE training.
Handles the unified config format with presets and combinations.
"""

import json
import os
from typing import Dict, Any, Optional

class ConfigLoader:
    """Load and manage VAE configurations from unified config file."""
    
    def __init__(self, config_path: str = "config/config_unified.json"):
        """Initialize with unified config file."""
        self.config_path = config_path
        self.unified_config = self._load_unified_config()
    
    def _load_unified_config(self) -> Dict[str, Any]:
        """Load the unified configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get_config(self, 
                   loss_preset: str = "mse_l1",
                   training_preset: str = "standard_training",
                   model_preset: str = "medium",
                   dataset_preset: str = "full",
                   custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a complete configuration by combining presets.
        
        Args:
            loss_preset: Loss function configuration preset
            training_preset: Training configuration preset  
            model_preset: Model architecture preset
            dataset_preset: Dataset size preset
            custom_overrides: Custom values to override any preset values
            
        Returns:
            Complete configuration dictionary
        """
        
        # Start with base config
        config = self.unified_config["base_config"].copy()
        
        # Apply model preset
        if model_preset in self.unified_config["model_presets"]:
            model_config = self.unified_config["model_presets"][model_preset]
            # Filter out documentation fields
            model_config = {k: v for k, v in model_config.items() if not k.startswith('_')}
            config.update(model_config)
        else:
            print(f"⚠️  Model preset '{model_preset}' not found, using base config")
        
        # Apply training preset
        if training_preset in self.unified_config["training_presets"]:
            training_config = self.unified_config["training_presets"][training_preset]
            # Filter out documentation fields
            training_config = {k: v for k, v in training_config.items() if not k.startswith('_')}
            config.update(training_config)
        else:
            print(f"⚠️  Training preset '{training_preset}' not found, using base config")
        
        # Apply loss preset
        if loss_preset in self.unified_config["loss_presets"]:
            loss_config = self.unified_config["loss_presets"][loss_preset]
            # Filter out documentation fields
            loss_config = {k: v for k, v in loss_config.items() if not k.startswith('_')}
            
            # Separate loss-specific config from general config
            loss_specific_keys = ['use_mse', 'use_l1', 'use_perceptual_loss', 'use_lpips', 
                                'mse_weight', 'l1_weight', 'perceptual_weight', 'generation_weight', 'lpips_weight']
            
            # Put loss-specific keys in loss_config
            config["loss_config"] = {k: v for k, v in loss_config.items() if k in loss_specific_keys}
            
            # Put general config keys (like perceptual scheduling) at top level
            general_keys = [k for k in loss_config.keys() if k not in loss_specific_keys]
            for key in general_keys:
                config[key] = loss_config[key]
        else:
            print(f"⚠️  Loss preset '{loss_preset}' not found, using base config")
            config["loss_config"] = {}
        
        # Apply dataset preset
        if dataset_preset in self.unified_config["dataset_presets"]:
            dataset_config = self.unified_config["dataset_presets"][dataset_preset]
            # Filter out documentation fields
            dataset_config = {k: v for k, v in dataset_config.items() if not k.startswith('_')}
            config.update(dataset_config)
        else:
            print(f"⚠️  Dataset preset '{dataset_preset}' not found, using base config")
        
        # Apply custom overrides
        if custom_overrides:
            config.update(custom_overrides)
        
        # Add metadata
        config["config_name"] = f"{loss_preset}_{training_preset}_{model_preset}_{dataset_preset}"
        config["model_name"] = config["config_name"]
        
        return config
    
    def list_presets(self) -> Dict[str, list]:
        """List all available presets."""
        def filter_presets(presets_dict):
            return [k for k in presets_dict.keys() if not k.startswith('_')]
        
        return {
            "loss_presets": filter_presets(self.unified_config["loss_presets"]),
            "training_presets": filter_presets(self.unified_config["training_presets"]),
            "model_presets": filter_presets(self.unified_config["model_presets"]),
            "dataset_presets": filter_presets(self.unified_config["dataset_presets"])
        }
    
    def get_preset_info(self, preset_type: str, preset_name: str) -> Dict[str, str]:
        """Get information about a specific preset."""
        if preset_type not in self.unified_config:
            return {"error": f"Preset type '{preset_type}' not found"}
        
        if preset_name not in self.unified_config[preset_type]:
            return {"error": f"Preset '{preset_name}' not found in {preset_type}"}
        
        preset = self.unified_config[preset_type][preset_name]
        if not isinstance(preset, dict):
            return {"error": f"Preset '{preset_name}' is not a dictionary"}
        
        return {k: v for k, v in preset.items() if k.startswith('_')}
    
    def print_preset_info(self, preset_type: str, preset_name: str):
        """Print detailed information about a preset."""
        info = self.get_preset_info(preset_type, preset_name)
        
        if "error" in info:
            print(f"❌ {info['error']}")
            return
        
        print(f"\n📋 {preset_type.title()} Preset: {preset_name}")
        print("-" * 50)
        
        for key, value in info.items():
            if key.startswith('_'):
                clean_key = key[1:].replace('_', ' ').title()
                print(f"{clean_key}: {value}")
    
    def print_all_presets(self):
        """Print information about all available presets."""
        print("🎯 Available VAE Configuration Presets")
        print("=" * 60)
        
        for preset_type, presets in self.list_presets().items():
            print(f"\n📁 {preset_type.replace('_', ' ').title()}:")
            print("-" * 30)
            
            for preset_name in presets:
                info = self.get_preset_info(preset_type, preset_name)
                if "error" in info:
                    print(f"  • {preset_name}: {info['error']}")
                else:
                    description = info.get('_description', 'No description')
                    print(f"  • {preset_name}: {description}")

def main():
    """Demo the configuration loader."""
    loader = ConfigLoader()
    
    # Print all available presets
    loader.print_all_presets()
    
    # Example: Get a specific configuration
    print("\n" + "=" * 60)
    print("🔧 Example Configuration Generation")
    print("=" * 60)
    
    config = loader.get_config(
        loss_preset="mse_l1",
        training_preset="standard_training", 
        model_preset="medium"
    )
    
    print(f"Generated config: {config['config_name']}")
    print(f"Loss config: {config['loss_config']}")
    print(f"Model size: {config['embedding_size']} embedding, {config['num_channels']} channels")
    print(f"Training: {config['max_epoch']} epochs, batch size {config['batch_size']}")

if __name__ == "__main__":
    main()
