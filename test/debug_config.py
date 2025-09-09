#!/usr/bin/env python3
"""Debug script to check the loaded configuration."""

from config_loader import ConfigLoader

def debug_config():
    """Debug the configuration loading."""
    print("🔍 Debugging Configuration Loading")
    print("=" * 50)
    
    # Load the same configuration as the training script
    config_loader = ConfigLoader("config/config_unified.json")
    config = config_loader.get_config(
        loss_preset="high_quality",
        training_preset="fast_high_quality_training", 
        model_preset="fast_high_quality",
        dataset_preset="full"
    )
    
    print(f"✅ Configuration loaded:")
    print(f"  • Input size: {config['input_img_size']}x{config['input_img_size']}")
    print(f"  • Embedding size: {config['embedding_size']}")
    print(f"  • Max epochs: {config['max_epoch']}")
    print(f"  • Batch size: {config['batch_size']}")
    print(f"  • Learning rate: {config['lr']}")
    
    # Print loss configuration
    loss_config = config.get('loss_config', {})
    print(f"\n🎯 Loss Configuration:")
    print(f"  • use_perceptual_loss: {loss_config.get('use_perceptual_loss', 'NOT SET')}")
    print(f"  • use_lpips: {loss_config.get('use_lpips', 'NOT SET')}")
    print(f"  • mse_weight: {loss_config.get('mse_weight', 'NOT SET')}")
    print(f"  • l1_weight: {loss_config.get('l1_weight', 'NOT SET')}")
    print(f"  • perceptual_weight: {loss_config.get('perceptual_weight', 'NOT SET')}")
    print(f"  • lpips_weight: {loss_config.get('lpips_weight', 'NOT SET')}")
    
    print(f"\n📋 Full loss_config:")
    print(loss_config)
    
    print(f"\n🔍 Top-level config keys:")
    for key in sorted(config.keys()):
        if not key.startswith('_'):
            print(f"  • {key}: {config[key]}")

if __name__ == "__main__":
    debug_config()
