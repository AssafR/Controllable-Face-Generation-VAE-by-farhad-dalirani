import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Demo script showing the unified configuration system.
"""

from config_loader import ConfigLoader
import os

def demo_config_system():
    """Demonstrate the unified configuration system."""
    
    print("🎯 Unified VAE Configuration System Demo")
    print("=" * 60)
    
    # Initialize loader with correct config path
    # Try relative path first (when running from test/ directory)
    config_path = "../config/config_unified.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = "config/config_unified.json"
    
    loader = ConfigLoader(config_path)
    
    # Show all available presets
    print("\n📋 Available Presets:")
    loader.print_all_presets()
    
    # Demo different configurations
    print("\n🔧 Configuration Examples:")
    print("=" * 40)
    
    configs_to_demo = [
        ("mse_only", "quick_test", "small", "Quick baseline test"),
        ("mse_l1", "standard_training", "medium", "Recommended for most users"),
        ("perceptual", "extended_training", "large", "High quality generation"),
        ("lpips", "extended_training", "large", "Best possible quality")
    ]
    
    for loss, training, model, description in configs_to_demo:
        print(f"\n📝 {description}")
        print("-" * 30)
        
        config = loader.get_config(
            loss_preset=loss,
            training_preset=training,
            model_preset=model
        )
        
        print(f"Config name: {config['config_name']}")
        print(f"Model: {config['embedding_size']}D latent, {config['num_channels']} channels")
        print(f"Training: {config['max_epoch']} epochs, batch {config['batch_size']}, lr {config['lr']}")
        
        loss_config = config['loss_config']
        active_losses = [k for k, v in loss_config.items() if k.startswith('use_') and v]
        print(f"Loss functions: {', '.join(active_losses)}")
        
        if any(loss_config.get(f'use_{loss}', False) for loss in ['mse', 'l1', 'perceptual_loss', 'lpips']):
            weights = []
            for loss_type in ['mse', 'l1', 'perceptual', 'lpips']:
                if loss_config.get(f'use_{loss_type}', False):
                    weight = loss_config.get(f'{loss_type}_weight', 0)
                    weights.append(f"{loss_type}={weight}")
            print(f"Loss weights: {', '.join(weights)}")
    
    print("\n🚀 Usage Commands:")
    print("=" * 30)
    print("# List all presets")
    print("uv run train_VAE_unified.py --list-presets")
    print()
    print("# Train with specific configuration")
    print("uv run train_VAE_unified.py --loss mse_l1 --training standard_training --model medium")
    print()
    print("# Quick test (1 epoch)")
    print("uv run train_VAE_unified.py --loss mse_only --training quick_test --model small")
    print()
    print("# High quality (best results)")
    print("uv run train_VAE_unified.py --loss lpips --training extended_training --model large")
    
    print("\n✅ Demo complete! The unified system is much easier to manage.")

if __name__ == "__main__":
    demo_config_system()
