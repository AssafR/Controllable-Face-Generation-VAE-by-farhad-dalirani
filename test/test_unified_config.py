import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Quick test of the unified configuration system.
"""

from config_loader import ConfigLoader
import torch
from variation_autoencoder_improved import VAE_pt

def test_config_system():
    """Test the unified configuration system with a minimal example."""
    
    print("🧪 Testing Unified Configuration System")
    print("=" * 50)
    
    # Initialize loader with correct config path
    # Try relative path first (when running from test/ directory)
    config_path = "../config/config_unified.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = "config/config_unified.json"
    
    loader = ConfigLoader(config_path)
    
    # Test different configurations
    configs_to_test = [
        ("mse_only", "quick_test", "small"),
        ("mse_l1", "quick_test", "medium"),
        ("perceptual", "quick_test", "large")
    ]
    
    for loss, training, model in configs_to_test:
        print(f"\n🔧 Testing: {loss} + {training} + {model}")
        print("-" * 40)
        
        # Get configuration
        config = loader.get_config(
            loss_preset=loss,
            training_preset=training,
            model_preset=model
        )
        
        print(f"Config name: {config['config_name']}")
        print(f"Embedding size: {config['embedding_size']}")
        print(f"Channels: {config['num_channels']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Epochs: {config['max_epoch']}")
        
        # Test model creation
        try:
            model_vae = VAE_pt(
                input_img_size=config["input_img_size"],
                embedding_size=config["embedding_size"],
                num_channels=config["num_channels"],
                beta=config["beta"],
                loss_config=config.get("loss_config", {})
            )
            
            # Test forward pass with dummy data
            dummy_input = torch.randn(2, 3, 64, 64)
            with torch.no_grad():
                emb_mean, emb_log_var, reconst = model_vae(dummy_input)
            
            print(f"✅ Model created successfully!")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Latent shape: {emb_mean.shape}")
            print(f"   Output shape: {reconst.shape}")
            print(f"   Loss config: {config['loss_config']}")
            
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
    
    print(f"\n🎯 Configuration System Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_config_system()
