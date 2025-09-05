import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Demo script showing dataset subset functionality.
"""

from config_loader import ConfigLoader
from utilities_pytorch import get_split_data
import time
import os

def demo_dataset_subsets():
    """Demonstrate dataset subset functionality."""
    
    print("📊 Dataset Subset Demo")
    print("=" * 50)
    
    # Initialize loader with correct config path
    # Try relative path first (when running from test/ directory)
    config_path = "../config/config_unified.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = "config/config_unified.json"
    
    loader = ConfigLoader(config_path)
    
    # Test different dataset sizes
    dataset_presets = ["tiny", "small", "medium", "large", "full"]
    
    for dataset_preset in dataset_presets:
        print(f"\n🔧 Testing dataset preset: {dataset_preset}")
        print("-" * 40)
        
        # Get configuration
        config = loader.get_config(
            loss_preset="mse_only",
            training_preset="quick_test",
            model_preset="small",
            dataset_preset=dataset_preset
        )
        
        print(f"Config: {config['config_name']}")
        print(f"Dataset subset: {config.get('dataset_subset', 'Full dataset')}")
        
        # Test dataset loading
        start_time = time.time()
        try:
            train_data, val_data = get_split_data(config)
            load_time = time.time() - start_time
            
            print(f"✅ Dataset loaded successfully!")
            print(f"   Train samples: {len(train_data)}")
            print(f"   Val samples: {len(val_data)}")
            print(f"   Load time: {load_time:.2f} seconds")
            
            # Show sample from dataset
            if len(train_data) > 0:
                sample = train_data[0]
                print(f"   Sample shape: {sample.shape}")
                print(f"   Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
            
        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
    
    print(f"\n🎯 Dataset Subset Demo Complete!")
    print("=" * 50)
    
    print("\n💡 Usage Examples:")
    print("-" * 30)
    print("# Quick test with tiny dataset (1000 samples)")
    print("uv run train_VAE_unified.py --loss mse_only --training quick_test --model small --dataset tiny")
    print()
    print("# Development with small dataset (10000 samples)")
    print("uv run train_VAE_unified.py --loss mse_l1 --training fast_training --model medium --dataset small")
    print()
    print("# Production with full dataset")
    print("uv run train_VAE_unified.py --loss perceptual --training extended_training --model large --dataset full")

if __name__ == "__main__":
    demo_dataset_subsets()
