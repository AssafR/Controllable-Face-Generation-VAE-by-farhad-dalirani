#!/usr/bin/env python3
"""
Test edge case resolutions to understand the exact limitations.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from variation_autoencoder_improved import VAE_pt

def test_edge_cases():
    """Test edge case resolutions."""
    
    print("🔍 Testing Edge Case Resolutions")
    print("=" * 50)
    
    # Test various resolutions
    test_cases = [
        (32, "Very small"),
        (48, "Small"),
        (64, "Standard"),
        (80, "Non-standard"),
        (96, "Non-standard"),
        (112, "Non-standard"),
        (128, "High-res"),
        (144, "Non-standard"),
        (160, "Non-standard"),
        (192, "Non-standard"),
        (224, "Non-standard"),
        (256, "Ultra high-res"),
        (320, "Non-standard"),
        (384, "Non-standard"),
        (448, "Non-standard"),
        (512, "Extreme high-res"),
        (640, "Non-standard"),
        (768, "Non-standard"),
        (1024, "Massive"),
    ]
    
    for resolution, description in test_cases:
        print(f"\n📐 Testing {resolution}x{resolution} ({description})")
        print("-" * 40)
        
        try:
            # Create model for this resolution
            model = VAE_pt(
                input_img_size=resolution,
                embedding_size=512,
                num_channels=128,
                beta=1.0
            )
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, resolution, resolution)
            
            with torch.no_grad():
                emb_mean, emb_log_var, reconst = model(dummy_input)
            
            # Calculate spatial dimensions
            spatial_size = resolution // 16
            expected_latent_size = 128 * 8 * spatial_size * spatial_size
            
            print(f"✅ SUCCESS!")
            print(f"   Spatial size: {spatial_size}x{spatial_size}")
            print(f"   Expected latent size: {expected_latent_size:,}")
            print(f"   Output shape: {reconst.shape}")
            
            # Check if output matches input
            if reconst.shape == dummy_input.shape:
                print(f"   ✅ Shape match: {reconst.shape}")
            else:
                print(f"   ❌ Shape mismatch: {reconst.shape} vs {dummy_input.shape}")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    print(f"\n🎯 Edge Case Testing Complete!")
    print("=" * 50)

def test_memory_usage():
    """Test memory usage for different resolutions."""
    
    print(f"\n💾 Testing Memory Usage")
    print("=" * 30)
    
    resolutions = [64, 128, 256, 512]
    
    for resolution in resolutions:
        print(f"\n📐 {resolution}x{resolution}:")
        
        try:
            model = VAE_pt(input_img_size=resolution, embedding_size=512, num_channels=128)
            
            # Calculate memory usage
            spatial_size = resolution // 16
            latent_size = 128 * 8 * spatial_size * spatial_size
            
            print(f"   Spatial size: {spatial_size}x{spatial_size}")
            print(f"   Latent size: {latent_size:,}")
            print(f"   Memory per image: ~{latent_size * 4 / 1024 / 1024:.1f} MB")
            
            # Test with small batch
            dummy_input = torch.randn(1, 3, resolution, resolution)
            with torch.no_grad():
                emb_mean, emb_log_var, reconst = model(dummy_input)
            
            print(f"   ✅ Works with batch size 1")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    test_edge_cases()
    test_memory_usage()
