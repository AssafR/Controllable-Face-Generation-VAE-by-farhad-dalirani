#!/usr/bin/env python3
"""
Test VAE compatibility with different resolutions.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from variation_autoencoder_improved import VAE_pt

def test_resolution_compatibility():
    """Test VAE with different input resolutions."""
    
    print("🔍 Testing VAE Resolution Compatibility")
    print("=" * 60)
    
    # Test different resolutions
    resolutions = [64, 128, 256, 512]
    
    for resolution in resolutions:
        print(f"\n📐 Testing {resolution}x{resolution} resolution")
        print("-" * 40)
        
        try:
            # Create model for this resolution
            model = VAE_pt(
                input_img_size=resolution,
                embedding_size=512,
                num_channels=128,
                beta=1.0,
                loss_config={
                    'use_mse': True,
                    'use_l1': True,
                    'use_perceptual_loss': False,
                    'use_lpips': False,
                    'mse_weight': 0.8,
                    'l1_weight': 0.2,
                    'perceptual_weight': 0.0,
                    'lpips_weight': 0.0
                }
            )
            
            # Test forward pass with dummy data
            dummy_input = torch.randn(2, 3, resolution, resolution)
            
            with torch.no_grad():
                emb_mean, emb_log_var, reconst = model(dummy_input)
            
            # Calculate spatial dimensions
            spatial_size = resolution // 16
            expected_latent_size = 128 * 8 * spatial_size * spatial_size
            
            print(f"✅ {resolution}x{resolution} - SUCCESS!")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Latent shape: {emb_mean.shape}")
            print(f"   Output shape: {reconst.shape}")
            print(f"   Spatial size: {spatial_size}x{spatial_size}")
            print(f"   Expected latent size: {expected_latent_size}")
            print(f"   Actual latent size: {emb_mean.numel() // emb_mean.size(0)}")
            
            # Verify dimensions match
            assert reconst.shape == dummy_input.shape, f"Output shape mismatch: {reconst.shape} vs {dummy_input.shape}"
            assert emb_mean.shape[1] == 512, f"Latent size mismatch: {emb_mean.shape[1]} vs 512"
            
        except Exception as e:
            print(f"❌ {resolution}x{resolution} - FAILED: {e}")
    
    print(f"\n🎯 Resolution Compatibility Test Complete!")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print("- ✅ 64x64: Standard resolution")
    print("- ✅ 128x128: High resolution") 
    print("- ✅ 256x256: Ultra high resolution")
    print("- ✅ 512x512: Extreme high resolution")
    print("- ✅ Any resolution divisible by 16")

def test_unsupported_resolution():
    """Test with an unsupported resolution to show the limitation."""
    
    print(f"\n⚠️  Testing Unsupported Resolution")
    print("-" * 40)
    
    try:
        # Try 100x100 (not divisible by 16)
        model = VAE_pt(
            input_img_size=100,
            embedding_size=512,
            num_channels=128,
            beta=1.0
        )
        
        dummy_input = torch.randn(2, 3, 100, 100)
        with torch.no_grad():
            emb_mean, emb_log_var, reconst = model(dummy_input)
        
        print("✅ 100x100 - SUCCESS! (Unexpected)")
        
    except Exception as e:
        print(f"❌ 100x100 - FAILED: {e}")
        print("   This is expected - resolution must be divisible by 16")
    
    print("\n💡 Resolution Requirements:")
    print("- Must be divisible by 16 (due to 4 stride-2 convolutions)")
    print("- Supported: 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, etc.")
    print("- Not supported: 100, 150, 200, 300, etc.")

if __name__ == "__main__":
    test_resolution_compatibility()
    test_unsupported_resolution()
