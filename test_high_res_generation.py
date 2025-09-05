#!/usr/bin/env python3
"""
Test high-resolution image generation from the trained model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import configure_gpu, display_image_grid
import os

def test_high_res_generation():
    """Test generating high-resolution images."""
    
    print("🎨 Testing High-Resolution Image Generation")
    print("=" * 50)
    
    # Configure GPU
    device = configure_gpu()
    
    # Load the trained high-resolution model
    model_path = "model_weights/mse_l1_quick_test_high_res_tiny.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📁 Loading model: {model_path}")
    
    # Create model with high-resolution configuration
    model = VAE_pt(
        input_img_size=128,
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
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    print(f"   Input size: 128x128")
    print(f"   Embedding size: 512")
    print(f"   Channels: 128")
    
    # Generate random images
    print("\n🎲 Generating random high-resolution images...")
    
    with torch.no_grad():
        # Generate 4 random images
        z = torch.randn(4, 512).to(device)
        generated = model.dec(z)
        generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
        
        print(f"Generated shape: {generated_np.shape}")
        print(f"Generated range: [{generated_np.min():.4f}, {generated_np.max():.4f}]")
        print(f"Generated mean: {generated_np.mean():.4f}")
        print(f"Generated std: {generated_np.std():.4f}")
        
        # Display images using improved grid display
        titles = [f"High-Res Generated {i+1}" for i in range(4)]
        
        display_image_grid(generated_np, 
                          titles=titles,
                          max_cols=2, 
                          figsize=(12, 8),
                          save_path="high_res_generated_samples.png")
        
        print("✅ High-resolution images saved to: high_res_generated_samples.png")
    
    print("\n🎯 High-Resolution Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_high_res_generation()
