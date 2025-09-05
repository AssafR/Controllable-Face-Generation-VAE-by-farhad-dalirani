#!/usr/bin/env python3
"""
Test script specifically for evaluating reconstruction quality.
Shows original vs reconstructed images in an improved grid layout.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from variation_autoencoder_pytorch import VAE_pt
from utilities_pytorch import get_split_data, configure_gpu, display_comparison_grid, display_image_grid
import json

def test_reconstruction_quality():
    """Test and display reconstruction quality with improved visualization."""
    
    print("🔍 Testing Reconstruction Quality")
    print("=" * 50)
    
    # Load config
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        config_path = "config/config.json"
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # Configure GPU
    device = configure_gpu()
    
    # Load model
    model_path = "model_weights/vae.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using train_VAE_pytorch.py")
        return
    
    model = VAE_pt(config["input_img_size"], config["embedding_size"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully from {model_path}")
    
    # Load validation data
    print("\n📊 Loading validation data...")
    train_data, val_data = get_split_data(config=config)
    print(f"✅ Loaded {len(val_data)} validation samples")
    
    # Test different numbers of images
    test_sizes = [6, 12, 20]
    
    for num_images in test_sizes:
        print(f"\n🔍 Testing Reconstruction with {num_images} images")
        print("-" * 40)
        
        # Get random sample of validation images
        indices = torch.randperm(len(val_data))[:num_images]
        real_images = torch.stack([val_data[i] for i in indices])
        
        print(f"✅ Selected {real_images.shape[0]} images for testing")
        print(f"✅ Image shape: {real_images.shape}")
        print(f"✅ Image range: [{real_images.min():.6f}, {real_images.max():.6f}]")
        
        # Reconstruct images
        with torch.no_grad():
            z_mean, z_log_var, z = model.enc(real_images.to(device))
            reconstructed = model.dec(z)
        
        # Convert to numpy for display
        real_np = real_images.permute(0, 2, 3, 1).cpu().numpy()
        recon_np = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
        
        print(f"✅ Reconstructed shape: {recon_np.shape}")
        print(f"✅ Reconstructed range: [{recon_np.min():.6f}, {recon_np.max():.6f}]")
        
        # Calculate quality metrics
        mse = np.mean((real_np - recon_np) ** 2)
        mae = np.mean(np.abs(real_np - recon_np))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        print(f"✅ Reconstruction MSE: {mse:.6f}")
        print(f"✅ Reconstruction MAE: {mae:.6f}")
        print(f"✅ PSNR: {psnr:.2f} dB")
        
        # Display comparison
        print(f"\n🖼️  Displaying {num_images} reconstruction pairs:")
        titles = [f"Pair {i+1}" for i in range(real_np.shape[0])]
        
        display_comparison_grid(real_np, recon_np,
                               titles=titles,
                               max_cols=4, 
                               figsize=(16, 8))
        
        # Save results
        save_path = f"reconstruction_test_{num_images}_images.png"
        display_comparison_grid(real_np, recon_np,
                               titles=titles,
                               max_cols=4, 
                               figsize=(16, 8),
                               save_path=save_path)
        print(f"✅ Results saved to: {save_path}")
    
    # Test with a larger sample for detailed analysis
    print(f"\n🔍 Detailed Analysis with 30 images")
    print("-" * 40)
    
    # Get larger sample
    indices = torch.randperm(len(val_data))[:30]
    real_images = torch.stack([val_data[i] for i in indices])
    
    with torch.no_grad():
        z_mean, z_log_var, z = model.enc(real_images.to(device))
        reconstructed = model.dec(z)
    
    real_np = real_images.permute(0, 2, 3, 1).cpu().numpy()
    recon_np = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
    
    # Calculate per-image metrics
    mse_per_image = np.mean((real_np - recon_np) ** 2, axis=(1, 2, 3))
    mae_per_image = np.mean(np.abs(real_np - recon_np), axis=(1, 2, 3))
    
    print(f"✅ Average MSE: {np.mean(mse_per_image):.6f} ± {np.std(mse_per_image):.6f}")
    print(f"✅ Average MAE: {np.mean(mae_per_image):.6f} ± {np.std(mae_per_image):.6f}")
    print(f"✅ Best reconstruction (lowest MSE): Image {np.argmin(mse_per_image) + 1}")
    print(f"✅ Worst reconstruction (highest MSE): Image {np.argmax(mse_per_image) + 1}")
    
    # Display all 30 images in a grid with slider
    print(f"\n🖼️  Displaying all 30 images with slider navigation:")
    titles = [f"Pair {i+1}\nMSE: {mse_per_image[i]:.4f}" for i in range(real_np.shape[0])]
    
    display_comparison_grid(real_np, recon_np,
                           titles=titles,
                           max_cols=5, 
                           figsize=(20, 12),
                           save_path="reconstruction_test_30_images.png")
    
    print("\n🎯 Reconstruction Quality Test Complete!")
    print("=" * 50)
    print("Features demonstrated:")
    print("✅ Side-by-side comparison grids")
    print("✅ Multiple test sizes (6, 12, 20, 30 images)")
    print("✅ Quality metrics (MSE, MAE, PSNR)")
    print("✅ Per-image analysis")
    print("✅ Slider navigation for large collections")
    print("✅ Automatic saving to files")

if __name__ == "__main__":
    test_reconstruction_quality()
