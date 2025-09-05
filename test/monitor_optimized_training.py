#!/usr/bin/env python3
"""
Monitor Optimized Fast High Quality Training Progress
Shows real-time training metrics and generates sample images.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import configure_gpu, display_image_grid
from config_loader import ConfigLoader
import json
import glob
import time

def monitor_optimized_training():
    """Monitor the optimized fast high quality training progress."""
    
    print("🔍 Monitoring Optimized Fast High Quality Training Progress")
    print("=" * 70)
    
    # Load configuration
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
    print(f"  • Batch size: {config['batch_size']} (optimized)")
    
    # Configure GPU
    device = configure_gpu()
    
    # Check for model checkpoints
    model_path = "model_weights/vae_optimized_fast_high_quality.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        print("Training may still be in progress...")
        
        # Check for any model files
        model_files = glob.glob("model_weights/*.pth")
        if model_files:
            print(f"📁 Found {len(model_files)} model files:")
            for f in model_files:
                print(f"  • {f}")
        else:
            print("📁 No model files found yet")
        
        # Check for sample images
        sample_files = glob.glob("sample_images/fast_high_quality_*.png")
        if sample_files:
            print(f"📸 Found {len(sample_files)} sample images:")
            for f in sample_files:
                print(f"  • {f}")
        
        return
    
    # Load model
    print(f"\n🏗️  Loading model from {model_path}...")
    model = VAE_pt(
        input_img_size=config['input_img_size'],
        embedding_size=config['embedding_size'],
        loss_config=config
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✅ Model loaded successfully")
    
    # Generate sample images
    print(f"\n🖼️  Generating sample images...")
    
    with torch.no_grad():
        # Generate random images
        z = torch.randn(16, config['embedding_size']).to(device)
        generated = model.dec(z)
        generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
        
        print(f"✅ Generated {generated_np.shape[0]} images")
        print(f"✅ Image shape: {generated_np.shape}")
        print(f"✅ Image range: [{generated_np.min():.6f}, {generated_np.max():.6f}]")
        
        # Display images
        titles = [f"Sample {i+1}" for i in range(16)]
        display_image_grid(generated_np, 
                          titles=titles,
                          max_cols=4, 
                          figsize=(20, 16),
                          save_path="optimized_fast_high_quality_monitor_samples.png")
        
        print(f"✅ Sample images saved: optimized_fast_high_quality_monitor_samples.png")
    
    # Check for TensorBoard logs
    log_dirs = glob.glob("runs/optimized_fast_high_quality_*")
    if log_dirs:
        latest_log = max(log_dirs, key=os.path.getctime)
        print(f"\n📊 TensorBoard logs found: {latest_log}")
        print(f"  • To view: tensorboard --logdir {latest_log}")
        print(f"  • Or run: uv run tensorboard --logdir {latest_log}")
    
    # Check for sample images from training
    sample_files = glob.glob("optimized_fast_high_quality_samples_epoch_*.png")
    if sample_files:
        print(f"\n📸 Training sample images found: {len(sample_files)}")
        latest_sample = max(sample_files, key=os.path.getctime)
        print(f"  • Latest: {latest_sample}")
    
    # Test reconstruction quality
    print(f"\n🔍 Testing reconstruction quality...")
    
    # Load some validation data
    from utilities_pytorch import get_split_data
    train_data, val_data = get_split_data(config=config)
    
    # Get a few validation images
    test_indices = torch.randperm(len(val_data))[:4]
    test_images = torch.stack([val_data[i] for i in test_indices])
    
    with torch.no_grad():
        # Reconstruct images
        z_mean, z_log_var, z = model.enc(test_images.to(device))
        reconstructed = model.dec(z)
        
        # Convert to numpy
        original_np = test_images.permute(0, 2, 3, 1).cpu().numpy()
        recon_np = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
        
        # Calculate metrics
        mse = np.mean((original_np - recon_np) ** 2)
        mae = np.mean(np.abs(original_np - recon_np))
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        print(f"✅ Reconstruction metrics:")
        print(f"  • MSE: {mse:.6f}")
        print(f"  • MAE: {mae:.6f}")
        print(f"  • PSNR: {psnr:.2f} dB")
        
        # Display reconstruction comparison
        from utilities_pytorch import display_comparison_grid
        display_comparison_grid(original_np, recon_np,
                               titles=[f"Pair {i+1}" for i in range(4)],
                               max_cols=2, 
                               figsize=(12, 8),
                               save_path="optimized_fast_high_quality_reconstruction_test.png")
        
        print(f"✅ Reconstruction comparison saved: optimized_fast_high_quality_reconstruction_test.png")
    
    print(f"\n🎯 Monitoring complete!")
    print("=" * 70)
    print("Next steps:")
    print("  • Check TensorBoard for training curves")
    print("  • Examine generated sample images")
    print("  • Review reconstruction quality")
    print("  • Continue training if needed")

if __name__ == "__main__":
    monitor_optimized_training()
