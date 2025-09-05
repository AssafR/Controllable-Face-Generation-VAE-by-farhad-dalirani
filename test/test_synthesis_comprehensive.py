import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Comprehensive test script for PyTorch VAE synthesis functionality.
Tests various synthesis functions and analyzes output quality.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from variation_autoencoder_pytorch import VAE_pt
from utilities_pytorch import get_split_data, configure_gpu
import json
import os

def test_model_loading():
    """Test if the model loads correctly."""
    print("=== Testing Model Loading ===")
    
    # Load config
    # Try relative path first (when running from test/ directory)
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = config_path
    
    with open(config_path, 'r') as file:
        config = json.load(f)
    
    # Configure GPU
    device = configure_gpu()
    
    # Load model
    model_path = "model_weights/vae.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None, None, None
    
    model = VAE_pt(config["input_img_size"], config["embedding_size"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully from {model_path}")
    print(f"✅ Model device: {next(model.parameters()).device}")
    print(f"✅ Model embedding size: {config['embedding_size']}")
    print(f"✅ Model input size: {config['input_img_size']}")
    
    return model, config, device

def test_random_generation(model, config, device, num_images=5):
    """Test random image generation."""
    print(f"\n=== Testing Random Generation ({num_images} images) ===")
    
    with torch.no_grad():
        # Generate random latent vectors
        z = torch.randn(num_images, config["embedding_size"]).to(device)
        
        # Generate images
        generated = model.dec(z)
        
        # Convert to numpy
        generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
        
        print(f"✅ Generated shape: {generated_np.shape}")
        print(f"✅ Generated range: [{generated_np.min():.6f}, {generated_np.max():.6f}]")
        print(f"✅ Generated mean: {generated_np.mean():.6f}")
        print(f"✅ Generated std: {generated_np.std():.6f}")
        
        # Check for any completely black or white images
        black_images = np.all(generated_np < 0.01, axis=(1, 2, 3))
        white_images = np.all(generated_np > 0.99, axis=(1, 2, 3))
        
        print(f"✅ Black images: {black_images.sum()}/{num_images}")
        print(f"✅ White images: {white_images.sum()}/{num_images}")
        
        return generated_np

def test_reconstruction(model, config, device, num_images=5):
    """Test image reconstruction."""
    print(f"\n=== Testing Reconstruction ({num_images} images) ===")
    
    # Load some real images
    train_data, val_data = get_split_data(config=config)
    
    # Get individual images from the dataset
    real_images = []
    for i in range(min(num_images, len(val_data))):
        img = val_data[i]  # Get individual image
        real_images.append(img)
    
    # Stack into a batch tensor
    real_images = torch.stack(real_images)
    
    if len(real_images) == 0:
        print("❌ No validation images found")
        return None
    
    # Images should now be in correct format [batch, channels, height, width]
    
    print(f"✅ Real images shape: {real_images.shape}")
    print(f"✅ Real images range: [{real_images.min():.6f}, {real_images.max():.6f}]")
    print(f"✅ Real images mean: {real_images.mean():.6f}")
    
    with torch.no_grad():
        # Encode and decode
        z_mean, z_log_var, z = model.enc(real_images.to(device))
        reconstructed = model.dec(z)
        
        # Convert to numpy
        real_np = real_images.permute(0, 2, 3, 1).cpu().numpy()
        recon_np = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
        
        print(f"✅ Reconstructed shape: {recon_np.shape}")
        print(f"✅ Reconstructed range: [{recon_np.min():.6f}, {recon_np.max():.6f}]")
        print(f"✅ Reconstructed mean: {recon_np.mean():.6f}")
        
        # Calculate reconstruction error
        mse = np.mean((real_np - recon_np) ** 2)
        print(f"✅ Reconstruction MSE: {mse:.6f}")
        
        return real_np, recon_np

def test_latent_arithmetic(model, config, device):
    """Test latent space arithmetic."""
    print(f"\n=== Testing Latent Arithmetic ===")
    
    # Create a simple attribute vector (random for now)
    attribute_vector = np.random.randn(config["embedding_size"])
    
    with torch.no_grad():
        # Generate base images
        z_base = torch.randn(3, config["embedding_size"]).to(device)
        base_images = model.dec(z_base)
        
        # Add attribute vector
        z_modified = z_base + torch.from_numpy(attribute_vector).float().to(device)
        modified_images = model.dec(z_modified)
        
        # Convert to numpy
        base_np = base_images.permute(0, 2, 3, 1).cpu().numpy()
        modified_np = modified_images.permute(0, 2, 3, 1).cpu().numpy()
        
        print(f"✅ Base images range: [{base_np.min():.6f}, {base_np.max():.6f}]")
        print(f"✅ Modified images range: [{modified_np.min():.6f}, {modified_np.max():.6f}]")
        
        # Check if there's a visible difference
        diff = np.mean(np.abs(base_np - modified_np))
        print(f"✅ Average difference: {diff:.6f}")
        
        return base_np, modified_np

def test_morphing(model, config, device):
    """Test image morphing."""
    print(f"\n=== Testing Image Morphing ===")
    
    with torch.no_grad():
        # Generate two different latent vectors
        z1 = torch.randn(2, config["embedding_size"]).to(device)
        z2 = torch.randn(2, config["embedding_size"]).to(device)
        
        # Morph between them
        alphas = [0.0, 0.5, 1.0]
        morphed_images = []
        
        for alpha in alphas:
            z_morphed = (1 - alpha) * z1 + alpha * z2
            images = model.dec(z_morphed)
            morphed_images.append(images.permute(0, 2, 3, 1).cpu().numpy())
        
        print(f"✅ Generated {len(morphed_images)} morphing levels")
        print(f"✅ Each level has {morphed_images[0].shape[0]} images")
        
        return morphed_images

def save_test_results(generated, real, recon, base, modified, morphed):
    """Save test results as images."""
    print(f"\n=== Saving Test Results ===")
    
    os.makedirs("test_results", exist_ok=True)
    
    # Save random generation
    if generated is not None:
        fig, axes = plt.subplots(1, min(5, generated.shape[0]), figsize=(15, 3))
        if generated.shape[0] == 1:
            axes = [axes]
        for i in range(min(5, generated.shape[0])):
            axes[i].imshow(np.clip(generated[i], 0, 1))
            axes[i].set_title(f"Generated {i+1}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig("test_results/random_generation.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved random generation results")
    
    # Save reconstruction comparison
    if real is not None and recon is not None:
        fig, axes = plt.subplots(2, min(3, real.shape[0]), figsize=(12, 8))
        if real.shape[0] == 1:
            axes = axes.reshape(2, 1)
        for i in range(min(3, real.shape[0])):
            axes[0, i].imshow(np.clip(real[i], 0, 1))
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')
            axes[1, i].imshow(np.clip(recon[i], 0, 1))
            axes[1, i].set_title(f"Reconstructed {i+1}")
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.savefig("test_results/reconstruction_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved reconstruction comparison")
    
    # Save latent arithmetic
    if base is not None and modified is not None:
        fig, axes = plt.subplots(2, min(3, base.shape[0]), figsize=(12, 8))
        if base.shape[0] == 1:
            axes = axes.reshape(2, 1)
        for i in range(min(3, base.shape[0])):
            axes[0, i].imshow(np.clip(base[i], 0, 1))
            axes[0, i].set_title(f"Base {i+1}")
            axes[0, i].axis('off')
            axes[1, i].imshow(np.clip(modified[i], 0, 1))
            axes[1, i].set_title(f"Modified {i+1}")
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.savefig("test_results/latent_arithmetic.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved latent arithmetic results")
    
    # Save morphing
    if morphed is not None and len(morphed) > 0:
        fig, axes = plt.subplots(len(morphed), min(2, morphed[0].shape[0]), figsize=(8, 4*len(morphed)))
        if len(morphed) == 1:
            axes = axes.reshape(1, -1)
        for i, morph_level in enumerate(morphed):
            for j in range(min(2, morph_level.shape[0])):
                axes[i, j].imshow(np.clip(morph_level[j], 0, 1))
                axes[i, j].set_title(f"Morph α={[0.0, 0.5, 1.0][i]} {j+1}")
                axes[i, j].axis('off')
        plt.tight_layout()
        plt.savefig("test_results/morphing.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved morphing results")

def main():
    """Run comprehensive synthesis tests."""
    print("🚀 Starting Comprehensive Synthesis Tests")
    print("=" * 50)
    
    # Test model loading
    model, config, device = test_model_loading()
    if model is None:
        return
    
    # Test random generation
    generated = test_random_generation(model, config, device, num_images=5)
    
    # Test reconstruction
    real, recon = test_reconstruction(model, config, device, num_images=3)
    
    # Test latent arithmetic
    base, modified = test_latent_arithmetic(model, config, device)
    
    # Test morphing
    morphed = test_morphing(model, config, device)
    
    # Save results
    save_test_results(generated, real, recon, base, modified, morphed)
    
    print("\n" + "=" * 50)
    print("✅ All tests completed! Check test_results/ folder for output images.")
    print("=" * 50)

if __name__ == "__main__":
    main()
