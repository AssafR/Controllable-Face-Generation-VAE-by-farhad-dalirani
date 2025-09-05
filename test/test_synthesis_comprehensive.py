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
from utilities_pytorch import get_split_data, configure_gpu, display_image_grid, display_comparison_grid, create_image_montage
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
        
        # Display generated images interactively
        print("\n🔍 Displaying Generated Images:")
        titles = [f"Generated {i+1}" for i in range(generated_np.shape[0])]
        display_image_grid(generated_np, 
                          titles=titles,
                          max_cols=5, 
                          figsize=(15, 6))
        
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
        
        # Display reconstruction comparison interactively
        print("\n🔍 Displaying Reconstruction Results:")
        display_comparison_grid(real_np, recon_np,
                               titles=[f"Pair {i+1}" for i in range(real_np.shape[0])],
                               max_cols=3, 
                               figsize=(15, 8))
        
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
    """Save test results as images using improved display functions."""
    print(f"\n=== Saving Test Results ===")
    
    os.makedirs("test_results", exist_ok=True)
    
    # Save random generation
    if generated is not None:
        titles = [f"Generated {i+1}" for i in range(generated.shape[0])]
        display_image_grid(generated, 
                          titles=titles,
                          max_cols=5, 
                          figsize=(15, 6),
                          save_path="test_results/random_generation.png")
        print("✅ Saved random generation results")
    
    # Save reconstruction comparison
    if real is not None and recon is not None:
        display_comparison_grid(real, recon,
                               titles=[f"Pair {i+1}" for i in range(real.shape[0])],
                               max_cols=3, 
                               figsize=(15, 8))
        print("✅ Saved reconstruction comparison")
    
    # Save latent arithmetic
    if base is not None and modified is not None:
        display_comparison_grid(base, modified,
                               titles=[f"Pair {i+1}" for i in range(base.shape[0])],
                               max_cols=3, 
                               figsize=(15, 8))
        print("✅ Saved latent arithmetic results")
    
    # Save morphing
    if morphed is not None and len(morphed) > 0:
        # Flatten all morphing levels into one array
        all_morphed = []
        morph_titles = []
        for i, morph_level in enumerate(morphed):
            for j in range(morph_level.shape[0]):
                all_morphed.append(morph_level[j])
                morph_titles.append(f"Morph α={[0.0, 0.5, 1.0][i]} {j+1}")
        
        display_image_grid(all_morphed, 
                          titles=morph_titles,
                          max_cols=6, 
                          figsize=(18, 10),
                          save_path="test_results/morphing.png")
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
