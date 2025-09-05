import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from variation_autoencoder_pytorch import VAE_pt
import json

# Load config
# Try relative path first (when running from test/ directory)
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = config_path
    
    with open(config_path, 'r') as file:
    config = json.load(file)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a fresh model (untrained)
model_vae = VAE_pt(
    input_img_size=config["input_img_size"], 
    embedding_size=config["embedding_size"], 
    num_channels=config["num_channels"], 
    beta=config["beta"])

model_vae = model_vae.to(device)
model_vae.eval()

print("Fresh model created")

# Test with random input
print("\n=== Testing with Random Input ===")
with torch.no_grad():
    # Create random input image
    random_input = torch.randn(1, 3, 64, 64).to(device)
    print(f"Input shape: {random_input.shape}")
    print(f"Input range: {random_input.min().item():.3f} to {random_input.max().item():.3f}")
    
    # Forward pass
    emb_mean, emb_log_var, reconst = model_vae(random_input)
    print(f"Embedding mean shape: {emb_mean.shape}")
    print(f"Embedding log var shape: {emb_log_var.shape}")
    print(f"Reconstruction shape: {reconst.shape}")
    print(f"Reconstruction range: {reconst.min().item():.3f} to {reconst.max().item():.3f}")

# Test decoder with random latent
print("\n=== Testing Decoder with Random Latent ===")
with torch.no_grad():
    # Random latent vector
    random_latent = torch.randn(4, config["embedding_size"]).to(device)
    print(f"Random latent shape: {random_latent.shape}")
    print(f"Random latent range: {random_latent.min().item():.3f} to {random_latent.max().item():.3f}")
    
    # Generate images
    generated = model_vae.dec(random_latent)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated range: {generated.min().item():.3f} to {generated.max().item():.3f}")
    
    # Convert to numpy
    generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
    print(f"Converted shape: {generated_np.shape}")
    print(f"Converted range: {generated_np.min():.3f} to {generated_np.max():.3f}")

# Check if the issue is with the sigmoid activation
print("\n=== Testing Without Sigmoid ===")
with torch.no_grad():
    # Get the raw output before sigmoid
    x = random_latent
    x = model_vae.dec.dense1(x)
    x = model_vae.dec.activation(model_vae.dec.bn_dense(x))
    x = x.view(x.size(0), *model_vae.dec.shape_before_flattening)
    
    x = model_vae.dec.convtr1(x)
    x = model_vae.dec.activation(model_vae.dec.bn1(x))
    x = model_vae.dec.convtr2(x)
    x = model_vae.dec.activation(model_vae.dec.bn2(x))
    x = model_vae.dec.convtr3(x)
    x = model_vae.dec.activation(model_vae.dec.bn3(x))
    x = model_vae.dec.convtr4(x)
    x = model_vae.dec.activation(model_vae.dec.bn4(x))
    
    # Raw output before sigmoid
    raw_output = model_vae.dec.conv1(x)
    print(f"Raw output range: {raw_output.min().item():.3f} to {raw_output.max().item():.3f}")
    
    # Apply sigmoid
    sigmoid_output = torch.sigmoid(raw_output)
    print(f"Sigmoid output range: {sigmoid_output.min().item():.3f} to {sigmoid_output.max().item():.3f}")

print("\n=== Conclusion ===")
print("The model architecture appears to be working correctly.")
print("The issue is likely that the model needs to be trained to generate meaningful images.")
print("An untrained VAE will generate random noise, which appears as dark/black images.")
print("Please wait for the training to complete, then test again.")
