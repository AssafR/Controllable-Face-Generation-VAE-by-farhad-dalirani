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

# Load model
model_vae = VAE_pt(
    input_img_size=config["input_img_size"], 
    embedding_size=config["embedding_size"], 
    num_channels=config["num_channels"], 
    beta=config["beta"])

model_path = "model_weights/vae.pth"
model_vae.load_state_dict(torch.load(model_path, map_location=device))
model_vae = model_vae.to(device)
model_vae.eval()

print("Model loaded successfully")

# Test 1: Generate random images
print("\n=== Test 1: Random Generation ===")
with torch.no_grad():
    # Generate random latent vectors
    random_latent = torch.randn(4, config["embedding_size"]).to(device)
    print(f"Random latent shape: {random_latent.shape}")
    print(f"Random latent range: {random_latent.min().item():.3f} to {random_latent.max().item():.3f}")
    
    # Generate images
    generated = model_vae.dec(random_latent)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated range: {generated.min().item():.3f} to {generated.max().item():.3f}")
    
    # Convert to numpy and display
    generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
    print(f"Converted shape: {generated_np.shape}")
    print(f"Converted range: {generated_np.min():.3f} to {generated_np.max():.3f}")

# Test 2: Check if images are all zeros or very small
print("\n=== Test 2: Image Analysis ===")
for i in range(4):
    img = generated_np[i]
    print(f"Image {i}: shape={img.shape}, min={img.min():.6f}, max={img.max():.6f}, mean={img.mean():.6f}")
    
    # Check if image is mostly black
    if img.max() < 0.1:
        print(f"  WARNING: Image {i} is very dark (max < 0.1)")
    if img.mean() < 0.05:
        print(f"  WARNING: Image {i} has very low mean brightness")

# Test 3: Try different normalization
print("\n=== Test 3: Different Normalization ===")
# Try without sigmoid (raw output)
with torch.no_grad():
    # Temporarily remove sigmoid from decoder
    raw_output = model_vae.dec.conv1(random_latent)
    print(f"Raw output range: {raw_output.min().item():.3f} to {raw_output.max().item():.3f}")
    
    # Apply sigmoid manually
    sigmoid_output = torch.sigmoid(raw_output)
    print(f"Sigmoid output range: {sigmoid_output.min().item():.3f} to {sigmoid_output.max().item():.3f}")

# Test 4: Check model weights
print("\n=== Test 4: Model Weight Analysis ===")
for name, param in model_vae.named_parameters():
    if 'conv1' in name and 'weight' in name:
        print(f"{name}: shape={param.shape}, range={param.min().item():.3f} to {param.max().item():.3f}")
        break

print("\n=== Test 5: Display Images ===")
# Display the generated images
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i in range(4):
    row, col = i // 2, i % 2
    axes[row, col].imshow(generated_np[i])
    axes[row, col].set_title(f'Generated {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('debug_generated_images.png')
print("Images saved to debug_generated_images.png")
plt.show()
