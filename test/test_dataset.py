import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities_pytorch import get_split_data
import json
import os

# Load config
# Try relative path first (when running from test/ directory)
config_path = "../config/config.json"
if not os.path.exists(config_path):
    # If not found, try absolute path (when running from main directory)
    config_path = "config/config.json"

with open(config_path, 'r') as file:
    config = json.load(file)

print("Loading dataset...")
train, validation = get_split_data(config=config)

print(f"Training samples: {len(train)}")
print(f"Validation samples: {len(validation)}")

# Test a few samples
print("\n=== Testing Dataset Samples ===")
for i in range(5):
    sample = train[i]
    print(f"Sample {i}:")
    print(f"  Type: {type(sample)}")
    print(f"  Shape: {sample.shape}")
    print(f"  Range: {sample.min().item():.6f} to {sample.max().item():.6f}")
    print(f"  Mean: {sample.mean().item():.6f}")
    print(f"  Std: {sample.std().item():.6f}")
    
    # Check if image is mostly black
    if sample.max().item() < 0.1:
        print(f"  WARNING: Sample {i} is very dark!")
    
    # Convert to numpy for display
    if isinstance(sample, torch.Tensor):
        img_np = sample.numpy()
        if img_np.shape[0] == 3:  # NCHW format
            img_np = img_np.transpose(1, 2, 0)  # Convert to NHWC
    
    # Display the image
    plt.figure(figsize=(10, 2))
    plt.subplot(1, 5, i+1)
    plt.imshow(img_np)
    plt.title(f'Sample {i}\nRange: {sample.min().item():.3f}-{sample.max().item():.3f}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('dataset_samples.png')
print("\nDataset samples saved to dataset_samples.png")
plt.show()

# Test with different normalization
print("\n=== Testing Different Normalization ===")
sample = train[0]
print(f"Original range: {sample.min().item():.6f} to {sample.max().item():.6f}")

# Try different normalization approaches
sample_denorm = sample * 255.0
print(f"Denormalized range: {sample_denorm.min().item():.6f} to {sample_denorm.max().item():.6f}")

# Check if the issue is with the dataset itself
print("\n=== Checking Dataset Directory ===")
import os
dataset_dir = os.path.join(config["dataset_dir"], "img_align_celeba")
if os.path.exists(dataset_dir):
    files = os.listdir(dataset_dir)
    print(f"Found {len(files)} files in dataset directory")
    if len(files) > 0:
        print(f"First few files: {files[:5]}")
    else:
        print("ERROR: Dataset directory is empty!")
else:
    print(f"ERROR: Dataset directory not found: {dataset_dir}")
