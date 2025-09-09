import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Debug script to test dataset loading directly
"""

import os
import jsonled 
import torch
from torchvision import transforms
from utilities_pytorch import CelebADataset

def debug_dataset_loading():
    """Debug the dataset loading process"""
    
    # Load config
    # Try relative path first (when running from test/ directory)
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = config_path
    
    with open(config_path, 'r') as file:
        config = json.load(f)
    
    print(f"Config: {config}")
    
    # Create the same transform as in get_split_data
    transform = transforms.Compose([
        transforms.Resize((config["input_img_size"], config["input_img_size"])),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float() / 255.0),  # Normalize to [0, 1]
    ])
    
    # Create dataset
    dataset_path = os.path.join(config["dataset_dir"], "img_align_celeba", "img_align_celeba")
    print(f"Dataset path: {dataset_path}")
    
    dataset = CelebADataset(dataset_path, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a few samples
    print("\n=== Testing Dataset Loading ===")
    for i in range(5):
        try:
            sample = dataset[i]
            print(f"Sample {i}:")
            print(f"  Type: {type(sample)}")
            print(f"  Shape: {sample.shape}")
            print(f"  Range: {sample.min():.6f} to {sample.max():.6f}")
            print(f"  Mean: {sample.mean():.6f}")
            print(f"  Std: {sample.std():.6f}")
            
            # Check if it's very dark
            if sample.max() < 0.1:
                print(f"  WARNING: Sample {i} is very dark!")
            else:
                print(f"  Sample {i} looks normal")
                
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
    
    # Test loading without transform
    print("\n=== Testing Without Transform ===")
    dataset_no_transform = CelebADataset(dataset_path, transform=None)
    sample_raw = dataset_no_transform[0]
    print(f"Raw sample type: {type(sample_raw)}")
    print(f"Raw sample size: {sample_raw.size}")
    print(f"Raw sample mode: {sample_raw.mode}")

if __name__ == "__main__":
    debug_dataset_loading()
