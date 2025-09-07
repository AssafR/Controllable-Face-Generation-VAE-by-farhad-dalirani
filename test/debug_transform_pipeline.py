import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Debug script to test transform pipeline step by step
"""

import os
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def debug_transform_pipeline():
    """Debug the transform pipeline step by step"""
    
    # Load config
    # Try relative path first (when running from test/ directory)
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = config_path
    
    with open(config_path, 'r') as file:
        config = json.load(f)
    
    # Load a sample image
    dataset_path = os.path.join(config["dataset_dir"], "img_align_celeba", "img_align_celeba")
    sample_file = "000001.jpg"
    image_path = os.path.join(dataset_path, sample_file)
    
    print(f"Loading image: {image_path}")
    
    # Load with PIL
    pil_image = Image.open(image_path).convert('RGB')
    print(f"PIL image size: {pil_image.size}")
    print(f"PIL image mode: {pil_image.mode}")
    
    # Convert to numpy to check values
    np_image = np.array(pil_image)
    print(f"NumPy image shape: {np_image.shape}")
    print(f"NumPy image range: {np_image.min()} to {np_image.max()}")
    print(f"NumPy image mean: {np_image.mean():.6f}")
    
    # Test each transform step
    print("\n=== Testing Transform Steps ===")
    
    # Step 1: Resize
    resize_transform = transforms.Resize((config["input_img_size"], config["input_img_size"]))
    resized = resize_transform(pil_image)
    print(f"After resize: {resized.size}")
    
    # Step 2: ToTensor
    totensor_transform = transforms.ToTensor()
    tensor_image = totensor_transform(resized)
    print(f"After ToTensor: shape {tensor_image.shape}, range {tensor_image.min():.6f} to {tensor_image.max():.6f}, mean {tensor_image.mean():.6f}")
    
    # Step 3: Convert to float and divide by 255
    float_image = tensor_image.float() / 255.0
    print(f"After float/255: range {float_image.min():.6f} to {float_image.max():.6f}, mean {float_image.mean():.6f}")
    
    # Test the complete transform
    print("\n=== Testing Complete Transform ===")
    complete_transform = transforms.Compose([
        transforms.Resize((config["input_img_size"], config["input_img_size"])),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float() / 255.0),
    ])
    
    final_image = complete_transform(pil_image)
    print(f"Final image: shape {final_image.shape}, range {final_image.min():.6f} to {final_image.max():.6f}, mean {final_image.mean():.6f}")
    
    # Check if there's a double normalization issue
    print("\n=== Checking for Double Normalization ===")
    print(f"ToTensor already normalizes to [0,1], so dividing by 255 again would make values very small!")
    print(f"ToTensor output range: {tensor_image.min():.6f} to {tensor_image.max():.6f}")
    print(f"After /255: {tensor_image.min()/255:.6f} to {tensor_image.max()/255:.6f}")
    
    # Test without the extra /255
    print("\n=== Testing Without Extra /255 ===")
    correct_transform = transforms.Compose([
        transforms.Resize((config["input_img_size"], config["input_img_size"])),
        transforms.ToTensor(),  # This already normalizes to [0,1]
    ])
    
    correct_image = correct_transform(pil_image) 
    print(f"Correct image: shape {correct_image.shape}, range {correct_image.min():.6f} to {correct_image.max():.6f}, mean {correct_image.mean():.6f}")
    
    # Visualize the results
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 5, 1)
    plt.imshow(np_image)
    plt.title(f"Original PIL\nRange: {np_image.min()}-{np_image.max()}")
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(tensor_image.permute(1, 2, 0))
    plt.title(f"After ToTensor\nRange: {tensor_image.min():.3f}-{tensor_image.max():.3f}")
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(float_image.permute(1, 2, 0))
    plt.title(f"After /255\nRange: {float_image.min():.6f}-{float_image.max():.6f}")
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.imshow(final_image.permute(1, 2, 0))
    plt.title(f"Final (wrong)\nRange: {final_image.min():.6f}-{final_image.max():.6f}")
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.imshow(correct_image.permute(1, 2, 0))
    plt.title(f"Correct\nRange: {correct_image.min():.3f}-{correct_image.max():.3f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_transform_pipeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to debug_transform_pipeline.png")

if __name__ == "__main__":
    debug_transform_pipeline()
