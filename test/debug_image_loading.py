import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Debug script to check image loading and preprocessing
"""

import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def test_raw_image_loading():
    """Test loading a raw image without any preprocessing"""
    
    # Find a sample image
    dataset_dir = "celeba-dataset/img_align_celeba/img_align_celeba"
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    
    if not image_files:
        print("No image files found!")
        return
    
    sample_file = image_files[0]
    image_path = os.path.join(dataset_dir, sample_file)
    
    print(f"Testing image: {sample_file}")
    print(f"Full path: {image_path}")
    
    # Load with PIL
    pil_image = Image.open(image_path).convert('RGB')
    print(f"PIL image size: {pil_image.size}")
    print(f"PIL image mode: {pil_image.mode}")
    
    # Convert to numpy
    np_image = np.array(pil_image)
    print(f"NumPy image shape: {np_image.shape}")
    print(f"NumPy image dtype: {np_image.dtype}")
    print(f"NumPy image range: {np_image.min()} to {np_image.max()}")
    print(f"NumPy image mean: {np_image.mean():.6f}")
    print(f"NumPy image std: {np_image.std():.6f}")
    
    # Test different normalization approaches
    print("\n=== Testing Different Normalizations ===")
    
    # Method 1: Direct division by 255
    norm1 = np_image.astype(np.float32) / 255.0
    print(f"Method 1 (div 255): range {norm1.min():.6f} to {norm1.max():.6f}, mean {norm1.mean():.6f}")
    
    # Method 2: PyTorch ToTensor (which divides by 255)
    transform_totensor = transforms.ToTensor()
    torch_image = transform_totensor(pil_image)
    print(f"Method 2 (ToTensor): range {torch_image.min():.6f} to {torch_image.max():.6f}, mean {torch_image.mean():.6f}")
    
    # Method 3: Check if there's a scaling issue
    print(f"Method 3 (check scaling): PIL max {np_image.max()}, should be 255")
    
    # Save the original image for visual inspection
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np_image)
    plt.title(f"Original PIL Image\nRange: {np_image.min()}-{np_image.max()}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(norm1)
    plt.title(f"Normalized (div 255)\nRange: {norm1.min():.6f}-{norm1.max():.6f}")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    torch_np = torch_image.permute(1, 2, 0).numpy()
    plt.imshow(torch_np)
    plt.title(f"PyTorch ToTensor\nRange: {torch_image.min():.6f}-{torch_image.max():.6f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_image_loading.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nImages saved to debug_image_loading.png")

if __name__ == "__main__":
    test_raw_image_loading()
