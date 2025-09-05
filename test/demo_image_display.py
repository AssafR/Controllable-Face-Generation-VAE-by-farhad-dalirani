#!/usr/bin/env python3
"""
Demo script showing the new improved image display capabilities.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from utilities_pytorch import get_split_data, display_image_grid, display_comparison_grid, create_image_montage
import json

def demo_image_display():
    """Demonstrate the new image display capabilities."""
    
    print("🎨 Image Display Capabilities Demo")
    print("=" * 50)
    
    # Load config
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        config_path = "config/config.json"
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    print("Loading dataset...")
    train, validation = get_split_data(config=config)
    
    print(f"✅ Dataset loaded: {len(train)} train, {len(validation)} val samples")
    
    # Get sample images
    print("\n📸 Getting sample images...")
    sample_images = []
    sample_titles = []
    
    for i in range(min(25, len(train))):
        sample = train[i]
        if isinstance(sample, torch.Tensor):
            img_np = sample.numpy()
            if img_np.shape[0] == 3:  # NCHW format
                img_np = img_np.transpose(1, 2, 0)  # Convert to NHWC
            
            sample_images.append(img_np)
            sample_titles.append(f'Sample {i+1}')
    
    print(f"✅ Got {len(sample_images)} sample images")
    
    # Demo 1: Small grid (fits in one screen)
    print("\n🔍 Demo 1: Small Grid (fits in one screen)")
    print("-" * 40)
    small_images = sample_images[:12]
    small_titles = sample_titles[:12]
    
    display_image_grid(small_images, 
                      titles=small_titles,
                      max_cols=4, 
                      figsize=(12, 8),
                      save_path="demo_small_grid.png")
    
    # Demo 2: Large grid with slider
    print("\n🔍 Demo 2: Large Grid with Slider")
    print("-" * 40)
    print("This will show all 25 images with a slider for navigation")
    
    display_image_grid(sample_images, 
                      titles=sample_titles,
                      max_cols=5, 
                      figsize=(15, 10),
                      save_path="demo_large_grid.png")
    
    # Demo 3: Comparison grid
    print("\n🔍 Demo 3: Comparison Grid")
    print("-" * 40)
    print("Showing original vs 'generated' (same images for demo)")
    
    # Use first 8 images for comparison
    orig_images = sample_images[:8]
    gen_images = sample_images[:8]  # Same for demo
    comp_titles = [f"Pair {i+1}" for i in range(8)]
    
    display_comparison_grid(orig_images, gen_images, 
                           titles=comp_titles,
                           max_cols=4, 
                           figsize=(16, 8))
    
    # Demo 4: Montage
    print("\n🔍 Demo 4: Image Montage")
    print("-" * 40)
    print("Creating a montage of all images")
    
    create_image_montage(sample_images, 
                        titles=sample_titles,
                        max_images=25,
                        figsize=(20, 16),
                        save_path="demo_montage.png")
    
    print("\n🎯 Image Display Demo Complete!")
    print("=" * 50)
    print("Features demonstrated:")
    print("✅ Small grid display (fits in one screen)")
    print("✅ Large grid with slider navigation")
    print("✅ Side-by-side comparison grid")
    print("✅ Image montage")
    print("✅ Automatic saving to files")
    print("✅ Responsive layout based on number of images")

if __name__ == "__main__":
    demo_image_display()
