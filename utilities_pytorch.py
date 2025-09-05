#!/usr/bin/env python3
"""
PyTorch utilities for VAE training and data handling.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import json

def configure_gpu():
    """Configure GPU settings for optimal performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU devices found: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU devices found. Running on CPU.")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    return device

class CelebADataset(Dataset):
    """CelebA dataset for PyTorch"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def data_preprocess(data):
    """Preprocess data for VAE training"""
    return (data - 0.5) * 2  # Normalize to [-1, 1]

def get_random_images(config, num_images=70):
    """Get random images from the dataset"""
    
    # Define transforms with proper normalization (no aggressive preprocessing)
    transform = transforms.Compose([
        transforms.Resize((config["input_img_size"], config["input_img_size"])),
        transforms.ToTensor(),  # This already normalizes to [0, 1]
        # No additional preprocessing - let the model learn from natural images
    ])
    
    # Create dataset
    dataset = CelebADataset(
        os.path.join(config["dataset_dir"], "img_align_celeba", "img_align_celeba"),
        transform=transform
    )
    
    # Get random indices
    indices = torch.randperm(len(dataset))[:num_images]
    images = [dataset[i] for i in indices]
    
    # Stack into batch and convert to numpy
    images_tensor = torch.stack(images)
    images_np = images_tensor.permute(0, 2, 3, 1).numpy()  # Convert from NCHW to NHWC
    
    return images_np

def get_split_data(config, shuffle=True, validation_split=0.2):
    """Return train and validation split"""
    
    # Define transforms with proper normalization (no aggressive preprocessing)
    transform = transforms.Compose([
        transforms.Resize((config["input_img_size"], config["input_img_size"])),
        transforms.ToTensor(),  # This already normalizes to [0, 1]
        # No additional preprocessing - let the model learn from natural images
    ])
    
    # Create full dataset
    full_dataset = CelebADataset(
        os.path.join(config["dataset_dir"], "img_align_celeba", "img_align_celeba"),
        transform=transform
    )
    
    # Apply dataset subset if specified
    dataset_subset = config.get("dataset_subset", None)
    if dataset_subset is not None and dataset_subset < len(full_dataset):
        print(f"📊 Using dataset subset: {dataset_subset} samples (from {len(full_dataset)} total)")
        # Create subset by randomly sampling indices
        subset_indices = torch.randperm(len(full_dataset))[:dataset_subset].tolist()
        full_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    elif dataset_subset is not None:
        print(f"📊 Dataset subset ({dataset_subset}) larger than available ({len(full_dataset)}), using full dataset")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )
    
    return train_dataset, val_dataset

def display_image_grid(images, titles=None, max_cols=4, figsize=(12, 8), save_path=None):
    """
    Display images in a grid with optional slider for large collections.
    
    Args:
        images: List or array of images (numpy arrays)
        titles: Optional list of titles for each image
        max_cols: Maximum number of columns in the grid
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    
    if len(images) == 0:
        print("No images to display")
        return
    
    num_images = len(images)
    num_cols = min(max_cols, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # If images fit in one screen, show them all
    if num_images <= max_cols * 6:  # Reasonable limit for one screen
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_images):
            row = i // num_cols
            col = i % num_cols
            
            if num_rows == 1 and num_cols == 1:
                ax = axes
            elif num_rows == 1:
                ax = axes[col]
            elif num_cols == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            ax.imshow(np.clip(images[i], 0, 1))
            ax.axis('off')
            
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
        
        # Hide empty subplots
        for i in range(num_images, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            if num_rows == 1 and num_cols == 1:
                axes.axis('off')
            elif num_rows == 1:
                axes[col].axis('off')
            elif num_cols == 1:
                axes[row].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Images saved to: {save_path}")
        
        plt.close()  # Close the figure to prevent crashes
    
    else:
        # Use slider for large collections
        display_image_grid_with_slider(images, titles, max_cols, figsize, save_path)

def display_image_grid_with_slider(images, titles=None, max_cols=4, figsize=(12, 8), save_path=None):
    """
    Display images in a grid with a slider for navigation.
    
    Args:
        images: List or array of images (numpy arrays)
        titles: Optional list of titles for each image
        max_cols: Maximum number of columns in the grid
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    
    num_images = len(images)
    num_cols = min(max_cols, num_images)
    images_per_page = max_cols * 4  # Show 4 rows by default
    
    # Create figure with slider
    fig = plt.figure(figsize=figsize)
    
    # Create main axes for images
    ax_images = plt.subplot2grid((10, 1), (0, 0), rowspan=8)
    
    # Create slider
    ax_slider = plt.subplot2grid((10, 1), (9, 0), rowspan=1)
    
    # Calculate number of pages
    num_pages = (num_images + images_per_page - 1) // images_per_page
    
    # Create slider
    slider = Slider(ax_slider, 'Page', 0, num_pages - 1, valinit=0, valfmt='%d')
    
    def update_display(page):
        """Update the display based on slider value"""
        page = int(page)
        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, num_images)
        
        # Clear previous images
        ax_images.clear()
        
        # Calculate grid for current page
        page_images = images[start_idx:end_idx]
        page_titles = titles[start_idx:end_idx] if titles else None
        
        if len(page_images) == 0:
            ax_images.text(0.5, 0.5, 'No images', ha='center', va='center', transform=ax_images.transAxes)
            ax_images.axis('off')
            return
        
        page_rows = (len(page_images) + num_cols - 1) // num_cols
        
        # Create subplot grid for current page
        for i, img in enumerate(page_images):
            row = i // num_cols
            col = i % num_cols
            
            # Calculate position in the main axes
            x_pos = col / num_cols
            y_pos = 1 - (row + 1) / page_rows
            width = 1 / num_cols
            height = 1 / page_rows
            
            # Create subplot
            ax_sub = fig.add_axes([x_pos, y_pos, width, height])
            ax_sub.imshow(np.clip(img, 0, 1))
            ax_sub.axis('off')
            
            if page_titles and i < len(page_titles):
                ax_sub.set_title(page_titles[i], fontsize=8)
        
        ax_images.axis('off')
        ax_images.set_title(f'Images {start_idx + 1}-{end_idx} of {num_images} (Page {page + 1}/{num_pages})')
    
    # Connect slider to update function
    slider.on_changed(update_display)
    
    # Initial display
    update_display(0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Images saved to: {save_path}")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Images saved to: {save_path}")
    plt.close()  # Close the figure to prevent crashes

def display_comparison_grid(original_images, generated_images, titles=None, max_cols=4, figsize=(15, 10), save_path=None):
    """
    Display original and generated images side by side for comparison.
    
    Args:
        original_images: List of original images
        generated_images: List of generated images
        titles: Optional list of titles
        max_cols: Maximum number of columns
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    
    if len(original_images) != len(generated_images):
        print(f"Warning: Mismatch in image counts - Original: {len(original_images)}, Generated: {len(generated_images)}")
        min_len = min(len(original_images), len(generated_images))
        original_images = original_images[:min_len]
        generated_images = generated_images[:min_len]
    
    num_images = len(original_images)
    num_cols = min(max_cols, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=figsize)
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_images):
        row = i // num_cols
        col = i % num_cols
        
        # Original image
        orig_col = col * 2
        gen_col = col * 2 + 1
        
        if num_rows == 1 and num_cols == 1:
            ax_orig = axes[0]
            ax_gen = axes[1]
        elif num_rows == 1:
            ax_orig = axes[orig_col]
            ax_gen = axes[gen_col]
        elif num_cols == 1:
            ax_orig = axes[row, 0]
            ax_gen = axes[row, 1]
        else:
            ax_orig = axes[row, orig_col]
            ax_gen = axes[row, gen_col]
        
        # Display original
        ax_orig.imshow(np.clip(original_images[i], 0, 1))
        ax_orig.set_title(f"Original {i+1}", fontsize=10)
        ax_orig.axis('off')
        
        # Display generated
        ax_gen.imshow(np.clip(generated_images[i], 0, 1))
        ax_gen.set_title(f"Generated {i+1}", fontsize=10)
        ax_gen.axis('off')
    
    # Hide empty subplots
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        orig_col = col * 2
        gen_col = col * 2 + 1
        
        if num_rows == 1 and num_cols == 1:
            pass  # Only 2 images, no empty subplots
        elif num_rows == 1:
            axes[orig_col].axis('off')
            axes[gen_col].axis('off')
        elif num_cols == 1:
            axes[row, 0].axis('off')
            axes[row, 1].axis('off')
        else:
            axes[row, orig_col].axis('off')
            axes[row, gen_col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison grid saved to: {save_path}")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Images saved to: {save_path}")
    plt.close()  # Close the figure to prevent crashes

def create_image_montage(images, titles=None, max_images=20, figsize=(15, 10), save_path=None):
    """
    Create a montage of images in a single figure.
    
    Args:
        images: List of images to display
        titles: Optional titles for images
        max_images: Maximum number of images to show
        figsize: Figure size
        save_path: Optional path to save
    """
    
    if len(images) == 0:
        print("No images to display")
        return
    
    # Limit number of images
    images = images[:max_images]
    if titles:
        titles = titles[:max_images]
    
    num_images = len(images)
    
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        if rows == 1 and cols == 1:
            ax = axes[0]
        elif rows == 1:
            ax = axes[col]
        elif cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        ax.imshow(np.clip(images[i], 0, 1))
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=8)
    
    # Hide empty subplots
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        
        if rows == 1 and cols == 1:
            pass
        elif rows == 1:
            axes[col].axis('off')
        elif cols == 1:
            axes[row].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Montage saved to: {save_path}")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Images saved to: {save_path}")
    plt.close()  # Close the figure to prevent crashes