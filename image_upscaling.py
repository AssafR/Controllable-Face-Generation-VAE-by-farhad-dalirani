#!/usr/bin/env python3
"""
Image upscaling utilities for better display in the GUI.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

def upscale_image_bicubic(image, scale_factor=4):
    """
    Upscale image using bicubic interpolation.
    
    Args:
        image: numpy array of shape (H, W, C) or (H, W)
        scale_factor: factor to upscale by (e.g., 4 for 64x64 -> 256x256)
    
    Returns:
        upscaled image as numpy array
    """
    if len(image.shape) == 3:
        # RGB image
        h, w, c = image.shape
        new_h, new_w = h * scale_factor, w * scale_factor
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        # Grayscale image
        h, w = image.shape
        new_h, new_w = h * scale_factor, w * scale_factor
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return upscaled

def upscale_image_lanczos(image, scale_factor=4):
    """
    Upscale image using Lanczos interpolation.
    
    Args:
        image: numpy array of shape (H, W, C) or (H, W)
        scale_factor: factor to upscale by
    
    Returns:
        upscaled image as numpy array
    """
    if len(image.shape) == 3:
        # RGB image
        h, w, c = image.shape
        new_h, new_w = h * scale_factor, w * scale_factor
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Grayscale image
        h, w = image.shape
        new_h, new_w = h * scale_factor, w * scale_factor
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    return upscaled

def upscale_image_pytorch(image, scale_factor=4):
    """
    Upscale image using PyTorch's interpolate function.
    
    Args:
        image: numpy array of shape (H, W, C)
        scale_factor: factor to upscale by
    
    Returns:
        upscaled image as numpy array
    """
    # Convert to tensor and add batch dimension
    if len(image.shape) == 3:
        # RGB image: (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    else:
        # Grayscale image: (H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    
    # Get current size
    _, _, h, w = tensor.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Upscale using bilinear interpolation
    upscaled_tensor = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Convert back to numpy
    if len(image.shape) == 3:
        # RGB: (1, C, H, W) -> (H, W, C)
        upscaled = upscaled_tensor.squeeze(0).permute(1, 2, 0).numpy()
    else:
        # Grayscale: (1, 1, H, W) -> (H, W)
        upscaled = upscaled_tensor.squeeze(0).squeeze(0).numpy()
    
    return upscaled

def upscale_generated_images(images, method='bicubic', scale_factor=4):
    """
    Upscale a batch of generated images.
    
    Args:
        images: numpy array of shape (N, H, W, C) or concatenated image
        method: 'bicubic', 'lanczos', or 'pytorch'
        scale_factor: factor to upscale by
    
    Returns:
        upscaled images as numpy array
    """
    if method == 'bicubic':
        upscale_func = upscale_image_bicubic
    elif method == 'lanczos':
        upscale_func = upscale_image_lanczos
    elif method == 'pytorch':
        upscale_func = upscale_image_pytorch
    else:
        raise ValueError(f"Unknown upscaling method: {method}")
    
    if len(images.shape) == 4:
        # Batch of images: (N, H, W, C)
        upscaled_images = []
        for i in range(images.shape[0]):
            upscaled = upscale_func(images[i], scale_factor)
            upscaled_images.append(upscaled)
        return np.array(upscaled_images)
    
    elif len(images.shape) == 3:
        # Single image: (H, W, C)
        return upscale_func(images, scale_factor)
    
    else:
        raise ValueError(f"Unsupported image shape: {images.shape}")

def test_upscaling():
    """Test different upscaling methods."""
    
    # Create a test 64x64 image
    test_image = np.random.rand(64, 64, 3).astype(np.float32)
    
    print("Testing upscaling methods on 64x64 image:")
    print(f"Original shape: {test_image.shape}")
    
    # Test different methods
    methods = ['bicubic', 'lanczos', 'pytorch']
    scale_factor = 4
    
    for method in methods:
        upscaled = upscale_generated_images(test_image, method=method, scale_factor=scale_factor)
        print(f"{method.capitalize()}: {upscaled.shape} (upscaled by {scale_factor}x)")
    
    print("\nRecommendation: Use 'bicubic' for good quality and speed balance.")

if __name__ == "__main__":
    test_upscaling()
