#!/usr/bin/env python3
"""
Loss Calculation System
Centralized loss calculation methods for VAE training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from perceptual_loss import create_perceptual_loss, get_perceptual_loss_info


class LossCalculator:
    """
    Centralized loss calculation system for VAE training.
    
    This class handles all loss calculations including reconstruction losses,
    perceptual losses, generation quality losses, and KL divergence.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize the loss calculator.
        
        Args:
            config: Training configuration dictionary
            device: Device to run calculations on
        """
        self.config = config
        self.device = device
        self.loss_config = config.get('loss_config', {})
        
        # Initialize perceptual loss if needed
        self.perceptual_loss_fn = None
        if self.loss_config.get('use_perceptual_loss', False):
            try:
                if get_perceptual_loss_info()['torchvision_available']:
                    # Determine optimization level based on GPU memory and config
                    gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3 if torch.cuda.is_available() else 0
                    aggressive_optimization = gpu_memory_gb < 12  # Use aggressive optimization for <12GB GPUs
                    full_vgg = gpu_memory_gb >= 16 and self.loss_config.get('full_vgg_perceptual', False)  # Use full VGG for >=16GB GPUs when enabled
                    ultra_full = gpu_memory_gb >= 20 and self.loss_config.get('ultra_full_vgg_perceptual', False)  # Use ultra-full VGG for >=20GB GPUs when enabled
                    
                    self.perceptual_loss_fn = create_perceptual_loss(device, aggressive_optimization=aggressive_optimization, full_vgg=full_vgg, ultra_full=ultra_full)
                    print(f"  ✅ Using VGG-based perceptual loss")
                else:
                    print(f"  ⚠️  VGG not available, using high-pass filter fallback")
            except Exception as e:
                print(f"  ⚠️  Error initializing perceptual loss: {e}")
    
    def calculate_mse_loss(self, images: torch.Tensor, reconst: torch.Tensor) -> torch.Tensor:
        """Calculate Mean Squared Error loss."""
        return F.mse_loss(reconst, images)
    
    def calculate_l1_loss(self, images: torch.Tensor, reconst: torch.Tensor) -> torch.Tensor:
        """Calculate L1 (Mean Absolute Error) loss."""
        return F.l1_loss(reconst, images)
    
    def calculate_perceptual_loss(self, images: torch.Tensor, reconst: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss using VGG features or high-pass filter.
        
        Args:
            images: Original images
            reconst: Reconstructed images
            
        Returns:
            Perceptual loss value
        """
        if not self.loss_config.get('use_perceptual_loss', False):
            return torch.tensor(0.0, device=images.device)
            
        try:
            # Use VGG perceptual loss if available
            if self.perceptual_loss_fn is not None:
                return self.perceptual_loss_fn.compute_loss(reconst, images)
            
            # Fallback to high-pass filter perceptual loss
            return self._calculate_highpass_perceptual_loss(images, reconst)
            
        except Exception as e:
            print(f"⚠️  Error calculating perceptual loss: {e}")
            return torch.tensor(0.0, device=images.device)
    
    def _calculate_highpass_perceptual_loss(self, images: torch.Tensor, reconst: torch.Tensor) -> torch.Tensor:
        """Calculate high-pass filter based perceptual loss as fallback."""
        # Convert to grayscale for texture analysis
        def to_grayscale(x):
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # High-pass filter to emphasize textures and edges
        def high_pass_filter(x):
            # Simple 3x3 high-pass kernel
            kernel = torch.tensor([[-1, -1, -1],
                                 [-1,  8, -1], 
                                 [-1, -1, -1]], dtype=torch.float32, device=x.device)
            kernel = kernel.view(1, 1, 3, 3)
            
            # Apply to each channel
            filtered = []
            for c in range(x.shape[1]):
                channel = x[:, c:c+1]
                filtered_channel = F.conv2d(channel, kernel, padding=1)
                filtered.append(filtered_channel)
            
            return torch.cat(filtered, dim=1)
        
        # Convert to grayscale and apply high-pass filter
        orig_gray = to_grayscale(images)
        recon_gray = to_grayscale(reconst)
        
        orig_filtered = high_pass_filter(orig_gray)
        recon_filtered = high_pass_filter(recon_gray)
        
        # L1 loss on filtered images (captures texture differences)
        perceptual_loss = F.l1_loss(orig_filtered, recon_filtered)
        
        return perceptual_loss
    
    def calculate_generation_quality_loss(self, reconst: torch.Tensor) -> torch.Tensor:
        """
        Calculate quality loss for generated images to encourage diversity and sharpness.
        
        Args:
            reconst: Reconstructed images
            
        Returns:
            Generation quality loss value
        """
        try:
            # Encourage sharp, diverse images through multiple quality metrics
            
            # 1. Edge sharpness loss (encourage sharp edges)
            edge_loss_val = self._calculate_edge_loss(reconst)
            
            # 2. Diversity loss (encourage variation within batch)
            diversity_loss_val = self._calculate_diversity_loss(reconst)
            
            # 3. Contrast loss (encourage good contrast)
            contrast_loss_val = self._calculate_contrast_loss(reconst)
            
            # Get configurable weights
            weights = self._get_generation_quality_weights()
            
            # Weighted combination
            quality_loss = (weights['edge'] * edge_loss_val + 
                          weights['diversity'] * diversity_loss_val + 
                          weights['contrast'] * contrast_loss_val)
            
            return quality_loss
            
        except Exception as e:
            print(f"⚠️  Error calculating generation quality loss: {e}")
            return torch.tensor(0.0, device=reconst.device)
    
    def _calculate_edge_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate edge sharpness loss."""
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device)
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Apply Sobel filters
        edges_x = F.conv2d(gray, sobel_x, padding=1)
        edges_y = F.conv2d(gray, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Convert to positive loss: minimize reciprocal to maximize edges
        edge_strength = torch.mean(edges)
        return 1.0 / (edge_strength + 1e-6)  # Minimize this to maximize edge strength
    
    def _calculate_diversity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate diversity loss to encourage variation within batch."""
        batch_size = x.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=x.device)
        
        # Flatten images
        x_flat = x.view(batch_size, -1)
        
        # Calculate pairwise L2 distances
        distances = torch.cdist(x_flat, x_flat, p=2)
        
        # Remove diagonal (self-comparison)
        mask = torch.eye(batch_size, device=x.device).bool()
        distances = distances[~mask]
        
        # Convert to positive loss: minimize reciprocal to maximize diversity
        diversity_strength = torch.mean(distances)
        return 1.0 / (diversity_strength + 1e-6)  # Minimize this to maximize diversity
    
    def _calculate_contrast_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate contrast loss to encourage good contrast."""
        # Calculate local contrast
        kernel = torch.ones(3, 3, dtype=torch.float32, device=x.device) / 9
        kernel = kernel.view(1, 1, 3, 3)
        
        # Convert to grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Local mean
        local_mean = F.conv2d(gray, kernel, padding=1)
        
        # Local variance (contrast)
        local_var = F.conv2d((gray - local_mean)**2, kernel, padding=1)
        
        # Convert to positive loss: minimize reciprocal to maximize contrast
        contrast_strength = torch.mean(local_var)
        return 1.0 / (contrast_strength + 1e-6)  # Minimize this to maximize contrast
    
    def calculate_kl_loss(self, model: nn.Module, emb_mean: torch.Tensor, emb_log_var: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence loss.
        
        Args:
            model: VAE model
            emb_mean: Encoder mean output
            emb_log_var: Encoder log variance output
            
        Returns:
            KL divergence loss value
        """
        return model.kl_loss(emb_mean, emb_log_var)
    
    def calculate_all_losses(self, images: torch.Tensor, reconst: torch.Tensor, 
                           emb_mean: torch.Tensor, emb_log_var: torch.Tensor,
                           model: nn.Module, loss_manager) -> Dict[str, torch.Tensor]:
        """
        Calculate all loss components and return comprehensive loss dictionary.
        
        Args:
            images: Original images
            reconst: Reconstructed images
            emb_mean: Encoder mean output
            emb_log_var: Encoder log variance output
            model: VAE model
            loss_manager: LossWeightManager instance
            
        Returns:
            Dictionary containing all loss components and total loss
        """
        # Basic reconstruction losses
        mse_loss = self.calculate_mse_loss(images, reconst)
        l1_loss = self.calculate_l1_loss(images, reconst)
        
        # Perceptual loss (reconstruction quality)
        perceptual_loss = self.calculate_perceptual_loss(images, reconst)
        
        # Generation quality loss (encourages diverse, sharp generated images)
        generation_quality_loss = self.calculate_generation_quality_loss(reconst)
        
        # Get current loss weights (with stage-based scheduling if enabled)
        mse_weight = loss_manager.get_weight('mse_weight')
        l1_weight = loss_manager.get_weight('l1_weight')
        generation_weight = loss_manager.get_weight('generation_weight')
        perceptual_weight = loss_manager.get_weight('perceptual_weight')
        
        # Combine reconstruction losses
        recon_loss = (mse_weight * mse_loss + 
                     l1_weight * l1_loss + 
                     perceptual_weight * perceptual_loss +
                     generation_weight * generation_quality_loss)
        
        # KL divergence loss
        kl_loss = self.calculate_kl_loss(model, emb_mean, emb_log_var)
        
        # Get current beta (with scheduling if enabled)
        current_beta = loss_manager.get_weight('beta')
        
        # Total loss
        total_loss = recon_loss + current_beta * kl_loss
        
        # Store loss components for logging
        loss_dict = {
            'loss': total_loss,
            'mse': mse_loss,
            'l1': l1_loss,
            'perceptual': perceptual_loss,
            'generation_quality': generation_quality_loss,
            'kl': kl_loss,
            'recon_loss': recon_loss,
            'beta': current_beta
        }
        
        return loss_dict
    
    def _get_generation_quality_weights(self) -> Dict[str, float]:
        """Get generation quality component weights from config or defaults."""
        # Check if custom weights are specified in config
        gen_qual_config = self.config.get('generation_quality_config', {})
        
        # Default weights
        default_weights = {
            'edge': 0.2,
            'diversity': 0.4,
            'contrast': 0.4
        }
        
        # Use config values if available, otherwise defaults
        weights = {
            'edge': gen_qual_config.get('edge_weight', default_weights['edge']),
            'diversity': gen_qual_config.get('diversity_weight', default_weights['diversity']),
            'contrast': gen_qual_config.get('contrast_weight', default_weights['contrast'])
        }
        
        # Normalize weights to ensure they sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def get_loss_info(self) -> Dict[str, Any]:
        """
        Get information about available loss functions.
        
        Returns:
            Dictionary with loss function information
        """
        return {
            'perceptual_available': self.perceptual_loss_fn is not None,
            'perceptual_type': 'VGG' if self.perceptual_loss_fn is not None else 'HighPass',
            'generation_quality_available': True,
            'generation_quality_weights': self._get_generation_quality_weights(),
            'mse_available': True,
            'l1_available': True,
            'kl_available': True
        }


def create_loss_calculator(config: Dict[str, Any], device: str = 'cuda') -> LossCalculator:
    """
    Factory function to create a loss calculator.
    
    Args:
        config: Training configuration dictionary
        device: Device to run calculations on
        
    Returns:
        LossCalculator instance
    """
    return LossCalculator(config, device)
