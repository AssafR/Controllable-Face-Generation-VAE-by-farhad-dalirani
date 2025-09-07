#!/usr/bin/env python3
"""
Perceptual Loss Implementation
Isolated VGG-based perceptual loss with memory optimization and fallback support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# Try to import torchvision for VGG perceptual loss
try:
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    models = None
    transforms = None


class VGGPerceptualLoss(nn.Module):
    """
    Memory-optimized VGG-based perceptual loss.
    
    This class provides a VGG19-based perceptual loss implementation with
    aggressive memory optimization options and graceful fallback when
    torchvision is not available.
    """
    
    def __init__(self, device: str = 'cuda', aggressive_optimization: bool = False):
        """
        Initialize VGG perceptual loss.
        
        Args:
            device: Device to run the model on
            aggressive_optimization: If True, use only 3 VGG layers for maximum memory savings
        """
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for VGG perceptual loss")
        
        self.device = device
        self.aggressive_optimization = aggressive_optimization
        
        # Load pre-trained VGG19 using modern API
        self._load_vgg_model()
        
        # Build minimal VGG with only necessary layers
        self._build_minimal_vgg()
        
        # Move to device
        self.to(device)
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def _load_vgg_model(self) -> None:
        """Load pre-trained VGG19 model with compatibility for different torchvision versions."""
        try:
            # Modern torchvision (0.13+) - preferred method
            full_vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
            print("  ✅ Using modern VGG19 weights API (torchvision 0.13+)")
        except AttributeError:
            # Fallback for torchvision 0.12-0.13
            try:
                full_vgg = models.vgg19(weights='IMAGENET1K_V1').features
                print("  ✅ Using VGG19 weights API (torchvision 0.12-0.13)")
            except TypeError:
                # Very old torchvision - use deprecated API with warning suppression
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    full_vgg = models.vgg19(pretrained=True).features
                print("  ⚠️  Using deprecated VGG19 API (torchvision <0.12) - consider upgrading")
        
        self.full_vgg = full_vgg
    
    def _build_minimal_vgg(self) -> None:
        """Build minimal VGG with only necessary layers for memory optimization."""
        if self.aggressive_optimization:
            # Ultra-aggressive optimization: only use 2-3 layers
            self.feature_layers = [1, 6, 11]  # conv1_2, conv2_2, conv3_4 only
            print("  🚀 Using aggressive VGG optimization (3 layers only)")
        else:
            # Standard optimization: use 5 layers
            self.feature_layers = [1, 6, 11, 20, 29]  # conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
            print("  ⚡ Using standard VGG optimization (5 layers)")
        
        # Build minimal VGG with only necessary layers
        self.vgg_layers = nn.ModuleList()
        for i, layer in enumerate(self.full_vgg):
            if i in self.feature_layers:
                self.vgg_layers.append(layer)
            elif i > max(self.feature_layers):
                break
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through VGG layers.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature maps from selected VGG layers
        """
        return self._extract_features(x)
    
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from VGG layers.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature tensors
        """
        features = []
        layer_idx = 0
        
        for i, layer in enumerate(self.full_vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
                layer_idx += 1
                if layer_idx >= len(self.vgg_layers):
                    break
        
        return features
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            pred: Predicted images tensor
            target: Target images tensor
            
        Returns:
            Perceptual loss value
        """
        # Normalize inputs to ImageNet mean/std if needed
        pred = self._normalize_input(pred)
        target = self._normalize_input(target)
        
        # Extract features
        pred_features = self.forward(pred)
        target_features = self.forward(target)
        
        # Compute loss
        loss = 0.0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to ImageNet statistics if needed.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Check if input is already normalized (values in [0, 1])
        if x.min() >= 0 and x.max() <= 1:
            # Normalize to ImageNet mean/std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
            x = (x - mean) / std
        
        return x
    
    def get_memory_usage(self) -> dict:
        """
        Get memory usage information for the VGG model.
        
        Returns:
            Dictionary with memory usage statistics
        """
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        
        device = next(self.parameters()).device
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        
        return {
            "cuda_available": True,
            "device": str(device),
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
            "feature_layers": len(self.feature_layers),
            "aggressive_optimization": self.aggressive_optimization
        }


class HighPassPerceptualLoss(nn.Module):
    """
    High-pass filter based perceptual loss as a lightweight alternative to VGG.
    
    This is a fallback option when VGG is not available or when memory is extremely limited.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize high-pass perceptual loss.
        
        Args:
            device: Device to run the model on
        """
        super().__init__()
        self.device = device
        
        # High-pass filter kernel (Laplacian)
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Create 3-channel kernel
        self.kernel = kernel.repeat(3, 1, 1, 1).to(device)
        
        print("  ⚡ Using high-pass filter perceptual loss (lightweight)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply high-pass filter to input.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            High-pass filtered tensor
        """
        return F.conv2d(x, self.kernel, padding=1, groups=3)
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute high-pass perceptual loss.
        
        Args:
            pred: Predicted images tensor
            target: Target images tensor
            
        Returns:
            High-pass perceptual loss value
        """
        pred_high = self.forward(pred)
        target_high = self.forward(target)
        
        return F.mse_loss(pred_high, target_high)


def create_perceptual_loss(device: str = 'cuda', 
                          aggressive_optimization: bool = False,
                          fallback_to_highpass: bool = True) -> nn.Module:
    """
    Factory function to create a perceptual loss module.
    
    Args:
        device: Device to run the model on
        aggressive_optimization: If True, use aggressive VGG optimization
        fallback_to_highpass: If True, fallback to high-pass filter when VGG unavailable
        
    Returns:
        Perceptual loss module
    """
    if TORCHVISION_AVAILABLE:
        return VGGPerceptualLoss(device, aggressive_optimization)
    elif fallback_to_highpass:
        print("  ⚠️  VGG not available, using high-pass filter fallback")
        return HighPassPerceptualLoss(device)
    else:
        raise ImportError("torchvision is required for VGG perceptual loss and fallback is disabled")


def get_perceptual_loss_info() -> dict:
    """
    Get information about available perceptual loss options.
    
    Returns:
        Dictionary with availability information
    """
    return {
        "torchvision_available": TORCHVISION_AVAILABLE,
        "vgg_available": TORCHVISION_AVAILABLE,
        "highpass_available": True,
        "recommended": "VGG" if TORCHVISION_AVAILABLE else "HighPass"
    }
