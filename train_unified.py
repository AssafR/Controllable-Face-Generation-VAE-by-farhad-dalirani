#!/usr/bin/env python3
"""
Unified VAE Training Script
Consolidates all training methods into a single, configurable system.
Supports all training presets through command-line arguments.
"""

import sys
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import time
import json

# Try to import torchvision for VGG perceptual loss
try:
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("⚠️  TensorBoard not available, logging disabled")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

import numpy as np
from tqdm import tqdm

# Import our modules
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import get_split_data, configure_gpu, display_image_grid, display_comparison_grid
from config_loader import ConfigLoader

class VGGPerceptualLoss(nn.Module):
    """Memory-optimized VGG-based perceptual loss."""
    
    def __init__(self, device='cuda', aggressive_optimization=False):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for VGG perceptual loss")
        
        # Load pre-trained VGG19 using modern API
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
        
        if aggressive_optimization:
            # Ultra-aggressive optimization: only use 2-3 layers
            self.feature_layers = [1, 6, 11]  # conv1_2, conv2_2, conv3_4 only
            print("  🚀 Using aggressive VGG optimization (3 layers only)")
        else:
            # Standard optimization: use 5 layers
            self.feature_layers = [1, 6, 11, 20, 29]  # conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
            print("  ⚡ Using standard VGG optimization (5 layers)")
        
        # Build minimal VGG with only necessary layers
        self.vgg_layers = nn.ModuleList()
        for i, layer in enumerate(full_vgg):
            if i <= max(self.feature_layers):
                self.vgg_layers.append(layer)
            else:
                break  # Stop after the last layer we need
        
        # Move to device
        self.vgg_layers = self.vgg_layers.to(device)
        
        # Freeze parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        # Normalize images to VGG input range
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Calculate memory savings
        original_params = sum(p.numel() for p in full_vgg.parameters())
        optimized_params = sum(p.numel() for p in self.vgg_layers.parameters())
        memory_savings = (1 - optimized_params / original_params) * 100
        print(f"  💾 VGG Memory Optimization: {memory_savings:.1f}% reduction ({optimized_params:,} vs {original_params:,} parameters)")
    
    def forward(self, x, y):
        """Calculate perceptual loss between x and y."""
        # Normalize to VGG input range
        x_norm = self.normalize(x)
        y_norm = self.normalize(y)
        
        # Extract features
        x_features = self._extract_features(x_norm)
        y_features = self._extract_features(y_norm)
        
        # Calculate L1 loss for each feature layer
        loss = 0.0
        for x_feat, y_feat in zip(x_features, y_features):
            loss += F.l1_loss(x_feat, y_feat)
        
        return loss / len(x_features)
    
    def _extract_features(self, x):
        """Extract features from optimized VGG layers."""
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

class UnifiedVAETrainer:
    """Unified VAE trainer that supports all training configurations."""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize loss history tracking
        self.loss_history = {
            'perceptual': [],
            'generation_quality': [],
            'kl': [],
            'total': []
        }
        
        # Initialize perceptual loss and adjust batch size
        self.perceptual_loss_fn = None
        self.original_batch_size = config['batch_size']
        self.user_override_batch_size = config.get('_user_override_batch_size', False)
        loss_config = config.get('loss_config', {})
        if loss_config.get('use_perceptual_loss', False):
            try:
                if TORCHVISION_AVAILABLE:
                    # Determine if we need aggressive optimization based on GPU memory
                    gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3 if torch.cuda.is_available() else 0
                    aggressive_optimization = gpu_memory_gb < 12  # Use aggressive optimization for <12GB GPUs
                    
                    self.perceptual_loss_fn = VGGPerceptualLoss(device, aggressive_optimization=aggressive_optimization)
                    # VGG uses significant memory, reduce batch size intelligently
                    self._adjust_batch_size_for_vgg(config, device)
                    print(f"  ✅ Using VGG-based perceptual loss")
                    if self.user_override_batch_size:
                        print(f"  ⚡ User specified batch size: {config['batch_size']} (VGG memory analysis applied)")
                    else:
                        print(f"  ⚡ Adjusted batch size from {self.original_batch_size} to {config['batch_size']} for VGG memory requirements")
                else:
                    print("  ⚠️  torchvision not available, using high-pass filter perceptual loss")
            except Exception as e:
                print(f"  ⚠️  Error initializing VGG perceptual loss: {e}")
                print("  ⚠️  Falling back to high-pass filter perceptual loss")
    
    def _adjust_batch_size_for_vgg(self, config, device):
        """Intelligently adjust batch size for VGG perceptual loss memory requirements."""
        if not torch.cuda.is_available():
            # CPU training - use smaller batch size
            config['batch_size'] = max(8, config['batch_size'] // 8)
            return
        
        # Get GPU memory info
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        # Estimate VGG memory usage (rough estimates)
        # Optimized VGG19: ~1-2GB for forward pass + features (vs 3-5GB for full VGG)
        # VAE model: ~1-2GB depending on resolution
        # Images: batch_size * height * width * 3 * 4 bytes
        
        image_memory_per_sample = config['input_img_size'] ** 2 * 3 * 4 / 1024**3  # GB per image
        
        # Conservative memory allocation
        available_memory = gpu_memory_gb * 0.8  # Use 80% of GPU memory
        
        # Determine VGG overhead based on optimization level
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        if gpu_memory_gb < 12:
            vgg_overhead = 1.5  # GB for aggressive VGG optimization (3 layers)
        else:
            vgg_overhead = 2.5  # GB for standard VGG optimization (5 layers)
        
        vae_overhead = 2.0  # GB for VAE model
        
        usable_memory = available_memory - vgg_overhead - vae_overhead
        
        if usable_memory > 0:
            max_batch_size = int(usable_memory / image_memory_per_sample)
            config['batch_size'] = max(8, min(config['batch_size'], max_batch_size))
        else:
            # Very limited memory - use minimum batch size
            config['batch_size'] = 8
        
        # Ensure batch size is reasonable
        if not self.user_override_batch_size:
            # Only cap if user didn't explicitly set batch size
            config['batch_size'] = max(8, min(config['batch_size'], 512))
        else:
            # User specified batch size - only ensure it's not too small
            config['batch_size'] = max(8, config['batch_size'])
        
        # Print memory analysis
        print(f"  📊 Memory Analysis:")
        print(f"    • GPU Memory: {gpu_memory_gb:.1f}GB total")
        print(f"    • VGG Overhead: ~4GB")
        print(f"    • VAE Overhead: ~2GB")
        print(f"    • Image Memory: {image_memory_per_sample:.3f}GB per sample")
        print(f"    • Recommended batch size: {config['batch_size']}")
        
        if config['batch_size'] < self.original_batch_size:
            print(f"    ⚠️  Reduced from {self.original_batch_size} due to VGG memory requirements")
        else:
            print(f"    ✅ No batch size reduction needed")
    
    def setup_logging(self):
        """Setup TensorBoard logging."""
        if TENSORBOARD_AVAILABLE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = self.config.get('config_name', 'unified')
            log_dir = f"runs/{config_name}_{timestamp}"
            self.writer = SummaryWriter(log_dir)
            print(f"  • Logging: {log_dir}")
        else:
            self.writer = None
            print(f"  • Logging: Disabled (TensorBoard not available)")
    
    def create_model(self):
        """Create and configure the VAE model."""
        print(f"\n🏗️  Creating model...")
        self.model = VAE_pt(
            input_img_size=self.config['input_img_size'],
            embedding_size=self.config['embedding_size'],
            loss_config=self.config
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ Model created:")
        print(f"  • Total parameters: {total_params:,}")
        print(f"  • Trainable parameters: {trainable_params:,}")
        print(f"  • Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        return self.model
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8, 
            min_lr=1e-7
        )
        
        print(f"✅ Training setup complete:")
        print(f"  • Optimizer: Adam (lr={self.config['lr']}, weight_decay=1e-5)")
        print(f"  • Scheduler: ReduceLROnPlateau (patience=8)")
        print(f"  • Batch size: {self.config['batch_size']}")
    
    def load_data(self):
        """Load and prepare the dataset."""
        print(f"\n📊 Loading dataset...")
        train_data, val_data = get_split_data(config=self.config)
        
        train_loader = DataLoader(
            train_data, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=0
        )
        
        print(f"✅ Dataset loaded:")
        print(f"  • Training samples: {len(train_data):,}")
        print(f"  • Validation samples: {len(val_data):,}")
        print(f"  • Training batches: {len(train_loader):,}")
        print(f"  • Validation batches: {len(val_loader):,}")
        
        return train_loader, val_loader
    
    def calculate_perceptual_loss(self, images, reconst):
        """Calculate perceptual loss using VGG features or high-pass filter."""
        loss_config = self.config.get('loss_config', {})
        if not loss_config.get('use_perceptual_loss', False):
            return 0.0
            
        try:
            # Use VGG perceptual loss if available
            if self.perceptual_loss_fn is not None:
                return self.perceptual_loss_fn(images, reconst)
            
            # Fallback to high-pass filter perceptual loss
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
            
        except Exception as e:
            print(f"⚠️  Error calculating perceptual loss: {e}")
            return 0.0
    
    def calculate_generation_quality_loss(self, reconst):
        """Calculate quality loss for generated images to encourage diversity and sharpness."""
        try:
            # Encourage sharp, diverse images through multiple quality metrics
            
            # 1. Edge sharpness loss (encourage sharp edges)
            def edge_loss(x):
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
                # Add small epsilon to prevent division by zero and stabilize gradients
                edge_strength = torch.mean(edges)
                return 1.0 / (edge_strength + 1e-6)  # Minimize this to maximize edge strength
            
            # 2. Diversity loss (encourage variation within batch)
            def diversity_loss(x):
                # Calculate pairwise differences between images in batch
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
            
            # 3. Contrast loss (encourage good contrast)
            def contrast_loss(x):
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
            
            # Combine quality losses
            edge_loss_val = edge_loss(reconst)
            diversity_loss_val = diversity_loss(reconst)
            contrast_loss_val = contrast_loss(reconst)
            
            # Weighted combination
            quality_loss = 0.4 * edge_loss_val + 0.3 * diversity_loss_val + 0.3 * contrast_loss_val
            
            return quality_loss
            
        except Exception as e:
            print(f"⚠️  Error calculating generation quality loss: {e}")
            return torch.tensor(0.0, device=reconst.device)
    
    def calculate_losses(self, images, reconst, emb_mean, emb_log_var):
        """Calculate all loss components."""
        # Basic reconstruction losses
        mse_loss = F.mse_loss(reconst, images)
        l1_loss = F.l1_loss(reconst, images)
        
        # Perceptual loss (reconstruction quality)
        perceptual_loss = self.calculate_perceptual_loss(images, reconst)
        
        # Generation quality loss (encourages diverse, sharp generated images)
        generation_quality_loss = self.calculate_generation_quality_loss(reconst)
        
        # Get loss weights
        loss_config = self.config.get('loss_config', {})
        mse_weight = loss_config.get('mse_weight', 0.5)
        l1_weight = loss_config.get('l1_weight', 0.3)
        perceptual_weight = loss_config.get('perceptual_weight', 0.2)
        generation_weight = loss_config.get('generation_weight', 0.1)  # New weight for generation quality
        
        # Combine reconstruction losses
        recon_loss = (mse_weight * mse_loss + 
                     l1_weight * l1_loss + 
                     perceptual_weight * perceptual_loss +
                     generation_weight * generation_quality_loss)
        
        # KL divergence loss
        kl_loss = self.model.kl_loss(emb_mean, emb_log_var)
        
        # Total loss
        total_loss = recon_loss + self.config['beta'] * kl_loss
        
        # Store loss components for logging
        loss_dict = {
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss,
            'generation_quality': generation_quality_loss.item() if isinstance(generation_quality_loss, torch.Tensor) else generation_quality_loss,
            'kl': kl_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_kl = 0.0
        train_l1 = 0.0
        train_perceptual = 0.0
        train_generation_quality = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['max_epoch']} [Train]", leave=False)
        
        for batch_idx, images in enumerate(train_pbar):
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            emb_mean, emb_log_var, reconst = self.model(images)
            
            # Calculate losses
            total_loss, loss_dict = self.calculate_losses(images, reconst, emb_mean, emb_log_var)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            train_loss += loss_dict['total']
            train_mse += loss_dict['mse']
            train_kl += loss_dict['kl']
            train_l1 += loss_dict['l1']
            train_perceptual += loss_dict['perceptual']
            train_generation_quality += loss_dict['generation_quality']
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.5f}',
                'MSE': f'{loss_dict["mse"]:.4f}',
                'L1': f'{loss_dict["l1"]:.4f}',
                'Perceptual': f'{loss_dict["perceptual"]:.4f}',
                'GenQual': f'{loss_dict["generation_quality"]:.4f}',
                'KL': f'{loss_dict["kl"]:.4f}'
            })
        
        # Calculate averages
        num_batches = len(train_loader)
        return {
            'loss': train_loss / num_batches,
            'mse': train_mse / num_batches,
            'kl': train_kl / num_batches,
            'l1': train_l1 / num_batches,
            'perceptual': train_perceptual / num_batches,
            'generation_quality': train_generation_quality / num_batches
        }
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_kl = 0.0
        val_l1 = 0.0
        val_perceptual = 0.0
        val_generation_quality = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config['max_epoch']} [Val]", leave=False)
            
            for batch_idx, images in enumerate(val_pbar):
                images = images.to(self.device)
                
                # Forward pass
                emb_mean, emb_log_var, reconst = self.model(images)
                
                # Calculate losses
                total_loss, loss_dict = self.calculate_losses(images, reconst, emb_mean, emb_log_var)
                
                val_loss += loss_dict['total']
                val_mse += loss_dict['mse']
                val_kl += loss_dict['kl']
                val_l1 += loss_dict['l1']
                val_perceptual += loss_dict['perceptual']
                val_generation_quality += loss_dict['generation_quality']
                
                val_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.6f}',
                    'MSE': f'{loss_dict["mse"]:.6f}',
                    'L1': f'{loss_dict["l1"]:.6f}',
                    'Perceptual': f'{loss_dict["perceptual"]:.6f}',
                    'GenQual': f'{loss_dict["generation_quality"]:.6f}',
                    'KL': f'{loss_dict["kl"]:.6f}'
                })
        
        # Calculate averages
        num_batches = len(val_loader)
        return {
            'loss': val_loss / num_batches,
            'mse': val_mse / num_batches,
            'kl': val_kl / num_batches,
            'l1': val_l1 / num_batches,
            'perceptual': val_perceptual / num_batches,
            'generation_quality': val_generation_quality / num_batches
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'config': self.config
        }
        
        # Save regular checkpoint
        config_name = self.config.get('config_name', 'unified')
        checkpoint_path = os.path.join(checkpoint_dir, f"{config_name}_training_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            os.makedirs(self.config['model_save_path'], exist_ok=True)
            best_model_path = f"{self.config['model_save_path']}/{config_name}_vae_best_model.pth"
            torch.save(self.model.state_dict(), best_model_path)
            print(f"  ✅ New best model saved: {best_model_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(checkpoint_dir, f"{config_name}_checkpoint_epoch_{epoch+1:03d}.pth")
            torch.save(checkpoint, periodic_path)
            print(f"  💾 Periodic checkpoint saved: {periodic_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if os.path.exists(checkpoint_path):
            print(f"🔄 Found checkpoint: {checkpoint_path}")
            print("   Resuming training from checkpoint...")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            self.patience_counter = checkpoint.get('patience_counter', 0)
            
            print(f"   ✅ Resumed from epoch {self.start_epoch}")
            print(f"   📊 Best validation loss: {self.best_val_loss:.6f}")
            print(f"   ⏳ Patience counter: {self.patience_counter}")
            
            # Display current configuration when resuming
            self.print_configuration("Resumed")
            return True
        else:
            print("🆕 No checkpoint found, starting fresh training")
            return False
    
    def clear_old_files(self):
        """Clear old training files for fresh training (only PNG files)."""
        print("🧹 Clearing old training files for fresh training...")
        
        # Check what will be deleted
        files_to_delete = []
        checkpoint_dir = "checkpoints"
        model_dir = self.config['model_save_path']
        sample_dir = "sample_images"
        
        # Count PNG files only
        png_count = 0
        if os.path.exists(checkpoint_dir):
            png_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.png')]
            png_count += len(png_files)
        if os.path.exists(model_dir):
            png_files = [f for f in os.listdir(model_dir) if f.endswith('.png')]
            png_count += len(png_files)
        if os.path.exists(sample_dir):
            png_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
            png_count += len(png_files)
        
        if png_count > 0:
            print(f"  📁 Will delete {png_count} PNG files from training directories")
            
            # Ask for confirmation
            response = input("  ❓ Continue? This will permanently delete old training images (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("  ❌ Cancelled. Use without --no-resume to continue existing training.")
                import sys
                sys.exit(0)
        else:
            print("  ℹ️  No old PNG files found to delete.")
        
        # Clear only PNG files from directories
        deleted_count = 0
        
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.png'):
                    os.remove(os.path.join(checkpoint_dir, file))
                    deleted_count += 1
        
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.png'):
                    os.remove(os.path.join(model_dir, file))
                    deleted_count += 1
        
        if os.path.exists(sample_dir):
            for file in os.listdir(sample_dir):
                if file.endswith('.png'):
                    os.remove(os.path.join(sample_dir, file))
                    deleted_count += 1
        
        if deleted_count > 0:
            print(f"  ✅ Deleted {deleted_count} PNG files")
        else:
            print("  ℹ️  No PNG files to delete")
        
        print("  🆕 Ready for fresh training!")
    
    def _assess_loss_behavior(self, train_metrics, val_metrics, epoch):
        """Concise assessment of loss behavior and quality indicators."""
        print(f"\n📈 TREND ANALYSIS:")
        print(f"{'─'*40}")
        
        # loss_history is now initialized in __init__
        
        # Store current values
        self.loss_history['perceptual'].append(train_metrics['perceptual'])
        self.loss_history['generation_quality'].append(train_metrics['generation_quality'])
        self.loss_history['kl'].append(train_metrics['kl'])
        self.loss_history['total'].append(train_metrics['loss'])
        
        # Only show trends if we have enough history
        if len(self.loss_history['total']) < 2:
            print(f"  📊 Collecting data... (need 2+ epochs for trends)")
            return
        
        # Perceptual trend (if enabled)
        if len(self.loss_history['perceptual']) > 1 and self.config.get('loss_config', {}).get('perceptual_weight', 0) > 0:
            perc_trend = self.loss_history['perceptual'][-1] - self.loss_history['perceptual'][-2]
            trend_icon = "📈" if perc_trend < 0 else "📉" if perc_trend > 0 else "➡️"
            print(f"  {trend_icon} Perceptual: {perc_trend:+.4f} change")
        
        # Generation quality trend (if enabled)
        if len(self.loss_history['generation_quality']) > 1 and self.config.get('loss_config', {}).get('generation_weight', 0) > 0:
            gen_trend = self.loss_history['generation_quality'][-1] - self.loss_history['generation_quality'][-2]
            trend_icon = "📈" if gen_trend < 0 else "📉" if gen_trend > 0 else "➡️"  # More negative = better
            print(f"  {trend_icon} GenQual: {gen_trend:+.4f} change")
        
        # KL trend
        if len(self.loss_history['kl']) > 1:
            kl_trend = self.loss_history['kl'][-1] - self.loss_history['kl'][-2]
            trend_icon = "📈" if kl_trend > 0 else "📉" if kl_trend < 0 else "➡️"
            print(f"  {trend_icon} KL: {kl_trend:+.4f} change")
        
        # Overall training trend
        if len(self.loss_history['total']) > 3:
            recent_losses = self.loss_history['total'][-3:]
            if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                print(f"  🟢 Overall: Converging (loss decreasing)")
            elif all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                print(f"  🔴 Overall: Diverging (loss increasing)")
            else:
                print(f"  🟡 Overall: Oscillating (loss fluctuating)")
        
        # Quick recommendations
        recommendations = []
        if train_metrics['kl'] < 0.0001:
            recommendations.append("🔴 POSTERIOR COLLAPSE - increase beta!")
        elif train_metrics['kl'] < 0.01:
            recommendations.append("🟡 KL weak - consider increasing beta")
        
        if self.config.get('loss_config', {}).get('perceptual_weight', 0) > 0 and train_metrics['perceptual'] < 0.001:
            recommendations.append("🟡 Perceptual weak - increase weight")
        
        if self.config.get('loss_config', {}).get('generation_weight', 0) > 0 and train_metrics['generation_quality'] > -0.001:
            recommendations.append("🟡 GenQual weak - increase weight")
        
        if len(self.loss_history['total']) > 5 and self.loss_history['total'][-1] > self.loss_history['total'][-5]:
            recommendations.append("🟡 Consider reducing learning rate")
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print(f"\n✅ All systems healthy! 🎉")
    
    def print_configuration(self, mode="Fresh"):
        """Print current configuration."""
        print(f"\n📋 Current Configuration ({mode}):")
        print(f"  • Input size: {self.config['input_img_size']}x{self.config['input_img_size']}")
        print(f"  • Embedding size: {self.config['embedding_size']}")
        print(f"  • Max epochs: {self.config['max_epoch']}")
        print(f"  • Batch size: {self.config['batch_size']}")
        print(f"  • Learning rate: {self.config['lr']}")
        print(f"  • Beta (KL weight): {self.config['beta']}")
        
        # Print loss configuration
        loss_config = self.config.get('loss_config', {})
        print(f"  • Loss weights: MSE={loss_config.get('mse_weight', 0)}, L1={loss_config.get('l1_weight', 0)}, Perceptual={loss_config.get('perceptual_weight', 0)}, GenQual={loss_config.get('generation_weight', 0)}")
        print(f"  • Loss components: MSE={loss_config.get('use_mse', False)}, L1={loss_config.get('use_l1', False)}, Perceptual={loss_config.get('use_perceptual_loss', False)}")
        print(f"  • Generation Quality: Edge Sharpness (40%), Diversity (30%), Contrast (30%)")
    
    def generate_samples(self, epoch, val_data):
        """Generate sample images."""
        print(f"  🖼️  Generating sample images...")
        
        # Create sample_images directory
        sample_dir = "sample_images"
        os.makedirs(sample_dir, exist_ok=True)
        
        with torch.no_grad():
            # Generate random images
            z = torch.randn(8, self.config['embedding_size']).to(self.device)
            generated = self.model.dec(z)
            generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
            
            # Log to tensorboard
            if self.writer is not None:
                self.writer.add_images('Generated/Images', generated, epoch)
            
            # Get clean config name for filename
            config_name = self.config.get('config_name', 'unified')
            
            # Save sample images
            sample_path = os.path.join(sample_dir, f"{config_name}_generated_epoch_{epoch+1:03d}.png")
            titles = [f"Epoch {epoch+1} - Sample {i+1}" for i in range(8)]
            display_image_grid(generated_np, 
                              titles=titles,
                              max_cols=4, 
                              figsize=(16, 8),
                              save_path=sample_path)
            print(f"  ✅ Sample images saved: {sample_path}")
            
            # Also generate reconstruction samples
            val_indices = torch.randperm(len(val_data))[:4]
            val_images = torch.stack([val_data[i] for i in val_indices]).to(self.device)
            z_mean, z_log_var, z = self.model.enc(val_images)
            reconstructed = self.model.dec(z)
            
            val_np = val_images.permute(0, 2, 3, 1).cpu().numpy()
            recon_np = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
            
            recon_path = os.path.join(sample_dir, f"{config_name}_reconstruction_epoch_{epoch+1:03d}.png")
            display_comparison_grid(val_np, recon_np,
                                   titles=[f"Epoch {epoch+1} - Pair {i+1}" for i in range(4)],
                                   max_cols=2, 
                                   figsize=(12, 8),
                                   save_path=recon_path)
            print(f"  ✅ Reconstruction samples saved: {recon_path}")
    
    def train(self, resume=True):
        """Main training loop."""
        print("🚀 Starting Unified VAE Training")
        print("=" * 70)
        
        # Print initial configuration
        self.print_configuration("Fresh")
        
        # Setup logging
        self.setup_logging()
        
        # Create model
        self.create_model()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Load data
        train_loader, val_loader = self.load_data()
        
        # Handle checkpoint loading/clearing
        config_name = self.config.get('config_name', 'unified')
        checkpoint_path = f"checkpoints/{config_name}_training_checkpoint.pth"
        if not resume:
            # Clear old checkpoints and model files for fresh start
            self.clear_old_files()
        elif resume and self.load_checkpoint(checkpoint_path):
            pass  # Configuration already printed
        
        # Training loop
        print(f"\n🎯 Starting training...")
        start_time = time.time()
        early_stopping_patience = 12
        
        for epoch in range(self.start_epoch, self.config['max_epoch']):
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
                self.writer.add_scalar('MSE/Train', train_metrics['mse'], epoch)
                self.writer.add_scalar('MSE/Val', val_metrics['mse'], epoch)
                self.writer.add_scalar('KL/Train', train_metrics['kl'], epoch)
                self.writer.add_scalar('KL/Val', val_metrics['kl'], epoch)
                self.writer.add_scalar('L1/Train', train_metrics['l1'], epoch)
                self.writer.add_scalar('L1/Val', val_metrics['l1'], epoch)
                self.writer.add_scalar('Perceptual/Train', train_metrics['perceptual'], epoch)
                self.writer.add_scalar('Perceptual/Val', val_metrics['perceptual'], epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Get GPU statistics
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
                gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
                gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100
            else:
                gpu_memory_allocated = gpu_memory_reserved = gpu_memory_total = gpu_utilization = 0
            
            # Print epoch summary with clean, scannable formatting
            print(f"\n{'='*60}")
            print(f"📊 EPOCH {epoch+1}/{self.config['max_epoch']} SUMMARY")
            print(f"{'='*60}")
            
            # Main metrics - easy to scan
            print(f"🎯 TOTAL LOSS: {train_metrics['loss']:.4f} (Train) | {val_metrics['loss']:.4f} (Val)")
            print(f"📈 LEARNING RATE: {current_lr:.2e}")
            print(f"💾 GPU MEMORY: {gpu_memory_allocated:.1f}GB / {gpu_memory_total:.1f}GB ({gpu_utilization:.0f}%)")
            
            print(f"\n🔍 LOSS BREAKDOWN:")
            print(f"{'─'*40}")
            
            # Loss components with clear formatting and status indicators
            loss_config = self.config.get('loss_config', {})
            mse_weight = loss_config.get('mse_weight', 0)
            l1_weight = loss_config.get('l1_weight', 0)
            perceptual_weight = loss_config.get('perceptual_weight', 0)
            generation_weight = loss_config.get('generation_weight', 0)
            
            if mse_weight > 0:
                mse_contrib = (train_metrics['mse'] * mse_weight) / train_metrics['loss'] * 100
                status = "✅" if 5 <= mse_contrib <= 50 else "⚠️"
                print(f"  {status} MSE:     {train_metrics['mse']:.4f} ({mse_contrib:.0f}% of total)")
            
            if l1_weight > 0:
                l1_contrib = (train_metrics['l1'] * l1_weight) / train_metrics['loss'] * 100
                status = "✅" if 5 <= l1_contrib <= 50 else "⚠️"
                print(f"  {status} L1:      {train_metrics['l1']:.4f} ({l1_contrib:.0f}% of total)")
            
            if perceptual_weight > 0:
                perc_contrib = (train_metrics['perceptual'] * perceptual_weight) / train_metrics['loss'] * 100
                status = "✅" if perc_contrib >= 10 else "⚠️"
                print(f"  {status} Perceptual: {train_metrics['perceptual']:.4f} ({perc_contrib:.0f}% of total)")
            
            if generation_weight > 0:
                gen_contrib = (train_metrics['generation_quality'] * generation_weight) / train_metrics['loss'] * 100
                status = "✅" if gen_contrib >= 5 else "⚠️"
                print(f"  {status} GenQual:  {train_metrics['generation_quality']:.4f} ({gen_contrib:.0f}% of total)")
            
            # KL divergence - critical for VAE health
            kl_contrib = (train_metrics['kl'] * self.config.get('beta', 1.0)) / train_metrics['loss'] * 100
            if train_metrics['kl'] < 0.0001:
                print(f"  🔴 KL:      {train_metrics['kl']:.6f} (POSTERIOR COLLAPSE!)")
            elif train_metrics['kl'] < 0.01:
                print(f"  🟡 KL:      {train_metrics['kl']:.4f} (Weak - increase beta)")
            else:
                print(f"  ✅ KL:      {train_metrics['kl']:.4f} (Healthy)")
            
            # Quick health check
            print(f"\n🏥 QUICK HEALTH CHECK:")
            print(f"{'─'*40}")
            
            # Perceptual health
            if perceptual_weight > 0:
                if train_metrics['perceptual'] > 0.01:
                    print(f"  🟢 Perceptual: Strong (>{train_metrics['perceptual']:.4f})")
                elif train_metrics['perceptual'] > 0.005:
                    print(f"  🟡 Perceptual: Good ({train_metrics['perceptual']:.4f})")
                else:
                    print(f"  🔴 Perceptual: Weak ({train_metrics['perceptual']:.4f}) - increase weight")
            
            # Generation quality health
            if generation_weight > 0:
                if train_metrics['generation_quality'] < -0.01:
                    print(f"  🟢 GenQual: Strong ({train_metrics['generation_quality']:.4f})")
                elif train_metrics['generation_quality'] < -0.005:
                    print(f"  🟡 GenQual: Good ({train_metrics['generation_quality']:.4f})")
                else:
                    print(f"  🔴 GenQual: Weak ({train_metrics['generation_quality']:.4f}) - increase weight")
            
            # Overall training health
            if len(self.loss_history.get('total', [])) > 3:
                recent_losses = self.loss_history['total'][-3:]
                if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                    print(f"  🟢 Training: Converging")
                elif all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                    print(f"  🔴 Training: Diverging - reduce LR")
                else:
                    print(f"  🟡 Training: Oscillating")
            
            # Comprehensive loss behavior assessment
            self._assess_loss_behavior(train_metrics, val_metrics, epoch)
            
            # Save best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                print(f"  ✅ New best model!")
            else:
                self.patience_counter += 1
                print(f"  ⏳ No improvement ({self.patience_counter}/{early_stopping_patience})")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Generate sample images for every epoch
            self.generate_samples(epoch, val_loader.dataset)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                break
            
            # Time estimation
            elapsed_time = time.time() - start_time
            if epoch > 0:
                avg_time_per_epoch = elapsed_time / (epoch + 1)
                remaining_epochs = self.config['max_epoch'] - (epoch + 1)
                estimated_remaining = remaining_epochs * avg_time_per_epoch
                print(f"  ⏱️  Time: {elapsed_time/3600:.1f}h elapsed, ~{estimated_remaining/3600:.1f}h remaining")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"  • Total time: {total_time/3600:.2f} hours")
        print(f"  • Best validation loss: {self.best_val_loss:.6f}")
        print(f"  • Final learning rate: {current_lr:.2e}")
        
        if self.writer is not None:
            self.writer.close()
        
        # Generate final test images
        print(f"\n🖼️  Generating final test images...")
        with torch.no_grad():
            z = torch.randn(16, self.config['embedding_size']).to(self.device)
            generated = self.model.dec(z)
            generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
            
            config_name = self.config.get('config_name', 'unified')
            final_path = f"{config_name}_final_samples.png"
            titles = [f"Final {i+1}" for i in range(16)]
            display_image_grid(generated_np, 
                              titles=titles,
                              max_cols=4, 
                              figsize=(20, 16),
                              save_path=final_path)
            print(f"  ✅ Final samples saved: {final_path}")


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Unified VAE Training Script')
    parser.add_argument('--loss-preset', type=str, default='high_quality',
                       choices=['mse_only', 'mse_l1', 'perceptual', 'lpips', 'custom_balanced', 
                               'high_quality', 'high_diversity', 'ultra_high_quality_loss', 'full_retrain_optimal'],
                       help='Loss function preset')
    parser.add_argument('--training-preset', type=str, default='fast_high_quality_training',
                       choices=['quick_test', 'fast_training', 'standard_training', 'extended_training',
                               'ultra_high_quality_training', 'fast_high_quality_training', 
                               'balanced_high_quality_training', 'compact_high_quality_training',
                               'full_retrain_training'],
                       help='Training configuration preset')
    parser.add_argument('--model-preset', type=str, default='fast_high_quality',
                       choices=['small', 'medium', 'large', 'high_res', 'ultra_high_quality',
                               'fast_high_quality', 'balanced_high_quality', 'compact_high_quality'],
                       help='Model architecture preset')
    parser.add_argument('--dataset-preset', type=str, default='full',
                       choices=['tiny', 'small', 'medium', 'large', 'full'],
                       help='Dataset size preset')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh training (ignore checkpoints)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (e.g., 128, 256, 384)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader("config/config_unified.json")
    config = config_loader.get_config(
        loss_preset=args.loss_preset,
        training_preset=args.training_preset,
        model_preset=args.model_preset,
        dataset_preset=args.dataset_preset
    )
    
    # Override batch size if specified
    if args.batch_size:
        config['batch_size'] = args.batch_size
        config['_user_override_batch_size'] = True
        print(f"📊 Batch size overridden to: {args.batch_size}")
    
    # Set config name for filenames (include both model and training presets)
    config['config_name'] = f"{args.model_preset}_{args.training_preset}"
    
    # Configure GPU
    device = configure_gpu() if args.device == 'cuda' else torch.device(args.device)
    print(f"✅ Using device: {device}")
    
    # Create trainer and start training
    trainer = UnifiedVAETrainer(config, device)
    trainer.train(resume=not args.no_resume)


if __name__ == "__main__":
    main()
