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

# Perceptual loss is now isolated in perceptual_loss.py

# TensorBoard support is now isolated in training_writer.py

import numpy as np
from tqdm import tqdm

# Import our modules
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import get_split_data, configure_gpu, display_image_grid, display_comparison_grid
from config_loader import ConfigLoader
from utils import create_directories, safe_remove_file
from loss_weight_manager import LossWeightManager
from training_reporter import TrainingReporter
from training_writer import create_training_writer
from perceptual_loss import create_perceptual_loss, get_perceptual_loss_info
from loss_calculator import create_loss_calculator
from training_utilities import create_training_utilities

# VGGPerceptualLoss class moved to perceptual_loss.py

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
        
        # Initialize centralized loss weight management
        self.loss_manager = LossWeightManager(config)
        
        # Initialize training reporter
        self.reporter = TrainingReporter(self.loss_manager)
        
        # Initialize loss calculator
        self.loss_calculator = create_loss_calculator(config, device)
        
        # Initialize training utilities
        self.utilities = create_training_utilities(config, device)
        
        # Initialize loss history tracking
        self.loss_history = {
            'perceptual': [],
            'generation_quality': [],
            'kl': [],
            'total': []
        }
        
        # Adjust batch size for VGG if needed
        loss_config = config.get('loss_config', {})
        if loss_config.get('use_perceptual_loss', False):
            self.utilities.adjust_batch_size_for_vgg(config, device)
    
    # Batch size adjustment moved to training_utilities.py
    
    def setup_logging(self):
        """Setup TensorBoard logging using the isolated writer."""
        log_dir = self.utilities.get_filename('log_dir')
        self.writer = create_training_writer(self.config, log_dir)
        
        if self.writer.is_enabled():
            print(f"  • Logging: {self.writer.get_log_dir()}")
        else:
            print(f"  • Logging: Disabled (TensorBoard not available)")
    
    # Accessor methods removed - use self.loss_manager.get_weight() directly
    
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
    
    # Perceptual loss calculation moved to loss_calculator.py
    
    # Generation quality loss calculation moved to loss_calculator.py
    
    def calculate_losses(self, images, reconst, emb_mean, emb_log_var):
        """Calculate all loss components using the centralized loss calculator."""
        loss_dict = self.loss_calculator.calculate_all_losses(
            images, reconst, emb_mean, emb_log_var, self.model, self.loss_manager
        )
        
        # Convert to the expected format for backward compatibility
        total_loss = loss_dict['loss']
        
        # Convert tensor values to scalars for logging
        loss_dict_scalar = {
            'mse': loss_dict['mse'].item(),
            'l1': loss_dict['l1'].item(),
            'perceptual': loss_dict['perceptual'].item() if isinstance(loss_dict['perceptual'], torch.Tensor) else loss_dict['perceptual'],
            'generation_quality': loss_dict['generation_quality'].item() if isinstance(loss_dict['generation_quality'], torch.Tensor) else loss_dict['generation_quality'],
            'kl': loss_dict['kl'].item(),
            'total': loss_dict['loss'].item()
        }
        
        return total_loss, loss_dict_scalar
    
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
            
            # Note: Mid-epoch samples are generated at the training loop level
        
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
    
    # save_checkpoint moved to training_utilities.py
    
    # load_checkpoint moved to training_utilities.py
    
    # get_filename moved to training_utilities.py
    
    # clear_old_files moved to training_utilities.py
        
        print("  🆕 Ready for fresh training!")
    
    # _assess_loss_behavior moved to training_utilities.py
    
    # print_configuration moved to training_utilities.py
    
    def generate_samples(self, epoch, val_data, suffix=""):
        """Generate sample images."""
        if suffix:
            print(f"  🖼️  Generating sample images ({suffix})...")
        else:
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
                self.writer.log_images('Generated/Images', generated, epoch)
            
            # Get clean config name for filename
            config_name = self.config.get('config_name', 'unified')
            
            # Save sample images
            sample_path = os.path.join(sample_dir, f"{config_name}_generated_epoch_{epoch+1:03d}{suffix}.png")
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
            
            recon_path = os.path.join(sample_dir, f"{config_name}_reconstruction_epoch_{epoch+1:03d}{suffix}.png")
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
        self.utilities.print_configuration("Fresh")
        
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
            self.utilities.clear_old_files()
        elif resume and self.utilities.load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler)[0]:
            pass  # Configuration already printed
        
        # Training loop with proper resource management
        print(f"\n🎯 Starting training...")
        start_time = time.time()
        early_stopping_patience = 12
        
        # Use context manager for proper writer cleanup
        try:
            for epoch in range(self.start_epoch, self.config['max_epoch']):
                # Update current epoch for weight scheduling
                self.loss_manager.set_epoch(epoch)
            
                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Generate mid-epoch samples for closer monitoring
                if epoch > 0 and epoch % 2 == 0:  # Every 2 epochs after the first
                    print(f"  🔍 Generating mid-epoch samples...")
                    self.generate_samples(epoch, val_loader.dataset, suffix="_mid_epoch")
                
                # Validation phase
                val_metrics = self.validate_epoch(val_loader, epoch)
            
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Logging
                # Log comprehensive metrics using the centralized writer
                if self.writer is not None:
                    # Get current weights for logging
                    weights = self.loss_manager.get_all_weights()
                    
                    # Log all epoch metrics
                    self.writer.log_epoch_metrics(epoch, train_metrics, val_metrics, weights)
                    self.writer.log_learning_rate(epoch, current_lr)
                    self.writer.log_beta(epoch, self.loss_manager.get_weight('beta'))
                    
                    # Log training stage if using stage-based training
                    if self.config.get('stage_based_training', False):
                        stage_name = self.loss_manager.get_current_stage_name()
                        self.writer.log_stage(epoch, stage_name)
                    
                    # Log GPU memory usage
                    if torch.cuda.is_available():
                        gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                        gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                        gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100
                        self.writer.log_gpu_memory(epoch, gpu_memory_allocated, gpu_memory_total, gpu_utilization)
            
                # Get GPU statistics
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                    gpu_memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
                    gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
                    gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100
                else:
                    gpu_memory_allocated = gpu_memory_reserved = gpu_memory_total = gpu_utilization = 0
                
                # Print epoch summary using the centralized reporter
                summary = self.reporter.format_epoch_summary(
                    epoch=epoch,
                    max_epochs=self.config['max_epoch'],
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    current_lr=current_lr,
                    gpu_memory_allocated=gpu_memory_allocated,
                    gpu_memory_total=gpu_memory_total,
                    gpu_utilization=gpu_utilization
                )
                print(summary)
            
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
                self.utilities.assess_loss_behavior(train_metrics, val_metrics, epoch)
                
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
                self.utilities.save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, train_metrics, val_metrics, is_best)
            
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
        
        finally:
            # Ensure writer is properly closed
            if self.writer is not None:
                self.writer.close()


def main():
    """Main function with command-line argument parsing."""
    # Load available presets dynamically from config
    config_loader = ConfigLoader('config/config_unified.json')
    # Filter out comment entries and other non-preset keys
    available_loss_presets = [k for k in config_loader.unified_config['loss_presets'].keys() if not k.startswith('_')]
    available_training_presets = [k for k in config_loader.unified_config['training_presets'].keys() if not k.startswith('_')]
    available_model_presets = [k for k in config_loader.unified_config['model_presets'].keys() if not k.startswith('_')]
    available_dataset_presets = [k for k in config_loader.unified_config['dataset_presets'].keys() if not k.startswith('_')]
    
    parser = argparse.ArgumentParser(description='Unified VAE Training Script')
    parser.add_argument('--loss-preset', type=str, default='high_quality',
                       choices=available_loss_presets,
                       help='Loss function preset')
    parser.add_argument('--training-preset', type=str, default='fast_high_quality_training',
                       choices=available_training_presets,
                       help='Training configuration preset')
    parser.add_argument('--model-preset', type=str, default='fast_high_quality',
                       choices=available_model_presets,
                       help='Model architecture preset')
    parser.add_argument('--dataset-preset', type=str, default='full',
                       choices=available_dataset_presets,
                       help='Dataset size preset')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh training (ignore checkpoints)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (e.g., 128, 256, 384)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                       help='Override learning rate (e.g., 0.001, 0.0001, 0.00001)')
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
    
    # Override learning rate if specified
    if args.lr:
        config['lr'] = args.lr
        print(f"📈 Learning rate overridden to: {args.lr}")
    
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
