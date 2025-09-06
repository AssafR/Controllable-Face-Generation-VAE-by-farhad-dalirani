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
        """Calculate perceptual loss using VAE encoder features."""
        loss_config = self.config.get('loss_config', {})
        if not loss_config.get('use_perceptual_loss', False):
            return 0.0
            
        try:
            # Extract features from VAE encoder intermediate layers
            def extract_encoder_features(x):
                features = []
                x = self.model.enc.conv1(x)
                x = self.model.enc.bn1(x)
                x = F.leaky_relu(x, 0.2)
                features.append(x)  # After first conv block
                
                x = self.model.enc.conv2(x)
                x = self.model.enc.bn2(x)
                x = F.leaky_relu(x, 0.2)
                
                x = self.model.enc.conv3(x)
                x = self.model.enc.bn3(x)
                x = F.leaky_relu(x, 0.2)
                features.append(x)  # After third conv block
                
                return features
            
            # Get encoder features for both images
            orig_features = extract_encoder_features(images)
            recon_features = extract_encoder_features(reconst)
            
            # Calculate perceptual loss as MSE of encoder features
            perceptual_loss = 0.0
            for orig_feat, recon_feat in zip(orig_features, recon_features):
                perceptual_loss += F.mse_loss(orig_feat, recon_feat)
            perceptual_loss /= len(orig_features)
            
            return perceptual_loss
            
        except Exception as e:
            print(f"⚠️  Error calculating perceptual loss: {e}")
            return 0.0
    
    def calculate_losses(self, images, reconst, emb_mean, emb_log_var):
        """Calculate all loss components."""
        # Basic reconstruction losses
        mse_loss = F.mse_loss(reconst, images)
        l1_loss = F.l1_loss(reconst, images)
        
        # Perceptual loss
        perceptual_loss = self.calculate_perceptual_loss(images, reconst)
        
        # Get loss weights
        loss_config = self.config.get('loss_config', {})
        mse_weight = loss_config.get('mse_weight', 0.5)
        l1_weight = loss_config.get('l1_weight', 0.3)
        perceptual_weight = loss_config.get('perceptual_weight', 0.2)
        
        # Combine reconstruction losses
        recon_loss = (mse_weight * mse_loss + 
                     l1_weight * l1_loss + 
                     perceptual_weight * perceptual_loss)
        
        # KL divergence loss
        kl_loss = self.model.kl_loss(emb_mean, emb_log_var)
        
        # Total loss
        total_loss = recon_loss + self.config['beta'] * kl_loss
        
        # Store loss components for logging
        loss_dict = {
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss,
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
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.5f}',
                'MSE': f'{loss_dict["mse"]:.4f}',
                'L1': f'{loss_dict["l1"]:.4f}',
                'Perceptual': f'{loss_dict["perceptual"]:.4f}',
                'KL': f'{loss_dict["kl"]:.4f}'
            })
        
        # Calculate averages
        num_batches = len(train_loader)
        return {
            'loss': train_loss / num_batches,
            'mse': train_mse / num_batches,
            'kl': train_kl / num_batches,
            'l1': train_l1 / num_batches,
            'perceptual': train_perceptual / num_batches
        }
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_kl = 0.0
        val_l1 = 0.0
        val_perceptual = 0.0
        
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
                
                val_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.6f}',
                    'MSE': f'{loss_dict["mse"]:.6f}',
                    'L1': f'{loss_dict["l1"]:.6f}',
                    'Perceptual': f'{loss_dict["perceptual"]:.6f}',
                    'KL': f'{loss_dict["kl"]:.6f}'
                })
        
        # Calculate averages
        num_batches = len(val_loader)
        return {
            'loss': val_loss / num_batches,
            'mse': val_mse / num_batches,
            'kl': val_kl / num_batches,
            'l1': val_l1 / num_batches,
            'perceptual': val_perceptual / num_batches
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
        checkpoint_path = os.path.join(checkpoint_dir, "training_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            os.makedirs(self.config['model_save_path'], exist_ok=True)
            best_model_path = f"{self.config['model_save_path']}/vae_best_model.pth"
            torch.save(self.model.state_dict(), best_model_path)
            print(f"  ✅ New best model saved: {best_model_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
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
        """Clear old checkpoints and model files for fresh training."""
        print("🧹 Clearing old checkpoints and model files for fresh training...")
        
        # Check what will be deleted
        files_to_delete = []
        checkpoint_dir = "checkpoints"
        model_dir = self.config['model_save_path']
        sample_dir = "sample_images"
        
        if os.path.exists(checkpoint_dir):
            files_to_delete.append(f"checkpoints/ ({len(os.listdir(checkpoint_dir))} files)")
        if os.path.exists(model_dir):
            files_to_delete.append(f"model_weights/ ({len(os.listdir(model_dir))} files)")
        if os.path.exists(sample_dir):
            files_to_delete.append(f"sample_images/ ({len(os.listdir(sample_dir))} files)")
        
        if files_to_delete:
            print("  📁 Will delete:")
            for item in files_to_delete:
                print(f"    • {item}")
            
            # Ask for confirmation
            response = input("  ❓ Continue? This will permanently delete old training data (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("  ❌ Cancelled. Use without --no-resume to continue existing training.")
                import sys
                sys.exit(0)
        else:
            print("  ℹ️  No old files found to delete.")
        
        # Clear directories
        if os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
            print(f"  ✅ Cleared checkpoints directory: {checkpoint_dir}")
        
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
            print(f"  ✅ Cleared model weights directory: {model_dir}")
        
        if os.path.exists(sample_dir):
            import shutil
            shutil.rmtree(sample_dir)
            print(f"  ✅ Cleared sample images directory: {sample_dir}")
        
        print("  🆕 Ready for fresh training!")
    
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
        print(f"  • Loss weights: MSE={loss_config.get('mse_weight', 0)}, L1={loss_config.get('l1_weight', 0)}, Perceptual={loss_config.get('perceptual_weight', 0)}")
        print(f"  • Loss components: MSE={loss_config.get('use_mse', False)}, L1={loss_config.get('use_l1', False)}, Perceptual={loss_config.get('use_perceptual_loss', False)}")
    
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
        checkpoint_path = "checkpoints/training_checkpoint.pth"
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
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['max_epoch']} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.6f} (MSE: {train_metrics['mse']:.6f}, KL: {train_metrics['kl']:.6f}, L1: {train_metrics['l1']:.6f}, Perceptual: {train_metrics['perceptual']:.6f})")
            print(f"  Val Loss:   {val_metrics['loss']:.6f} (MSE: {val_metrics['mse']:.6f}, KL: {val_metrics['kl']:.6f}, L1: {val_metrics['l1']:.6f}, Perceptual: {val_metrics['perceptual']:.6f})")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  GPU Memory: {gpu_memory_allocated:.2f}GB / {gpu_memory_total:.1f}GB ({gpu_utilization:.1f}%)")
            
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
                               'high_quality', 'ultra_high_quality_loss', 'full_retrain_optimal'],
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
    
    # Set config name for filenames
    config['config_name'] = f"{args.loss_preset}_{args.training_preset}_{args.model_preset}"
    
    # Configure GPU
    device = configure_gpu() if args.device == 'cuda' else torch.device(args.device)
    print(f"✅ Using device: {device}")
    
    # Create trainer and start training
    trainer = UnifiedVAETrainer(config, device)
    trainer.train(resume=not args.no_resume)


if __name__ == "__main__":
    main()
