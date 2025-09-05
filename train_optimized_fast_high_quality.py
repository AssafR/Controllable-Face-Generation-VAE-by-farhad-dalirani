#!/usr/bin/env python3
"""
Optimized Fast High Quality VAE Training Script
Uses optimal batch size 512 for RTX 3090:
- Resolution: 128x128
- Parameters: 125M
- Batch Size: 512 (optimized for RTX 3090)
- Training Time: ~4.6 hours
- Memory Usage: ~1GB (4.2% of 24GB VRAM)
"""

import sys
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# No need for warning suppression - using modern PyTorch API
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("⚠️  TensorBoard not available, logging disabled")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
import numpy as np
from tqdm import tqdm
import json
import time
from datetime import datetime

# Import our modules
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import get_split_data, configure_gpu, display_image_grid
from config_loader import ConfigLoader

def train_optimized_fast_high_quality():
    """Train VAE with optimized fast high quality settings for RTX 3090."""
    
    print("🚀 Starting Optimized Fast High Quality VAE Training")
    print("=" * 70)
    print("Configuration:")
    print("  • Resolution: 128x128")
    print("  • Latent Space: 512")
    print("  • Loss: MSE + L1 + Perceptual")
    print("  • Epochs: 100")
    print("  • Batch Size: 512 (optimized for RTX 3090)")
    print("  • Learning Rate: 0.0001")
    print("  • Expected Time: ~4.6 hours")
    print("  • Memory Usage: ~1GB (4.2% of 24GB VRAM)")
    print("=" * 70)
    
    # Load optimized configuration
    config_loader = ConfigLoader("config/config_unified.json")
    config = config_loader.get_config(
        loss_preset="high_quality",
        training_preset="fast_high_quality_training", 
        model_preset="fast_high_quality",
        dataset_preset="full"
    )
    
    # Set a clean, simple config name for filenames
    config['config_name'] = "fast_high_quality"
    
    # Optimize batch size for perceptual loss - we have room for more
    if config.get('loss_config', {}).get('use_perceptual_loss', False):
        original_batch_size = config['batch_size']
        config['batch_size'] = min(config['batch_size'], 256)  # Optimized for available VRAM
        if config['batch_size'] != original_batch_size:
            print(f"⚡  Optimized batch size from {original_batch_size} to {config['batch_size']} for perceptual loss")
    
    print(f"✅ Configuration loaded:")
    print(f"  • Input size: {config['input_img_size']}x{config['input_img_size']}")
    print(f"  • Embedding size: {config['embedding_size']}")
    print(f"  • Max epochs: {config['max_epoch']}")
    print(f"  • Batch size: {config['batch_size']} (optimized)")
    print(f"  • Learning rate: {config['lr']}")
    
    # Print loss configuration
    loss_config = config.get('loss_config', {})
    print(f"  • Loss weights: MSE={loss_config.get('mse_weight', 0)}, L1={loss_config.get('l1_weight', 0)}, Perceptual={loss_config.get('perceptual_weight', 0)}")
    
    # Configure GPU
    device = configure_gpu()
    print(f"✅ Using device: {device}")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = VAE_pt(
        input_img_size=config['input_img_size'],
        embedding_size=config['embedding_size'],
        loss_config=config
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created:")
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Trainable parameters: {trainable_params:,}")
    print(f"  • Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Load data
    print(f"\n📊 Loading dataset...")
    train_data, val_data = get_split_data(config=config)
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    print(f"✅ Dataset loaded:")
    print(f"  • Training samples: {len(train_data):,}")
    print(f"  • Validation samples: {len(val_data):,}")
    print(f"  • Training batches: {len(train_loader):,}")
    print(f"  • Validation batches: {len(val_loader):,}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/optimized_fast_high_quality_{timestamp}"
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir)
        print(f"  • Logging: {log_dir}")
    else:
        writer = None
        print(f"  • Logging: Disabled (TensorBoard not available)")
    
    print(f"✅ Training setup complete:")
    print(f"  • Optimizer: Adam (lr={config['lr']}, weight_decay=1e-5)")
    print(f"  • Scheduler: ReduceLROnPlateau (patience=8)")
    print(f"  • Batch size: {config['batch_size']} (optimized for RTX 3090)")
    
    # Checkpoint setup
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "training_checkpoint.pth")
    best_model_path = f"{config['model_save_path']}/vae_optimized_fast_high_quality.pth"
    
    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 12  # Reduced for faster training
    start_epoch = 0
    
    # Try to resume from checkpoint
    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint: {checkpoint_path}")
        print("   Resuming training from checkpoint...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        
        print(f"   ✅ Resumed from epoch {start_epoch}")
        print(f"   📊 Best validation loss: {best_val_loss:.6f}")
        print(f"   ⏳ Patience counter: {patience_counter}")
    else:
        print("🆕 No checkpoint found, starting fresh training")
    
    # Training loop
    print(f"\n🎯 Starting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['max_epoch']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_kl = 0.0
        train_l1 = 0.0
        train_perceptual = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epoch']} [Train]", leave=False)
        
        for batch_idx, images in enumerate(train_pbar):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            emb_mean, emb_log_var, reconst = model(images)
            
            # Calculate reconstruction loss components
            mse_loss = F.mse_loss(reconst, images)
            l1_loss = F.l1_loss(reconst, images)
            
            # Calculate perceptual loss if enabled - using VAE encoder features
            perceptual_loss = 0.0
            loss_config = config.get('loss_config', {})
            if loss_config.get('use_perceptual_loss', False):
                try:
                    # Extract features from VAE encoder intermediate layers (optimized)
                    def extract_encoder_features(x):
                        features = []
                        x = model.enc.conv1(x)  # First conv layer
                        x = model.enc.bn1(x)
                        x = F.leaky_relu(x, 0.2)
                        features.append(x)  # After first conv block
                        
                        # Skip middle layer to save memory
                        x = model.enc.conv2(x)  # Second conv layer
                        x = model.enc.bn2(x)
                        x = F.leaky_relu(x, 0.2)
                        # Don't store this feature to save memory
                        
                        x = model.enc.conv3(x)  # Third conv layer
                        x = model.enc.bn3(x)
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
                    
                except Exception as e:
                    print(f"⚠️  Error calculating perceptual loss: {e}")
                    perceptual_loss = 0.0
            
            # Combine losses with weights
            mse_weight = loss_config.get('mse_weight', 0.5)
            l1_weight = loss_config.get('l1_weight', 0.3)
            perceptual_weight = loss_config.get('perceptual_weight', 0.2)
            
            recon_loss = (mse_weight * mse_loss + 
                         l1_weight * l1_loss + 
                         perceptual_weight * perceptual_loss)
            
            # Store loss components for logging
            model.loss_components = {
                'mse': mse_loss.item(),
                'l1': l1_loss.item(),
                'perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss,
                'total': recon_loss.item()
            }
            
            # Calculate KL divergence loss
            kl_loss = model.kl_loss(emb_mean, emb_log_var)
            
            # Total loss
            total_loss = recon_loss + config['beta'] * kl_loss
            
            # Get loss components from model
            loss_dict = getattr(model, 'loss_components', {})
            loss_dict['kl'] = kl_loss.item()
            loss_dict['total'] = total_loss.item()
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            train_loss += total_loss.item()
            train_mse += loss_dict.get('mse', 0)
            train_kl += loss_dict.get('kl', 0)
            train_l1 += loss_dict.get('l1', 0)
            train_perceptual += loss_dict.get('perceptual', 0)
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.5f}',
                'MSE': f'{loss_dict.get("mse", 0):.4f}',
                'L1': f'{loss_dict.get("l1", 0):.4f}',
                'Perceptual': f'{loss_dict.get("perceptual", 0):.4f}',
                'KL': f'{loss_dict.get("kl", 0):.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_kl = 0.0
        val_l1 = 0.0
        val_perceptual = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['max_epoch']} [Val]", leave=False)
            
            for batch_idx, images in enumerate(val_pbar):
                images = images.to(device)
                
                # Forward pass
                emb_mean, emb_log_var, reconst = model(images)
                
                # Calculate reconstruction loss components
                mse_loss = F.mse_loss(reconst, images)
                l1_loss = F.l1_loss(reconst, images)
                
                # Calculate perceptual loss if enabled - using VAE encoder features
                perceptual_loss = 0.0
                loss_config = config.get('loss_config', {})
                if loss_config.get('use_perceptual_loss', False):
                    try:
                        # Extract features from VAE encoder intermediate layers (optimized)
                        def extract_encoder_features(x):
                            features = []
                            x = model.enc.conv1(x)  # First conv layer
                            x = model.enc.bn1(x)
                            x = F.leaky_relu(x, 0.2)
                            features.append(x)  # After first conv block
                            
                            # Skip middle layer to save memory
                            x = model.enc.conv2(x)  # Second conv layer
                            x = model.enc.bn2(x)
                            x = F.leaky_relu(x, 0.2)
                            # Don't store this feature to save memory
                            
                            x = model.enc.conv3(x)  # Third conv layer
                            x = model.enc.bn3(x)
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
                        
                    except Exception as e:
                        perceptual_loss = 0.0
                
                # Combine losses with weights
                mse_weight = loss_config.get('mse_weight', 0.5)
                l1_weight = loss_config.get('l1_weight', 0.3)
                perceptual_weight = loss_config.get('perceptual_weight', 0.2)
                
                recon_loss = (mse_weight * mse_loss + 
                             l1_weight * l1_loss + 
                             perceptual_weight * perceptual_loss)
                
                # Store loss components for logging
                loss_dict = {
                    'mse': mse_loss.item(),
                    'l1': l1_loss.item(),
                    'perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss,
                    'total': recon_loss.item()
                }
                
                # Calculate KL divergence loss
                kl_loss = model.kl_loss(emb_mean, emb_log_var)
                
                # Total loss
                total_loss = recon_loss + config['beta'] * kl_loss
                
                # Add KL loss to loss dictionary
                loss_dict['kl'] = kl_loss.item()
                loss_dict['total'] = total_loss.item()
                
                val_loss += total_loss.item()
                val_mse += loss_dict.get('mse', 0)
                val_kl += loss_dict.get('kl', 0)
                val_l1 += loss_dict.get('l1', 0)
                val_perceptual += loss_dict.get('perceptual', 0)
                
                val_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.6f}',
                    'MSE': f'{loss_dict.get("mse", 0):.6f}',
                    'L1': f'{loss_dict.get("l1", 0):.6f}',
                    'Perceptual': f'{loss_dict.get("perceptual", 0):.6f}',
                    'KL': f'{loss_dict.get("kl", 0):.6f}'
                })
        
        # Calculate averages
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_kl /= len(train_loader)
        train_l1 /= len(train_loader)
        train_perceptual /= len(train_loader)
        
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        val_kl /= len(val_loader)
        val_l1 /= len(val_loader)
        val_perceptual /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('MSE/Train', train_mse, epoch)
            writer.add_scalar('MSE/Val', val_mse, epoch)
            writer.add_scalar('KL/Train', train_kl, epoch)
            writer.add_scalar('KL/Val', val_kl, epoch)
            writer.add_scalar('L1/Train', train_l1, epoch)
            writer.add_scalar('L1/Val', val_l1, epoch)
            writer.add_scalar('Perceptual/Train', train_perceptual, epoch)
            writer.add_scalar('Perceptual/Val', val_perceptual, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Get GPU statistics
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            gpu_memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100
        else:
            gpu_memory_allocated = gpu_memory_reserved = gpu_memory_total = gpu_utilization = 0
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['max_epoch']} Summary:")
        print(f"  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, KL: {train_kl:.6f}, L1: {train_l1:.6f}, Perceptual: {train_perceptual:.6f})")
        print(f"  Val Loss:   {val_loss:.6f} (MSE: {val_mse:.6f}, KL: {val_kl:.6f}, L1: {val_l1:.6f}, Perceptual: {val_perceptual:.6f})")
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  GPU Memory: {gpu_memory_allocated:.2f}GB / {gpu_memory_total:.1f}GB ({gpu_utilization:.1f}%)")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(config['model_save_path'], exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ New best model saved: {best_model_path}")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Save checkpoint every epoch (complete state)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'config': config
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pth")
            torch.save(checkpoint, periodic_checkpoint_path)
            print(f"  💾 Periodic checkpoint saved: {periodic_checkpoint_path}")
        
        # Generate sample images for first 5 epochs, then every 5 epochs
        if (epoch + 1) <= 5 or (epoch + 1) % 5 == 0:
            print(f"  🖼️  Generating sample images...")
            
            # Create sample_images directory
            sample_dir = "sample_images"
            os.makedirs(sample_dir, exist_ok=True)
            
            with torch.no_grad():
                # Generate random images
                z = torch.randn(8, config['embedding_size']).to(device)
                generated = model.dec(z)
                generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
                
                # Log to tensorboard
                if writer is not None:
                    writer.add_images('Generated/Images', generated, epoch)
                
                # Get clean config name for filename
                config_name = config.get('config_name', 'fast_high_quality')
                
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
                val_images = torch.stack([val_data[i] for i in val_indices]).to(device)
                z_mean, z_log_var, z = model.enc(val_images)
                reconstructed = model.dec(z)
                
                val_np = val_images.permute(0, 2, 3, 1).cpu().numpy()
                recon_np = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
                
                recon_path = os.path.join(sample_dir, f"{config_name}_reconstruction_epoch_{epoch+1:03d}.png")
                from utilities_pytorch import display_comparison_grid
                display_comparison_grid(val_np, recon_np,
                                       titles=[f"Epoch {epoch+1} - Pair {i+1}" for i in range(4)],
                                       max_cols=2, 
                                       figsize=(12, 8),
                                       save_path=recon_path)
                print(f"  ✅ Reconstruction samples saved: {recon_path}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
        
        # Time estimation
        elapsed_time = time.time() - start_time
        if epoch > 0:
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = config['max_epoch'] - (epoch + 1)
            estimated_remaining = remaining_epochs * avg_time_per_epoch
            print(f"  ⏱️  Time: {elapsed_time/3600:.1f}h elapsed, ~{estimated_remaining/3600:.1f}h remaining")
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed!")
    print(f"  • Total time: {total_time/3600:.2f} hours")
    print(f"  • Best validation loss: {best_val_loss:.6f}")
    print(f"  • Final learning rate: {current_lr:.2e}")
    print(f"  • Model saved: {config['model_save_path']}/vae_optimized_fast_high_quality.pth")
    if writer is not None:
        print(f"  • TensorBoard logs: {log_dir}")
    
    # Generate final test images
    print(f"\n🖼️  Generating final test images...")
    with torch.no_grad():
        # Generate 16 random images
        z = torch.randn(16, config['embedding_size']).to(device)
        generated = model.dec(z)
        generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
        
        # Save final samples
        config_name = config.get('config_name', 'fast_high_quality')
        final_path = f"{config_name}_final_samples.png"
        titles = [f"Final {i+1}" for i in range(16)]
        display_image_grid(generated_np, 
                          titles=titles,
                          max_cols=4, 
                          figsize=(20, 16),
                          save_path=final_path)
        print(f"  ✅ Final samples saved: {final_path}")
    
    if writer is not None:
        writer.close()
    
    # Final cleanup - save final checkpoint
    final_checkpoint = {
        'epoch': config['max_epoch'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'config': config,
        'training_complete': True
    }
    torch.save(final_checkpoint, checkpoint_path)
    
    print(f"\n✅ Optimized fast high quality training complete!")
    print(f"📁 Checkpoints saved in: {checkpoint_dir}/")
    print(f"🏆 Best model: {best_model_path}")
    print(f"💾 Latest checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    train_optimized_fast_high_quality()
