#!/usr/bin/env python3
"""
Unified VAE training script using the new configuration system.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import get_split_data, configure_gpu
from config_loader import ConfigLoader
import argparse

def train_variational_autoencoder_unified(config, device='cuda'):
    """
    Train the VAE using unified configuration system.
    """
    
    print("🚀 Starting Unified VAE Training")
    print("=" * 50)
    print(f"Configuration: {config['config_name']}")
    print(f"Architecture:")
    print(f"  - Embedding size: {config['embedding_size']}")
    print(f"  - Channels: {config['num_channels']}")
    print(f"  - Beta: {config['beta']}")
    print(f"  - Input size: {config['input_img_size']}x{config['input_img_size']}")
    
    # Print loss configuration
    loss_config = config.get("loss_config", {})
    print(f"  - Loss configuration:")
    print(f"    * MSE: {loss_config.get('use_mse', True)} (weight: {loss_config.get('mse_weight', 0.8)})")
    print(f"    * L1: {loss_config.get('use_l1', True)} (weight: {loss_config.get('l1_weight', 0.2)})")
    print(f"    * Perceptual: {loss_config.get('use_perceptual_loss', False)} (weight: {loss_config.get('perceptual_weight', 0.1)})")
    print(f"    * LPIPS: {loss_config.get('use_lpips', False)} (weight: {loss_config.get('lpips_weight', 0.1)})")
    
    print(f"Training:")
    print(f"  - Epochs: {config['max_epoch']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['lr']}")
    print("=" * 50)
    
    # Create VAE model
    model = VAE_pt(
        input_img_size=config["input_img_size"],
        embedding_size=config["embedding_size"],
        num_channels=config["num_channels"],
        beta=config["beta"],
        loss_config=loss_config
    )
    
    model = model.to(device)
    model.train()
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["lr"],
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Load dataset
    print("Loading dataset...")
    train_data, val_data = get_split_data(config=config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✅ Dataset loaded: {len(train_data)} train, {len(val_data)} val samples")
    print(f"✅ Training with batch size: {config['batch_size']}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(config["max_epoch"]):
        print(f"\nEpoch {epoch+1}/{config['max_epoch']}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, images in enumerate(train_pbar):
            images = images.to(device)
            
            # Training step
            metrics = model.train_step(images, optimizer)
            train_losses.append(metrics["loss"])
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f"{metrics['loss']:.4f}",
                'Recon': f"{metrics['reconstruction_loss']:.4f}",
                'KL': f"{metrics['kl_loss']:.4f}"
            })
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch_idx, images in enumerate(val_pbar):
                images = images.to(device)
                
                # Validation step
                metrics = model.test_step(images)
                val_losses.append(metrics["loss"])
                
                val_pbar.set_postfix({'Loss': f"{metrics['loss']:.4f}"})
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save model
            os.makedirs(config["model_save_path"], exist_ok=True)
            model_name = config.get("model_name", "vae_unified")
            torch.save(model.state_dict(), os.path.join(config["model_save_path"], f"{model_name}.pth"))
            print(f"✅ New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{max_patience})")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"🛑 Early stopping after {epoch+1} epochs")
            break
        
        # Generate sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("Generating sample images...")
            with torch.no_grad():
                # Generate random images
                z = torch.randn(5, config["embedding_size"]).to(device)
                generated = model.dec(z)
                generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
                
                print(f"Sample range: [{generated_np.min():.4f}, {generated_np.max():.4f}]")
                print(f"Sample mean: {generated_np.mean():.4f}")
    
    print("\n" + "=" * 50)
    print("✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 50)

def main():
    """Main function with command line arguments"""
    
    parser = argparse.ArgumentParser(description='Train VAE with unified configuration system')
    parser.add_argument('--loss', type=str, default='mse_l1',
                       help='Loss function preset (mse_only, mse_l1, perceptual, lpips, custom_balanced, high_quality)')
    parser.add_argument('--training', type=str, default='standard_training',
                       help='Training preset (quick_test, fast_training, standard_training, extended_training)')
    parser.add_argument('--model', type=str, default='medium',
                       help='Model preset (small, medium, large, high_res)')
    parser.add_argument('--dataset', type=str, default='full',
                       help='Dataset preset (tiny, small, medium, large, full)')
    parser.add_argument('--config', type=str, default='config/config_unified.json',
                       help='Path to unified config file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--list-presets', action='store_true',
                       help='List all available presets and exit')
    
    args = parser.parse_args()
    
    # Initialize config loader
    try:
        loader = ConfigLoader(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # List presets if requested
    if args.list_presets:
        loader.print_all_presets()
        return
    
    # Configure GPU
    if args.device == 'auto':
        device = configure_gpu()
    else:
        device = torch.device(args.device)
    
    # Get configuration
    try:
        config = loader.get_config(
            loss_preset=args.loss,
            training_preset=args.training,
            model_preset=args.model,
            dataset_preset=args.dataset
        )
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("\nAvailable presets:")
        loader.print_all_presets()
        return
    
    print(f"Using configuration: {config['config_name']}")
    
    # Train the model
    train_variational_autoencoder_unified(config, device)

if __name__ == "__main__":
    main()
