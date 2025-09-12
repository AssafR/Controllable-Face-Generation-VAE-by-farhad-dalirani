#!/usr/bin/env python3
"""
Improved training script with better architecture and loss functions to reduce blurriness.
"""

import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import get_split_data, configure_gpu

def train_variational_autoencoder_improved(config, device='cuda'):
    """
    Train the improved VAE with better architecture and loss functions.
    """
    
    print("🚀 Starting Improved VAE Training")
    print("=" * 50)
    print(f"Architecture improvements:")
    print(f"  - Embedding size: {config['embedding_size']} (increased from 200)")
    print(f"  - Beta: {config['beta']} (reduced from 2000)")
    
    # Print loss configuration
    loss_config = config.get("loss_config", {})
    print(f"  - Loss configuration:")
    print(f"    * MSE: {loss_config.get('use_mse', True)} (weight: {loss_config.get('mse_weight', 0.8)})")
    print(f"    * L1: {loss_config.get('use_l1', True)} (weight: {loss_config.get('l1_weight', 0.2)})")
    print(f"    * Perceptual: {loss_config.get('use_perceptual_loss', False)} (weight: {loss_config.get('perceptual_weight', 0.1)})")
    print(f"    * LPIPS: {loss_config.get('use_lpips', False)} (weight: {loss_config.get('lpips_weight', 0.1)})")
    print("=" * 50)
    
    # Create improved VAE model
    model = VAE_pt(
        input_img_size=config["input_img_size"],
        embedding_size=config["embedding_size"],
        num_channels=config["num_channels"],
        beta=config["beta"],
        loss_config=config.get("loss_config", {})
    )
    
    model = model.to(device)
    model.train()
    
    # Create optimizer with lower learning rate
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.get("lr", 0.0001),  # Lower learning rate
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Load dataset
    print("Loading dataset...")
    train_data, val_data = get_split_data(config=config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
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
        for batch_idx, (images, _) in enumerate(train_pbar):
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
            for batch_idx, (images, _) in enumerate(val_pbar):
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
            torch.save(model.state_dict(), os.path.join(config["model_save_path"], "vae_improved.pth"))
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
    """Main function"""
    
    # Configure GPU
    device = configure_gpu()
    
    # Load config
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override with improved settings
    config.update({
        "embedding_size": 512,  # Increased from 200
        "beta": 1.0,  # Reduced from 2000
        "lr": 0.0001,  # Lower learning rate
        "max_epoch": 100,  # More epochs
        "use_perceptual_loss": True,
        "use_lpips": False
    })
    
    print("Improved VAE Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train the model
    train_variational_autoencoder_improved(config, device)

if __name__ == "__main__":
    main()
