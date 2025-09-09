import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("TensorBoard not available, using simple logging instead")
    TENSORBOARD_AVAILABLE = False
from variation_autoencoder_pytorch import VAE_pt
from utilities_pytorch import get_split_data

# GPU Configuration
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
    
    # Print PyTorch version and device info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    
    return device

def save_pytorch_as_tensorflow(pytorch_model, save_path):
    """Save PyTorch model info for TensorFlow compatibility"""
    print("\n" + "="*60)
    print("MODEL FORMAT COMPATIBILITY NOTICE")
    print("="*60)
    print("PyTorch models have been saved in .pth format.")
    print("Existing GUI and synthesis scripts expect .keras format.")
    print("\nTo use the trained PyTorch models, you have these options:")
    print("\n1. RECOMMENDED: Use original TensorFlow training for final deployment")
    print("   - Run: uv run train_VAE.py")
    print("   - This will create .keras files for existing scripts")
    print("\n2. Convert existing scripts to use PyTorch models")
    print("   - Modify gui.py and synthesis.py to load .pth files")
    print("   - Use VAE_pt instead of VAE class")
    print("\n3. Manual weight conversion (complex)")
    print("   - Convert PyTorch weights to TensorFlow format")
    print("   - Requires careful mapping of layer weights")
    print("\nCurrent PyTorch models saved:")
    print(f"   - {os.path.join(save_path, 'vae.pth')}")
    print(f"   - {os.path.join(save_path, 'encoder.pth')}")
    print(f"   - {os.path.join(save_path, 'decoder.pth')}")
    print("="*60)

def train_variational_autoencoder(config, train, validation, device):
    """ Train Variational Autoencoder with debugging """

    # Create Variational Autoencoder
    model_vae = VAE_pt(
        input_img_size=config["input_img_size"], 
        embedding_size=config["embedding_size"], 
        num_channels=config["num_channels"], 
        beta=config["beta"])
    
    model_vae = model_vae.to(device)

    # Optimizer
    optimizer = optim.Adam(model_vae.parameters(), lr=config["lr"])

    # Create data loaders with increased batch size for GPU
    batch_size = config.get("batch_size", 256)  # Increased default batch size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Training metrics
    best_val_loss = float('inf')
    checkpoint_path = "./model_weights/checkpoint/"
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(config["model_save_path"], exist_ok=True)

    print(f"Starting training for {config['max_epoch']} epochs with batch size {batch_size}...")
    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(validation)}")
    
    for epoch in range(config["max_epoch"]):
        # Training phase
        model_vae.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        num_batches = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["max_epoch"]} [Train]', 
                         leave=False, ncols=100)
        for batch_idx, batch_data in enumerate(train_pbar):
            if isinstance(batch_data, (list, tuple)):
                data = batch_data[0]  # Handle (data, labels) format
            else:
                data = batch_data
            
            # Convert to PyTorch tensor and move to device
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            
            # Ensure data is in NCHW format (batch_size, channels, height, width)
            if data.dim() == 4 and data.shape[-1] == 3:  # NHWC format
                data = data.permute(0, 3, 1, 2)  # Convert to NCHW
            
            data = data.to(device)
            
            # Debug: Print data statistics
            if batch_idx == 0:
                print(f"\n=== Batch {batch_idx} Debug Info ===")
                print(f"Data shape: {data.shape}")
                print(f"Data range: {data.min().item():.6f} to {data.max().item():.6f}")
                print(f"Data mean: {data.mean().item():.6f}")
                print(f"Data std: {data.std().item():.6f}")
            
            # Training step
            metrics = model_vae.train_step(data, optimizer)
            
            train_loss += metrics["loss"]
            train_recon_loss += metrics["reconstruction_loss"]
            train_kl_loss += metrics["kl_loss"]
            num_batches += 1
            
            # Update progress bar with current metrics
            train_pbar.set_postfix({
                'Loss': f'{metrics["loss"]:.4f}',
                'Recon': f'{metrics["reconstruction_loss"]:.4f}',
                'KL': f'{metrics["kl_loss"]:.4f}'
            })
            
            # Debug: Print detailed metrics every 50 batches
            if batch_idx % 50 == 0:
                print(f"\n=== Batch {batch_idx} Detailed Metrics ===")
                print(f"Total Loss: {metrics['loss']:.6f}")
                print(f"Reconstruction Loss: {metrics['reconstruction_loss']:.6f}")
                print(f"KL Loss: {metrics['kl_loss']:.6f}")
                
                # Test generation during training
                with torch.no_grad():
                    test_latent = torch.randn(1, config["embedding_size"]).to(device)
                    test_generated = model_vae.dec(test_latent)
                    test_np = test_generated.permute(0, 2, 3, 1).cpu().numpy()
                    print(f"Test generation range: {test_np.min():.6f} to {test_np.max():.6f}")
                    print(f"Test generation mean: {test_np.mean():.6f}")

        # Validation phase
        model_vae.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_batches = 0
        
        # Validation loop with progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["max_epoch"]} [Val]', 
                       leave=False, ncols=100)
        with torch.no_grad():
            for batch_data in val_pbar:
                if isinstance(batch_data, (list, tuple)):
                    data = batch_data[0]
                else:
                    data = batch_data
                
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                
                if data.dim() == 4 and data.shape[-1] == 3:  # NHWC format
                    data = data.permute(0, 3, 1, 2)  # Convert to NCHW
                
                data = data.to(device)
                
                metrics = model_vae.test_step(data)
                
                val_loss += metrics["loss"]
                val_recon_loss += metrics["reconstruction_loss"]
                val_kl_loss += metrics["kl_loss"]
                val_batches += 1
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'Loss': f'{metrics["loss"]:.4f}',
                    'Recon': f'{metrics["reconstruction_loss"]:.4f}',
                    'KL': f'{metrics["kl_loss"]:.4f}'
                })

        # Calculate average losses
        avg_train_loss = train_loss / num_batches
        avg_train_recon = train_recon_loss / num_batches
        avg_train_kl = train_kl_loss / num_batches
        avg_val_loss = val_loss / val_batches
        avg_val_recon = val_recon_loss / val_batches
        avg_val_kl = val_kl_loss / val_batches

        print(f'\nEpoch {epoch+1}/{config["max_epoch"]} - '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Test final generation
        print(f"\n=== Final Generation Test ===")
        with torch.no_grad():
            test_latent = torch.randn(4, config["embedding_size"]).to(device)
            test_generated = model_vae.dec(test_latent)
            test_np = test_generated.permute(0, 2, 3, 1).cpu().numpy()
            print(f"Final generation range: {test_np.min():.6f} to {test_np.max():.6f}")
            print(f"Final generation mean: {test_np.mean():.6f}")
            
            # Save a test image
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            for i in range(4):
                row, col = i // 2, i % 2
                axes[row, col].imshow(test_np[i])
                axes[row, col].set_title(f'Generated {i}')
                axes[row, col].axis('off')
            plt.tight_layout()
            plt.savefig('debug_training_result.png')
            print("Test images saved to debug_training_result.png")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(checkpoint_path, "best_model.pth"))
            print(f'New best model saved with validation loss: {avg_val_loss:.4f}')

    # Save final models
    torch.save(model_vae.state_dict(), os.path.join(config["model_save_path"], "vae.pth"))
    torch.save(model_vae.enc.state_dict(), os.path.join(config["model_save_path"], "encoder.pth"))
    torch.save(model_vae.dec.state_dict(), os.path.join(config["model_save_path"], "decoder.pth"))
    
    print("Training completed!")

if __name__ == '__main__':
    
    # Configure GPU settings
    device = configure_gpu()
    
    # Path to config file
    config_path = 'config/config.json'

    # Open and read the config file
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    train, validation = get_split_data(config=config)

    train_variational_autoencoder(config=config, train=train, validation=validation, device=device)
