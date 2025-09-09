#!/usr/bin/env python3
"""
Script to retrain the VAE model with higher resolution images for better display quality.
"""

import json
import os
import shutil

def create_high_res_config():
    """Create a new config file for higher resolution training."""
    
    # Load current config
    with open("config/config.json", 'r') as f:
        config = json.load(f)
    
    # Create high-resolution config
    high_res_config = config.copy()
    high_res_config["input_img_size"] = 128  # Increase from 64 to 128
    high_res_config["max_epoch"] = 100  # More epochs for higher resolution
    high_res_config["batch_size"] = 128  # Smaller batch size due to higher resolution
    high_res_config["lr"] = 0.0003  # Slightly lower learning rate for stability
    
    # Save high-resolution config
    with open("config/config_high_res.json", 'w') as f:
        json.dump(high_res_config, f, indent=4)
    
    print("✅ Created high-resolution config: config/config_high_res.json")
    print(f"   - Input size: {high_res_config['input_img_size']}x{high_res_config['input_img_size']}")
    print(f"   - Batch size: {high_res_config['batch_size']}")
    print(f"   - Max epochs: {high_res_config['max_epoch']}")
    print(f"   - Learning rate: {high_res_config['lr']}")
    
    return high_res_config

def create_high_res_training_script():
    """Create a modified training script for high resolution."""
    
    # Read the original training script
    with open("train_VAE_pytorch.py", 'r') as f:
        content = f.read()
    
    # Modify to use high-res config
    modified_content = content.replace(
        'config_path = "config/config.json"',
        'config_path = "config/config_high_res.json"'
    ).replace(
        'model_save_path = os.path.join(config["model_save_path"], "vae.pth")',
        'model_save_path = os.path.join(config["model_save_path"], "vae_high_res.pth")'
    )
    
    # Save modified training script
    with open("train_VAE_pytorch_high_res.py", 'w') as f:
        f.write(modified_content)
    
    print("✅ Created high-resolution training script: train_VAE_pytorch_high_res.py")

def create_high_res_gui():
    """Create a modified GUI for high resolution display."""
    
    # Read the original GUI
    with open("gui_pytorch.py", 'r') as f:
        content = f.read()
    
    # Modify to use high-res model and config
    modified_content = content.replace(
        'config_path = \'config/config.json\'',
        'config_path = \'config/config_high_res.json\''
    ).replace(
        'model_path = os.path.join(config["model_save_path"], "vae.pth")',
        'model_path = os.path.join(config["model_save_path"], "vae_high_res.pth")'
    ).replace(
        'width=800',
        'width=1000'  # Larger display for higher resolution images
    )
    
    # Save modified GUI
    with open("gui_pytorch_high_res.py", 'w') as f:
        f.write(modified_content)
    
    print("✅ Created high-resolution GUI: gui_pytorch_high_res.py")

def main():
    """Main function to set up high-resolution training."""
    
    print("🚀 Setting up High-Resolution Training")
    print("=" * 50)
    
    # Create high-resolution config
    config = create_high_res_config()
    
    # Create high-resolution training script
    create_high_res_training_script()
    
    # Create high-resolution GUI
    create_high_res_gui()
    
    print("\n" + "=" * 50)
    print("✅ High-resolution setup complete!")
    print("\nTo train with higher resolution:")
    print("1. Run: uv run train_VAE_pytorch_high_res.py")
    print("2. Run: uv run streamlit run gui_pytorch_high_res.py")
    print("\nNote: This will take longer and use more GPU memory.")
    print("The 128x128 images will be much clearer in the GUI!")
    print("=" * 50)

if __name__ == "__main__":
    main()
