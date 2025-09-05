#!/usr/bin/env python3
"""
Monitor training progress by checking the latest model and generating sample images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from variation_autoencoder_pytorch import VAE_pt
from utilities_pytorch import configure_gpu
import json
import os
import time

def check_training_progress():
    """Check if training is in progress and show current model status."""
    
    print("🔍 Checking Training Progress")
    print("=" * 40)
    
    # Load config
    with open("config/config.json", 'r') as f:
        config = json.load(f)
    
    # Configure GPU
    device = configure_gpu()
    
    # Check if model file exists
    model_path = "model_weights/vae.pth"
    if not os.path.exists(model_path):
        print("❌ No model file found. Training may not have started yet.")
        return
    
    # Check file modification time
    mod_time = os.path.getmtime(model_path)
    mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
    print(f"✅ Model file last modified: {mod_time_str}")
    
    # Load model
    try:
        model = VAE_pt(config["input_img_size"], config["embedding_size"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
        
        # Test model with random generation
        print("\n🧪 Testing Model Quality")
        print("-" * 30)
        
        with torch.no_grad():
            # Generate random images
            z = torch.randn(5, config["embedding_size"]).to(device)
            generated = model.dec(z)
            generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
            
            print(f"Generated shape: {generated_np.shape}")
            print(f"Generated range: [{generated_np.min():.6f}, {generated_np.max():.6f}]")
            print(f"Generated mean: {generated_np.mean():.6f}")
            print(f"Generated std: {generated_np.std():.6f}")
            
            # Check if images are still dark
            if generated_np.mean() < 0.01:
                print("⚠️  Images are still very dark - training may need more epochs")
            elif generated_np.mean() > 0.1:
                print("✅ Images have reasonable brightness - training is progressing well!")
            else:
                print("🔄 Images are improving but may need more training")
            
            # Save sample images
            os.makedirs("training_samples", exist_ok=True)
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                axes[i].imshow(np.clip(generated_np[i], 0, 1))
                axes[i].set_title(f"Sample {i+1}")
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig(f"training_samples/checkpoint_{int(time.time())}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Sample images saved to training_samples/")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n" + "=" * 40)
    print("💡 Training is in progress. Check back in a few minutes!")
    print("💡 You can run this script again to monitor progress.")

if __name__ == "__main__":
    check_training_progress()
