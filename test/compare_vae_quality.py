import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Compare VAE quality between original and improved models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from variation_autoencoder_pytorch import VAE_pt as VAE_original
from variation_autoencoder_improved import VAE_pt as VAE_improved
from utilities_pytorch import configure_gpu
import json
import os

def test_model_quality(model, config, device, model_name, num_samples=5):
    """Test model quality and generate sample images."""
    
    print(f"\n🧪 Testing {model_name}")
    print("-" * 40)
    
    model.eval()
    
    with torch.no_grad():
        # Generate random images
        z = torch.randn(num_samples, config["embedding_size"]).to(device)
        generated = model.dec(z)
        generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
        
        print(f"Generated shape: {generated_np.shape}")
        print(f"Generated range: [{generated_np.min():.6f}, {generated_np.max():.6f}]")
        print(f"Generated mean: {generated_np.mean():.6f}")
        print(f"Generated std: {generated_np.std():.6f}")
        
        # Calculate sharpness (Laplacian variance)
        sharpness_scores = []
        for i in range(num_samples):
            img = generated_np[i]
            if len(img.shape) == 3:
                img_gray = np.mean(img, axis=2)
            else:
                img_gray = img
            
            # Calculate Laplacian variance (higher = sharper)
            laplacian = np.var(np.gradient(np.gradient(img_gray, axis=0), axis=0) + 
                              np.gradient(np.gradient(img_gray, axis=1), axis=1))
            sharpness_scores.append(laplacian)
        
        avg_sharpness = np.mean(sharpness_scores)
        print(f"Average sharpness: {avg_sharpness:.6f}")
        
        # Save sample images
        os.makedirs("comparison_samples", exist_ok=True)
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            axes[i].imshow(np.clip(generated_np[i], 0, 1))
            axes[i].set_title(f"{model_name} {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"comparison_samples/{model_name.lower().replace(' ', '_')}_samples.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'range': (generated_np.min(), generated_np.max()),
            'mean': generated_np.mean(),
            'std': generated_np.std(),
            'sharpness': avg_sharpness
        }

def main():
    """Main comparison function"""
    
    print("🔍 VAE Quality Comparison")
    print("=" * 50)
    
    # Configure GPU
    device = configure_gpu()
    
    # Load config
    # Try relative path first (when running from test/ directory)
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = config_path
    
    with open(config_path, 'r') as file:
        config = json.load(f)
    
    results = {}
    
    # Test original model
    if os.path.exists("model_weights/vae.pth"):
        print("Loading original model...")
        model_original = VAE_original(
            config["input_img_size"], 
            config["embedding_size"]
        )
        model_original.load_state_dict(torch.load("model_weights/vae.pth", map_location=device))
        model_original.to(device)
        
        results['Original'] = test_model_quality(
            model_original, config, device, "Original VAE"
        )
    else:
        print("❌ Original model not found")
    
    # Test improved model
    if os.path.exists("model_weights/vae_improved.pth"):
        print("Loading improved model...")
        # Use improved config
        improved_config = config.copy()
        improved_config["embedding_size"] = 512
        
        model_improved = VAE_improved(
            improved_config["input_img_size"], 
            improved_config["embedding_size"]
        )
        model_improved.load_state_dict(torch.load("model_weights/vae_improved.pth", map_location=device))
        model_improved.to(device)
        
        results['Improved'] = test_model_quality(
            model_improved, improved_config, device, "Improved VAE"
        )
    else:
        print("❌ Improved model not found - run train_VAE_improved.py first")
    
    # Print comparison
    print("\n📊 Comparison Results")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Range: [{metrics['range'][0]:.4f}, {metrics['range'][1]:.4f}]")
        print(f"  Mean: {metrics['mean']:.4f}")
        print(f"  Std: {metrics['std']:.4f}")
        print(f"  Sharpness: {metrics['sharpness']:.6f}")
    
    if len(results) == 2:
        print(f"\n🎯 Improvements:")
        orig = results['Original']
        impr = results['Improved']
        
        print(f"  Sharpness: {((impr['sharpness'] - orig['sharpness']) / orig['sharpness'] * 100):+.1f}%")
        print(f"  Mean brightness: {((impr['mean'] - orig['mean']) / orig['mean'] * 100):+.1f}%")
        print(f"  Std (contrast): {((impr['std'] - orig['std']) / orig['std'] * 100):+.1f}%")
    
    print("\n✅ Comparison complete! Check comparison_samples/ folder for images.")

if __name__ == "__main__":
    main()
