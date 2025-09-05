import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Compare different loss function configurations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import configure_gpu
import json
import os
import glob

def test_loss_configuration(config_path, device, num_samples=5):
    """Test a specific loss configuration."""
    
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    print(f"\n🧪 Testing {config_name}")
    print("-" * 40)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    model_path = f"model_weights/{config_name}.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    model = VAE_pt(
        input_img_size=config["input_img_size"],
        embedding_size=config["embedding_size"],
        num_channels=config["num_channels"],
        beta=config["beta"],
        loss_config=config.get("loss_config", {})
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Print loss configuration
    loss_config = config.get("loss_config", {})
    print(f"Loss config: MSE={loss_config.get('use_mse', True)}, "
          f"L1={loss_config.get('use_l1', True)}, "
          f"Perceptual={loss_config.get('use_perceptual_loss', False)}, "
          f"LPIPS={loss_config.get('use_lpips', False)}")
    
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
        os.makedirs("loss_comparison", exist_ok=True)
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            axes[i].imshow(np.clip(generated_np[i], 0, 1))
            axes[i].set_title(f"{config_name} {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"loss_comparison/{config_name}_samples.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'config_name': config_name,
            'range': (generated_np.min(), generated_np.max()),
            'mean': generated_np.mean(),
            'std': generated_np.std(),
            'sharpness': avg_sharpness,
            'loss_config': loss_config
        }

def main():
    """Main comparison function"""
    
    print("🔍 Loss Function Comparison")
    print("=" * 50)
    
    # Configure GPU
    device = configure_gpu()
    
    # Find all config files
    config_files = glob.glob("../config/config_*.json") if os.path.exists("../config/") else glob.glob("config/config_*.json")
    config_files = [f for f in config_files if not f.endswith("config.json")]  # Exclude base config
    
    if not config_files:
        print("❌ No config files found in config/ directory")
        print("Available configs should be:")
        print("  - config_mse_only.json")
        print("  - config_mse_l1.json") 
        print("  - config_perceptual.json")
        print("  - config_lpips.json")
        return
    
    print(f"Found {len(config_files)} configuration files")
    
    results = []
    
    # Test each configuration
    for config_path in config_files:
        result = test_loss_configuration(config_path, device)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No models found to compare")
        return
    
    # Print comparison table
    print("\n📊 Comparison Results")
    print("=" * 80)
    print(f"{'Config':<20} {'Range':<20} {'Mean':<8} {'Std':<8} {'Sharpness':<12}")
    print("-" * 80)
    
    for result in results:
        range_str = f"[{result['range'][0]:.3f}, {result['range'][1]:.3f}]"
        print(f"{result['config_name']:<20} {range_str:<20} {result['mean']:<8.3f} {result['std']:<8.3f} {result['sharpness']:<12.6f}")
    
    # Find best configurations
    if len(results) > 1:
        print(f"\n🎯 Best Results:")
        
        # Best sharpness
        best_sharpness = max(results, key=lambda x: x['sharpness'])
        print(f"  Sharpest: {best_sharpness['config_name']} (sharpness: {best_sharpness['sharpness']:.6f})")
        
        # Best contrast (std)
        best_contrast = max(results, key=lambda x: x['std'])
        print(f"  Best contrast: {best_contrast['config_name']} (std: {best_contrast['std']:.3f})")
        
        # Most balanced (closest to 0.5 mean)
        best_balanced = min(results, key=lambda x: abs(x['mean'] - 0.5))
        print(f"  Most balanced: {best_balanced['config_name']} (mean: {best_balanced['mean']:.3f})")
    
    print(f"\n✅ Comparison complete! Check loss_comparison/ folder for images.")
    print(f"📁 Generated {len(results)} comparison images")

if __name__ == "__main__":
    main()
