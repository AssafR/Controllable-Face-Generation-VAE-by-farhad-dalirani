#!/usr/bin/env python3
"""
Compare model sizes and training times for different high-quality presets.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from variation_autoencoder_improved import VAE_pt
from config_loader import ConfigLoader

def calculate_model_size(embedding_size, num_channels, input_size):
    """Calculate approximate model size and parameters."""
    
    # Create a temporary model to count parameters
    model = VAE_pt(
        input_img_size=input_size,
        embedding_size=embedding_size,
        num_channels=num_channels,
        loss_config={
            'use_mse': True,
            'use_l1': True,
            'use_perceptual_loss': True,
            'use_lpips': False,
            'mse_weight': 0.5,
            'l1_weight': 0.3,
            'perceptual_weight': 0.2,
            'lpips_weight': 0.0
        }
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (in MB)
    model_size_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float32
    
    return total_params, trainable_params, model_size_mb

def estimate_training_time(embedding_size, num_channels, input_size, batch_size, max_epochs, dataset_size=200000):
    """Estimate training time based on model complexity and settings."""
    
    # Base time per epoch (in hours) - rough estimates
    base_time_per_epoch = 0.1  # hours
    
    # Complexity factors
    resolution_factor = (input_size / 64) ** 2  # Quadratic scaling with resolution
    channel_factor = (num_channels / 128) ** 1.5  # Sub-quadratic scaling with channels
    embedding_factor = (embedding_size / 512) ** 1.2  # Sub-linear scaling with embedding
    batch_factor = 64 / batch_size  # Inverse scaling with batch size
    
    # Calculate time per epoch
    time_per_epoch = base_time_per_epoch * resolution_factor * channel_factor * embedding_factor * batch_factor
    
    # Total training time
    total_time = time_per_epoch * max_epochs
    
    return time_per_epoch, total_time

def main():
    """Compare different high-quality model configurations."""
    
    print("🔍 High-Quality VAE Model Size Comparison")
    print("=" * 80)
    
    # Load configuration
    config_loader = ConfigLoader("config/config_unified.json")
    
    # Define configurations to compare
    configs = [
        {
            "name": "Ultra High Quality (Current)",
            "model_preset": "ultra_high_quality",
            "training_preset": "ultra_high_quality_training",
            "loss_preset": "ultra_high_quality_loss"
        },
        {
            "name": "Fast High Quality",
            "model_preset": "fast_high_quality", 
            "training_preset": "fast_high_quality_training",
            "loss_preset": "high_quality"
        },
        {
            "name": "Balanced High Quality",
            "model_preset": "balanced_high_quality",
            "training_preset": "balanced_high_quality_training", 
            "loss_preset": "high_quality"
        },
        {
            "name": "Compact High Quality",
            "model_preset": "compact_high_quality",
            "training_preset": "compact_high_quality_training",
            "loss_preset": "high_quality"
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Analyzing: {config['name']}")
        print("-" * 50)
        
        # Load configuration
        full_config = config_loader.get_config(
            loss_preset=config['loss_preset'],
            training_preset=config['training_preset'],
            model_preset=config['model_preset'],
            dataset_preset="full"
        )
        
        # Extract key parameters
        input_size = full_config['input_img_size']
        embedding_size = full_config['embedding_size']
        num_channels = full_config['num_channels']
        batch_size = full_config['batch_size']
        max_epochs = full_config['max_epoch']
        lr = full_config['lr']
        
        # Calculate model size
        total_params, trainable_params, model_size_mb = calculate_model_size(
            embedding_size, num_channels, input_size
        )
        
        # Estimate training time
        time_per_epoch, total_time = estimate_training_time(
            embedding_size, num_channels, input_size, batch_size, max_epochs
        )
        
        # Store results
        result = {
            'name': config['name'],
            'input_size': input_size,
            'embedding_size': embedding_size,
            'num_channels': num_channels,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'lr': lr,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'time_per_epoch': time_per_epoch,
            'total_time': total_time
        }
        results.append(result)
        
        # Print details
        print(f"  Resolution: {input_size}x{input_size}")
        print(f"  Embedding Size: {embedding_size}")
        print(f"  Channels: {num_channels}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Max Epochs: {max_epochs}")
        print(f"  Learning Rate: {lr}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Model Size: {model_size_mb:.1f} MB")
        print(f"  Time per Epoch: {time_per_epoch:.2f} hours")
        print(f"  Total Training Time: {total_time:.1f} hours")
    
    # Summary comparison
    print(f"\n📈 Summary Comparison")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Params (M)':<12} {'Size (MB)':<12} {'Time (h)':<12} {'Quality':<15}")
    print("-" * 80)
    
    for result in results:
        params_m = result['total_params'] / 1_000_000
        print(f"{result['name']:<25} {params_m:<12.1f} {result['model_size_mb']:<12.1f} {result['total_time']:<12.1f} {'High' if result['input_size'] >= 128 else 'Good'}")
    
    # Recommendations
    print(f"\n💡 Recommendations")
    print("=" * 80)
    
    # Find the most balanced option
    balanced = min(results[1:], key=lambda x: x['total_time'] + x['model_size_mb']/100)
    fastest = min(results[1:], key=lambda x: x['total_time'])
    smallest = min(results[1:], key=lambda x: x['total_params'])
    
    print(f"🚀 Fastest Training: {fastest['name']}")
    print(f"   • {fastest['total_time']:.1f} hours training time")
    print(f"   • {fastest['total_params']/1_000_000:.1f}M parameters")
    print(f"   • {fastest['input_size']}x{fastest['input_size']} resolution")
    
    print(f"\n⚖️  Most Balanced: {balanced['name']}")
    print(f"   • {balanced['total_time']:.1f} hours training time")
    print(f"   • {balanced['total_params']/1_000_000:.1f}M parameters")
    print(f"   • {balanced['input_size']}x{balanced['input_size']} resolution")
    
    print(f"\n💾 Smallest Model: {smallest['name']}")
    print(f"   • {smallest['total_params']/1_000_000:.1f}M parameters")
    print(f"   • {smallest['model_size_mb']:.1f} MB memory")
    print(f"   • {smallest['input_size']}x{smallest['input_size']} resolution")
    
    print(f"\n🎯 Suggested Choice: {balanced['name']}")
    print(f"   • Good balance of quality and speed")
    print(f"   • Reasonable training time ({balanced['total_time']:.1f}h)")
    print(f"   • Manageable model size ({balanced['total_params']/1_000_000:.1f}M params)")
    print(f"   • High resolution output ({balanced['input_size']}x{balanced['input_size']})")

if __name__ == "__main__":
    main()
