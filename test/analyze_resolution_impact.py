#!/usr/bin/env python3
"""
Analyze how resolution changes affect parameter count in VAE models.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from variation_autoencoder_improved import VAE_pt

def analyze_resolution_impact():
    """Analyze how different resolutions affect parameter count."""
    
    print("🔍 VAE Parameter Count Analysis by Resolution")
    print("=" * 70)
    
    # Test configurations
    configs = [
        {
            "name": "64x64, 512 embedding, 128 channels",
            "input_size": 64,
            "embedding_size": 512,
            "num_channels": 128
        },
        {
            "name": "128x128, 512 embedding, 128 channels", 
            "input_size": 128,
            "embedding_size": 512,
            "num_channels": 128
        },
        {
            "name": "256x256, 512 embedding, 128 channels",
            "input_size": 256,
            "embedding_size": 512,
            "num_channels": 128
        },
        {
            "name": "64x64, 1024 embedding, 128 channels",
            "input_size": 64,
            "embedding_size": 1024,
            "num_channels": 128
        },
        {
            "name": "128x128, 1024 embedding, 128 channels",
            "input_size": 128,
            "embedding_size": 1024,
            "num_channels": 128
        },
        {
            "name": "64x64, 512 embedding, 256 channels",
            "input_size": 64,
            "embedding_size": 512,
            "num_channels": 256
        },
        {
            "name": "128x128, 512 embedding, 256 channels",
            "input_size": 128,
            "embedding_size": 512,
            "num_channels": 256
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Analyzing: {config['name']}")
        print("-" * 50)
        
        # Create model
        model = VAE_pt(
            input_img_size=config['input_size'],
            embedding_size=config['embedding_size'],
            num_channels=config['num_channels'],
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
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / 1024 / 1024
        
        # Calculate encoder/decoder breakdown
        encoder_params = sum(p.numel() for p in model.enc.parameters())
        decoder_params = sum(p.numel() for p in model.dec.parameters())
        
        result = {
            'name': config['name'],
            'input_size': config['input_size'],
            'embedding_size': config['embedding_size'],
            'num_channels': config['num_channels'],
            'total_params': total_params,
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'model_size_mb': model_size_mb
        }
        results.append(result)
        
        print(f"  Resolution: {config['input_size']}x{config['input_size']}")
        print(f"  Embedding Size: {config['embedding_size']}")
        print(f"  Channels: {config['num_channels']}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Encoder Parameters: {encoder_params:,}")
        print(f"  Decoder Parameters: {decoder_params:,}")
        print(f"  Model Size: {model_size_mb:.1f} MB")
    
    # Analysis
    print(f"\n📈 Resolution Impact Analysis")
    print("=" * 70)
    
    # Compare 64x64 vs 128x128 with same embedding/channels
    base_64 = next(r for r in results if r['input_size'] == 64 and r['embedding_size'] == 512 and r['num_channels'] == 128)
    base_128 = next(r for r in results if r['input_size'] == 128 and r['embedding_size'] == 512 and r['num_channels'] == 128)
    
    param_increase = ((base_128['total_params'] - base_64['total_params']) / base_64['total_params']) * 100
    
    print(f"🔍 Resolution Impact (512 embedding, 128 channels):")
    print(f"  • 64×64:  {base_64['total_params']:,} parameters")
    print(f"  • 128×128: {base_128['total_params']:,} parameters")
    print(f"  • Increase: {param_increase:.1f}%")
    print(f"  • Additional params: {base_128['total_params'] - base_64['total_params']:,}")
    
    # Compare embedding size impact
    embed_512_64 = next(r for r in results if r['input_size'] == 64 and r['embedding_size'] == 512 and r['num_channels'] == 128)
    embed_1024_64 = next(r for r in results if r['input_size'] == 64 and r['embedding_size'] == 1024 and r['num_channels'] == 128)
    
    embed_increase = ((embed_1024_64['total_params'] - embed_512_64['total_params']) / embed_512_64['total_params']) * 100
    
    print(f"\n🔍 Embedding Size Impact (64×64, 128 channels):")
    print(f"  • 512 embedding:  {embed_512_64['total_params']:,} parameters")
    print(f"  • 1024 embedding: {embed_1024_64['total_params']:,} parameters")
    print(f"  • Increase: {embed_increase:.1f}%")
    print(f"  • Additional params: {embed_1024_64['total_params'] - embed_512_64['total_params']:,}")
    
    # Compare channel impact
    chan_128_64 = next(r for r in results if r['input_size'] == 64 and r['embedding_size'] == 512 and r['num_channels'] == 128)
    chan_256_64 = next(r for r in results if r['input_size'] == 64 and r['embedding_size'] == 512 and r['num_channels'] == 256)
    
    chan_increase = ((chan_256_64['total_params'] - chan_128_64['total_params']) / chan_128_64['total_params']) * 100
    
    print(f"\n🔍 Channel Count Impact (64×64, 512 embedding):")
    print(f"  • 128 channels:  {chan_128_64['total_params']:,} parameters")
    print(f"  • 256 channels: {chan_256_64['total_params']:,} parameters")
    print(f"  • Increase: {chan_increase:.1f}%")
    print(f"  • Additional params: {chan_256_64['total_params'] - chan_128_64['total_params']:,}")
    
    # Summary table
    print(f"\n📊 Summary Comparison")
    print("=" * 70)
    print(f"{'Configuration':<40} {'Params (M)':<12} {'Size (MB)':<12}")
    print("-" * 70)
    
    for result in results:
        params_m = result['total_params'] / 1_000_000
        print(f"{result['name']:<40} {params_m:<12.1f} {result['model_size_mb']:<12.1f}")
    
    # Key insights
    print(f"\n💡 Key Insights")
    print("=" * 70)
    print(f"✅ Resolution has MINIMAL impact on parameters:")
    print(f"   • 64×64 → 128×128: Only {param_increase:.1f}% increase")
    print(f"   • Most parameters are in the dense layers (embedding space)")
    print(f"   • Convolutional layers scale with channels, not resolution")
    
    print(f"\n✅ Embedding size has MAJOR impact:")
    print(f"   • 512 → 1024 embedding: {embed_increase:.1f}% increase")
    print(f"   • Dense layers dominate parameter count")
    
    print(f"\n✅ Channel count has MODERATE impact:")
    print(f"   • 128 → 256 channels: {chan_increase:.1f}% increase")
    print(f"   • Affects convolutional layers")
    
    print(f"\n🎯 Recommendation for 128×128:")
    print(f"   • Use 512 embedding + 128 channels = {base_128['total_params']/1_000_000:.1f}M params")
    print(f"   • Only {param_increase:.1f}% more than 64×64 version")
    print(f"   • 4x more pixels with minimal parameter increase!")

if __name__ == "__main__":
    analyze_resolution_impact()
