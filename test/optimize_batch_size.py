#!/usr/bin/env python3
"""
Optimize batch size for Fast High Quality VAE on RTX 3090.
Tests different batch sizes to find the optimal one for training speed and memory usage.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import psutil
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import get_split_data, configure_gpu
from config_loader import ConfigLoader

def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return allocated, reserved, total
    return 0, 0, 0

def test_batch_size(batch_size, model, device, train_data, num_iterations=5):
    """Test a specific batch size and measure performance."""
    
    print(f"\n🧪 Testing batch size: {batch_size}")
    print("-" * 40)
    
    try:
        # Create data loader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        sample_batch = sample_batch.to(device)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Measure memory before
        mem_alloc_before, mem_reserved_before, mem_total = get_gpu_memory_info()
        
        # Warm up
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        # Time multiple iterations
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            
            # Forward pass
            emb_mean, emb_log_var, reconst = model(sample_batch)
            
            # Calculate loss
            recon_loss = model.reconstruction_loss(sample_batch, reconst)
            kl_loss = model.kl_loss(emb_mean, emb_log_var)
            total_loss = recon_loss + kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Clear gradients
            optimizer.zero_grad()
        
        # Measure memory after
        mem_alloc_after, mem_reserved_after, _ = get_gpu_memory_info()
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate throughput
        images_per_second = batch_size / avg_time
        images_per_hour = images_per_second * 3600
        
        # Memory usage
        memory_used = mem_alloc_after - mem_alloc_before
        memory_efficiency = (memory_used / mem_total) * 100
        
        print(f"  ✅ Success!")
        print(f"  • Average time per batch: {avg_time:.3f}s")
        print(f"  • Min time: {min_time:.3f}s, Max time: {max_time:.3f}s")
        print(f"  • Throughput: {images_per_second:.1f} images/sec")
        print(f"  • Throughput: {images_per_hour:.0f} images/hour")
        print(f"  • Memory used: {memory_used:.2f} GB")
        print(f"  • Memory efficiency: {memory_efficiency:.1f}% of total GPU memory")
        
        return {
            'batch_size': batch_size,
            'success': True,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'throughput_per_sec': images_per_second,
            'throughput_per_hour': images_per_hour,
            'memory_used_gb': memory_used,
            'memory_efficiency_pct': memory_efficiency
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ❌ Out of memory!")
            print(f"  • Error: {str(e)[:100]}...")
            return {
                'batch_size': batch_size,
                'success': False,
                'error': 'out_of_memory'
            }
        else:
            print(f"  ❌ Runtime error: {str(e)}")
            return {
                'batch_size': batch_size,
                'success': False,
                'error': str(e)
            }
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return {
            'batch_size': batch_size,
            'success': False,
            'error': str(e)
        }

def estimate_training_time(throughput_per_hour, dataset_size, epochs):
    """Estimate total training time."""
    total_images = dataset_size * epochs
    hours = total_images / throughput_per_hour
    return hours

def main():
    """Find optimal batch size for Fast High Quality VAE."""
    
    print("🔍 Batch Size Optimization for Fast High Quality VAE")
    print("=" * 60)
    print("GPU: RTX 3090 (24GB VRAM)")
    print("Model: 128×128, 512 embedding, 128 channels (~125M params)")
    print("=" * 60)
    
    # Configure GPU
    device = configure_gpu()
    
    # Load configuration
    config_loader = ConfigLoader("config/config_unified.json")
    config = config_loader.get_config(
        loss_preset="high_quality",
        training_preset="fast_high_quality_training", 
        model_preset="fast_high_quality",
        dataset_preset="small"  # Use small dataset for testing
    )
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = VAE_pt(
        input_img_size=config['input_img_size'],
        embedding_size=config['embedding_size'],
        loss_config=config
    ).to(device)
    
    # Load small dataset for testing
    print(f"\n📊 Loading test dataset...")
    train_data, val_data = get_split_data(config=config)
    print(f"  • Test samples: {len(train_data):,}")
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    
    print(f"\n🧪 Testing batch sizes...")
    results = []
    
    for batch_size in batch_sizes:
        result = test_batch_size(batch_size, model, device, train_data)
        results.append(result)
        
        # Clear GPU memory between tests
        torch.cuda.empty_cache()
        
        # Stop if we hit memory limit
        if not result['success'] and 'out_of_memory' in result.get('error', ''):
            print(f"  ⚠️  Stopping at batch size {batch_size} due to memory limit")
            break
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print(f"\n❌ No successful batch sizes found!")
        return
    
    # Find optimal batch size
    print(f"\n📈 Analysis Results")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Time (s)':<10} {'Throughput':<15} {'Memory (GB)':<12} {'Efficiency':<12}")
    print("-" * 80)
    
    best_throughput = 0
    best_efficiency = 0
    best_balanced = 0
    
    for result in successful_results:
        print(f"{result['batch_size']:<12} {result['avg_time']:<10.3f} {result['throughput_per_hour']:<15.0f} {result['memory_used_gb']:<12.2f} {result['memory_efficiency_pct']:<12.1f}%")
        
        # Track best performers
        if result['throughput_per_hour'] > best_throughput:
            best_throughput = result['throughput_per_hour']
            best_throughput_batch = result['batch_size']
        
        if result['memory_efficiency_pct'] > best_efficiency:
            best_efficiency = result['memory_efficiency_pct']
            best_efficiency_batch = result['batch_size']
        
        # Balanced score (throughput * efficiency)
        balanced_score = result['throughput_per_hour'] * (result['memory_efficiency_pct'] / 100)
        if balanced_score > best_balanced:
            best_balanced = balanced_score
            best_balanced_batch = result['batch_size']
    
    # Recommendations
    print(f"\n💡 Recommendations")
    print("=" * 60)
    
    print(f"🚀 Best Throughput: Batch size {best_throughput_batch}")
    best_throughput_result = next(r for r in successful_results if r['batch_size'] == best_throughput_batch)
    print(f"   • {best_throughput_result['throughput_per_hour']:.0f} images/hour")
    print(f"   • {best_throughput_result['avg_time']:.3f}s per batch")
    print(f"   • {best_throughput_result['memory_used_gb']:.2f} GB memory")
    
    print(f"\n💾 Best Memory Efficiency: Batch size {best_efficiency_batch}")
    best_efficiency_result = next(r for r in successful_results if r['batch_size'] == best_efficiency_batch)
    print(f"   • {best_efficiency_result['memory_efficiency_pct']:.1f}% memory efficiency")
    print(f"   • {best_efficiency_result['throughput_per_hour']:.0f} images/hour")
    print(f"   • {best_efficiency_result['memory_used_gb']:.2f} GB memory")
    
    print(f"\n⚖️  Best Balanced: Batch size {best_balanced_batch}")
    best_balanced_result = next(r for r in successful_results if r['batch_size'] == best_balanced_batch)
    print(f"   • {best_balanced_result['throughput_per_hour']:.0f} images/hour")
    print(f"   • {best_balanced_result['memory_efficiency_pct']:.1f}% memory efficiency")
    print(f"   • {best_balanced_result['memory_used_gb']:.2f} GB memory")
    
    # Training time estimates
    print(f"\n⏱️  Training Time Estimates (Full Dataset)")
    print("-" * 60)
    
    dataset_size = 162080  # Full CelebA training set
    epochs = 100
    
    for result in successful_results:
        if result['batch_size'] in [best_throughput_batch, best_efficiency_batch, best_balanced_batch]:
            hours = estimate_training_time(result['throughput_per_hour'], dataset_size, epochs)
            print(f"  Batch {result['batch_size']:3d}: {hours:.1f} hours ({hours/24:.1f} days)")
    
    # Final recommendation
    print(f"\n🎯 Final Recommendation")
    print("=" * 60)
    print(f"Use batch size {best_balanced_batch} for optimal balance of:")
    print(f"  • Speed: {best_balanced_result['throughput_per_hour']:.0f} images/hour")
    print(f"  • Memory: {best_balanced_result['memory_efficiency_pct']:.1f}% GPU utilization")
    print(f"  • Training time: {estimate_training_time(best_balanced_result['throughput_per_hour'], dataset_size, epochs):.1f} hours")
    
    # Update config recommendation
    print(f"\n🔧 Configuration Update")
    print("-" * 60)
    print(f"Update your training config to use batch_size: {best_balanced_batch}")
    print(f"This will be more efficient than the default batch_size: 128")

if __name__ == "__main__":
    main()
