#!/usr/bin/env python3
"""
Checkpoint Management Script
Helps manage training checkpoints and resume training
"""

import os
import torch
import argparse
from datetime import datetime

def list_checkpoints(checkpoint_dir="checkpoints"):
    """List all available checkpoints"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(filepath)
            size = os.path.getsize(filepath) / (1024*1024)  # MB
            checkpoints.append((file, filepath, mtime, size))
    
    checkpoints.sort(key=lambda x: x[2], reverse=True)  # Sort by modification time
    
    print(f"📁 Available checkpoints in {checkpoint_dir}:")
    print("=" * 80)
    for file, filepath, mtime, size in checkpoints:
        timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"📄 {file}")
        print(f"   📅 Modified: {timestamp}")
        print(f"   📊 Size: {size:.1f} MB")
        print(f"   📍 Path: {filepath}")
        print()
    
    return checkpoints

def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
        print("=" * 60)
        
        if 'epoch' in checkpoint:
            print(f"📊 Epoch: {checkpoint['epoch'] + 1}")
        
        if 'best_val_loss' in checkpoint:
            print(f"🏆 Best validation loss: {checkpoint['best_val_loss']:.6f}")
        
        if 'patience_counter' in checkpoint:
            print(f"⏳ Patience counter: {checkpoint['patience_counter']}")
        
        if 'training_complete' in checkpoint:
            print(f"✅ Training complete: {checkpoint['training_complete']}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"⚙️  Configuration:")
            print(f"   • Max epochs: {config.get('max_epoch', 'N/A')}")
            print(f"   • Learning rate: {config.get('learning_rate', 'N/A')}")
            print(f"   • Batch size: {config.get('batch_size', 'N/A')}")
            print(f"   • Input size: {config.get('input_img_size', 'N/A')}x{config.get('input_img_size', 'N/A')}")
            print(f"   • Embedding size: {config.get('embedding_size', 'N/A')}")
        
        print(f"\n📦 Checkpoint contains:")
        for key in checkpoint.keys():
            if key == 'model_state_dict':
                print(f"   • {key}: Model parameters")
            elif key == 'optimizer_state_dict':
                print(f"   • {key}: Optimizer state")
            elif key == 'scheduler_state_dict':
                print(f"   • {key}: Learning rate scheduler state")
            else:
                print(f"   • {key}: {type(checkpoint[key]).__name__}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")

def clean_old_checkpoints(checkpoint_dir="checkpoints", keep_last=5):
    """Clean old periodic checkpoints, keeping only the last N"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find periodic checkpoints (not the main training_checkpoint.pth)
    periodic_checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(filepath)
            periodic_checkpoints.append((file, filepath, mtime))
    
    # Sort by modification time (newest first)
    periodic_checkpoints.sort(key=lambda x: x[2], reverse=True)
    
    # Keep only the last N checkpoints
    to_remove = periodic_checkpoints[keep_last:]
    
    if not to_remove:
        print(f"✅ No old checkpoints to clean (keeping last {keep_last})")
        return
    
    print(f"🧹 Cleaning old checkpoints (keeping last {keep_last}):")
    for file, filepath, mtime in to_remove:
        try:
            os.remove(filepath)
            print(f"   🗑️  Removed: {file}")
        except Exception as e:
            print(f"   ❌ Error removing {file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage training checkpoints")
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument("--inspect", type=str, help="Inspect a specific checkpoint")
    parser.add_argument("--clean", action="store_true", help="Clean old periodic checkpoints")
    parser.add_argument("--keep", type=int, default=5, help="Number of periodic checkpoints to keep (default: 5)")
    parser.add_argument("--dir", type=str, default="checkpoints", help="Checkpoint directory (default: checkpoints)")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args.dir)
    elif args.inspect:
        inspect_checkpoint(args.inspect)
    elif args.clean:
        clean_old_checkpoints(args.dir, args.keep)
    else:
        print("🔧 Checkpoint Management Tool")
        print("Use --help for available options")
        print("\nExamples:")
        print("  python manage_checkpoints.py --list")
        print("  python manage_checkpoints.py --inspect checkpoints/training_checkpoint.pth")
        print("  python manage_checkpoints.py --clean --keep 3")

if __name__ == "__main__":
    main()
