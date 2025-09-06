#!/usr/bin/env python3
"""
Simplified VAE Training Script
Easy-to-use wrapper for the unified training system.
"""

import subprocess
import sys
import argparse

def main():
    """Main function with simplified argument parsing."""
    parser = argparse.ArgumentParser(description='Simplified VAE Training Script')
    parser.add_argument('--mode', type=str, default='fast_high_quality',
                       choices=['quick_test', 'fast_high_quality', 'ultra_high_quality', 
                               'balanced_high_quality', 'full_retrain', 'resume_fix'],
                       help='Training mode')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (e.g., 128, 256)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh training (delete old checkpoints and model files)')
    
    args = parser.parse_args()
    
    # Map modes to unified training arguments
    mode_mappings = {
        'quick_test': {
            'loss_preset': 'mse_l1',
            'training_preset': 'quick_test',
            'model_preset': 'small',
            'dataset_preset': 'tiny'
        },
        'fast_high_quality': {
            'loss_preset': 'high_quality',
            'training_preset': 'fast_high_quality_training',
            'model_preset': 'fast_high_quality',
            'dataset_preset': 'full'
        },
        'ultra_high_quality': {
            'loss_preset': 'ultra_high_quality_loss',
            'training_preset': 'ultra_high_quality_training',
            'model_preset': 'ultra_high_quality',
            'dataset_preset': 'full'
        },
        'balanced_high_quality': {
            'loss_preset': 'high_quality',
            'training_preset': 'balanced_high_quality_training',
            'model_preset': 'balanced_high_quality',
            'dataset_preset': 'full'
        },
        'full_retrain': {
            'loss_preset': 'full_retrain_optimal',
            'training_preset': 'full_retrain_training',
            'model_preset': 'fast_high_quality',
            'dataset_preset': 'full'
        },
        'resume_fix': {
            'loss_preset': 'high_quality',
            'training_preset': 'fast_high_quality_training',
            'model_preset': 'fast_high_quality',
            'dataset_preset': 'full'
        }
    }
    
    if args.mode not in mode_mappings:
        print(f"❌ Unknown mode: {args.mode}")
        print("Available modes: quick_test, fast_high_quality, ultra_high_quality, balanced_high_quality, full_retrain, resume_fix")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, 'train_unified.py']
    cmd.extend(['--loss-preset', mode_mappings[args.mode]['loss_preset']])
    cmd.extend(['--training-preset', mode_mappings[args.mode]['training_preset']])
    cmd.extend(['--model-preset', mode_mappings[args.mode]['model_preset']])
    cmd.extend(['--dataset-preset', mode_mappings[args.mode]['dataset_preset']])
    
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    
    if args.no_resume:
        cmd.append('--no-resume')
    
    # Print mode description
    mode_descriptions = {
        'quick_test': 'Quick test (1 epoch, small model, tiny dataset)',
        'fast_high_quality': 'Fast high quality (100 epochs, 128x128, balanced settings)',
        'ultra_high_quality': 'Ultra high quality (200 epochs, 256x256, best quality)',
        'balanced_high_quality': 'Balanced high quality (150 epochs, good quality-speed balance)',
        'full_retrain': 'Full retrain (150 epochs, optimal settings for fresh start)',
        'resume_fix': 'Resume fix (aggressive settings to fix blurry images)'
    }
    
    print(f"🚀 Starting VAE Training: {mode_descriptions[args.mode]}")
    if args.batch_size:
        print(f"📊 Batch Size Override: {args.batch_size}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    
    # Run the unified training script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
