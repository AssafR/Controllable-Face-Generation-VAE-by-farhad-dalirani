#!/usr/bin/env python3
"""
Migration script to move from separate config files to unified config.
"""

import os
import shutil
from pathlib import Path

def migrate_configs():
    """Migrate from separate config files to unified config."""
    
    print("🔄 Migrating to Unified Configuration System")
    print("=" * 50)
    
    # List of old config files to archive
    old_configs = [
        "config/config_mse_only.json",
        "config/config_mse_l1.json", 
        "config/config_perceptual.json",
        "config/config_lpips.json"
    ]
    
    # Create archive directory
    archive_dir = "config/archive"
    os.makedirs(archive_dir, exist_ok=True)
    
    print("📁 Archiving old configuration files...")
    
    for config_file in old_configs:
        if os.path.exists(config_file):
            # Move to archive
            filename = os.path.basename(config_file)
            archive_path = os.path.join(archive_dir, filename)
            shutil.move(config_file, archive_path)
            print(f"  ✅ Moved {config_file} → {archive_path}")
        else:
            print(f"  ⚠️  {config_file} not found, skipping")
    
    print(f"\n📋 Old configs archived in: {archive_dir}")
    print("✅ Migration complete!")
    
    print("\n🎯 New Usage Examples:")
    print("=" * 30)
    print("# List all available presets")
    print("uv run train_VAE_unified.py --list-presets")
    print()
    print("# Train with specific presets")
    print("uv run train_VAE_unified.py --loss mse_l1 --training standard_training --model medium")
    print()
    print("# Quick test")
    print("uv run train_VAE_unified.py --loss mse_only --training quick_test --model small")
    print()
    print("# High quality training")
    print("uv run train_VAE_unified.py --loss lpips --training extended_training --model large")

if __name__ == "__main__":
    migrate_configs()
