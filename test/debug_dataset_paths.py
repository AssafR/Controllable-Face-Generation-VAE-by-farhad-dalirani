import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Debug script to check dataset paths
"""

import os
import glob
import json

def debug_dataset_paths():
    """Debug what paths are being used for the dataset"""
    
    # Try relative path first (when running from test/ directory)
    config_path = "../config/config.json"
    if not os.path.exists(config_path):
        # If not found, try absolute path (when running from main directory)
        config_path = config_path
    
    with open(config_path, 'r') as file:
        config = json.load(f)
    print(f"Config dataset_dir: {config['dataset_dir']}")
    
    # Test different path combinations
    base_path = config["dataset_dir"]
    print(f"Base path: {base_path}")
    print(f"Base path exists: {os.path.exists(base_path)}")
    
    if os.path.exists(base_path):
        print(f"Contents of {base_path}:")
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            print(f"  {item} ({'dir' if os.path.isdir(item_path) else 'file'})")
    
    # Test the path we're using
    dataset_path = os.path.join(base_path, "img_align_celeba", "img_align_celeba")
    print(f"\nDataset path: {dataset_path}")
    print(f"Dataset path exists: {os.path.exists(dataset_path)}")
    
    if os.path.exists(dataset_path):
        print(f"Contents of {dataset_path}:")
        items = os.listdir(dataset_path)
        print(f"  Total items: {len(items)}")
        print(f"  First 10 items: {items[:10]}")
        
        # Check for .jpg files
        jpg_files = [f for f in items if f.endswith('.jpg')]
        print(f"  JPG files: {len(jpg_files)}")
        if jpg_files:
            print(f"  First 5 JPG files: {jpg_files[:5]}")
    
    # Test glob pattern
    glob_pattern = os.path.join(dataset_path, "*.jpg")
    print(f"\nGlob pattern: {glob_pattern}")
    glob_files = glob.glob(glob_pattern)
    print(f"Glob found {len(glob_files)} files")
    if glob_files:
        print(f"First 5 glob files: {[os.path.basename(f) for f in glob_files[:5]]}")

if __name__ == "__main__":
    debug_dataset_paths()
