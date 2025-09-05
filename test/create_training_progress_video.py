#!/usr/bin/env python3
"""
Create a training progress video from sample images.
Shows the evolution of generated and reconstructed images over training epochs.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_training_progress_video():
    """Create a video showing training progress."""
    
    print("🎬 Creating Training Progress Video")
    print("=" * 50)
    
    sample_dir = "sample_images"
    
    if not os.path.exists(sample_dir):
        print(f"❌ Sample images directory not found: {sample_dir}")
        print("Make sure training is running and generating samples...")
        return
    
    # Find all generated images (clean filename format)
    generated_files = sorted(glob.glob(os.path.join(sample_dir, "fast_high_quality_generated_epoch_*.png")))
    reconstruction_files = sorted(glob.glob(os.path.join(sample_dir, "fast_high_quality_reconstruction_epoch_*.png")))
    
    print(f"📁 Found {len(generated_files)} generated image files")
    print(f"📁 Found {len(reconstruction_files)} reconstruction image files")
    
    if len(generated_files) == 0:
        print("❌ No generated images found. Training may not have started yet.")
        return
    
    # Create video from generated images
    if len(generated_files) > 1:
        create_video_from_images(generated_files, "training_progress_generated.mp4", "Generated Images Progress")
    
    # Create video from reconstruction images
    if len(reconstruction_files) > 1:
        create_video_from_images(reconstruction_files, "training_progress_reconstruction.mp4", "Reconstruction Progress")
    
    # Create a combined progress summary
    create_progress_summary(generated_files, reconstruction_files)
    
    print(f"\n✅ Training progress videos created!")
    print(f"  • Generated images: training_progress_generated.mp4")
    print(f"  • Reconstruction: training_progress_reconstruction.mp4")
    print(f"  • Summary: training_progress_summary.png")

def create_video_from_images(image_files, output_path, title):
    """Create a video from a list of image files."""
    
    print(f"\n🎥 Creating video: {output_path}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, channels = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 2  # 2 frames per second
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each image to video
    for i, image_file in enumerate(image_files):
        # Read image
        image = cv2.imread(image_file)
        
        # Add epoch number overlay
        epoch_num = os.path.basename(image_file).split('_')[2].split('.')[0]
        cv2.putText(image, f"Epoch {epoch_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        video_writer.write(image)
        
        # Duplicate frame for longer display
        video_writer.write(image)
    
    video_writer.release()
    print(f"  ✅ Video saved: {output_path}")

def create_progress_summary(generated_files, reconstruction_files):
    """Create a summary image showing progress over time."""
    
    print(f"\n📊 Creating progress summary...")
    
    # Select key epochs (every 10th epoch)
    key_epochs = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    
    # Filter to available epochs
    available_generated = [f for i, f in enumerate(generated_files) if i in key_epochs]
    available_reconstruction = [f for i, f in enumerate(reconstruction_files) if i in key_epochs]
    
    if len(available_generated) == 0:
        print("  ⚠️  No generated images available for summary")
        return
    
    # Create summary figure
    fig, axes = plt.subplots(2, min(len(available_generated), 6), figsize=(20, 8))
    if len(available_generated) == 1:
        axes = axes.reshape(2, 1)
    
    # Plot generated images
    for i, image_file in enumerate(available_generated[:6]):
        if i < axes.shape[1]:
            image = Image.open(image_file)
            axes[0, i].imshow(image)
            epoch_num = os.path.basename(image_file).split('_')[2].split('.')[0]
            axes[0, i].set_title(f"Generated - Epoch {epoch_num}")
            axes[0, i].axis('off')
    
    # Plot reconstruction images
    for i, image_file in enumerate(available_reconstruction[:6]):
        if i < axes.shape[1]:
            image = Image.open(image_file)
            axes[1, i].imshow(image)
            epoch_num = os.path.basename(image_file).split('_')[2].split('.')[0]
            axes[1, i].set_title(f"Reconstruction - Epoch {epoch_num}")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("training_progress_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Summary saved: training_progress_summary.png")

def monitor_sample_directory():
    """Monitor the sample_images directory for new files."""
    
    sample_dir = "sample_images"
    
    if not os.path.exists(sample_dir):
        print(f"📁 Creating sample_images directory...")
        os.makedirs(sample_dir, exist_ok=True)
        return
    
    # Count files
    generated_files = glob.glob(os.path.join(sample_dir, "fast_high_quality_generated_epoch_*.png"))
    reconstruction_files = glob.glob(os.path.join(sample_dir, "fast_high_quality_reconstruction_epoch_*.png"))
    
    print(f"📊 Sample Images Directory Status:")
    print(f"  • Generated images: {len(generated_files)}")
    print(f"  • Reconstruction images: {len(reconstruction_files)}")
    
    if len(generated_files) > 0:
        latest_generated = max(generated_files, key=os.path.getctime)
        latest_reconstruction = max(reconstruction_files, key=os.path.getctime) if reconstruction_files else None
        
        print(f"  • Latest generated: {os.path.basename(latest_generated)}")
        if latest_reconstruction:
            print(f"  • Latest reconstruction: {os.path.basename(latest_reconstruction)}")
        
        # Show file sizes
        gen_size = os.path.getsize(latest_generated) / 1024 / 1024
        print(f"  • Generated file size: {gen_size:.1f} MB")
        
        if latest_reconstruction:
            recon_size = os.path.getsize(latest_reconstruction) / 1024 / 1024
            print(f"  • Reconstruction file size: {recon_size:.1f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create training progress videos")
    parser.add_argument("--monitor", action="store_true", help="Monitor sample directory")
    parser.add_argument("--video", action="store_true", help="Create progress videos")
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_sample_directory()
    elif args.video:
        create_training_progress_video()
    else:
        print("🎬 Training Progress Video Creator")
        print("=" * 40)
        print("Usage:")
        print("  --monitor  : Check sample directory status")
        print("  --video    : Create progress videos")
        print("\nExample:")
        print("  uv run test/create_training_progress_video.py --monitor")
        print("  uv run test/create_training_progress_video.py --video")
