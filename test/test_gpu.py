import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3
"""
Simple GPU detection and TensorFlow configuration test script.
Run this to check if your GPU is properly detected and configured.
"""

import tensorflow as tf
import os

def test_gpu():
    """Test GPU detection and configuration"""
    print("=" * 60)
    print("GPU DETECTION AND TENSORFLOW CONFIGURATION TEST")
    print("=" * 60)
    
    # Print TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras backend: {tf.keras.backend.backend()}")
    
    # Check available devices
    print("\n--- Available Physical Devices ---")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"  {device.device_type}: {device.name}")
    
    # Check GPU devices specifically
    print("\n--- GPU Devices ---")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU device(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            
        # Try to configure GPU memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled successfully")
        except RuntimeError as e:
            print(f"✗ Error enabling GPU memory growth: {e}")
            
        # Test GPU computation
        print("\n--- Testing GPU Computation ---")
        try:
            with tf.device('/GPU:0'):
                # Create a simple tensor and perform computation
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"Matrix multiplication result: {c}")
                print("✓ GPU computation successful")
        except RuntimeError as e:
            print(f"✗ GPU computation failed: {e}")
            
    else:
        print("No GPU devices found!")
        print("\n--- Possible Issues ---")
        print("1. CUDA drivers not installed")
        print("2. TensorFlow not compiled with GPU support")
        print("3. GPU not compatible with current TensorFlow version")
        print("4. Environment variables not set correctly")
    
    # Check environment variables
    print("\n--- Environment Variables ---")
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # Check if running on CPU or GPU
    print("\n--- Current Device ---")
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow is configured to use GPU")
        # Test which device is actually being used
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0, 3.0])
            print(f"Test tensor device: {test_tensor.device}")
    else:
        print("TensorFlow is running on CPU only")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_gpu()

