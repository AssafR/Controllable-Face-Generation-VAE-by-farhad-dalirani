# GPU Troubleshooting Guide for Controllable Face Generation VAE

## Overview
This guide helps you resolve GPU-related issues when running the Controllable Face Generation VAE project. The project has been updated with proper GPU configuration, but you may still encounter issues depending on your system setup.

## Quick Fixes Applied

### 1. GPU Configuration Added
The following files have been updated with proper GPU configuration:

- `train_VAE.py` - Training script with GPU memory management
- `synthesis.py` - Image generation script with GPU support
- `gui.py` - Streamlit interface with GPU configuration

### 2. Key GPU Settings Implemented
- **Memory Growth**: Prevents TensorFlow from allocating all GPU memory at once
- **Device Detection**: Automatically detects available GPUs
- **Mixed Precision**: Enables FP16 for better GPU performance (training only)

## Testing GPU Detection

### Run the GPU Test Script
```bash
python test_gpu.py
```

This script will:
- Check TensorFlow version
- Detect available devices
- Test GPU computation
- Show environment variables
- Verify device placement

## Common Issues and Solutions

### Issue 1: "No GPU devices found"

**Symptoms:**
- Script shows "No GPU devices found. Running on CPU."
- Training/inference runs slowly

**Solutions:**
1. **Check CUDA Installation:**
   ```bash
   nvidia-smi
   ```
   If this fails, install NVIDIA drivers.

2. **Verify TensorFlow GPU Support:**
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

3. **Check TensorFlow Version:**
   - Ensure you have TensorFlow 2.20.0+ (as specified in pyproject.toml)
   - GPU support requires specific CUDA/cuDNN versions

### Issue 2: "CUDA out of memory"

**Symptoms:**
- RuntimeError: CUDA out of memory
- GPU memory allocation fails

**Solutions:**
1. **Reduce Batch Size:**
   Edit `config/config.json`:
   ```json
   {
       "batch_size": 64  // Reduce from 128
   }
   ```

2. **Enable Memory Growth (Already implemented):**
   The code now automatically enables memory growth.

3. **Monitor GPU Memory:**
   ```bash
   watch -n 1 nvidia-smi
   ```

### Issue 3: "TensorFlow not using GPU"

**Symptoms:**
- GPU detected but computations run on CPU
- No speed improvement

**Solutions:**
1. **Check Device Placement:**
   ```python
   import tensorflow as tf
   print("GPU available:", tf.config.list_physical_devices('GPU'))
   print("Current device:", tf.config.get_visible_devices())
   ```

2. **Force GPU Usage:**
   ```python
   with tf.device('/GPU:0'):
       # Your model operations here
   ```

3. **Verify Environment Variables:**
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   ```

## System Requirements

### NVIDIA GPU Requirements
- **Compute Capability**: 3.5 or higher
- **Memory**: 4GB+ recommended for training
- **Drivers**: Latest NVIDIA drivers

### Software Requirements
- **CUDA**: 11.8 or 12.0
- **cuDNN**: Compatible with your CUDA version
- **TensorFlow**: 2.20.0+ (GPU version)

## Installation Steps

### 1. Install NVIDIA Drivers
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535

# Windows: Download from NVIDIA website
```

### 2. Install CUDA Toolkit
```bash
# Download from NVIDIA website or use package manager
# Example for Ubuntu:
sudo apt install nvidia-cuda-toolkit
```

### 3. Install cuDNN
```bash
# Download from NVIDIA developer website
# Extract and copy to CUDA installation directory
```

### 4. Install TensorFlow GPU
```bash
# The project already specifies this in pyproject.toml
pip install tensorflow[gpu]
# or
pip install tensorflow-gpu
```

## Environment Variables

Set these environment variables for optimal GPU performance:

```bash
# Linux/macOS
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# Windows
set CUDA_VISIBLE_DEVICES=0
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_CPP_MIN_LOG_LEVEL=2
```

## Performance Optimization

### 1. Mixed Precision Training
The training script now enables mixed precision (FP16) for better GPU performance.

### 2. Memory Management
- GPU memory growth is enabled
- Batch size can be adjusted based on GPU memory

### 3. Data Pipeline
- Use `tf.data.Dataset` for efficient data loading
- Enable prefetching and caching

## Monitoring and Debugging

### 1. GPU Usage Monitoring
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Detailed GPU info
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

### 2. TensorFlow Debugging
```python
# Enable TensorFlow logging
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

# Check device placement
print(tf.config.get_visible_devices())
```

### 3. Memory Profiling
```python
# Check GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"GPU: {gpu.name}")
        print(f"Memory growth: {tf.config.experimental.get_memory_growth(gpu)}")
```

## Troubleshooting Checklist

- [ ] NVIDIA drivers installed and working (`nvidia-smi` works)
- [ ] CUDA toolkit installed and in PATH
- [ ] cuDNN installed and compatible
- [ ] TensorFlow GPU version installed
- [ ] GPU detected by TensorFlow (`tf.config.list_physical_devices('GPU')`)
- [ ] Environment variables set correctly
- [ ] GPU memory sufficient for batch size
- [ ] No conflicting TensorFlow installations

## Getting Help

If you're still experiencing issues:

1. **Run the test script**: `python test_gpu.py`
2. **Check system requirements** against the list above
3. **Review error messages** for specific failure points
4. **Check TensorFlow documentation** for your specific version
5. **Verify CUDA compatibility** with your TensorFlow version

## Additional Resources

- [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [TensorFlow Performance Guide](https://www.tensorflow.org/guide/performance)
- [GPU Memory Management](https://www.tensorflow.org/guide/gpu#memory_management)

