# Unified VAE Training System

This document describes the new unified training system that consolidates all training methods into a single, configurable codebase.

## Overview

The unified system replaces multiple training scripts (`train_optimized_fast_high_quality.py`, `train_fast_high_quality.py`, `train_ultra_high_quality.py`, etc.) with two main scripts:

1. **`train_unified.py`** - Full-featured training script with all options
2. **`train.py`** - Simplified wrapper for common use cases

## Quick Start

### Simple Training (Recommended)

```bash
# Fast high quality training (most common)
uv run train.py --mode fast_high_quality

# Quick test (1 epoch, small model)
uv run train.py --mode quick_test

# Ultra high quality (best results, slowest)
uv run train.py --mode ultra_high_quality

# Fresh retrain with optimal settings
uv run train.py --mode full_retrain --no-resume
```

### Advanced Training

```bash
# Full control with unified script
uv run train_unified.py --loss-preset high_quality --training-preset fast_high_quality_training --model-preset fast_high_quality --dataset-preset full

# Resume from checkpoint with aggressive settings to fix blurry images
uv run train_unified.py --loss-preset high_quality --training-preset fast_high_quality_training --model-preset fast_high_quality --dataset-preset full
```

## Training Modes

### `train.py` Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `quick_test` | 1 epoch, small model, tiny dataset | Debugging, architecture testing |
| `fast_high_quality` | 100 epochs, 128x128, balanced settings | **Most users, good quality** |
| `ultra_high_quality` | 200 epochs, 256x256, best quality | Research, maximum quality |
| `balanced_high_quality` | 150 epochs, good quality-speed balance | Production use |
| `full_retrain` | 150 epochs, optimal fresh start settings | Fresh training from scratch |
| `resume_fix` | Aggressive settings to fix blurry images | Fixing existing training |

### Configuration Presets

The system uses the existing `config/config_unified.json` with these presets:

#### Loss Presets
- `mse_only` - Pure MSE loss (fastest, blurriest)
- `mse_l1` - MSE + L1 loss (good balance)
- `perceptual` - MSE + L1 + Perceptual (high quality)
- `high_quality` - Balanced high quality (recommended)
- `ultra_high_quality_loss` - All losses including LPIPS (best quality)
- `full_retrain_optimal` - Optimal for fresh training

#### Training Presets
- `quick_test` - 1 epoch, small batch
- `fast_training` - 20 epochs, large batch
- `standard_training` - 50 epochs, balanced
- `extended_training` - 100 epochs, careful settings
- `fast_high_quality_training` - 100 epochs, optimized
- `ultra_high_quality_training` - 200 epochs, best quality
- `full_retrain_training` - 150 epochs, fresh start

#### Model Presets
- `small` - 64x64, 256 latent, 64 channels
- `medium` - 64x64, 512 latent, 128 channels
- `fast_high_quality` - 128x128, 512 latent, 128 channels
- `ultra_high_quality` - 256x256, 1024 latent, 256 channels
- `balanced_high_quality` - 128x128, 768 latent, 160 channels

## Features

### Unified Codebase
- **Single source of truth** for all training logic
- **Consistent behavior** across all training modes
- **Easy maintenance** and bug fixes
- **No code duplication**

### Configuration Display
- **Clear parameter visibility** at training start
- **Resume configuration display** when loading checkpoints
- **Beta (KL weight) and loss weights** always shown

### Advanced Features
- **Automatic checkpointing** and resuming
- **TensorBoard logging** (if available)
- **Sample image generation** during training
- **Early stopping** with patience
- **GPU memory monitoring**
- **Time estimation** and progress tracking

### Loss Function Support
- **MSE Loss** - Pixel-level accuracy
- **L1 Loss** - Edge sharpness
- **Perceptual Loss** - Texture and detail preservation
- **LPIPS Loss** - Best perceptual quality (if available)
- **KL Divergence** - Latent space regularization

## Migration from Old Scripts

### Before (Multiple Scripts)
```bash
# Different scripts for different purposes
uv run train_optimized_fast_high_quality.py
uv run train_fast_high_quality.py
uv run train_ultra_high_quality.py
uv run train_VAE_unified.py
```

### After (Unified System)
```bash
# Single script with different modes
uv run train.py --mode fast_high_quality
uv run train.py --mode ultra_high_quality
uv run train_unified.py --loss-preset high_quality --training-preset ultra_high_quality_training
```

## Configuration Examples

### Fix Blurry Images (Resume Training)
```bash
# Use aggressive settings to fix posterior collapse
uv run train.py --mode resume_fix
```

### Fresh Training (Optimal Settings)
```bash
# Start completely fresh with optimal configuration
uv run train.py --mode full_retrain --no-resume
```

### Quick Testing
```bash
# Fast 1-epoch test with small model
uv run train.py --mode quick_test
```

### Production Training
```bash
# High quality with reasonable training time
uv run train.py --mode fast_high_quality
```

## Benefits

1. **Maintainability** - Single codebase instead of 9+ scripts
2. **Consistency** - Same features across all training modes
3. **Flexibility** - Easy to add new configurations
4. **Debugging** - Centralized logging and error handling
5. **Documentation** - Single source of truth for training logic

## Troubleshooting

### Blurry Images
- Use `--mode resume_fix` for aggressive settings
- Use `--mode full_retrain --no-resume` for fresh start
- Check that Beta (KL weight) is > 1.0

### Out of Memory
- Use smaller model presets (`small`, `medium`)
- Use smaller dataset presets (`tiny`, `small`)
- Reduce batch size in training presets

### Slow Training
- Use `fast_high_quality` mode for balanced quality/speed
- Use `quick_test` mode for rapid iteration
- Check GPU utilization and memory usage

## Future Enhancements

- [ ] Add more loss function combinations
- [ ] Support for different optimizers
- [ ] Automatic hyperparameter tuning
- [ ] Distributed training support
- [ ] Model architecture search
