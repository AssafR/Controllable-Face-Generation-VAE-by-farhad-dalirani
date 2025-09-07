# Ultra High Quality VAE Training

## 🎯 **Training Configuration**

### **Model Architecture**
- **Resolution**: 256x256 pixels (4x higher than standard 64x64)
- **Latent Space**: 1024 dimensions (2x larger than standard 512)
- **Channels**: 256 (2x more than standard 128)
- **Total Parameters**: ~50M+ parameters

### **Loss Function (Ultra High Quality)**
- **MSE Loss**: 40% weight - pixel-level accuracy
- **L1 Loss**: 20% weight - edge sharpness
- **Perceptual Loss**: 20% weight - realistic textures and features
- **LPIPS Loss**: 20% weight - perceptual similarity (best quality)

### **Training Parameters**
- **Epochs**: 200 (extended training)
- **Batch Size**: 64 (smaller for better gradient estimates)
- **Learning Rate**: 0.00005 (very low for stable training)
- **Optimizer**: Adam with weight decay (1e-5)
- **Scheduler**: ReduceLROnPlateau (patience=10)
- **Gradient Clipping**: Max norm 1.0
- **Early Stopping**: Patience 20 epochs

### **Dataset**
- **Full CelebA**: ~200K images
- **Resolution**: 256x256 (upscaled from original)
- **Preprocessing**: Proper normalization, no double normalization

## 🚀 **Expected Results**

### **Quality Improvements**
- **Sharpness**: Much sharper than 64x64 models
- **Detail**: Fine facial features and textures
- **Realism**: Better perceptual similarity to real images
- **Consistency**: More stable generation across different latent vectors

### **Training Time**
- **Estimated Duration**: 8-12 hours on RTX 3090
- **Memory Usage**: ~8-12 GB VRAM
- **Checkpoints**: Saved every epoch with best model tracking

## 📊 **Monitoring**

### **Real-time Metrics**
- **Loss Components**: MSE, KL, L1, Perceptual, LPIPS
- **Learning Rate**: Adaptive scheduling
- **Validation**: Separate validation loss tracking
- **Samples**: Generated images every 10 epochs

### **KL Scheduling and Stability**

We use adaptive KL control to maintain the KL contribution within a target band and prevent early dominance that can cause posterior collapse.

#### Cyclical KL Annealing (Bowman et al., 2016)

We also support optional cyclical KL annealing to periodically allow reconstruction losses (MSE/L1/perceptual) to recover while maintaining a gentle upward trend in KL pressure.

- Reference: Bowman, Samuel R., Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, and Samy Bengio. "Generating Sentences from a Continuous Space." Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP) Workshop on Representation Learning for NLP (CoNLL 2016).
- Implementation notes:
  - A short cooldown at the start of each cycle reduces beta.
  - A slowly rising beta floor avoids collapsing back to zero pressure.
  - Adaptive control remains active, using smoothened KL contribution.

Configuration keys (in `training_presets`):

```
"kl_cycle_enabled": true,
"kl_cycle_period": 8,
"kl_cooldown_epochs": 2,
"kl_cooldown_reduction": 0.25,
"kl_beta_floor": 0.00005,
"kl_floor_growth": 1.02
```

### **TensorBoard Logging**
- **Location**: `runs/ultra_high_quality_YYYYMMDD_HHMMSS/`
- **View**: `tensorboard --logdir runs/ultra_high_quality_*`
- **Metrics**: All loss components, learning rate, sample images

### **Output Files**
- **Model**: `model_weights/vae_ultra_high_quality.pth`
- **Samples**: `ultra_high_quality_samples_epoch_*.png`
- **Final**: `ultra_high_quality_final_samples.png`

## 🔧 **Usage**

### **Start Training**
```bash
uv run train_ultra_high_quality.py
```

### **Monitor Progress**
```bash
uv run test/monitor_ultra_high_quality_training.py
```

### **View TensorBoard**
```bash
uv run tensorboard --logdir runs/ultra_high_quality_*
```

## ⚠️ **Requirements**

### **Hardware**
- **GPU**: RTX 3090 or better (24GB VRAM recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 10GB+ free space

### **Software**
- **PyTorch**: 2.0+ with CUDA support
- **Dependencies**: LPIPS, OpenCV, TensorBoard
- **Python**: 3.8+

## 🎨 **Quality Comparison**

| Aspect | Standard (64x64) | Ultra High Quality (256x256) |
|--------|------------------|------------------------------|
| Resolution | 64x64 | 256x256 |
| Latent Space | 512 | 1024 |
| Loss Function | MSE only | MSE + L1 + Perceptual + LPIPS |
| Training Time | 2-4 hours | 8-12 hours |
| Image Quality | Blurry, low detail | Sharp, high detail |
| Perceptual Quality | Poor | Excellent |

## 🔍 **Troubleshooting**

### **Common Issues**
1. **Out of Memory**: Reduce batch size to 32 or 16
2. **Slow Training**: Check GPU utilization, ensure CUDA is working
3. **Poor Quality**: Verify dataset preprocessing, check loss weights
4. **Training Stalls**: Monitor learning rate, check gradient norms

### **Monitoring Commands**
```bash
# Check GPU usage
nvidia-smi

# Monitor training logs
tail -f training.log

# Check model checkpoints
ls -la model_weights/

# View latest samples
ls -la ultra_high_quality_samples_epoch_*.png
```

## 📈 **Expected Timeline**

- **Epochs 1-20**: Rapid initial improvement
- **Epochs 21-50**: Steady quality gains
- **Epochs 51-100**: Fine-tuning and refinement
- **Epochs 101-200**: Marginal improvements, potential early stopping

## 🎯 **Success Metrics**

### **Quantitative**
- **PSNR**: >25 dB (vs ~16 dB for standard)
- **LPIPS**: <0.3 (lower is better)
- **MSE**: <0.01 (pixel-level accuracy)

### **Qualitative**
- **Sharpness**: Clear facial features
- **Realism**: Photorealistic appearance
- **Consistency**: Stable generation across samples
- **Diversity**: Variety in generated faces

---

**Note**: This is a research-grade training configuration designed for maximum quality. The training time and resource requirements are significantly higher than standard configurations, but the results should be substantially better.
