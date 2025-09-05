# Configurable Loss Functions for VAE

## **Overview**

The VAE now supports configurable loss functions through the configuration file, allowing you to experiment with different combinations to reduce blurriness and improve image quality.

## **Available Loss Functions**

### 1. **MSE Loss** (Mean Squared Error)
- **Purpose**: Smooth reconstructions
- **Pros**: Stable training, good for overall structure
- **Cons**: Tends to produce blurry images
- **Weight**: Typically 0.6-1.0

### 2. **L1 Loss** (Mean Absolute Error)
- **Purpose**: Sharp edges and details
- **Pros**: Preserves fine details, reduces blurriness
- **Cons**: Can be less stable than MSE
- **Weight**: Typically 0.2-0.4

### 3. **Perceptual Loss** (VGG-based)
- **Purpose**: Realistic textures and high-level features
- **Pros**: More perceptually accurate, better textures
- **Cons**: Computationally expensive, requires pre-trained VGG
- **Weight**: Typically 0.1-0.3

### 4. **LPIPS Loss** (Learned Perceptual Image Patch Similarity)
- **Purpose**: Most accurate perceptual similarity
- **Pros**: Best perceptual quality, learned similarity
- **Cons**: Most expensive, requires additional dependency
- **Weight**: Typically 0.1-0.3

## **Configuration Examples**

### **MSE Only** (Original, Blurry)
```json
{
    "loss_config": {
        "use_mse": true,
        "use_l1": false,
        "use_perceptual_loss": false,
        "use_lpips": false,
        "mse_weight": 1.0,
        "l1_weight": 0.0,
        "perceptual_weight": 0.0,
        "lpips_weight": 0.0
    }
}
```

### **MSE + L1** (Balanced, Less Blurry)
```json
{
    "loss_config": {
        "use_mse": true,
        "use_l1": true,
        "use_perceptual_loss": false,
        "use_lpips": false,
        "mse_weight": 0.8,
        "l1_weight": 0.2,
        "perceptual_weight": 0.0,
        "lpips_weight": 0.0
    }
}
```

### **MSE + L1 + Perceptual** (High Quality)
```json
{
    "loss_config": {
        "use_mse": true,
        "use_l1": true,
        "use_perceptual_loss": true,
        "use_lpips": false,
        "mse_weight": 0.6,
        "l1_weight": 0.2,
        "perceptual_weight": 0.2,
        "lpips_weight": 0.0
    }
}
```

### **MSE + L1 + LPIPS** (Best Quality)
```json
{
    "loss_config": {
        "use_mse": true,
        "use_l1": true,
        "use_perceptual_loss": false,
        "use_lpips": true,
        "mse_weight": 0.6,
        "l1_weight": 0.2,
        "perceptual_weight": 0.0,
        "lpips_weight": 0.2
    }
}
```

## **Usage**

### **1. Train with Specific Configuration**
```bash
# Train with MSE + L1
uv run train_VAE_configurable.py --config config/config_mse_l1.json

# Train with Perceptual loss
uv run train_VAE_configurable.py --config config/config_perceptual.json

# Train with LPIPS
uv run train_VAE_configurable.py --config config/config_lpips.json
```

### **2. Compare All Configurations**
```bash
# First train all configurations
uv run train_VAE_configurable.py --config config/config_mse_only.json
uv run train_VAE_configurable.py --config config/config_mse_l1.json
uv run train_VAE_configurable.py --config config/config_perceptual.json
uv run train_VAE_configurable.py --config config/config_lpips.json

# Then compare them
uv run compare_loss_functions.py
```

### **3. Create Custom Configuration**
```bash
# Copy an existing config
cp config/config_mse_l1.json config/config_custom.json

# Edit the loss_config section
# Then train
uv run train_VAE_configurable.py --config config/config_custom.json
```

## **Configuration Parameters**

### **Loss Function Switches**
- `use_mse`: Enable/disable MSE loss
- `use_l1`: Enable/disable L1 loss  
- `use_perceptual_loss`: Enable/disable VGG perceptual loss
- `use_lpips`: Enable/disable LPIPS loss

### **Loss Weights**
- `mse_weight`: Weight for MSE loss (0.0-1.0)
- `l1_weight`: Weight for L1 loss (0.0-1.0)
- `perceptual_weight`: Weight for perceptual loss (0.0-1.0)
- `lpips_weight`: Weight for LPIPS loss (0.0-1.0)

### **Important Notes**
- **Weights should sum to 1.0** for balanced training
- **At least one loss function must be enabled**
- **Perceptual and LPIPS are computationally expensive**
- **LPIPS requires additional installation**: `uv add lpips`

## **Expected Results**

### **MSE Only**
- ✅ Stable training
- ❌ Very blurry images
- ❌ Poor detail preservation

### **MSE + L1**
- ✅ Sharper images
- ✅ Better edge preservation
- ✅ Good balance of quality and speed

### **MSE + L1 + Perceptual**
- ✅ Realistic textures
- ✅ Better high-level features
- ⚠️ Slower training

### **MSE + L1 + LPIPS**
- ✅ Best perceptual quality
- ✅ Most realistic images
- ⚠️ Slowest training

## **Recommendations**

### **For Quick Testing**
Use `config_mse_l1.json` - good balance of quality and speed

### **For Best Quality**
Use `config_lpips.json` - best perceptual quality

### **For Fast Training**
Use `config_mse_only.json` - fastest but blurriest

### **For Custom Needs**
Create your own config with specific weight combinations

## **Troubleshooting**

### **Training Instability**
- Reduce learning rate
- Increase MSE weight
- Reduce L1 weight

### **Too Blurry**
- Increase L1 weight
- Add perceptual loss
- Reduce MSE weight

### **Too Noisy**
- Increase MSE weight
- Reduce L1 weight
- Add regularization

### **Out of Memory**
- Reduce batch size
- Disable perceptual/LPIPS loss
- Use smaller embedding size

## **Performance Comparison**

| Configuration | Training Speed | Image Quality | Sharpness | Stability |
|---------------|----------------|---------------|-----------|-----------|
| MSE Only | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| MSE + L1 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| + Perceptual | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| + LPIPS | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

Choose the configuration that best fits your needs! 🎯
