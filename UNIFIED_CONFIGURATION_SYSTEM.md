# Unified VAE Configuration System

## **Overview**

The VAE now uses a **unified configuration system** that combines all settings into a single, well-documented JSON file. This eliminates the need for multiple separate config files and provides much better management and flexibility.

## **Why Unified Configuration?**

### **Problems with Separate Files:**
- ❌ **Scattered settings**: Multiple JSON files to maintain
- ❌ **Inconsistent structure**: Each file had different formats
- ❌ **Hard to compare**: Difficult to see differences between configs
- ❌ **Poor documentation**: Limited context and explanations
- ❌ **Maintenance burden**: Changes required updating multiple files

### **Benefits of Unified System:**
- ✅ **Single source of truth**: All settings in one place
- ✅ **Preset system**: Easy combinations of different aspects
- ✅ **Rich documentation**: Built-in descriptions and use cases
- ✅ **Easy comparison**: Side-by-side preset comparison
- ✅ **Flexible combinations**: Mix and match any presets
- ✅ **Better maintainability**: Update once, use everywhere

## **Configuration Structure**

The unified config file (`config/config_unified.json`) contains:

### **1. Base Configuration**
```json
"base_config": {
    "embedding_size": 512,
    "num_channels": 128,
    "beta": 1.0,
    "input_img_size": 64,
    "batch_size": 128,
    "lr": 0.0001,
    "max_epoch": 100,
    "dataset_dir": "celeba-dataset",
    "model_save_path": "model_weights/"
}
```

### **2. Loss Function Presets**
- **`mse_only`**: Pure MSE (fastest, blurriest)
- **`mse_l1`**: MSE + L1 (balanced, recommended)
- **`perceptual`**: MSE + L1 + Perceptual (high quality)
- **`lpips`**: MSE + L1 + LPIPS (best quality)
- **`custom_balanced`**: Custom balanced weights
- **`high_quality`**: Best balance without LPIPS

### **3. Training Presets**
- **`quick_test`**: 1 epoch, small batch (debugging)
- **`fast_training`**: 20 epochs, larger batch (quick results)
- **`standard_training`**: 50 epochs, balanced (normal use)
- **`extended_training`**: 100 epochs, careful (best quality)

### **4. Model Presets**
- **`small`**: 256D latent, 64 channels (fast inference)
- **`medium`**: 512D latent, 128 channels (balanced)
- **`large`**: 1024D latent, 256 channels (best quality)
- **`high_res`**: 512D latent, 128 channels, 128x128 images

### **5. Dataset Presets**
- **`tiny`**: 1,000 samples (debugging, architecture testing)
- **`small`**: 10,000 samples (development, quick experiments)
- **`medium`**: 50,000 samples (balanced testing)
- **`large`**: 100,000 samples (good quality training)
- **`full`**: All available samples (production training)

## **Usage Examples**

### **1. List All Available Presets**
```bash
uv run train_VAE_unified.py --list-presets
```

### **2. Train with Specific Presets**
```bash
# Quick debugging (tiny dataset, 1 epoch)
uv run train_VAE_unified.py --loss mse_only --training quick_test --model small --dataset tiny

# Development testing (small dataset, fast training)
uv run train_VAE_unified.py --loss mse_l1 --training fast_training --model medium --dataset small

# Recommended for most users (full dataset, standard training)
uv run train_VAE_unified.py --loss mse_l1 --training standard_training --model medium --dataset full

# High quality generation (full dataset, extended training)
uv run train_VAE_unified.py --loss perceptual --training extended_training --model large --dataset full

# Best possible quality (full dataset, all features)
uv run train_VAE_unified.py --loss lpips --training extended_training --model large --dataset full
```

### **3. Programmatic Usage**
```python
from config_loader import ConfigLoader

# Initialize loader
loader = ConfigLoader()

# Get specific configuration
config = loader.get_config(
    loss_preset="mse_l1",
    training_preset="standard_training",
    model_preset="medium",
    dataset_preset="full"
)

# Use config for training
train_variational_autoencoder_unified(config, device)
```

## **Configuration Combinations**

### **Quick Testing**
```bash
uv run train_VAE_unified.py --loss mse_only --training quick_test --model small --dataset tiny
```
- **Use case**: Debugging, quick validation, architecture testing
- **Time**: ~1 minute
- **Quality**: Baseline (blurry)
- **Dataset**: 1,000 samples

### **Fast Results**
```bash
uv run train_VAE_unified.py --loss mse_l1 --training fast_training --model medium --dataset small
```
- **Use case**: Quick results, limited time, development
- **Time**: ~10 minutes
- **Quality**: Good (sharper than MSE only)
- **Dataset**: 10,000 samples

### **Standard Training**
```bash
uv run train_VAE_unified.py --loss mse_l1 --training standard_training --model medium --dataset full
```
- **Use case**: Normal training, good results
- **Time**: ~2 hours
- **Quality**: Very good (recommended)
- **Dataset**: Full dataset (~200k samples)

### **High Quality**
```bash
uv run train_VAE_unified.py --loss perceptual --training extended_training --model large --dataset full
```
- **Use case**: High quality generation
- **Time**: ~4 hours
- **Quality**: Excellent (realistic textures)
- **Dataset**: Full dataset (~200k samples)

### **Best Quality**
```bash
uv run train_VAE_unified.py --loss lpips --training extended_training --model large --dataset full
```
- **Use case**: Research, best possible quality
- **Time**: ~6 hours
- **Quality**: Outstanding (most realistic)
- **Dataset**: Full dataset (~200k samples)

## **Dataset Subset Benefits**

### **Why Use Dataset Subsets?**

#### **Development Benefits:**
- ⚡ **Faster iteration**: Test changes quickly with tiny datasets
- 🔧 **Debugging**: Isolate issues without waiting for full training
- 💾 **Resource efficiency**: Use less memory and storage
- 🧪 **Experimentation**: Try different architectures rapidly

#### **Training Time Comparison:**
| Dataset Size | Samples | Training Time | Use Case |
|--------------|---------|---------------|----------|
| `tiny` | 1,000 | ~1 minute | Debugging, architecture testing |
| `small` | 10,000 | ~10 minutes | Development, quick experiments |
| `medium` | 50,000 | ~30 minutes | Balanced testing |
| `large` | 100,000 | ~1 hour | Good quality validation |
| `full` | ~200,000 | ~2-6 hours | Production training |

#### **Memory Usage:**
- **Tiny**: ~50MB RAM, ~100MB disk
- **Small**: ~200MB RAM, ~500MB disk  
- **Medium**: ~800MB RAM, ~2GB disk
- **Large**: ~1.5GB RAM, ~4GB disk
- **Full**: ~3GB RAM, ~8GB disk

### **When to Use Each Subset:**

#### **`tiny` (1,000 samples)**
- ✅ Debugging code changes
- ✅ Testing new architectures
- ✅ Validating data loading
- ✅ Quick proof-of-concept

#### **`small` (10,000 samples)**
- ✅ Development and experimentation
- ✅ Testing different loss functions
- ✅ Hyperparameter tuning
- ✅ Limited compute resources

#### **`medium` (50,000 samples)**
- ✅ Balanced quality vs speed
- ✅ Testing model improvements
- ✅ Validation before full training
- ✅ Moderate quality requirements

#### **`large` (100,000 samples)**
- ✅ Good quality without full dataset
- ✅ Production-like testing
- ✅ Limited time but need quality
- ✅ Resource constraints

#### **`full` (All samples)**
- ✅ Production training
- ✅ Best possible quality
- ✅ Research applications
- ✅ Final model deployment

## **Migration from Old System**

### **Automatic Migration**
```bash
# Archive old config files
uv run migrate_to_unified_config.py
```

### **Manual Migration**
Old separate files are moved to `config/archive/`:
- `config_mse_only.json` → `config/archive/config_mse_only.json`
- `config_mse_l1.json` → `config/archive/config_mse_l1.json`
- `config_perceptual.json` → `config/archive/config_perceptual.json`
- `config_lpips.json` → `config/archive/config_lpips.json`

## **Advanced Usage**

### **Custom Overrides**
```python
# Get base config
config = loader.get_config(
    loss_preset="mse_l1",
    training_preset="standard_training",
    model_preset="medium"
)

# Override specific values
config["batch_size"] = 256
config["lr"] = 0.0005
config["max_epoch"] = 30

# Train with custom settings
train_variational_autoencoder_unified(config, device)
```

### **Creating New Presets**
Add new presets to `config/config_unified.json`:

```json
"loss_presets": {
    "my_custom_loss": {
        "_description": "My custom loss configuration",
        "_use_case": "Specific requirements",
        "use_mse": true,
        "use_l1": true,
        "use_perceptual_loss": false,
        "use_lpips": false,
        "mse_weight": 0.6,
        "l1_weight": 0.4,
        "perceptual_weight": 0.0,
        "lpips_weight": 0.0
    }
}
```

## **Configuration Validation**

The system automatically validates configurations:
- ✅ **Required fields**: All necessary parameters present
- ✅ **Loss weights**: Sum to reasonable values
- ✅ **Preset existence**: All referenced presets exist
- ✅ **Type checking**: Correct data types for all parameters

## **Performance Comparison**

| Configuration | Training Time | Image Quality | Memory Usage | Recommended For |
|---------------|---------------|---------------|--------------|-----------------|
| `mse_only` + `quick_test` + `small` | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Debugging |
| `mse_l1` + `fast_training` + `medium` | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Quick results |
| `mse_l1` + `standard_training` + `medium` | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **Most users** |
| `perceptual` + `extended_training` + `large` | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | High quality |
| `lpips` + `extended_training` + `large` | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Research |

## **Troubleshooting**

### **Common Issues**

#### **"Preset not found"**
```bash
# List available presets
uv run train_VAE_unified.py --list-presets

# Check preset names are correct
uv run train_VAE_unified.py --loss mse_l1 --training standard_training --model medium
```

#### **"Config file not found"**
```bash
# Check config file exists
ls config/config_unified.json

# Use absolute path if needed
uv run train_VAE_unified.py --config /full/path/to/config_unified.json
```

#### **"Out of memory"**
```bash
# Use smaller model
uv run train_VAE_unified.py --loss mse_l1 --training standard_training --model small

# Reduce batch size in config
```

### **Getting Help**
```bash
# Show help
uv run train_VAE_unified.py --help

# List all presets
uv run train_VAE_unified.py --list-presets

# Demo the system
uv run demo_unified_config.py
```

## **Benefits Summary**

### **For Users:**
- 🎯 **Easy selection**: Choose from predefined combinations
- 📚 **Clear documentation**: Know what each preset does
- ⚡ **Quick setup**: No need to understand all parameters
- 🔄 **Easy switching**: Try different combinations easily

### **For Developers:**
- 🛠️ **Single maintenance**: Update one file, affects all
- 📊 **Easy comparison**: Side-by-side preset analysis
- 🔧 **Flexible system**: Easy to add new presets
- 🧪 **Better testing**: Consistent configuration across tests

### **For Research:**
- 📈 **Reproducible**: Exact configurations saved
- 🔬 **Comparable**: Easy to compare different approaches
- 📝 **Documented**: Clear rationale for each choice
- 🎛️ **Configurable**: Easy to experiment with new combinations

The unified configuration system makes VAE training much more manageable and user-friendly! 🚀
