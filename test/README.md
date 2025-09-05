# Test Directory

This directory contains all test, demo, debug, and comparison scripts for the VAE project.

## **Script Categories**

### **🧪 Test Scripts**
- **`test_unified_config.py`** - Test the unified configuration system
- **`test_dataset.py`** - Test dataset loading and preprocessing
- **`test_gpu.py`** - Test GPU availability and configuration
- **`test_model_architecture.py`** - Test model architecture and forward passes
- **`test_synthesis_comprehensive.py`** - Comprehensive synthesis testing

### **🎯 Demo Scripts**
- **`demo_unified_config.py`** - Demonstrate the unified configuration system
- **`demo_dataset_subsets.py`** - Demonstrate dataset subset functionality

### **🔍 Debug Scripts**
- **`debug_dataset_loading.py`** - Debug dataset loading issues
- **`debug_dataset_paths.py`** - Debug dataset path resolution
- **`debug_image_loading.py`** - Debug image loading and preprocessing
- **`debug_synthesis.py`** - Debug synthesis functionality
- **`debug_transform_pipeline.py`** - Debug data transformation pipeline

### **📊 Comparison Scripts**
- **`compare_loss_functions.py`** - Compare different loss function configurations
- **`compare_vae_quality.py`** - Compare VAE model quality

### **🛠️ Utility Scripts**
- **`update_imports.py`** - Update import paths when moving files to test directory

## **Usage**

### **Running Tests from Main Directory**
```bash
# Run individual test files
uv run test/test_unified_config.py
uv run test/demo_dataset_subsets.py
uv run test/compare_loss_functions.py

# Run debug scripts
uv run test/debug_dataset_loading.py
uv run test/debug_synthesis.py
```

### **Running Tests from Test Directory**
```bash
cd test
uv run test_unified_config.py
uv run demo_unified_config.py
uv run compare_loss_functions.py
```

## **Script Descriptions**

### **Configuration Testing**
- **`test_unified_config.py`**: Tests the unified configuration system with different preset combinations
- **`demo_unified_config.py`**: Demonstrates all available presets and their descriptions
- **`demo_dataset_subsets.py`**: Shows dataset subset functionality with timing and memory usage

### **Model Testing**
- **`test_model_architecture.py`**: Tests model creation and forward passes
- **`test_gpu.py`**: Tests GPU availability and configuration
- **`test_synthesis_comprehensive.py`**: Comprehensive testing of synthesis functions

### **Debugging**
- **`debug_dataset_loading.py`**: Debug dataset loading and path issues
- **`debug_image_loading.py`**: Debug image loading and preprocessing
- **`debug_synthesis.py`**: Debug synthesis and generation issues
- **`debug_transform_pipeline.py`**: Debug data transformation pipeline

### **Comparison and Analysis**
- **`compare_loss_functions.py`**: Compare different loss function configurations
- **`compare_vae_quality.py`**: Compare VAE model quality and performance

## **Adding New Test Scripts**

When adding new test scripts to this directory:

1. **Place the script** in the `test/` directory
2. **Run the import updater**:
   ```bash
   cd test
   uv run update_imports.py
   ```
3. **Update config paths** if using `ConfigLoader`:
   ```python
   loader = ConfigLoader("../config/config_unified.json")
   ```
4. **Test the script** to ensure it works from the test directory

## **Import Path Handling**

All test scripts have been updated to work from the `test/` directory by:

1. **Adding parent directory to Python path**:
   ```python
   import sys
   import os
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```

2. **Using relative paths for config files**:
   ```python
   loader = ConfigLoader("../config/config_unified.json")
   ```

## **File Organization**

```
test/
├── README.md                    # This file
├── update_imports.py           # Utility to update import paths
├── test_*.py                   # Test scripts
├── demo_*.py                   # Demo scripts
├── debug_*.py                  # Debug scripts
├── compare_*.py                # Comparison scripts
└── monitor_*.py                # Monitoring scripts
```

## **Best Practices**

1. **Use descriptive names** that indicate the script's purpose
2. **Add docstrings** explaining what each script does
3. **Include usage examples** in the script comments
4. **Test from both directories** (main and test) to ensure compatibility
5. **Update this README** when adding new scripts

## **Troubleshooting**

### **Import Errors**
If you get import errors when running from the test directory:
```bash
cd test
uv run update_imports.py
```

### **Config File Not Found**
Make sure to use relative paths for config files:
```python
loader = ConfigLoader("../config/config_unified.json")
```

### **Module Not Found**
Ensure the parent directory is in the Python path (handled by `update_imports.py`).

This organization keeps the main project directory clean while providing easy access to all testing and debugging tools! 🧪
