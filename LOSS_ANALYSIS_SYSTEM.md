# Unified Extensible Loss Analysis System

A comprehensive, extensible system for analyzing loss behavior during VAE training. This system provides three distinct analysis approaches and can be easily extended with new analysis methods.

## Overview

The Loss Analysis System is designed to provide deep insights into training behavior by analyzing loss patterns from multiple perspectives:

1. **Standard Loss Analysis** - Observes if loss is stuck even with weight changes
2. **Constant Weight Analysis** - Calculates loss based on fixed reference weights
3. **Pareto Criterion Analysis** - Multi-objective optimization analysis

## Features

- **Extensible Architecture**: Easy to add new analysis methods
- **Unified Interface**: Single system coordinates all analysis methods
- **Comprehensive Logging**: Detailed analysis results saved to JSON
- **Health Scoring**: Quantitative health assessment (0.0 to 1.0)
- **Smart Recommendations**: Actionable suggestions based on analysis
- **Integration Ready**: Works seamlessly with existing training loops

## Quick Start

### Basic Usage

```python
from loss_analysis_system import create_loss_analysis_system, AnalysisMethod

# Create analysis system
config = {
    'stuck_threshold': 0.001,
    'stuck_patience': 5,
    'trend_window': 10,
    'enable_logging': True
}
analysis_system = create_loss_analysis_system(config)

# Run analysis for an epoch
analysis_results = analysis_system.analyze_epoch(
    epoch, train_metrics, val_metrics, current_weights
)

# Get summary
summary = analysis_system.get_analysis_summary(analysis_results)
print(f"Health Score: {summary['overall_health_score']:.3f}")
```

### Integration with Training

```python
# In your training loop
for epoch in range(max_epochs):
    # ... training code ...
    
    # Run loss analysis
    current_weights = loss_manager.get_all_weights()
    analysis_results = analysis_system.analyze_epoch(
        epoch, train_metrics, val_metrics, current_weights
    )
    
    # Print recommendations
    recommendations = analysis_system.get_combined_recommendations(analysis_results)
    for rec in recommendations:
        print(f"💡 {rec}")
```

## Analysis Methods

### 1. Standard Loss Analysis

**Purpose**: Detects if training is stuck regardless of weight changes.

**Key Features**:
- Tracks loss trends over time
- Detects stuck training periods
- Analyzes weight change impact
- Provides trend analysis (improving/worsening/plateaued)

**Configuration**:
```python
config = {
    'stuck_threshold': 0.001,    # Minimum improvement to not be stuck
    'stuck_patience': 5,         # Epochs without improvement
    'trend_window': 10           # Window for trend analysis
}
```

**Health Score Factors**:
- Trend direction (improving = +0.3, worsening = -0.3)
- Stuck status (stuck = -0.2)
- Improvement rate (good = +0.2, bad = -0.2)

### 2. Constant Weight Analysis

**Purpose**: Provides stable evaluation using fixed reference weights.

**Key Features**:
- Uses constant reference weights for evaluation
- Tracks reference loss trends
- Measures training stability
- Independent of weight scheduling

**Configuration**:
```python
config = {
    'reference_weights': {
        'mse_weight': 1.0,
        'l1_weight': 0.2,
        'perceptual_weight': 0.01,
        'generation_weight': 0.001,
        'beta': 0.0
    }
}
```

**Health Score Factors**:
- Reference loss trend (improving = +0.3, worsening = -0.3)
- Stability (high stability = +0.4)
- Improvement rate (good = +0.2, bad = -0.2)

### 3. Pareto Criterion Analysis

**Purpose**: Multi-objective optimization analysis for balanced training.

**Key Features**:
- Evaluates Pareto optimality
- Tracks Pareto front evolution
- Measures objective balance
- Multi-objective recommendations

**Configuration**:
```python
config = {
    'pareto_objectives': ['mse', 'l1', 'perceptual', 'kl'],
    'pareto_weights': {
        'mse': 1.0,
        'l1': 1.0,
        'perceptual': 1.0,
        'kl': 1.0
    },
    'pareto_tolerance': 0.01
}
```

**Health Score Factors**:
- Pareto optimality (optimal = +0.3)
- Distance to front (close = +0.2, far = -0.2)
- Objective balance (balanced = +0.3)

## Configuration

### Basic Configuration

```python
config = {
    # General parameters
    'stuck_threshold': 0.001,        # Minimum improvement threshold
    'stuck_patience': 5,             # Epochs before considering stuck
    'trend_window': 10,              # Window for trend analysis
    
    # Logging
    'enable_logging': True,          # Enable JSON logging
    # Paths are resolved relative to config['log_dir'] when no directory is provided
    'log_file': 'loss_analysis.json', # Log file path (e.g., logs/loss_analysis.json)
    'save_plots': True,              # Save analysis plots
    'plot_dir': 'loss_analysis_plots' # Plot directory (e.g., logs/loss_analysis_plots)
}
```

### Advanced Configuration

```python
config = {
    # Standard analysis
    'stuck_threshold': 0.0005,
    'stuck_patience': 3,
    'trend_window': 15,
    
    # Constant weight analysis
    'reference_weights': {
        'mse_weight': 1.0,
        'l1_weight': 0.2,
        'perceptual_weight': 0.01,
        'generation_weight': 0.001,
        'beta': 0.0
    },
    
    # Pareto analysis
    'pareto_objectives': ['mse', 'l1', 'perceptual', 'kl'],
    'pareto_weights': {
        'mse': 1.0,
        'l1': 0.8,
        'perceptual': 1.2,
        'kl': 1.0
    },
    'pareto_tolerance': 0.005
}
```

## Integration with Training Loops

### Unified Training Script

The system is already integrated with `train_unified.py`:

```bash
# Enable loss analysis
python train_unified.py --enable-loss-analysis --loss-analysis-interval 5

# Use specific presets
python train_unified.py --loss-preset high_quality --enable-loss-analysis
```

### Custom Integration

```python
class MyTrainer:
    def __init__(self, config):
        self.analysis_system = create_loss_analysis_system(config.get('loss_analysis', {}))
    
    def train_epoch(self, epoch):
        # ... training code ...
        
        # Run analysis
        analysis_results = self.analysis_system.analyze_epoch(
            epoch, train_metrics, val_metrics, current_weights
        )
        
        # Use results
        if analysis_results:
            summary = self.analysis_system.get_analysis_summary(analysis_results)
            if summary['overall_health_score'] < 0.3:
                print("⚠️ Training health is poor - consider adjustments")
```

## Extending the System

### Adding Custom Analysis Methods

```python
from loss_analysis_system import BaseLossAnalyzer, AnalysisMethod, LossAnalysisResult
from datetime import datetime

class MyCustomAnalyzer(BaseLossAnalyzer):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your analyzer
    
    def analyze(self, epoch, train_metrics, val_metrics, current_weights):
        # Implement your analysis logic
        analysis_data = {
            'custom_metric': self._calculate_custom_metric(train_metrics),
            'custom_trend': self._analyze_custom_trend(epoch)
        }
        
        recommendations = self.get_recommendations(analysis_data)
        health_score = self.calculate_health_score(analysis_data)
        
        return LossAnalysisResult(
            method=AnalysisMethod.CUSTOM,  # Define this enum value
            epoch=epoch,
            analysis_data=analysis_data,
            recommendations=recommendations,
            health_score=health_score,
            timestamp=datetime.now()
        )
    
    def get_recommendations(self, analysis_data):
        # Generate recommendations based on your analysis
        return ["Custom recommendation 1", "Custom recommendation 2"]
    
    def calculate_health_score(self, analysis_data):
        # Calculate health score (0.0 to 1.0)
        return 0.5  # Your logic here

# Register custom analyzer
analysis_system.add_custom_analyzer(AnalysisMethod.CUSTOM, MyCustomAnalyzer(config))
```

## API Reference

### UnifiedLossAnalysisSystem

Main system class that coordinates all analysis methods.

#### Methods

- `analyze_epoch(epoch, train_metrics, val_metrics, current_weights, methods=None)`
  - Run analysis for a given epoch
  - Returns dictionary of analysis results

- `get_combined_recommendations(results)`
  - Get combined recommendations from all methods
  - Returns list of unique recommendations

- `get_overall_health_score(results)`
  - Get overall health score from all methods
  - Returns average health score (0.0 to 1.0)

- `get_analysis_summary(results)`
  - Get comprehensive summary of all results
  - Returns detailed summary dictionary

### BaseLossAnalyzer

Abstract base class for analysis methods.

#### Methods to Override

- `analyze(epoch, train_metrics, val_metrics, current_weights)`
  - Main analysis method (required)
  - Returns LossAnalysisResult

- `get_recommendations(analysis_data)`
  - Generate recommendations (optional)
  - Returns list of recommendation strings

- `calculate_health_score(analysis_data)`
  - Calculate health score (optional)
  - Returns float between 0.0 and 1.0

## Examples

### Running the Example

```bash
# Run the demonstration script
python example_loss_analysis.py

# This will simulate 20 epochs of training and show analysis results
```

### Configuration Examples

See `config/config_loss_analysis.json` for detailed configuration examples including presets for different use cases:

- **basic**: Minimal analysis for quick training
- **detailed**: Comprehensive analysis with logging
- **research**: Full analysis with all features enabled

## Troubleshooting

### Common Issues

1. **No analysis results**: Check that metrics contain expected loss components
2. **Poor health scores**: Review recommendations and adjust training parameters
3. **Logging errors**: Ensure write permissions for log directory
4. **Memory issues**: Reduce trend_window or disable logging for large datasets

### Debug Mode

Enable debug logging to see detailed analysis information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To add new analysis methods:

1. Create a new class inheriting from `BaseLossAnalyzer`
2. Implement the required methods
3. Add the new method to the `AnalysisMethod` enum
4. Register it with the system using `add_custom_analyzer()`

## License

This system is part of the Controllable Face Generation VAE project and follows the same license terms.
