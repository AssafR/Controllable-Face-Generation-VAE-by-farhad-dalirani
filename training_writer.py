#!/usr/bin/env python3
"""
Training Writer System
Isolated TensorBoard and logging functionality with fallback support.
"""

import os
from typing import Dict, Any, Optional, Union
from datetime import datetime

# TensorBoard support with graceful fallback
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TrainingWriter:
    """Centralized training logging and visualization with TensorBoard support."""
    
    def __init__(self, config: Dict[str, Any], log_dir: Optional[str] = None):
        """
        Initialize the training writer.
        
        Args:
            config: Training configuration dictionary
            log_dir: Optional custom log directory
        """
        self.config = config
        self.writer = None
        self.log_dir = log_dir
        self.enabled = False
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging with TensorBoard if available."""
        if TENSORBOARD_AVAILABLE:
            if self.log_dir is None:
                # Generate log directory name
                config_name = self.config.get('config_name', 'unified')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_dir = f"runs/{config_name}_{timestamp}"
            
            # Create log directory
            os.makedirs(self.log_dir, exist_ok=True)
            
            try:
                self.writer = SummaryWriter(self.log_dir)
                self.enabled = True
                print(f"  ✅ TensorBoard logging enabled: {self.log_dir}")
            except Exception as e:
                print(f"  ⚠️  TensorBoard initialization failed: {e}")
                self.enabled = False
        else:
            print("  ⚠️  TensorBoard not available, logging disabled")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value.
        
        Args:
            tag: Tag name for the scalar
            value: Scalar value to log
            step: Step number
        """
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_scalar(tag, value, step)
            except Exception as e:
                print(f"  ⚠️  Failed to log scalar {tag}: {e}")
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """
        Log multiple scalars with the same step.
        
        Args:
            main_tag: Main tag for the group
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Step number
        """
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_scalars(main_tag, tag_scalar_dict, step)
            except Exception as e:
                print(f"  ⚠️  Failed to log scalars {main_tag}: {e}")
    
    def log_image(self, tag: str, img_tensor, step: int, **kwargs) -> None:
        """
        Log an image.
        
        Args:
            tag: Tag name for the image
            img_tensor: Image tensor
            step: Step number
            **kwargs: Additional arguments for add_image
        """
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_image(tag, img_tensor, step, **kwargs)
            except Exception as e:
                print(f"  ⚠️  Failed to log image {tag}: {e}")
    
    def log_images(self, tag: str, img_tensor, step: int, **kwargs) -> None:
        """
        Log multiple images.
        
        Args:
            tag: Tag name for the images
            img_tensor: Image tensor (batch of images)
            step: Step number
            **kwargs: Additional arguments for add_images
        """
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_images(tag, img_tensor, step, **kwargs)
            except Exception as e:
                print(f"  ⚠️  Failed to log images {tag}: {e}")
    
    def log_histogram(self, tag: str, values, step: int, **kwargs) -> None:
        """
        Log a histogram.
        
        Args:
            tag: Tag name for the histogram
            values: Values to create histogram from
            step: Step number
            **kwargs: Additional arguments for add_histogram
        """
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_histogram(tag, values, step, **kwargs)
            except Exception as e:
                print(f"  ⚠️  Failed to log histogram {tag}: {e}")
    
    def log_text(self, tag: str, text_string: str, step: int) -> None:
        """
        Log text.
        
        Args:
            tag: Tag name for the text
            text_string: Text to log
            step: Step number
        """
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_text(tag, text_string, step)
            except Exception as e:
                print(f"  ⚠️  Failed to log text {tag}: {e}")
    
    def log_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]) -> None:
        """
        Log hyperparameters and metrics.
        
        Args:
            hparam_dict: Dictionary of hyperparameters
            metric_dict: Dictionary of metrics
        """
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_hparams(hparam_dict, metric_dict)
            except Exception as e:
                print(f"  ⚠️  Failed to log hyperparameters: {e}")
    
    def log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], weights: Dict[str, float]) -> None:
        """
        Log comprehensive epoch metrics.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            weights: Current loss weights dictionary
        """
        if not self.enabled:
            return
        
        # Log main losses
        self.log_scalars('Loss/Total', {
            'Train': train_metrics.get('loss', 0),
            'Validation': val_metrics.get('loss', 0)
        }, epoch)
        
        # Log individual loss components
        loss_components = ['mse', 'l1', 'perceptual', 'generation_quality', 'kl']
        for component in loss_components:
            if component in train_metrics:
                self.log_scalars(f'Loss/{component.title()}', {
                    'Train': train_metrics[component],
                    'Validation': val_metrics.get(component, 0)
                }, epoch)
        
        # Log weights
        weight_dict = {f'Weight/{k.replace("_weight", "").title()}': v for k, v in weights.items() if v > 0}
        if weight_dict:
            self.log_scalars('Weights', weight_dict, epoch)
    
    def log_learning_rate(self, epoch: int, lr: float) -> None:
        """Log learning rate."""
        self.log_scalar('Learning_Rate', lr, epoch)
    
    def log_beta(self, epoch: int, beta: float) -> None:
        """Log beta value."""
        self.log_scalar('Beta', beta, epoch)
    
    def log_stage(self, epoch: int, stage_name: str) -> None:
        """Log current training stage."""
        # Convert stage name to numeric value for plotting
        stage_mapping = {
            'reconstruction': 0,
            'perceptual_introduction': 1,
            'quality_enhancement': 2,
            'final_polish': 3,
            'standard': -1
        }
        stage_value = stage_mapping.get(stage_name, -1)
        self.log_scalar('Training_Stage', stage_value, epoch)
    
    def log_gpu_memory(self, epoch: int, allocated: float, total: float, utilization: float) -> None:
        """Log GPU memory usage."""
        self.log_scalars('GPU_Memory', {
            'Allocated_GB': allocated,
            'Total_GB': total,
            'Utilization_Percent': utilization
        }, epoch)
    
    def log_model_weights(self, epoch: int, model) -> None:
        """Log model weight histograms."""
        if not self.enabled:
            return
        
        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.log_histogram(f'Model_Weights/{name}', param.data, epoch)
        except Exception as e:
            print(f"  ⚠️  Failed to log model weights: {e}")
    
    def log_gradients(self, epoch: int, model) -> None:
        """Log gradient histograms."""
        if not self.enabled:
            return
        
        try:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.log_histogram(f'Gradients/{name}', param.grad.data, epoch)
        except Exception as e:
            print(f"  ⚠️  Failed to log gradients: {e}")
    
    def close(self) -> None:
        """Close the writer and flush all logs."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.close()
                print(f"  ✅ TensorBoard logs saved to: {self.log_dir}")
            except Exception as e:
                print(f"  ⚠️  Error closing TensorBoard writer: {e}")
    
    def is_enabled(self) -> bool:
        """Check if logging is enabled."""
        return self.enabled
    
    def get_log_dir(self) -> Optional[str]:
        """Get the log directory path."""
        return self.log_dir
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class NoOpWriter:
    """No-operation writer for when TensorBoard is not available."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        """Return a no-op function for any method call."""
        return lambda *args, **kwargs: None
    
    def is_enabled(self) -> bool:
        return False
    
    def get_log_dir(self) -> Optional[str]:
        return None


def create_training_writer(config: Dict[str, Any], log_dir: Optional[str] = None) -> Union[TrainingWriter, NoOpWriter]:
    """
    Factory function to create a training writer.
    
    Args:
        config: Training configuration dictionary
        log_dir: Optional custom log directory
        
    Returns:
        TrainingWriter instance if TensorBoard is available, NoOpWriter otherwise
    """
    if TENSORBOARD_AVAILABLE:
        return TrainingWriter(config, log_dir)
    else:
        return NoOpWriter()
