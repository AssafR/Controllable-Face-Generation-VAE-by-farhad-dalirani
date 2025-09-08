#!/usr/bin/env python3
"""
Training Utilities
Centralized utilities for VAE training including configuration, checkpoints, and file management.
"""

import os
import json
import torch
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import time


class ConfigurationManager:
    """Handles configuration display and management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def print_configuration(self, mode: str = "Fresh") -> None:
        """Print comprehensive training configuration."""
        print(f"\n{'='*70}")
        print(f"🔧 {mode.upper()} TRAINING CONFIGURATION")
        print(f"{'='*70}")
        
        # Model configuration
        print(f"\n🏗️  MODEL CONFIGURATION:")
        print(f"  • Architecture: VAE (PyTorch)")
        print(f"  • Embedding size: {self.config['embedding_size']}")
        print(f"  • Image size: {self.config['input_img_size']}x{self.config['input_img_size']}")
        print(f"  • Channels: {self.config['num_channels']}")
        
        # Training configuration
        print(f"\n🎯 TRAINING CONFIGURATION:")
        print(f"  • Epochs: {self.config['max_epoch']}")
        print(f"  • Batch size: {self.config['batch_size']}")
        print(f"  • Learning rate: {self.config['lr']}")
        
        # Loss configuration
        self._print_loss_configuration()
        
        # Loss analysis configuration
        self._print_loss_analysis_configuration()
        
        # Dataset configuration
        self._print_dataset_configuration()
        
        # Hardware configuration
        self._print_hardware_configuration()
        
        print(f"\n{'='*70}")
    
    def _print_loss_analysis_configuration(self) -> None:
        """Print loss analysis configuration details."""
        loss_analysis_config = self.config.get('loss_analysis', {})
        if not loss_analysis_config:
            return
        
        print(f"\n📊 LOSS ANALYSIS CONFIGURATION:")
        
        # Analysis methods
        methods = self.config.get('loss_analysis_methods', ['standard', 'constant_weight', 'pareto'])
        method_icons = {
            'standard': '📈',
            'constant_weight': '⚖️', 
            'pareto': '🎯'
        }
        method_names = {
            'standard': 'Standard',
            'constant_weight': 'Constant Weight',
            'pareto': 'Pareto Criterion'
        }
        
        method_display = []
        for method in methods:
            icon = method_icons.get(method, '📊')
            name = method_names.get(method, method.replace('_', ' ').title())
            method_display.append(f"{icon} {name}")
        
        print(f"  • Analysis methods: {', '.join(method_display)}")
        
        # Analysis parameters
        if 'stuck_threshold' in loss_analysis_config:
            print(f"  • Stuck threshold: {loss_analysis_config['stuck_threshold']}")
        if 'stuck_patience' in loss_analysis_config:
            print(f"  • Stuck patience: {loss_analysis_config['stuck_patience']} epochs")
        if 'trend_window' in loss_analysis_config:
            print(f"  • Trend window: {loss_analysis_config['trend_window']} epochs")
        
        # Reporting interval
        interval = self.config.get('loss_analysis_interval', 5)
        print(f"  • Report interval: Every {interval} epochs")
        
        # Logging
        if loss_analysis_config.get('enable_logging', False):
            log_file = loss_analysis_config.get('log_file', 'loss_analysis.json')
            print(f"  • Logging: Enabled ({log_file})")
        else:
            print(f"  • Logging: Disabled")
    
    def _print_loss_configuration(self) -> None:
        """Print loss configuration details."""
        loss_config = self.config.get('loss_config', {})
        print(f"\n⚖️  LOSS CONFIGURATION:")
        
        # Loss weights
        print(f"  • Loss weights: MSE={loss_config.get('mse_weight', 0)}, L1={loss_config.get('l1_weight', 0)}")
        print(f"  • Perceptual: {loss_config.get('perceptual_weight', 0):.3f}, GenQual: {loss_config.get('generation_weight', 0):.3f}")
        
        # Loss components
        print(f"  • Loss components: MSE={loss_config.get('use_mse', False)}, L1={loss_config.get('use_l1', False)}")
        print(f"  • Perceptual: {loss_config.get('use_perceptual_loss', False)}, GenQual: {loss_config.get('use_generation_quality', False)}")
        
        # Generation quality breakdown (if enabled)
        if loss_config.get('use_generation_quality', False):
            # Use default weights for display (actual weights managed by LossWeightManager)
            gen_qual_weights = {'edge': 0.4, 'diversity': 0.3, 'contrast': 0.3}
            print(f"  • GenQual breakdown: Edge ({gen_qual_weights['edge']:.0%}), Diversity ({gen_qual_weights['diversity']:.0%}), Contrast ({gen_qual_weights['contrast']:.0%})")
        
        # Beta configuration
        beta = self.config.get('beta', 1.0)
        beta_schedule = self.config.get('beta_schedule', False)
        if beta_schedule:
            beta_start = self.config.get('beta_start', 0.1)
            beta_end = self.config.get('beta_end', 2.0)
            print(f"  • Beta: {beta_start:.1f} → {beta_end:.1f} (scheduled)")
        else:
            print(f"  • Beta: {beta:.1f} (fixed)")
        
        # Stage-based training
        if self.config.get('stage_based_training', False):
            print(f"  • Stage-based training: ENABLED")
            stages = self.config.get('stages', {})
            for stage_name, stage_config in stages.items():
                start_epoch, end_epoch = stage_config['epochs']
                print(f"    - {stage_name}: Epochs {start_epoch}-{end_epoch}")
        else:
            print(f"  • Stage-based training: DISABLED")
    
    def _print_dataset_configuration(self) -> None:
        """Print dataset configuration details."""
        print(f"\n📊 DATASET CONFIGURATION:")
        print(f"  • Dataset: {self.config.get('dataset', 'Unknown')}")
        print(f"  • Data path: {self.config.get('data_path', 'Unknown')}")
        print(f"  • Train/Val split: {self.config.get('train_split', 0.8):.1%}/{1-self.config.get('train_split', 0.8):.1%}")
    
    def _print_hardware_configuration(self) -> None:
        """Print hardware configuration details."""
        print(f"\n💻 HARDWARE CONFIGURATION:")
        print(f"  • Device: {self.config.get('device', 'Unknown')}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  • GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print(f"  • GPU: Not available")


class CheckpointManager:
    """Handles checkpoint saving and loading."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        self.best_val_loss = float('inf')
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, 
                       train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                       is_best: bool = False) -> str:
        """
        Save training checkpoint.
        
        Args:
            model: VAE model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        config_name = self.config.get('config_name', 'unified')
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = f"{checkpoint_dir}/{config_name}_training_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if applicable
        if is_best:
            self.best_val_loss = val_metrics['loss']
            best_path = f"{checkpoint_dir}/{config_name}_best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✅ Best model saved: {best_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler) -> Tuple[bool, int, Dict[str, float]]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: VAE model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Tuple of (success, epoch, metrics)
        """
        if not os.path.exists(checkpoint_path):
            return False, 0, {}
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            # Load metrics
            train_metrics = checkpoint.get('train_metrics', {})
            val_metrics = checkpoint.get('val_metrics', {})
            
            print(f"  ✅ Checkpoint loaded: {checkpoint_path}")
            print(f"  • Epoch: {epoch}")
            print(f"  • Best validation loss: {self.best_val_loss:.6f}")
            
            return True, epoch, {'train': train_metrics, 'val': val_metrics}
            
        except Exception as e:
            print(f"  ⚠️  Error loading checkpoint: {e}")
            return False, 0, {}


class FileManager:
    """Handles file naming and management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.config_name = config.get('config_name', 'unified')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_filename(self, file_type: str, epoch: Optional[int] = None, 
                    suffix: str = "", extension: str = "png") -> str:
        """
        Generate standardized filename for different file types.
        
        Args:
            file_type: Type of file (samples, model, log_dir, etc.)
            epoch: Epoch number (optional)
            suffix: Additional suffix (optional)
            extension: File extension (default: png)
            
        Returns:
            Generated filename
        """
        if file_type == "samples":
            if epoch is not None:
                return f"{self.config_name}_epoch_{epoch:03d}{suffix}.{extension}"
            else:
                return f"{self.config_name}_samples{suffix}.{extension}"
        
        elif file_type == "model":
            if epoch is not None:
                return f"{self.config_name}_model_epoch_{epoch:03d}.pth"
            else:
                return f"{self.config_name}_model.pth"
        
        elif file_type == "log_dir":
            return f"runs/{self.config_name}_{self.timestamp}"
        
        elif file_type == "checkpoint":
            return f"checkpoints/{self.config_name}_training_checkpoint.pth"
        
        elif file_type == "final_samples":
            return f"{self.config_name}_final_samples.{extension}"
        
        else:
            return f"{self.config_name}_{file_type}{suffix}.{extension}"
    
    def clear_old_files(self) -> None:
        """Clear old model files and checkpoints for fresh start."""
        print(f"  🧹 Clearing old files for fresh start...")
        
        # Files to clear
        files_to_clear = [
            self.get_filename("checkpoint"),
            self.get_filename("model"),
            self.get_filename("samples"),
            self.get_filename("final_samples")
        ]
        
        # Clear log directory
        log_dir = self.get_filename("log_dir")
        if os.path.exists(log_dir):
            import shutil
            shutil.rmtree(log_dir)
            print(f"    ✅ Cleared log directory: {log_dir}")
        
        # Clear individual files
        cleared_count = 0
        for file_path in files_to_clear:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"    ✅ Removed: {file_path}")
                    cleared_count += 1
                except Exception as e:
                    print(f"    ⚠️  Could not remove {file_path}: {e}")
        
        if cleared_count == 0:
            print(f"    ℹ️  No old files found to clear")
        else:
            print(f"    ✅ Cleared {cleared_count} old files")


class TrainingUtilities:
    """Main utilities class that combines all utility functionality."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize sub-managers
        self.config_manager = ConfigurationManager(config)
        self.checkpoint_manager = CheckpointManager(config, device)
        self.file_manager = FileManager(config)
    
    def print_configuration(self, mode: str = "Fresh") -> None:
        """Print training configuration."""
        self.config_manager.print_configuration(mode)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int,
                       train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                       is_best: bool = False) -> str:
        """Save training checkpoint."""
        return self.checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, epoch, train_metrics, val_metrics, is_best
        )
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler) -> Tuple[bool, int, Dict[str, float]]:
        """Load training checkpoint."""
        return self.checkpoint_manager.load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    
    def get_filename(self, file_type: str, epoch: Optional[int] = None,
                    suffix: str = "", extension: str = "png") -> str:
        """Generate filename."""
        return self.file_manager.get_filename(file_type, epoch, suffix, extension)
    
    def clear_old_files(self) -> None:
        """Clear old files."""
        self.file_manager.clear_old_files()
    
    def adjust_batch_size_for_vgg(self, config: Dict[str, Any], device: str) -> None:
        """Intelligently adjust batch size for VGG perceptual loss memory requirements."""
        if not torch.cuda.is_available():
            # CPU training - use smaller batch size
            if config['batch_size'] > 8:
                config['batch_size'] = 8
                print(f"    ⚠️  Reduced batch size to {config['batch_size']} for CPU training")
            return
        
        # GPU training - adjust based on available memory
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        if gpu_memory_gb < 8:
            # Low memory GPU - use very small batch size
            if config['batch_size'] > 4:
                config['batch_size'] = 4
                print(f"    ⚠️  Reduced batch size to {config['batch_size']} for low memory GPU ({gpu_memory_gb:.1f}GB)")
        elif gpu_memory_gb < 12:
            # Medium memory GPU - use moderate batch size
            if config['batch_size'] > 8:
                config['batch_size'] = 8
                print(f"    ⚠️  Reduced batch size to {config['batch_size']} for medium memory GPU ({gpu_memory_gb:.1f}GB)")
        elif gpu_memory_gb < 16:
            # High memory GPU - use larger batch size but still conservative
            if config['batch_size'] > 16:
                config['batch_size'] = 16
                print(f"    ⚠️  Reduced batch size to {config['batch_size']} for high memory GPU ({gpu_memory_gb:.1f}GB)")
        
        if config['batch_size'] < self.config.get('original_batch_size', config['batch_size']):
            print(f"    ⚠️  Reduced from {self.config.get('original_batch_size', config['batch_size'])} due to VGG memory requirements")
        else:
            print(f"    ✅ No batch size reduction needed")
    
    def assess_loss_behavior(self, train_metrics: Dict[str, float], 
                           val_metrics: Dict[str, float], epoch: int) -> None:
        """
        DEPRECATED: Assess and report on loss behavior patterns.
        
        This method is deprecated. Loss behavior assessment is now handled by the 
        loss analysis system which provides more comprehensive and intelligent analysis.
        """
        # This method is deprecated - use loss analysis system instead
        # Only provide basic health check as fallback
        if train_metrics.get('kl', 0) < 0.001:
            print(f"  ⚠️  KL collapse detected (KL loss very low) - consider increasing beta")
        # Other checks are now handled by the loss analysis system


def create_training_utilities(config: Dict[str, Any], device: str = 'cuda') -> TrainingUtilities:
    """
    Factory function to create training utilities.
    
    Args:
        config: Training configuration dictionary
        device: Device to run on
        
    Returns:
        TrainingUtilities instance
    """
    return TrainingUtilities(config, device)
