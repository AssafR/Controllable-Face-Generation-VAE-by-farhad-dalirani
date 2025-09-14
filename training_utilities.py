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
import torch
import torch.optim as optim
class RunIdManager:
    """Generates and holds a single run_id per process (single source of truth)."""

    def __init__(self, config: Dict[str, Any]):
        # Honor pre-set run_id (e.g., from CLI) to avoid overriding explicit namespaces
        preconfigured = config.get('run_id')
        if preconfigured:
            self.run_id = str(preconfigured)
        else:
            base = config.get('config_name', 'unified')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_id = f"{base}_{timestamp}"

    def get_run_id(self) -> str:
        return self.run_id



class ConfigurationManager:
    """Handles configuration display and management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def _get_genqual_weights(self) -> Dict[str, float]:
        """Get normalized GenQual weights from config (single source of truth)."""
        gq_cfg = self.config.get('generation_quality_config', {}) or {}
        edge_w = float(gq_cfg.get('edge_weight', 0.05))
        div_w = float(gq_cfg.get('diversity_weight', 0.85))
        ctr_w = float(gq_cfg.get('contrast_weight', 0.10))
        total_w = edge_w + div_w + ctr_w
        if total_w > 0:
            return {"edge": edge_w/total_w, "diversity": div_w/total_w, "contrast": ctr_w/total_w}
        else:
            return {"edge": edge_w, "diversity": div_w, "contrast": ctr_w}
    
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
        # Early stopping and convergence control
        patience = self.config.get('early_stopping_patience', 12)
        conv_disabled = bool(self.config.get('disable_convergence_stop', False))
        conv_status = 'DISABLED' if conv_disabled else 'ENABLED (phase-aware)'
        stuck_disabled = bool(self.config.get('disable_stuck_stop', True))
        stuck_status = 'DISABLED' if stuck_disabled else 'ENABLED'
        print(f"  • Early stopping patience: {patience} epochs")
        print(f"  • Convergence stop: {conv_status}")
        print(f"  • Stuck-training stop: {stuck_status}")
        
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
        enabled = bool(self.config.get('enable_loss_analysis', False))
        loss_analysis_config = self.config.get('loss_analysis', {}) if enabled else {}
        if not enabled:
            print(f"\n📊 LOSS ANALYSIS: DISABLED")
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
        if enabled:
            interval = self.config.get('loss_analysis_interval', 5)
            print(f"  • Report interval: Every {interval} epochs")
        
        # Logging
        if loss_analysis_config.get('enable_logging', False):
            base_dir = self.config.get('log_dir', 'logs')
            lf = loss_analysis_config.get('log_file', 'loss_analysis.json')
            log_file = lf if os.path.dirname(lf) else os.path.join(base_dir, lf)
            print(f"  • Logging: Enabled ({log_file})")
        else:
            print(f"  • Logging: Disabled")
    
    def _print_loss_configuration(self) -> None:
        """Print loss configuration details."""
        loss_config = self.config.get('loss_config', {})
        print(f"\n⚖️  LOSS CONFIGURATION:")
        
        # Loss weights
        print(f"  • Loss weights: MSE={loss_config.get('mse_weight', 0)}, L1={loss_config.get('l1_weight', 0)}")
        
        # Perceptual and GenQual weights with VGG type
        perceptual_weight = loss_config.get('perceptual_weight', 0)
        genqual_weight = loss_config.get('generation_weight', 0)
        
        # Get VGG type for perceptual loss display
        vgg_type_short = ""
        if loss_config.get('use_perceptual_loss', False):
            vgg_type = self._get_vgg_loss_type_from_config()
            if vgg_type:
                vgg_type_short = f" ({vgg_type.split('(')[0].split()[-1].lower()})"
        
        print(f"  • Perceptual{vgg_type_short}: {perceptual_weight:.3f}, GenQual: {genqual_weight:.3f}")
        
        # Loss components
        print(f"  • Loss components: MSE={loss_config.get('use_mse', False)}, L1={loss_config.get('use_l1', False)}")
        
        # Perceptual loss with type
        perceptual_enabled = loss_config.get('use_perceptual_loss', False)
        if perceptual_enabled:
            vgg_type = self._get_vgg_loss_type_from_config()
            if vgg_type:
                # Extract the type from "VGG Full (12 layers)" -> "full"
                vgg_type_short = vgg_type.split('(')[0].split()[-1].lower()
                print(f"  • Perceptual: {perceptual_enabled} ({vgg_type_short})")
            else:
                print(f"  • Perceptual: {perceptual_enabled}")
        else:
            print(f"  • Perceptual: {perceptual_enabled}")
        
        print(f"  • GenQual: {loss_config.get('use_generation_quality', False)}")
        
        # Generation quality breakdown (if enabled)
        if loss_config.get('use_generation_quality', False):
            weights = self._get_genqual_weights()
            print(f"  • GenQual breakdown: Edge ({weights['edge']:.0%}), Diversity ({weights['diversity']:.0%}), Contrast ({weights['contrast']:.0%})")
        
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
    
    def _get_vgg_loss_type_from_config(self) -> str:
        """Get VGG loss type from configuration flags (single source of truth)."""
        loss_config = self.config.get('loss_config', {})
        
        # Check if perceptual loss is enabled
        if not loss_config.get('use_perceptual_loss', False):
            return None
        
        # Determine VGG type based on configuration flags
        if loss_config.get('ultra_full_vgg_perceptual', False):
            return "VGG Ultra-Full (16 layers)"
        elif loss_config.get('full_vgg_perceptual', False):
            return "VGG Full (12 layers)"
        else:
            # Check GPU memory to determine if aggressive optimization is used
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory_gb < 12:
                    return "VGG Aggressive (3 layers)"
                else:
                    return "VGG Standard (5 layers)"
            else:
                return "VGG Standard (5 layers)"


class ExperimentLogger:
    """Centralized experiment logger (single source of truth).

    Writes JSONL entries to experiments_log.jsonl, capturing command line,
    start/resume mode, checkpoint details, and config diffs when resuming.
    """

    def __init__(self, config: Dict[str, Any], log_path: str = None):
        self.config = config
        base_dir = config.get('log_dir', 'logs')
        self.log_path = log_path or os.path.join(base_dir, 'experiments_log.jsonl')
        # Ensure directory exists if a nested path is provided
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def _append(self, record: Dict[str, Any]) -> None:
        try:
            # Add standard fields
            safe_record = dict(record)
            safe_record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
            safe_record.setdefault("config_name", self.config.get("config_name", "unified"))
            if self.config.get("run_id"):
                safe_record.setdefault("run_id", self.config.get("run_id"))
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  ⚠️  Experiment logging failed: {e}")

    def log_run_start(self, command_line: str, mode: str, extras: Optional[Dict[str, Any]] = None) -> None:
        # Use ConfigurationManager to get GenQual weights (single source of truth)
        config_manager = ConfigurationManager(self.config)
        record = {
            "event": "run_start",
            "mode": mode,  # Fresh | Resume
            "command_line": command_line,
            "cwd": os.getcwd(),
            "genqual_weights": config_manager._get_genqual_weights(),
        }
        if extras:
            record.update(extras)
        self._append(record)

    def log_resume_details(self, checkpoint_path: str, epoch: int, best_val_loss: float,
                           checkpoint_config: Optional[Dict[str, Any]],
                           current_config: Optional[Dict[str, Any]],
                           config_diff: Optional[Dict[str, Any]]) -> None:
        record = {
            "event": "resume",
            "checkpoint_path": checkpoint_path,
            "restored_epoch": epoch,
            "best_val_loss": best_val_loss,
            "config_changed": bool(config_diff) if config_diff is not None else False,
            "config_diff": config_diff or {},
        }
        # For reproducibility, also store compact copies of configs
        if checkpoint_config is not None:
            record["checkpoint_config_snapshot"] = _shallow_config_view(checkpoint_config)
        if current_config is not None:
            record["current_config_snapshot"] = _shallow_config_view(current_config)
        self._append(record)

    def log_run_end(self, status: str, best_val_loss: float, epochs_completed: int,
                    notes: Optional[str] = None) -> None:
        record = {
            "event": "run_end",
            "status": status,  # completed | interrupted | failed | early_stopped
            "best_val_loss": best_val_loss,
            "epochs_completed": epochs_completed,
        }
        if notes:
            record["notes"] = notes
        self._append(record)

    def log_image_saved(self, file_path: str, kind: str, epoch: Optional[int] = None) -> None:
        """Log a saved image (generated or reconstruction)."""
        record = {
            "event": "image_saved",
            "path": file_path,
            "kind": kind,  # generated | reconstruction
        }
        if epoch is not None:
            record["epoch"] = int(epoch)
        self._append(record)


def _shallow_config_view(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact, reproducible subset of config for logging.

    Keeps key training-relevant fields without huge nested structures.
    """
    keys = [
        "config_name", "embedding_size", "num_channels", "input_img_size", "batch_size", "lr",
        "max_epoch", "beta", "beta_schedule", "beta_start", "beta_end",
        "perceptual_schedule", "perceptual_start", "perceptual_end", "perceptual_warmup_epochs",
        "generation_schedule", "generation_start", "generation_end", "generation_warmup_epochs",
        "early_stopping_patience", "enable_loss_analysis", "loss_analysis_interval",
        "loss_analysis_methods", "cycle_training", "cycle_period", "cycle_recon_epochs",
        "cycle_variational_epochs",
    ]
    loss_cfg = cfg.get("loss_config", {})
    result = {k: cfg.get(k) for k in keys if k in cfg}
    result["loss_config"] = {
        k: loss_cfg.get(k) for k in [
            "use_mse", "use_l1", "use_perceptual_loss", "use_generation_quality",
            "mse_weight", "l1_weight", "perceptual_weight", "generation_weight",
            "full_vgg_perceptual", "ultra_full_vgg_perceptual",
        ] if k in loss_cfg
    }
    return result


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
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
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
            'config': self.config,
            'run_id': self.config.get('run_id')
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
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler) -> Tuple[bool, int, Dict[str, float], Dict[str, Any]]:
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
            return False, 0, {}, {}
        
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
            loaded_config = checkpoint.get('config', {})
            
            print(f"  ✅ Checkpoint loaded: {checkpoint_path}")
            print(f"  • Epoch: {epoch}")
            print(f"  • Best validation loss: {self.best_val_loss:.6f}")
            
            return True, epoch, {'train': train_metrics, 'val': val_metrics}, loaded_config
            
        except Exception as e:
            print(f"  ⚠️  Error loading checkpoint: {e}")
            return False, 0, {}, {}


class FileManager:
    """Handles file naming and management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.config_name = config.get('config_name', 'unified')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = config.get('run_id')
    
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
            if self.run_id:
                return f"checkpoints/{self.config_name}_{self.run_id}_training_checkpoint.pth"
            return f"checkpoints/{self.config_name}_training_checkpoint.pth"
        
        elif file_type == "final_samples":
            return f"{self.config_name}_final_samples.{extension}"
        
        else:
            return f"{self.config_name}_{file_type}{suffix}.{extension}"

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find latest checkpoint path for this config, considering run_id variants.

        Preference order:
        1) run_id-specific file if it exists
        2) newest checkpoints/{config_name}_*_training_checkpoint.pth by mtime
        3) legacy checkpoints/{config_name}_training_checkpoint.pth
        """
        import glob
        # 1) Exact run_id
        if self.run_id:
            candidate = f"checkpoints/{self.config_name}_{self.run_id}_training_checkpoint.pth"
            if os.path.exists(candidate):
                return candidate
        # 2) Any run_id variant
        pattern = f"checkpoints/{self.config_name}_*_training_checkpoint.pth"
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]
        # 3) Legacy name
        legacy = f"checkpoints/{self.config_name}_training_checkpoint.pth"
        if os.path.exists(legacy):
            return legacy
        return None
    
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
        
        # Also remove any old run variants for this config
        import glob
        run_variants = glob.glob(f"checkpoints/{self.config_name}_*_training_checkpoint.pth")
        files_to_clear.extend([p for p in run_variants if p not in files_to_clear])

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
        self.run_id_manager = RunIdManager(config)
        # Persist run_id into config for downstream consumers
        self.config['run_id'] = self.run_id_manager.get_run_id()
        self.config_manager = ConfigurationManager(config)
        self.checkpoint_manager = CheckpointManager(config, device)
        self.file_manager = FileManager(config)
        self.experiment_logger = ExperimentLogger(config)
    
    def print_configuration(self, mode: str = "Fresh") -> None:
        """Print training configuration."""
        self.config_manager.print_configuration(mode)
        # If StrategyController is present via LossWeightManager (accessed at runtime), show strategy snapshot.
        try:
            from loss_weight_manager import LossWeightManager  # local import to avoid cycles
            # Best-effort: create a transient manager to query strategy config snapshot
            temp_lwm = LossWeightManager(self.config)
            strategy = getattr(temp_lwm, 'strategy_controller', None)
            if strategy is not None:
                info = strategy.get_display_info()
                print(f"\n🧭 STRATEGY CONFIGURATION:")
                if info.get('cycle_enabled'):
                    preset = self.config.get('cycle_preset', 'custom')
                    print(f"  • Alternating cycle: ENABLED (preset={preset}, period={info.get('cycle_period')}, recon={info.get('cycle_recon_epochs')}, var={info.get('cycle_variational_epochs')})")
                    print(f"  • Alternating cycle affects: MSE, L1, Perceptual, Generation, Beta")
                else:
                    print(f"  • Alternating cycle: DISABLED")
                # Priority snapshot (if any)
                if 'mse_priority_active' in info or 'priority_phase' in info:
                    print(f"  • Priority: {info}")
        except Exception:
            pass
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int,
                       train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                       is_best: bool = False) -> str:
        """Save training checkpoint."""
        return self.checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, epoch, train_metrics, val_metrics, is_best
        )

    def create_optimizer(self, model_parameters, config: Dict[str, Any]):
        """Create optimizer from config (single source of truth).

        Supports: adam (default), adamw, sgd, rmsprop. Uses config['optimizer_params']
        and ensures 'lr' defaults to config['lr'] if not provided.
        Returns (optimizer, optimizer_name).
        """
        opt_name = str(config.get('optimizer', 'adam')).lower()
        opt_params = dict(config.get('optimizer_params', {}))
        opt_params.setdefault('lr', config.get('lr', 1e-3))
        if opt_name == 'sgd':
            optimizer = optim.SGD(model_parameters, **opt_params)
        elif opt_name == 'adamw':
            optimizer = optim.AdamW(model_parameters, **opt_params)
        elif opt_name == 'rmsprop':
            optimizer = optim.RMSprop(model_parameters, **opt_params)
        else:
            optimizer = optim.Adam(model_parameters, **opt_params)
            opt_name = 'adam'
        return optimizer, opt_name
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler) -> Tuple[bool, int, Dict[str, float], Dict[str, Any]]:
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
