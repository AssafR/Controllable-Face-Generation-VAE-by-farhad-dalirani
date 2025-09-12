#!/usr/bin/env python3
"""
Unified VAE Training Script
Consolidates all training methods into a single, configurable system.
Supports all training presets through command-line arguments.
"""

import sys
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import time
import json
from typing import Dict, Any

# Ensure UTF-8 safe console output on Windows (avoid UnicodeEncodeError)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# Perceptual loss is now isolated in perceptual_loss.py

# TensorBoard support is now isolated in training_writer.py

import numpy as np
from tqdm import tqdm

# Import our modules
from variation_autoencoder_improved import VAE_pt
from utilities_pytorch import get_split_data, configure_gpu, display_image_grid, display_comparison_grid
from config_loader import ConfigLoader
from utils import create_directories, safe_remove_file
from loss_weight_manager import LossWeightManager
from training_reporter import TrainingReporter
from training_writer import create_training_writer
from perceptual_loss import create_perceptual_loss, get_perceptual_loss_info
from loss_calculator import create_loss_calculator
from training_utilities import create_training_utilities
from loss_analysis_system import create_loss_analysis_system, AnalysisMethod

# VGGPerceptualLoss class moved to perceptual_loss.py

class UnifiedVAETrainer:
    """Unified VAE trainer that supports all training configurations."""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        # Best validation loss tracking (replaced outdated eval_weights mechanism)
        self.best_ref_val = float('inf')
        # Track acceleration state to reset patience on activation/deactivation
        self.prev_accel_active = False
        
        # Initialize centralized loss weight management
        self.loss_manager = LossWeightManager(config)
        
        # Initialize training reporter
        self.reporter = TrainingReporter(self.loss_manager)
        
        # Initialize loss calculator
        self.loss_calculator = create_loss_calculator(config, device)
        
        # Initialize training utilities
        self.utilities = create_training_utilities(config, device)
        
        # Initialize loss analysis system
        self.loss_analysis_system = create_loss_analysis_system(config.get('loss_analysis', {}))
        
        # Initialize loss history tracking
        self.loss_history = {
            'perceptual': [],
            'generation_quality': [],
            'kl': [],
            'total': []
        }
        
        # Adjust batch size for VGG if needed
        loss_config = config.get('loss_config', {})
        if loss_config.get('use_perceptual_loss', False):
            self.utilities.adjust_batch_size_for_vgg(config, device)
    
    # Batch size adjustment moved to training_utilities.py
    
    def setup_logging(self):
        """Setup TensorBoard logging using the isolated writer."""
        log_dir = self.utilities.get_filename('log_dir')
        self.writer = create_training_writer(self.config, log_dir)
        
        if self.writer.is_enabled():
            print(f"  • Logging: {self.writer.get_log_dir()}")
        else:
            print(f"  • Logging: Disabled (TensorBoard not available)")
    
    # Accessor methods removed - use self.loss_manager.get_weight() directly
    
    def create_model(self):
        """Create and configure the VAE model."""
        print(f"\n🏗️  Creating model...")
        self.model = VAE_pt(
            input_img_size=self.config['input_img_size'],
            embedding_size=self.config['embedding_size'],
            loss_config=self.config
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ Model created:")
        print(f"  • Total parameters: {total_params:,}")
        print(f"  • Trainable parameters: {trainable_params:,}")
        print(f"  • Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        return self.model
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8, 
            min_lr=1e-7
        )
        
        print(f"✅ Training setup complete:")
        print(f"  • Optimizer: Adam (lr={self.config['lr']}, weight_decay=1e-5)")
        print(f"  • Scheduler: ReduceLROnPlateau (patience=8)")
        print(f"  • Batch size: {self.config['batch_size']}")
    
    def load_data(self):
        """Load and prepare the dataset."""
        print(f"\n📊 Loading dataset...")
        train_data, val_data = get_split_data(config=self.config)
        
        train_loader = DataLoader(
            train_data, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=0
        )
        
        print(f"✅ Dataset loaded:")
        print(f"  • Training samples: {len(train_data):,}")
        print(f"  • Validation samples: {len(val_data):,}")
        print(f"  • Training batches: {len(train_loader):,}")
        print(f"  • Validation batches: {len(val_loader):,}")
        
        return train_loader, val_loader
    
    # Perceptual loss calculation moved to loss_calculator.py
    
    # Generation quality loss calculation moved to loss_calculator.py
    
    def calculate_losses(self, images, reconst, emb_mean, emb_log_var):
        """Calculate all loss components using the centralized loss calculator."""
        loss_dict = self.loss_calculator.calculate_all_losses(
            images, reconst, emb_mean, emb_log_var, self.model, self.loss_manager
        )
        
        # Convert to the expected format for backward compatibility
        total_loss = loss_dict['loss']
        
        # Convert tensor values to scalars for logging
        loss_dict_scalar = {
            'loss': loss_dict['loss'].item(),
            'mse': loss_dict['mse'].item(),
            'l1': loss_dict['l1'].item(),
            'perceptual': loss_dict['perceptual'].item() if isinstance(loss_dict['perceptual'], torch.Tensor) else loss_dict['perceptual'],
            'generation_quality': loss_dict['generation_quality'].item() if isinstance(loss_dict['generation_quality'], torch.Tensor) else loss_dict['generation_quality'],
            'kl': loss_dict['kl'].item(),
            'total': loss_dict['loss'].item()
        }
        
        return total_loss, loss_dict_scalar
    
    def _process_epoch(self, data_loader, epoch, is_training=True):
        """
        Unified epoch processing for both training and validation.
        
        Args:
            data_loader: DataLoader for the epoch
            epoch: Current epoch number
            is_training: Whether this is training (True) or validation (False)
            
        Returns:
            Dictionary of averaged metrics
        """
        if is_training:
            self.model.train()
            mode = "Train"
        else:
            self.model.eval()
            mode = "Val"
        
        # Initialize accumulators
        metrics = {
            'loss': 0.0, 'mse': 0.0, 'kl': 0.0, 'l1': 0.0, 
            'perceptual': 0.0, 'generation_quality': 0.0
        }
        
        # Create progress bar
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.config['max_epoch']} [{mode}]", leave=False)
        
        # Process batches
        with torch.no_grad() if not is_training else torch.enable_grad():
            for batch_idx, images in enumerate(pbar):
                images = images.to(self.device)
                
                if is_training:
                    # Zero gradients for training
                    self.optimizer.zero_grad()
                
                # Forward pass
                emb_mean, emb_log_var, reconst = self.model(images)
                
                # Calculate losses
                total_loss, loss_dict = self.calculate_losses(images, reconst, emb_mean, emb_log_var)
                
                if is_training:
                    # Backward pass and optimization for training
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Accumulate losses
                for key in metrics:
                    metrics[key] += loss_dict[key]
                
                # Update progress bar with loss percentages
                self._update_progress_bar(pbar, total_loss, loss_dict)
        
        # Calculate averages
        num_batches = len(data_loader)
        return {key: value / num_batches for key, value in metrics.items()}
    
    def _update_progress_bar(self, pbar, total_loss, loss_dict):
        """Update progress bar with loss percentages."""
        total_loss_val = total_loss.item()
        
        # Calculate weighted losses for percentage display
        weighted_losses = {
            'mse': loss_dict["mse"] * self.loss_manager.get_weight('mse_weight'),
            'l1': loss_dict["l1"] * self.loss_manager.get_weight('l1_weight'),
            'perceptual': loss_dict["perceptual"] * self.loss_manager.get_weight('perceptual_weight'),
            'generation_quality': loss_dict["generation_quality"] * self.loss_manager.get_weight('generation_weight'),
            'kl': loss_dict["kl"] * self.loss_manager.get_weight('beta')
        }
        
        # Calculate percentages
        percentages = {}
        for key, weighted_loss in weighted_losses.items():
            percentages[key] = (weighted_loss / total_loss_val * 100) if total_loss_val > 0 else 0
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_val:.4f}',
            'MSE': f'{percentages["mse"]:.0f}%',
            'L1': f'{percentages["l1"]:.0f}%',
            'Perceptual': f'{percentages["perceptual"]:.0f}%',
            'GenQual': f'{percentages["generation_quality"]:.0f}%',
            'KL': f'{percentages["kl"]:.0f}%'
        })
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        return self._process_epoch(train_loader, epoch, is_training=True)
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch."""
        return self._process_epoch(val_loader, epoch, is_training=False)
    
    # save_checkpoint moved to training_utilities.py
    
    # load_checkpoint moved to training_utilities.py
    
    # get_filename moved to training_utilities.py
    
    # clear_old_files moved to training_utilities.py
        
        print("  🆕 Ready for fresh training!")
    
    # _assess_loss_behavior moved to training_utilities.py
    
    # print_configuration moved to training_utilities.py
    
    def get_detailed_loss_analysis(self, epoch: int, train_metrics: Dict[str, float], 
                                  val_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Get detailed loss analysis for the current epoch.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            
        Returns:
            Dictionary containing detailed analysis results
        """
        current_weights = self.loss_manager.get_all_weights()
        selected_methods = self.config.get('loss_analysis_methods', ['standard', 'constant_weight', 'pareto'])
        # Convert string method names to AnalysisMethod enums
        from loss_analysis_system import AnalysisMethod
        method_enums = [AnalysisMethod(method) for method in selected_methods]
        
        analysis_results = self.loss_analysis_system.analyze_epoch(
            epoch, train_metrics, val_metrics, current_weights, method_enums
        )
        
        return self.loss_analysis_system.get_analysis_summary(analysis_results)
    
    def print_loss_analysis_report(self, epoch: int, train_metrics: Dict[str, float], 
                                  val_metrics: Dict[str, float]):
        """
        Print a detailed loss analysis report.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
        """
        analysis_summary = self.get_detailed_loss_analysis(epoch, train_metrics, val_metrics)
        
        print(f"\n📊 DETAILED LOSS ANALYSIS REPORT - EPOCH {epoch+1}")
        print("=" * 60)
        
        # Overall health score
        overall_health = analysis_summary['overall_health_score']
        health_icon = "🟢" if overall_health > 0.7 else "🟡" if overall_health > 0.4 else "🔴"
        print(f"{health_icon} Overall Health Score: {overall_health:.3f}")
        
        # Method-specific results
        method_results = analysis_summary.get('method_results', {})
        for method_name, result in method_results.items():
            method_icon = "📈" if method_name == "standard" else "⚖️" if method_name == "constant_weight" else "🎯"
            print(f"\n{method_icon} {method_name.replace('_', ' ').title()} Analysis:")
            print(f"  Health Score: {result['health_score']:.3f}")
            
            if result['recommendations']:
                print(f"  Recommendations:")
                for rec in result['recommendations']:
                    print(f"    • {rec}")
        
        # Combined recommendations
        recommendations = analysis_summary.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Combined Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("=" * 60)
    
    def _apply_loss_analysis_decisions(self, analysis_results, epoch):
        """
        Apply in-training decisions based on loss analysis results.
        
        Args:
            analysis_results: Results from loss analysis system
            epoch: Current epoch number
        """
        from loss_analysis_system import AnalysisMethod
        
        # Check if MSE priority phase is active
        mse_priority_active = (hasattr(self.loss_manager, 'mse_priority_phase') and 
                              self.loss_manager.mse_priority_phase and 
                              epoch < self.loss_manager.mse_priority_epochs)
        
        # Acceleration decisions based on standard analysis
        if AnalysisMethod.STANDARD in analysis_results:
            standard_analysis = analysis_results[AnalysisMethod.STANDARD]
            is_stuck = standard_analysis.analysis_data.get('is_stuck', False)
            stuck_epochs = standard_analysis.analysis_data.get('stuck_epochs', 0)
            trend = standard_analysis.analysis_data.get('trend', 'unknown')
            
            # Apply acceleration based on loss analysis (more conservative during MSE priority)
            if mse_priority_active:
                # Only activate acceleration if really stuck during MSE priority
                if is_stuck and stuck_epochs >= 5 and not self.loss_manager.is_acceleration_active():
                    self.loss_manager.acceleration_active = True
                    self.loss_manager.stuck_epochs = stuck_epochs
                    print(f"  ⚡ Acceleration activated during MSE priority (stuck for {stuck_epochs} epochs)")
                elif not is_stuck and self.loss_manager.is_acceleration_active():
                    self.loss_manager.acceleration_active = False
                    self.loss_manager.stuck_epochs = 0
                    print(f"  ⚡ Acceleration deactivated during MSE priority (training recovered)")
            else:
                # Normal acceleration logic
                if is_stuck and not self.loss_manager.is_acceleration_active():
                    self.loss_manager.acceleration_active = True
                    self.loss_manager.stuck_epochs = stuck_epochs
                    print(f"  ⚡ Acceleration activated by loss analysis (stuck for {stuck_epochs} epochs)")
                elif not is_stuck and self.loss_manager.is_acceleration_active():
                    self.loss_manager.acceleration_active = False
                    self.loss_manager.stuck_epochs = 0
                    print(f"  ⚡ Acceleration deactivated by loss analysis (training recovered)")
        
        # Learning rate adjustments based on trend analysis
        if AnalysisMethod.STANDARD in analysis_results:
            standard_analysis = analysis_results[AnalysisMethod.STANDARD]
            trend = standard_analysis.analysis_data.get('trend', 'unknown')
            improvement_rate = standard_analysis.analysis_data.get('improvement_rate', 0)
            
            # Adjust learning rate based on trend (more conservative during MSE priority)
            if mse_priority_active:
                # Only reduce LR if really diverging during MSE priority
                if trend == 'worsening' and improvement_rate < -0.15:  # 15% worsening (vs 5%)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = current_lr * 0.9  # Smaller reduction (vs 0.8)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"  🔻 Learning rate reduced during MSE priority: {current_lr:.2e} → {new_lr:.2e}")
            else:
                # Normal LR adjustment logic
                if trend == 'worsening' and improvement_rate < -0.05:  # 5% worsening
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = current_lr * 0.8  # Reduce by 20%
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"  🔻 Learning rate reduced due to diverging trend: {current_lr:.2e} → {new_lr:.2e}")
                elif trend == 'plateaued' and improvement_rate > -0.001:  # Very small improvement
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = min(current_lr * 1.1, current_lr * 2.0)  # Increase by 10%, max 2x
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"  🔺 Learning rate increased due to plateau: {current_lr:.2e} → {new_lr:.2e}")
        
        # Weight adjustment suggestions based on Pareto analysis
        if AnalysisMethod.PARETO in analysis_results:
            pareto_analysis = analysis_results[AnalysisMethod.PARETO]
            is_pareto_optimal = pareto_analysis.analysis_data.get('is_pareto_optimal', False)
            objective_balance = pareto_analysis.analysis_data.get('objective_balance', 1.0)
            
            # Suggest weight adjustments if not Pareto optimal
            if not is_pareto_optimal and objective_balance < 0.5:
                print(f"  🎯 Pareto analysis suggests weight rebalancing (balance: {objective_balance:.2f})")
        
        # Early stopping suggestions based on constant weight analysis
        if AnalysisMethod.CONSTANT_WEIGHT in analysis_results:
            constant_analysis = analysis_results[AnalysisMethod.CONSTANT_WEIGHT]
            trend = constant_analysis.analysis_data.get('trend', 'unknown')
            stability = constant_analysis.analysis_data.get('stability', 0.5)
            
            # Suggest early stopping if reference loss is stable and high
            if trend == 'stable' and stability > 0.8:
                print(f"  ⏹️  Constant weight analysis suggests training may have converged")
    
    def _check_early_stopping(self, analysis_results, epoch, early_stopping_patience):
        """
        Check if training should stop early based on loss analysis.
        
        Args:
            analysis_results: Results from loss analysis system
            epoch: Current epoch number
            early_stopping_patience: Maximum patience for early stopping
            
        Returns:
            True if training should stop, False otherwise
        """
        from loss_analysis_system import AnalysisMethod
        
        # Use traditional patience counter as fallback
        if self.patience_counter >= early_stopping_patience:
            return True
        
        # Use loss analysis for more intelligent early stopping
        if analysis_results:
            # Check constant weight analysis for convergence
            if AnalysisMethod.CONSTANT_WEIGHT in analysis_results:
                constant_analysis = analysis_results[AnalysisMethod.CONSTANT_WEIGHT]
                trend = constant_analysis.analysis_data.get('trend', 'unknown')
                stability = constant_analysis.analysis_data.get('stability', 0.5)
                
                # Stop if reference loss is stable and high (converged)
                if trend == 'stable' and stability > 0.9:
                    print(f"  ⏹️  Early stopping: Reference loss converged (stability: {stability:.2f})")
                    return True
            
            # Check standard analysis for stuck training
            if AnalysisMethod.STANDARD in analysis_results:
                standard_analysis = analysis_results[AnalysisMethod.STANDARD]
                is_stuck = standard_analysis.analysis_data.get('is_stuck', False)
                stuck_epochs = standard_analysis.analysis_data.get('stuck_epochs', 0)
                
                # Stop if stuck for too long
                if is_stuck and stuck_epochs >= early_stopping_patience:
                    print(f"  ⏹️  Early stopping: Training stuck for {stuck_epochs} epochs")
                    return True
            
            # Check overall health score
            analysis_summary = self.loss_analysis_system.get_analysis_summary(analysis_results)
            overall_health = analysis_summary.get('overall_health_score', 0.5)
            
            # Stop if health is consistently poor
            if overall_health < 0.2 and epoch > 10:  # Only after 10 epochs
                print(f"  ⏹️  Early stopping: Poor training health ({overall_health:.2f})")
                return True
        
        return False
    
    def generate_samples(self, epoch, val_data, suffix=""):
        """Generate sample images."""
        if suffix:
            print(f"  🖼️  Generating sample images ({suffix})...")
        else:
            print(f"  🖼️  Generating sample images...")
        
        # Create sample_images directory
        sample_dir = "sample_images"
        os.makedirs(sample_dir, exist_ok=True)
        
        with torch.no_grad():
            # Generate random images
            z = torch.randn(8, self.config['embedding_size']).to(self.device)
            generated = self.model.dec(z)
            generated_np = generated.permute(0, 2, 3, 1).cpu().numpy()
            
            # Log to tensorboard
            if self.writer is not None:
                self.writer.log_images('Generated/Images', generated, epoch)
            
            # Get clean config name for filename
            config_name = self.config.get('config_name', 'unified')
            
            # Save sample images
            sample_path = os.path.join(sample_dir, f"{config_name}_epoch_{epoch+1:03d}_generated{suffix}.png")
            titles = [f"Epoch {epoch+1} - Sample {i+1}" for i in range(8)]
            display_image_grid(generated_np, 
                              titles=titles,
                              max_cols=4, 
                              figsize=(16, 8),
                              save_path=sample_path)
            print(f"  ✅ Sample images saved: {sample_path}")
            
            # Also generate reconstruction samples
            val_indices = torch.randperm(len(val_data))[:4]
            val_images = torch.stack([val_data[i] for i in val_indices]).to(self.device)
            z_mean, z_log_var, z = self.model.enc(val_images)
            reconstructed = self.model.dec(z)
            
            val_np = val_images.permute(0, 2, 3, 1).cpu().numpy()
            recon_np = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
            
            recon_path = os.path.join(sample_dir, f"{config_name}_epoch_{epoch+1:03d}_reconstruction{suffix}.png")
            display_comparison_grid(val_np, recon_np,
                                   titles=[f"Epoch {epoch+1} - Pair {i+1}" for i in range(4)],
                                   max_cols=2, 
                                   figsize=(12, 8),
                                   save_path=recon_path)
            print(f"  ✅ Reconstruction samples saved: {recon_path}")
    
    def train(self, resume=True):
        """Main training loop."""
        print("🚀 Starting Unified VAE Training")
        print("=" * 70)
        
        # Setup training environment
        train_loader, val_loader = self._setup_training_environment(resume)
        
        # Run training loop
        self._run_training_loop(train_loader, val_loader)
    
    def _setup_training_environment(self, resume=True):
        """Setup the training environment including model, data, and checkpoints."""
        # Print initial configuration
        self.utilities.print_configuration("Fresh")
        
        # Setup logging
        self.setup_logging()
        
        # Create model
        self.create_model()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Load data
        train_loader, val_loader = self.load_data()
        
        # Handle checkpoint loading/clearing
        config_name = self.config.get('config_name', 'unified')
        checkpoint_path = f"checkpoints/{config_name}_training_checkpoint.pth"
        if not resume:
            # Clear old checkpoints and model files for fresh start
            self.utilities.clear_old_files()
        elif resume and self.utilities.load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler)[0]:
            pass  # Configuration already printed
        
        return train_loader, val_loader
    
    def _run_training_loop(self, train_loader, val_loader):
        """Run the main training loop."""
        print(f"\n🎯 Starting training...")
        start_time = time.time()
        early_stopping_patience = 12
        
        # Use context manager for proper writer cleanup
        try:
            for epoch in range(self.start_epoch, self.config['max_epoch']):
                # Process a single epoch
                should_stop = self._process_epoch(epoch, train_loader, val_loader, start_time, early_stopping_patience)
                if should_stop:
                    break
            
            # Training completed successfully
            print(f"\n🎉 Training completed successfully!")
            print(f"📊 Total training time: {(time.time() - start_time)/3600:.2f} hours")
            print(f"🏆 Best validation loss: {self.best_ref_val:.6f}")
            
            # Generate final samples
            print(f"\n🎨 Generating final samples...")
            final_path = f"sample_images/final_samples_{self.config.get('config_name', 'unified')}.png"
            self.generate_samples(self.config['max_epoch']-1, val_loader.dataset, 
                                  save_path=final_path)
            print(f"  ✅ Final samples saved: {final_path}")
        
        finally:
            # Ensure writer is properly closed
            if self.writer is not None:
                self.writer.close()
    
    def _process_epoch(self, epoch, train_loader, val_loader, start_time, early_stopping_patience):
        """Process a single training epoch with all analysis and decisions."""
                # Update current epoch for weight scheduling
                self.loss_manager.set_epoch(epoch)
            
                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Generate mid-epoch samples for closer monitoring
                if epoch > 0 and epoch % 2 == 0:  # Every 2 epochs after the first
                    print(f"  🔍 Generating mid-epoch samples...")
                    self.generate_samples(epoch, val_loader.dataset, suffix="_mid_epoch")
                
                # Validation phase
                val_metrics = self.validate_epoch(val_loader, epoch)
            
        # Learning rate scheduling - use validation loss for basic scheduling
        val_loss = val_metrics.get('loss', 0)
        self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Logging
        self._log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr)
        
        # Get GPU statistics and print epoch summary
        gpu_stats = self._get_gpu_statistics()
        self._print_epoch_summary(epoch, train_metrics, val_metrics, current_lr, gpu_stats)
        
        # Handle acceleration state changes
        self._handle_acceleration_state_changes(early_stopping_patience)
        
        # Run loss analysis and apply decisions
        analysis_results = self._run_loss_analysis(epoch, train_metrics, val_metrics)
        
        # Save best model and checkpoint
        is_best = self._save_best_model(epoch, val_metrics, analysis_results)
        self.utilities.save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, train_metrics, val_metrics, is_best)
    
        # Generate sample images for every epoch
        self.generate_samples(epoch, val_loader.dataset)
        
        # Print detailed loss analysis report at configured interval
        analysis_interval = self.config.get('loss_analysis_interval', 5)
        if epoch % analysis_interval == 0 or epoch == self.config['max_epoch'] - 1:
            self.print_loss_analysis_report(epoch, train_metrics, val_metrics)
        
        # Early stopping check
        should_stop = self._check_early_stopping(analysis_results, epoch, early_stopping_patience)
        if should_stop:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            return True
        
        # Time estimation
        self._print_time_estimation(start_time, epoch)
        
        return False
    
    def _log_epoch_metrics(self, epoch, train_metrics, val_metrics, current_lr):
        """Log epoch metrics to tensorboard."""
                if self.writer is not None:
                    # Get current weights for logging
                    weights = self.loss_manager.get_all_weights()
                    
                    # Log all epoch metrics
                    self.writer.log_epoch_metrics(epoch, train_metrics, val_metrics, weights)
                    self.writer.log_learning_rate(epoch, current_lr)
                    self.writer.log_beta(epoch, self.loss_manager.get_weight('beta'))
                    
                    # Log training stage if using stage-based training
                    if self.config.get('stage_based_training', False):
                        stage_name = self.loss_manager.get_current_stage_name()
                        self.writer.log_stage(epoch, stage_name)
                    
                    # Log GPU memory usage
                    if torch.cuda.is_available():
                        gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                        gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                        gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100
                        self.writer.log_gpu_memory(epoch, gpu_memory_allocated, gpu_memory_total, gpu_utilization)
            
    def _get_gpu_statistics(self):
        """Get GPU memory statistics."""
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                    gpu_memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
                    gpu_memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
                    gpu_utilization = (gpu_memory_allocated / gpu_memory_total) * 100
                else:
                    gpu_memory_allocated = gpu_memory_reserved = gpu_memory_total = gpu_utilization = 0
                
        return {
            'allocated': gpu_memory_allocated,
            'reserved': gpu_memory_reserved,
            'total': gpu_memory_total,
            'utilization': gpu_utilization
        }
    
    def _print_epoch_summary(self, epoch, train_metrics, val_metrics, current_lr, gpu_stats):
        """Print epoch summary and control information."""
                # Print epoch summary using the centralized reporter
                summary = self.reporter.format_epoch_summary(
                    epoch=epoch,
                    max_epochs=self.config['max_epoch'],
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    current_lr=current_lr,
            gpu_memory_allocated=gpu_stats['allocated'],
            gpu_memory_total=gpu_stats['total'],
            gpu_utilization=gpu_stats['utilization']
                )
                print(summary)
                
                # Compact control summary: patience and acceleration state
                accel_info = self.loss_manager.get_acceleration_info()
                accel_on = 'on' if accel_info.get('active', False) else 'off'
                accel_factor = accel_info.get('factor', 1.0)
        print(f"  🔧 Control: Patience {self.patience_counter}/{12} | Accel: {accel_on} ({accel_factor:.1f}x)")
                
    def _handle_acceleration_state_changes(self, early_stopping_patience):
        """Handle acceleration state changes and reset patience if needed."""
                accel_active = self.loss_manager.is_acceleration_active()
                if accel_active != self.prev_accel_active:
                    self.patience_counter = 0
                    state = "activated" if accel_active else "deactivated"
                    print(f"  ♻️ Early-stopping patience reset ({state})")
                    # Optional: reduce LR when acceleration activates to help recovery
                    if accel_active and self.config.get('lr_on_acceleration', False):
                        factor = float(self.config.get('lr_accel_factor', 0.5))
                        lr_min = float(self.config.get('lr_min', 1e-6))
                        for g in self.optimizer.param_groups:
                            g['lr'] = max(lr_min, g['lr'] * factor)
                        new_lr = self.optimizer.param_groups[0]['lr']
                        print(f"  🔻 Learning rate reduced on acceleration: {new_lr:.2e}")
                    self.prev_accel_active = accel_active
            
    def _run_loss_analysis(self, epoch, train_metrics, val_metrics):
        """Run loss analysis and apply decisions."""
                # Comprehensive loss behavior assessment
                self.utilities.assess_loss_behavior(train_metrics, val_metrics, epoch)
                
        # Run loss analysis using the unified system
        current_weights = self.loss_manager.get_all_weights()
        selected_methods = self.config.get('loss_analysis_methods', ['standard', 'constant_weight', 'pareto'])
        # Convert string method names to AnalysisMethod enums
        from loss_analysis_system import AnalysisMethod
        method_enums = [AnalysisMethod(method) for method in selected_methods]
        
        # Prepare additional context for loss analysis using the centralized manager
        additional_context = self.loss_manager.priority_manager.get_context_for_analysis(epoch)
        
        analysis_results = self.loss_analysis_system.analyze_epoch(
            epoch, train_metrics, val_metrics, current_weights, method_enums, additional_context
        )
        
        # Use loss analysis for in-training decisions
        if analysis_results:
            # Get user-friendly summary
            user_summary = self.loss_analysis_system.get_user_friendly_summary(analysis_results)
            
            # Use loss analysis for in-training decisions
            self._apply_loss_analysis_decisions(analysis_results, epoch)
            
            # Show user-friendly analysis results
            active_methods = list(analysis_results.keys())
            method_icons = {
                'standard': '📈',
                'constant_weight': '⚖️', 
                'pareto': '🎯'
            }
            method_display = [f"{method_icons.get(method.value, '📊')}{method.value}" for method in active_methods]
            
            print(f"  📊 Loss Analysis ({', '.join(method_display)}): {user_summary['status']} ({user_summary['health_score']:.2f})")
            print(f"  💬 {user_summary['message']}")
            
            # Show priority actions
            if user_summary['priority_actions']:
                print(f"  🎯 Priority Actions:")
                for action in user_summary['priority_actions']:
                    priority_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(action['priority'], '⚪')
                    print(f"    {priority_icon} {action['action']} - {action['reason']}")
            
            # Show health indicators
            if user_summary['health_indicators']:
                print(f"  📋 Health Indicators: {'; '.join(user_summary['health_indicators'])}")
        
        return analysis_results
    
    def _save_best_model(self, epoch, val_metrics, analysis_results):
        """Save best model based on validation loss and analysis."""
        val_loss = val_metrics.get('loss', 0)
        is_best = val_loss < self.best_ref_val
        
        # Enhanced best model detection using loss analysis
        if analysis_results and AnalysisMethod.STANDARD in analysis_results:
            standard_analysis = analysis_results[AnalysisMethod.STANDARD]
            trend = standard_analysis.analysis_data.get('trend', 'unknown')
            improvement_rate = standard_analysis.analysis_data.get('improvement_rate', 0)
            
            # Consider it "best" if loss is lower OR if analysis shows improvement trend
            if not is_best and trend == 'improving' and improvement_rate > 0.01:
                is_best = True
                print(f"  ✅ Analysis-based best model! (trend: {trend}, rate: {improvement_rate:.3f})")
        
                if is_best:
            self.best_ref_val = val_loss
                    self.patience_counter = 0
            print(f"  ✅ New best model! (validation loss = {val_loss:.6f})")
                else:
                    self.patience_counter += 1
            print(f"  ⏳ No improvement ({self.patience_counter}/12)")
        
        return is_best
    
    def _print_time_estimation(self, start_time, epoch):
        """Print time estimation for remaining training."""
                elapsed_time = time.time() - start_time
                if epoch > 0:
                    avg_time_per_epoch = elapsed_time / (epoch + 1)
                    remaining_epochs = self.config['max_epoch'] - (epoch + 1)
                    estimated_remaining = remaining_epochs * avg_time_per_epoch
                    print(f"  ⏱️  Time: {elapsed_time/3600:.1f}h elapsed, ~{estimated_remaining/3600:.1f}h remaining")
            

def main():
    """Main function with command-line argument parsing."""
    # Load available presets dynamically from config
    config_loader = ConfigLoader('config/config_unified.json')
    # Filter out comment entries and other non-preset keys
    available_loss_presets = [k for k in config_loader.unified_config['loss_presets'].keys() if not k.startswith('_')]
    available_training_presets = [k for k in config_loader.unified_config['training_presets'].keys() if not k.startswith('_')]
    available_model_presets = [k for k in config_loader.unified_config['model_presets'].keys() if not k.startswith('_')]
    available_dataset_presets = [k for k in config_loader.unified_config['dataset_presets'].keys() if not k.startswith('_')]
    
    parser = argparse.ArgumentParser(description='Unified VAE Training Script')
    parser.add_argument('--loss-preset', choices=available_loss_presets, default='balanced',
                       help='Loss configuration preset (default: balanced)')
    parser.add_argument('--training-preset', choices=available_training_presets, default='standard_training',
                       help='Training configuration preset (default: standard_training)')
    parser.add_argument('--model-preset', choices=available_model_presets, default='standard',
                       help='Model configuration preset (default: standard)')
    parser.add_argument('--dataset-preset', choices=available_dataset_presets, default='celeba',
                       help='Dataset configuration preset (default: celeba)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start training from scratch (clear checkpoints)')
    parser.add_argument('--loss-analysis-interval', type=int, default=5,
                       help='Interval for detailed loss analysis reports (default: 5)')
    parser.add_argument('--enable-loss-analysis', action='store_true',
                       help='Enable loss analysis system')
    parser.add_argument('--loss-analysis-methods', nargs='+', 
                       choices=['standard', 'constant_weight', 'pareto'], 
                       default=['standard', 'constant_weight', 'pareto'],
                       help='Loss analysis methods to use (default: all)')
    parser.add_argument('--loss-analysis-preset', choices=['none', 'basic', 'standard', 'detailed', 'research'], 
                       default='standard',
                       help='Loss analysis configuration preset (default: standard)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader('config/config_unified.json')
    config = config_loader.get_config(
        loss_preset=args.loss_preset,
        training_preset=args.training_preset,
        model_preset=args.model_preset,
        dataset_preset=args.dataset_preset,
        loss_analysis_preset=args.loss_analysis_preset
    )
    
    # Override with command line arguments
    config['loss_analysis_interval'] = args.loss_analysis_interval
    config['enable_loss_analysis'] = args.enable_loss_analysis
    config['loss_analysis_methods'] = args.loss_analysis_methods
    
    # Create and run trainer
    trainer = UnifiedVAETrainer(config)
    trainer.train(resume=not args.no_resume)


if __name__ == "__main__":
    main()
        print(f"  🔧 Control: Patience {self.patience_counter}/{12} | Accel: {accel_on} ({accel_factor:.1f}x)")
    
    def _handle_acceleration_state_changes(self, early_stopping_patience):
        """Handle acceleration state changes and reset patience if needed."""
        accel_active = self.loss_manager.is_acceleration_active()
        if accel_active != self.prev_accel_active:
            self.patience_counter = 0
            state = "activated" if accel_active else "deactivated"
            print(f"  ♻️ Early-stopping patience reset ({state})")
            # Optional: reduce LR when acceleration activates to help recovery
            if accel_active and self.config.get('lr_on_acceleration', False):
                factor = float(self.config.get('lr_accel_factor', 0.5))
                lr_min = float(self.config.get('lr_min', 1e-6))
                for g in self.optimizer.param_groups:
                    g['lr'] = max(lr_min, g['lr'] * factor)
                new_lr = self.optimizer.param_groups[0]['lr']
                print(f"  🔻 Learning rate reduced on acceleration: {new_lr:.2e}")
            self.prev_accel_active = accel_active
    
    def _run_loss_analysis(self, epoch, train_metrics, val_metrics):
        """Run loss analysis and apply decisions."""
        # Comprehensive loss behavior assessment
        self.utilities.assess_loss_behavior(train_metrics, val_metrics, epoch)
        
        # Run loss analysis using the unified system
        current_weights = self.loss_manager.get_all_weights()
        selected_methods = self.config.get('loss_analysis_methods', ['standard', 'constant_weight', 'pareto'])
        # Convert string method names to AnalysisMethod enums
        from loss_analysis_system import AnalysisMethod
        method_enums = [AnalysisMethod(method) for method in selected_methods]
        
        # Prepare additional context for loss analysis using the centralized manager
        additional_context = self.loss_manager.priority_manager.get_context_for_analysis(epoch)
        
        analysis_results = self.loss_analysis_system.analyze_epoch(
            epoch, train_metrics, val_metrics, current_weights, method_enums, additional_context
        )
        
        # Use loss analysis for in-training decisions
        if analysis_results:
            # Get user-friendly summary
            user_summary = self.loss_analysis_system.get_user_friendly_summary(analysis_results)
            
            # Use loss analysis for in-training decisions
            self._apply_loss_analysis_decisions(analysis_results, epoch)
            
            # Show user-friendly analysis results
            active_methods = list(analysis_results.keys())
            method_icons = {
                'standard': '📈',
                'constant_weight': '⚖️', 
                'pareto': '🎯'
            }
            method_display = [f"{method_icons.get(method.value, '📊')}{method.value}" for method in active_methods]
            
            print(f"  📊 Loss Analysis ({', '.join(method_display)}): {user_summary['status']} ({user_summary['health_score']:.2f})")
            print(f"  💬 {user_summary['message']}")
            
            # Show priority actions
            if user_summary['priority_actions']:
                print(f"  🎯 Priority Actions:")
                for action in user_summary['priority_actions']:
                    priority_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(action['priority'], '⚪')
                    print(f"    {priority_icon} {action['action']} - {action['reason']}")
            
            # Show health indicators
            if user_summary['health_indicators']:
                print(f"  📋 Health Indicators: {'; '.join(user_summary['health_indicators'])}")
        
        return analysis_results
    
    def _save_best_model(self, epoch, val_metrics, analysis_results):
        """Save best model based on validation loss and analysis."""
        val_loss = val_metrics.get('loss', 0)
        is_best = val_loss < self.best_ref_val
        
        # Enhanced best model detection using loss analysis
        if analysis_results and AnalysisMethod.STANDARD in analysis_results:
            standard_analysis = analysis_results[AnalysisMethod.STANDARD]
            trend = standard_analysis.analysis_data.get('trend', 'unknown')
            improvement_rate = standard_analysis.analysis_data.get('improvement_rate', 0)
            
            # Consider it "best" if loss is lower OR if analysis shows improvement trend
            if not is_best and trend == 'improving' and improvement_rate > 0.01:
                is_best = True
                print(f"  ✅ Analysis-based best model! (trend: {trend}, rate: {improvement_rate:.3f})")
        
        if is_best:
            self.best_ref_val = val_loss
            self.patience_counter = 0
            print(f"  ✅ New best model! (validation loss = {val_loss:.6f})")
        else:
            self.patience_counter += 1
            print(f"  ⏳ No improvement ({self.patience_counter}/12)")
        
        return is_best
    
    def _print_time_estimation(self, start_time, epoch):
        """Print time estimation for remaining training."""
        elapsed_time = time.time() - start_time
        if epoch > 0:
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = self.config['max_epoch'] - (epoch + 1)
            estimated_remaining = remaining_epochs * avg_time_per_epoch
            print(f"  ⏱️  Time: {elapsed_time/3600:.1f}h elapsed, ~{estimated_remaining/3600:.1f}h remaining")


def main():
    """Main function with command-line argument parsing."""
    # Load available presets dynamically from config
    config_loader = ConfigLoader('config/config_unified.json')
    # Filter out comment entries and other non-preset keys
    available_loss_presets = [k for k in config_loader.unified_config['loss_presets'].keys() if not k.startswith('_')]
    available_training_presets = [k for k in config_loader.unified_config['training_presets'].keys() if not k.startswith('_')]
    available_model_presets = [k for k in config_loader.unified_config['model_presets'].keys() if not k.startswith('_')]
    available_dataset_presets = [k for k in config_loader.unified_config['dataset_presets'].keys() if not k.startswith('_')]
    
    parser = argparse.ArgumentParser(description='Unified VAE Training Script')
    parser.add_argument('--loss-preset', type=str, default='high_quality',
                       choices=available_loss_presets,
                       help='Loss function preset')
    parser.add_argument('--training-preset', type=str, default='fast_high_quality_training',
                       choices=available_training_presets,
                       help='Training configuration preset')
    parser.add_argument('--model-preset', type=str, default='fast_high_quality',
                       choices=available_model_presets,
                       help='Model architecture preset')
    parser.add_argument('--dataset-preset', type=str, default='full',
                       choices=available_dataset_presets,
                       help='Dataset size preset')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh training (ignore checkpoints)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (e.g., 128, 256, 384)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                       help='Override learning rate (e.g., 0.001, 0.0001, 0.00001)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--loss-analysis-interval', type=int, default=5,
                       help='Interval for detailed loss analysis reports (epochs)')
    parser.add_argument('--enable-loss-analysis', action='store_true',
                       help='Enable detailed loss analysis system')
    parser.add_argument('--loss-analysis-methods', nargs='+', 
                       choices=['standard', 'constant_weight', 'pareto'],
                       default=['standard', 'constant_weight', 'pareto'],
                       help='Loss analysis methods to use (default: all)')
    parser.add_argument('--loss-analysis-preset', type=str, default='standard',
                       choices=['none', 'basic', 'standard', 'detailed', 'research'],
                       help='Loss analysis configuration preset')
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader("config/config_unified.json")
    config = config_loader.get_config(
        loss_preset=args.loss_preset,
        training_preset=args.training_preset,
        model_preset=args.model_preset,
        dataset_preset=args.dataset_preset,
        loss_analysis_preset=args.loss_analysis_preset
    )
    
    # Override batch size if specified
    if args.batch_size:
        config['batch_size'] = args.batch_size
        config['_user_override_batch_size'] = True
        print(f"📊 Batch size overridden to: {args.batch_size}")
    
    # Override learning rate if specified
    if args.lr:
        config['lr'] = args.lr
        print(f"📈 Learning rate overridden to: {args.lr}")
    
    # Set config name for filenames (include both model and training presets)
    config['config_name'] = f"{args.model_preset}_{args.training_preset}"
    
    # Configure loss analysis
    if args.enable_loss_analysis or config.get('loss_analysis', {}).get('enabled', False):
        # Use preset configuration if available, otherwise use command line args
        if 'loss_analysis' in config and config['loss_analysis'].get('enabled', False):
            # Use preset configuration
            analysis_config = config['loss_analysis']
            print(f"📊 Loss analysis enabled via preset '{args.loss_analysis_preset}'")
            if 'methods' in analysis_config:
                print(f"📊 Analysis methods: {', '.join(analysis_config['methods'])}")
            if 'interval' in analysis_config:
                print(f"📊 Report interval: Every {analysis_config['interval']} epochs")
        else:
            # Fallback to command line configuration
            config['loss_analysis'] = config.get('loss_analysis', {})
            config['loss_analysis']['enable_logging'] = True
            config['loss_analysis']['save_plots'] = True
            config['loss_analysis_interval'] = args.loss_analysis_interval
            config['loss_analysis_methods'] = args.loss_analysis_methods
            print(f"📊 Loss analysis enabled via command line (interval: {args.loss_analysis_interval} epochs)")
            print(f"📊 Analysis methods: {', '.join(args.loss_analysis_methods)}")
    else:
        print(f"📊 Loss analysis disabled")
    
    # Configure GPU
    device = configure_gpu() if args.device == 'cuda' else torch.device(args.device)
    print(f"✅ Using device: {device}")
    
    # Create trainer and start training
    trainer = UnifiedVAETrainer(config, device)
    trainer.train(resume=not args.no_resume)


if __name__ == "__main__":
    main()
