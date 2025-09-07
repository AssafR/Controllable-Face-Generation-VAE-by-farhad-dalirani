#!/usr/bin/env python3
"""
Training Reporter System
Centralized, extensible training progress reporting and analysis.
"""

from typing import Dict, Any, List, Optional
from loss_weight_manager import LossWeightManager


class TrainingReporter:
    """Centralized training progress reporting and analysis."""
    
    def __init__(self, loss_manager: LossWeightManager):
        """
        Initialize the training reporter.
        
        Args:
            loss_manager: LossWeightManager instance for weight information
        """
        self.loss_manager = loss_manager
        
        # Define loss type icons and display names
        self.loss_icons = {
            'mse_weight': '📐',
            'l1_weight': '📏',
            'perceptual_weight': '🎨',
            'generation_weight': '🎯',
            'lpips_weight': '🔍',
            'beta': '🔄'
        }
        
        self.loss_display_names = {
            'mse_weight': 'MSE',
            'l1_weight': 'L1',
            'perceptual_weight': 'Perceptual',
            'generation_weight': 'Generation',
            'lpips_weight': 'LPIPS',
            'beta': 'Beta'
        }
        
        # Health status thresholds
        self.health_thresholds = {
            'perceptual': {'strong': 10, 'good': 5},
            'generation': {'strong': 5, 'good': 2},
            'kl': {'critical': 0.0001, 'weak': 0.01}
        }
    
    def get_loss_contribution_analysis(self, train_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive loss contribution analysis.
        
        Args:
            train_metrics: Dictionary of training metrics
            
        Returns:
            Dictionary with contribution analysis for each loss type
        """
        weights = self.loss_manager.get_all_weights()
        analysis = {}
        total_loss = train_metrics.get('loss', 0)
        
        # Calculate contribution for each loss type
        for loss_type, weight in weights.items():
            if weight > 0 and loss_type in train_metrics:
                raw_loss = train_metrics[loss_type]
                weighted_loss = raw_loss * weight
                contribution_percent = (weighted_loss / total_loss) * 100 if total_loss > 0 else 0
                
                analysis[loss_type] = {
                    'raw_value': raw_loss,
                    'weight': weight,
                    'weighted_value': weighted_loss,
                    'contribution_percent': contribution_percent
                }
        
        return analysis
    
    def get_health_status(self, train_metrics: Dict[str, float]) -> Dict[str, Dict[str, str]]:
        """
        Get health status for all loss components.
        
        Args:
            train_metrics: Dictionary of training metrics
            
        Returns:
            Dictionary with health status for each component
        """
        analysis = self.get_loss_contribution_analysis(train_metrics)
        health_status = {}
        
        # Perceptual loss health
        if 'perceptual_weight' in analysis:
            perc_contrib = analysis['perceptual_weight']['contribution_percent']
            thresholds = self.health_thresholds['perceptual']
            
            if perc_contrib >= thresholds['strong']:
                health_status['perceptual'] = {
                    'status': 'strong', 
                    'message': f'Strong (>{perc_contrib:.0f}%)'
                }
            elif perc_contrib >= thresholds['good']:
                health_status['perceptual'] = {
                    'status': 'good', 
                    'message': f'Good ({perc_contrib:.0f}%)'
                }
            else:
                health_status['perceptual'] = {
                    'status': 'weak', 
                    'message': f'Weak ({perc_contrib:.0f}%) - increase weight'
                }
        
        # Generation quality health
        if 'generation_weight' in analysis:
            gen_contrib = analysis['generation_weight']['contribution_percent']
            thresholds = self.health_thresholds['generation']
            
            if gen_contrib >= thresholds['strong']:
                health_status['generation'] = {
                    'status': 'strong', 
                    'message': f'Strong (>{gen_contrib:.0f}%)'
                }
            else:
                health_status['generation'] = {
                    'status': 'weak', 
                    'message': f'Weak ({gen_contrib:.0f}%) - increase weight'
                }
        
        # KL divergence health
        kl_value = train_metrics.get('kl', 0)
        thresholds = self.health_thresholds['kl']
        
        if kl_value < thresholds['critical']:
            health_status['kl'] = {
                'status': 'critical', 
                'message': 'POSTERIOR COLLAPSE!'
            }
        elif kl_value < thresholds['weak']:
            health_status['kl'] = {
                'status': 'weak', 
                'message': 'Weak - increase beta'
            }
        else:
            health_status['kl'] = {
                'status': 'healthy', 
                'message': 'Healthy'
            }
        
        return health_status
    
    def get_status_icon(self, status: str) -> str:
        """Get status icon based on health status."""
        status_icons = {
            'strong': '✅',
            'good': '🟢',
            'healthy': '✅',
            'weak': '⚠️',
            'critical': '🔴'
        }
        return status_icons.get(status, '❓')
    
    def get_loss_icon(self, loss_type: str) -> str:
        """Get icon for loss type."""
        return self.loss_icons.get(loss_type, '📊')
    
    def get_loss_display_name(self, loss_type: str) -> str:
        """Get display name for loss type."""
        return self.loss_display_names.get(loss_type, loss_type.replace('_', ' ').title())
    
    def get_type_label(self, info: Dict[str, Any]) -> str:
        """Get type label for weight info."""
        if info['type'] == 'stage-based':
            return f"stage-based ({info['stage']})"
        elif info['type'] == 'scheduled':
            # Use higher precision so small scheduled values (e.g., 0.001) are visible
            return f"scheduled ({info['start']:.4f} → {info['end']:.4f})"
        else:
            return 'fixed'
    
    def format_epoch_summary(self, epoch: int, max_epochs: int, train_metrics: Dict[str, float], 
                            val_metrics: Dict[str, float], current_lr: float, 
                            gpu_memory_allocated: float, gpu_memory_total: float, 
                            gpu_utilization: float) -> str:
        """
        Format complete epoch summary.
        
        Args:
            epoch: Current epoch number
            max_epochs: Maximum number of epochs
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            current_lr: Current learning rate
            gpu_memory_allocated: Allocated GPU memory in GB
            gpu_memory_total: Total GPU memory in GB
            gpu_utilization: GPU utilization percentage
            
        Returns:
            Formatted epoch summary string
        """
        lines = []
        
        # Header
        lines.append(f"\n{'='*60}")
        lines.append(f"📊 EPOCH {epoch+1}/{max_epochs} SUMMARY")
        lines.append(f"{'='*60}")
        
        # Main metrics
        lines.append(f"🎯 TOTAL LOSS: {train_metrics['loss']:.4f} (Train) | {val_metrics['loss']:.4f} (Val)")
        lines.append(f"📈 LEARNING RATE: {current_lr:.2e}")
        
        # Training stage
        if self.loss_manager.stage_based:
            current_stage = self.loss_manager.get_current_stage_name()
            lines.append(f"🎭 TRAINING STAGE: {current_stage.upper()}")
        
        # Beta and perceptual weights
        beta_info = self.loss_manager.get_weight_info('beta')
        perceptual_info = self.loss_manager.get_weight_info('perceptual_weight')
        
        lines.append(f"🔄 BETA: {beta_info['value']:.3f} ({self.get_type_label(beta_info)})")
        lines.append(f"🎨 PERCEPTUAL: {perceptual_info['value']:.3f} ({self.get_type_label(perceptual_info)})")
        
        # GPU memory
        lines.append(f"💾 GPU MEMORY: {gpu_memory_allocated:.1f}GB / {gpu_memory_total:.1f}GB ({gpu_utilization:.0f}%)")
        
        # Current weights
        lines.append(f"\n⚖️  CURRENT WEIGHTS:")
        lines.append(f"{'─'*40}")
        
        all_weights = self.loss_manager.get_all_weights()
        weight_info = {loss_type: self.loss_manager.get_weight_info(loss_type) for loss_type in all_weights.keys()}
        
        for loss_type, weight in all_weights.items():
            if weight > 0:  # Only show non-zero weights
                info = weight_info[loss_type]
                icon = self.get_loss_icon(loss_type)
                display_name = self.get_loss_display_name(loss_type)
                type_label = self.get_type_label(info)
                lines.append(f"  {icon} {display_name:<12} {weight:.3f} ({type_label})")
        
        # Loss breakdown
        lines.append(f"\n🔍 LOSS BREAKDOWN:")
        lines.append(f"{'─'*40}")
        
        health_status = self.get_health_status(train_metrics)
        total_loss = train_metrics.get('loss', 0)
        
        # Show all loss components with proper weighting
        loss_components = {
            'mse': ('📐', 'MSE', 'mse_weight'),
            'l1': ('📏', 'L1', 'l1_weight'),
            'perceptual': ('🎨', 'Perceptual', 'perceptual_weight'),
            'generation_quality': ('🎯', 'Generation', 'generation_weight')
        }
        
        # Display each loss component with proper weighting
        for loss_key, (icon, display_name, weight_key) in loss_components.items():
            if loss_key in train_metrics:
                raw_value = train_metrics[loss_key]
                weight = self.loss_manager.get_weight(weight_key)
                weighted_value = raw_value * weight
                contribution = (weighted_value / total_loss) * 100 if total_loss > 0 else 0
                
                # Get health status for this loss type
                if loss_key in health_status:
                    status_icon = self.get_status_icon(health_status[loss_key]['status'])
                    lines.append(f"  {status_icon} {icon} {display_name:<12} {raw_value:.4f} (raw) × {weight:.3f} = {weighted_value:.4f} ({contribution:.0f}% of total)")
                else:
                    lines.append(f"  ✅ {icon} {display_name:<12} {raw_value:.4f} (raw) × {weight:.3f} = {weighted_value:.4f} ({contribution:.0f}% of total)")
        
        # Show reconstruction total
        if 'recon_loss' in train_metrics:
            recon_value = train_metrics['recon_loss']
            recon_contrib = (recon_value / total_loss) * 100 if total_loss > 0 else 0
            lines.append(f"  ✅ 🔄 Recon Total    {recon_value:.4f} ({recon_contrib:.0f}% of total)")
        
        # Update adaptive MSE/L1 control
        if 'mse' in train_metrics and 'l1' in train_metrics:
            mse_value = train_metrics['mse']
            l1_value = train_metrics['l1']
            mse_weight = self.loss_manager.get_weight('mse_weight')
            l1_weight = self.loss_manager.get_weight('l1_weight')
            mse_weighted = mse_value * mse_weight
            l1_weighted = l1_value * l1_weight
            mse_contrib = (mse_weighted / total_loss) * 100 if total_loss > 0 else 0
            l1_contrib = (l1_weighted / total_loss) * 100 if total_loss > 0 else 0
            
            # Update adaptive MSE/L1 control
            self.loss_manager.update_mse_l1_contribution(mse_contrib / 100.0, l1_contrib / 100.0)
        
        # Check for stuck training and accelerate scheduled updates if needed
        acceleration_applied = self.loss_manager.check_stuck_training(total_loss)
        if acceleration_applied:
            print(f"  ⚡ Schedule acceleration activated - weights will update faster")
        
        # Show acceleration status in summary
        if self.loss_manager.is_acceleration_active():
            accel_info = self.loss_manager.get_acceleration_info()
            print(f"  ⚡ Acceleration: {accel_info['factor']:.1f}x speed (stuck {accel_info['stuck_epochs']}/{accel_info['patience']} epochs)")
        
        # KL divergence - special handling with beta weight
        kl_value = train_metrics.get('kl', 0)
        beta_weight = self.loss_manager.get_weight('beta')
        kl_weighted = kl_value * beta_weight
        kl_contrib = (kl_weighted / total_loss) * 100 if total_loss > 0 else 0
        
        # Update adaptive KL control
        self.loss_manager.update_kl_contribution(kl_contrib / 100.0)  # Convert percentage to fraction
        
        if 'kl' in health_status:
            status_icon = self.get_status_icon(health_status['kl']['status'])
            lines.append(f"  {status_icon} 🔄 KL:           {kl_value:.4f} (raw) × {beta_weight:.3f} = {kl_weighted:.4f} ({kl_contrib:.0f}% of total)")
        else:
            lines.append(f"  ✅ 🔄 KL:           {kl_value:.4f} (raw) × {beta_weight:.3f} = {kl_weighted:.4f} ({kl_contrib:.0f}% of total)")
        
        # Quick health check
        lines.append(f"\n🏥 QUICK HEALTH CHECK:")
        lines.append(f"{'─'*40}")
        
        for loss_type, status in health_status.items():
            status_icon = self.get_status_icon(status['status'])
            lines.append(f"  {status_icon} {loss_type.title()}: {status['message']}")
        
        return '\n'.join(lines)
    
    def add_loss_type(self, loss_type: str, icon: str, display_name: str) -> None:
        """
        Add a new loss type to the reporter.
        
        Args:
            loss_type: Name of the loss type
            icon: Icon to display for this loss type
            display_name: Human-readable display name
        """
        self.loss_icons[loss_type] = icon
        self.loss_display_names[loss_type] = display_name
    
    def update_health_thresholds(self, loss_type: str, thresholds: Dict[str, float]) -> None:
        """
        Update health thresholds for a loss type.
        
        Args:
            loss_type: Type of loss to update thresholds for
            thresholds: Dictionary of threshold values
        """
        self.health_thresholds[loss_type] = thresholds
