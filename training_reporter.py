#!/usr/bin/env python3
"""
Training Reporter System
Centralized, extensible training progress reporting and analysis.
"""

from typing import Dict, Any, List, Optional
import torch
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
        # Track last printed strategy phase to avoid spamming
        self._last_strategy_phase_name: Optional[str] = None
    
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
            label = f"scheduled ({info['start']:.4f} → {info['end']:.4f})"
            # Append progress percentage through warmup if available
            if 'progress' in info:
                pct = int(round(info['progress'] * 100))
                label += f", {pct}%"
            # For cyclical KL show phase up/down and position
            if 'phase' in info:
                phase = info['phase']
                if 'cycle_pos' in info and 'cycle_period' in info:
                    label += f", {phase} (pos {info['cycle_pos']}/{info['cycle_period']})"
                else:
                    label += f", {phase}"
            return label
        else:
            return 'fixed'
    
    def _get_vgg_loss_type(self) -> str:
        """Get the VGG loss type being used (single source of truth)."""
        try:
            # Get loss config from the loss manager
            loss_config = self.loss_manager.loss_calculator.loss_config
            
            # Check if perceptual loss is enabled
            if not loss_config.get('use_perceptual_loss', False):
                return None
            
            # Determine VGG type based on configuration flags (same logic as startup)
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
        except Exception:
            return None
    
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
        
        # Training stage and strategy phase
        if self.loss_manager.stage_based:
            current_stage = self.loss_manager.get_current_stage_name()
            lines.append(f"🎭 TRAINING STAGE: {current_stage.upper()}")
        # Strategy phase from centralized controller
        try:
            strategy = getattr(self.loss_manager, 'strategy_controller', None)
            if strategy is not None:
                s_info = strategy.get_display_info()
                if s_info.get('cycle_enabled'):
                    phase = s_info.get('phase', 'none')
                    pos = s_info.get('phase_pos', 0)
                    period = s_info.get('cycle_period', 0)
                    # Summarize boosts/reductions for this phase and render inline
                    try:
                        effect_names = {
                            'mse_weight': 'MSE',
                            'l1_weight': 'L1',
                            'perceptual_weight': 'Perceptual',
                            'generation_weight': 'Generation',
                            'beta': 'Beta',
                        }
                        ups, downs, neutrals = [], [], []
                        for key in ['mse_weight', 'l1_weight', 'perceptual_weight', 'generation_weight', 'beta']:
                            mult = getattr(strategy, 'get_multipliers')(key).get('cycle', 1.0)
                            name = effect_names[key]
                            if mult > 1.05:
                                ups.append(name)
                            elif mult < 0.95:
                                downs.append(name)
                            else:
                                neutrals.append(name)
                        summary_parts = []
                        if ups:
                            summary_parts.append(f"↑ {' '.join(ups)}")
                        if downs:
                            summary_parts.append(f"↓ {' '.join(downs)}")
                        if neutrals:
                            summary_parts.append(f"= {' '.join(neutrals)}")
                        if summary_parts:
                            lines.append(f"🧭 STRATEGY (this epoch): {phase} (pos {pos}/{period}) — {' | '.join(summary_parts)}")
                        else:
                            lines.append(f"🧭 STRATEGY (this epoch): {phase} (pos {pos}/{period})")
                    except Exception:
                        lines.append(f"🧭 STRATEGY (this epoch): {phase} (pos {pos}/{period})")
                    # Verbose phase header printed once per phase switch
                    if phase != self._last_strategy_phase_name:
                        self._last_strategy_phase_name = phase
                        lines.append(f"\n🧪 PHASE STRATEGY BREAKDOWN ({phase.upper()}):")
                        lines.append(f"{'─'*60}")
                        # Show base≈ × multipliers = final for key losses
                        key_losses = ['mse_weight', 'l1_weight', 'perceptual_weight', 'generation_weight', 'beta']
                        current_weights = self.loss_manager.get_all_weights()
                        for lt in key_losses:
                            if lt in current_weights:
                                final_val = float(current_weights[lt])
                                mults = strategy.get_multipliers(lt)
                                combined = max(1e-12, mults.get('combined', 1.0))
                                base_est = final_val / combined
                                name = self.get_loss_display_name(lt)
                                lines.append(
                                    f"  {name:<12} base≈{base_est:.3f} × pri×{mults['priority']:.2f} × cyc×{mults['cycle']:.2f} = {final_val:.3f}"
                                )
        except Exception:
            pass
        
        # Beta and perceptual weights
        beta_info = self.loss_manager.get_weight_info('beta')
        perceptual_info = self.loss_manager.get_weight_info('perceptual_weight')
        
        lines.append(f"🔄 BETA: {beta_info['value']:.3f} ({self.get_type_label(beta_info)})")
        
        # Perceptual with VGG type
        vgg_type = self._get_vgg_loss_type()
        if vgg_type:
            # Extract the type from "VGG Full (12 layers)" -> "full"
            vgg_type_short = vgg_type.split('(')[0].split()[-1].lower()
            lines.append(f"🎨 PERCEPTUAL: {perceptual_info['value']:.3f} ({self.get_type_label(perceptual_info)}) [{vgg_type_short}]")
        else:
            lines.append(f"🎨 PERCEPTUAL: {perceptual_info['value']:.3f} ({self.get_type_label(perceptual_info)})")
        
        # GPU memory
        lines.append(f"💾 GPU MEMORY: {gpu_memory_allocated:.1f}GB / {gpu_memory_total:.1f}GB ({gpu_utilization:.0f}%)")
        
        # Current weights
        lines.append(f"\n⚖️  CURRENT WEIGHTS:")
        lines.append(f"{'─'*40}")
        
        all_weights = self.loss_manager.get_all_weights()
        weight_info = {loss_type: self.loss_manager.get_weight_info(loss_type) for loss_type in all_weights.keys()}
        
        # Check for active priority phases using the centralized manager
        priority_info = self.loss_manager.priority_manager.get_display_info(epoch)
        
        for loss_type, info in priority_info.items():
            if info['is_active']:
                lines.append(f"  🎯 {info['message']}")
                lines.append(f"  📊 Other objectives suppressed during {loss_type.upper()} priority phase")
            elif info['is_transition']:
                lines.append(f"  🔄 {info['message']}")
                lines.append(f"  📊 Other objectives gradually ramping up")
        
        for loss_type, weight in all_weights.items():
            if weight > 0:  # Only show non-zero weights
                info = weight_info[loss_type]
                icon = self.get_loss_icon(loss_type)
                display_name = self.get_loss_display_name(loss_type)
                type_label = self.get_type_label(info)
                
                # Add priority indicators using the centralized manager
                priority_info = self.loss_manager.priority_manager.get_display_info(epoch)
                for loss_type_name, info in priority_info.items():
                    if info['is_active'] and loss_type == f"{loss_type_name}_weight":
                        type_label += f" (PRIORITY ×{info['multiplier']:.1f})"
                
                # Strategy breakdown (base × multipliers = final)
                base_label = 'scheduled'
                if info['type'] == 'stage-based':
                    base_label = f"stage:{info['stage']}"
                # Estimate base from final / combined multiplier (safe if >0)
                try:
                    strategy = getattr(self.loss_manager, 'strategy_controller', None)
                    if strategy is not None:
                        mults = strategy.get_multipliers(loss_type)
                        combined = max(1e-12, mults.get('combined', 1.0))
                        base_est = weight / combined
                        lines.append(
                            f"  {icon} {display_name:<12} {weight:.3f} ({type_label}) => base≈{base_est:.3f} × pri×{mults['priority']:.2f} × cyc×{mults['cycle']:.2f}"
                        )
                    else:
                        lines.append(f"  {icon} {display_name:<12} {weight:.3f} ({type_label})")
                except Exception:
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
        
        # Get VGG type for perceptual loss display
        vgg_type = self._get_vgg_loss_type()
        vgg_type_short = ""
        if vgg_type:
            vgg_type_short = f" ({vgg_type.split('(')[0].split()[-1].lower()})"
        
        # Display each loss component with proper weighting
        for loss_key, (icon, display_name, weight_key) in loss_components.items():
            if loss_key in train_metrics:
                raw_value = train_metrics[loss_key]
                weight = self.loss_manager.get_weight(weight_key)
                weighted_value = raw_value * weight
                contribution = (weighted_value / total_loss) * 100 if total_loss > 0 else 0
                
                # Add VGG type to perceptual loss display name
                display_name_with_type = display_name
                if loss_key == 'perceptual' and vgg_type_short:
                    display_name_with_type = f"{display_name}{vgg_type_short}"
                
                # Get health status for this loss type
                if loss_key in health_status:
                    status_icon = self.get_status_icon(health_status[loss_key]['status'])
                    lines.append(f"  {status_icon} {icon} {display_name_with_type:<12} {raw_value:.4f} (raw) × {weight:.3f} = {weighted_value:.4f} ({contribution:.0f}% of total)")
                else:
                    lines.append(f"  ✅ {icon} {display_name_with_type:<12} {raw_value:.4f} (raw) × {weight:.3f} = {weighted_value:.4f} ({contribution:.0f}% of total)")
        
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
        
        # Note: Stuck detection and acceleration are now handled by the loss analysis system
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
        
        # Quick health check and strategy recommendations
        lines.append(f"\n🏥 QUICK HEALTH CHECK:")
        lines.append(f"{'─'*40}")
        # Show early-stopping controls summary inline for clarity
        try:
            cfg = getattr(self.loss_manager, 'config', {})
            patience = int(cfg.get('early_stopping_patience', 12))
            conv_disabled = bool(cfg.get('disable_convergence_stop', False))
            conv_status = 'off' if conv_disabled else 'on (phase-aware)'
            stuck_disabled = bool(cfg.get('disable_stuck_stop', True))
            stuck_status = 'off' if stuck_disabled else 'on'
            lines.append(f"  🔧 Control: Patience {patience} | Convergence stop: {conv_status} | Stuck stop: {stuck_status}")
        except Exception:
            pass
        
        for loss_type, status in health_status.items():
            status_icon = self.get_status_icon(status['status'])
            lines.append(f"  {status_icon} {loss_type.title()}: {status['message']}")
        # Strategy-linked recommendations
        try:
            strategy = getattr(self.loss_manager, 'strategy_controller', None)
            if strategy is not None:
                rec = strategy.get_recommendations()
                for msg in rec.get('messages', []):
                    lines.append(f"  💡 {msg}")
        except Exception:
            pass
        
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
