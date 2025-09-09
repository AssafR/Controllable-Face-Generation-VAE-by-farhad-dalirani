#!/usr/bin/env python3
"""
Unified Loss Weight Management System
Centralized, consistent scheduling for all loss weights (beta, perceptual, etc.)
"""

from typing import Dict, Any, List, Optional
import math
from mse_priority_manager import LossPriorityManager
from training_strategies import StrategyController


class LossWeightManager:
    """
    Unified loss weight management with consistent scheduling patterns.
    
    This class provides a single, consistent interface for all weight scheduling:
    - Beta (KL divergence weight)
    - Perceptual loss weight
    - Generation quality weight
    - All other loss weights
    
    All scheduling follows the same pattern:
    - Linear interpolation between start and end values
    - Configurable warmup periods
    - Consistent API for all weight types
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the loss weight manager.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.current_epoch = 0
        self.max_epochs = config.get('max_epoch', 100)
        self.loss_config = config.get('loss_config', {})
        
        # Define all supported loss types and their default values
        self.supported_losses = {
            'mse_weight': 0.0,
            'l1_weight': 0.0,
            'perceptual_weight': 0.0,
            'generation_weight': 0.0,
            'lpips_weight': 0.0,
            'beta': 1.0
        }
        
        # Adaptive KL control parameters
        self.adaptive_kl = config.get('adaptive_kl_control', False)
        self.kl_target_min = config.get('kl_target_min', 0.10)  # 10% minimum
        self.kl_target_max = config.get('kl_target_max', 0.20)  # 20% maximum
        self.kl_adjustment_rate = config.get('kl_adjustment_rate', 0.1)  # 10% adjustment per epoch
        self.kl_initial_beta = config.get('kl_initial_beta', 0.001)  # Start with small beta
        # Optional delay (in epochs) before applying any KL, to avoid early collapse
        self.kl_delay_epochs = config.get('kl_delay_epochs', 0)
        self.current_beta = self.kl_initial_beta if self.adaptive_kl else 0.0
        self.kl_history = []
        # Optional cyclical KL annealing (Bowman et al., 2016) with gentle rising floor
        # Reference: Bowman, Samuel R., et al. "Generating Sentences from a Continuous Space." CoNLL 2016.
        self.kl_cycle_enabled = config.get('kl_cycle_enabled', False)
        self.kl_cycle_period = config.get('kl_cycle_period', 8)  # epochs per cycle
        self.kl_cooldown_epochs = config.get('kl_cooldown_epochs', 2)  # early epochs in cycle for cooldown
        self.kl_cooldown_reduction = config.get('kl_cooldown_reduction', 0.3)  # reduce beta by 30% during cooldown
        self.kl_beta_floor = config.get('kl_beta_floor', self.kl_initial_beta)  # minimum beta floor
        self.kl_floor_growth = config.get('kl_floor_growth', 1.02)  # floor grows 2% per cycle
        
        # Adaptive MSE/L1 control parameters
        self.adaptive_mse_l1 = config.get('adaptive_mse_l1_control', False)
        self.mse_target_min = config.get('mse_target_min', 0.60)  # 60% minimum for MSE
        self.mse_target_max = config.get('mse_target_max', 0.80)  # 80% maximum for MSE
        self.l1_target_min = config.get('l1_target_min', 0.10)  # 10% minimum for L1
        self.l1_target_max = config.get('l1_target_max', 0.30)  # 30% maximum for L1
        self.mse_l1_adjustment_rate = config.get('mse_l1_adjustment_rate', 0.1)  # 10% adjustment per epoch
        self.current_mse_weight = self.loss_config.get('mse_weight', 1.0)
        self.current_l1_weight = self.loss_config.get('l1_weight', 0.1)
        self.mse_history = []
        self.l1_history = []
        
        # Stuck training detection and schedule acceleration
        # NOTE: Stuck detection is now handled by the loss analysis system
        self.stuck_detection = False  # Disabled - use loss analysis system instead
        self.stuck_patience = config.get('stuck_patience', 5)  # epochs without improvement
        self.acceleration_factor = config.get('acceleration_factor', 2.0)  # 2x speed when stuck
        self.best_loss = float('inf')
        self.stuck_epochs = 0
        self.acceleration_active = False
        self.original_epoch = 0  # Track original epoch for schedule calculations
        
        # Stage-based training configuration
        self.stage_based = config.get('stage_based_training', False)
        self.stages = config.get('stages', {})
        
        # MSE priority phase configuration
        self.mse_priority_phase = config.get('mse_priority_phase', False)
        self.mse_priority_epochs = config.get('mse_priority_epochs', 10)
        self.mse_priority_multiplier = config.get('mse_priority_multiplier', 2.0)
        
        # Initialize loss priority manager and centralized strategy controller
        self.priority_manager = LossPriorityManager(self)
        self.strategy_controller = StrategyController(config, self.priority_manager)
        
        # Cycle strategy is fully centralized in StrategyController
    
    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for weight calculations."""
        self.current_epoch = epoch
        self.original_epoch = epoch
        # Inform centralized strategy controller
        self.strategy_controller.set_epoch(epoch)
        # On cycle boundary, gently raise the KL beta floor
        if self.kl_cycle_enabled and self.current_epoch > 0 and (self.current_epoch % max(self.kl_cycle_period, 1) == 0):
            self.kl_beta_floor = min(1.0, self.kl_beta_floor * self.kl_floor_growth)
        
        # Cycle phase handled by StrategyController
    
    def update_kl_contribution(self, kl_contribution: float) -> None:
        """
        Update KL contribution for adaptive control.
        
        Args:
            kl_contribution: KL loss contribution as percentage of total loss
        """
        if not self.adaptive_kl:
            return
            
        self.kl_history.append(kl_contribution)
        
        # Keep only recent history (last 5 epochs)
        if len(self.kl_history) > 5:
            self.kl_history = self.kl_history[-5:]
        
        # Calculate average KL contribution over recent epochs
        avg_kl_contrib = sum(self.kl_history) / len(self.kl_history)
        
        # Adjust beta weight based on KL contribution
        adjustment_rate = self.kl_adjustment_rate
        # Under acceleration, speed reductions (when KL too high) but soften increases (when KL too low)
        if self.acceleration_active:
            if avg_kl_contrib > self.kl_target_max:
                adjustment_rate *= self.acceleration_factor
            else:
                adjustment_rate /= max(self.acceleration_factor, 1.0)
        
        if avg_kl_contrib > self.kl_target_max:
            # KL too high, reduce beta
            self.current_beta *= (1.0 - adjustment_rate)
        elif avg_kl_contrib < self.kl_target_min:
            # KL too low, increase beta (possibly softened during acceleration)
            self.current_beta *= (1.0 + adjustment_rate)
        
        # Apply cyclical cooldown early in each cycle to allow reconstruction to recover
        if self.kl_cycle_enabled and self.current_epoch >= self.kl_delay_epochs:
            cycle_pos = self.current_epoch % max(self.kl_cycle_period, 1)
            if cycle_pos < self.kl_cooldown_epochs:
                self.current_beta *= (1.0 - self.kl_cooldown_reduction)

        # Enforce KL beta floor and bounds
        if self.kl_cycle_enabled:
            self.current_beta = max(self.kl_beta_floor, self.current_beta)
        self.current_beta = max(0.0, min(self.current_beta, 1.0))
    
    def update_mse_l1_contribution(self, mse_contribution: float, l1_contribution: float) -> None:
        """
        Update MSE and L1 contributions for adaptive control.
        
        Args:
            mse_contribution: MSE loss contribution as percentage of total loss
            l1_contribution: L1 loss contribution as percentage of total loss
        """
        if not self.adaptive_mse_l1:
            return
            
        self.mse_history.append(mse_contribution)
        self.l1_history.append(l1_contribution)
        
        # Keep only recent history (last 5 epochs)
        if len(self.mse_history) > 5:
            self.mse_history = self.mse_history[-5:]
            self.l1_history = self.l1_history[-5:]
        
        # Calculate average contributions over recent history
        avg_mse_contrib = sum(self.mse_history) / len(self.mse_history)
        avg_l1_contrib = sum(self.l1_history) / len(self.l1_history)
        
        # Adjust MSE weight based on its contribution
        adjustment_rate = self.mse_l1_adjustment_rate
        if self.acceleration_active:
            adjustment_rate *= self.acceleration_factor
            
        if avg_mse_contrib > self.mse_target_max:
            # MSE too high, reduce MSE weight
            self.current_mse_weight *= (1.0 - adjustment_rate)
        elif avg_mse_contrib < self.mse_target_min:
            # MSE too low, increase MSE weight
            self.current_mse_weight *= (1.0 + adjustment_rate)
        
        # Adjust L1 weight based on its contribution
        if avg_l1_contrib > self.l1_target_max:
            # L1 too high, reduce L1 weight
            self.current_l1_weight *= (1.0 - adjustment_rate)
        elif avg_l1_contrib < self.l1_target_min:
            # L1 too low, increase L1 weight
            self.current_l1_weight *= (1.0 + adjustment_rate)
        
        # Keep weights within reasonable bounds
        self.current_mse_weight = max(0.1, min(self.current_mse_weight, 2.0))
        self.current_l1_weight = max(0.01, min(self.current_l1_weight, 1.0))
    
    def check_stuck_training(self, current_loss: float) -> bool:
        """
        DEPRECATED: Check if training is stuck and accelerate scheduled updates if needed.
        
        This method is deprecated. Stuck detection is now handled by the loss analysis system.
        Use the loss analysis system for all training decisions.
        
        Args:
            current_loss: Current epoch's total loss
            
        Returns:
            True if acceleration was applied, False otherwise
        """
        # This method is deprecated - stuck detection now handled by loss analysis system
        return False
    
    def get_effective_epoch(self) -> int:
        """
        Get the effective epoch for schedule calculations.
        If acceleration is active, use accelerated epoch progression.
        """
        if self.acceleration_active:
            # Accelerate the schedule by the acceleration factor
            return int(self.original_epoch * self.acceleration_factor)
        return self.original_epoch
    
    def is_acceleration_active(self) -> bool:
        """Check if schedule acceleration is currently active."""
        return self.acceleration_active
    
    def get_acceleration_info(self) -> dict:
        """Get information about current acceleration status."""
        return {
            'active': self.acceleration_active,
            'factor': self.acceleration_factor if self.acceleration_active else 1.0,
            'stuck_epochs': self.stuck_epochs,
            'patience': self.stuck_patience
        }
    
    def get_weight(self, loss_type: str) -> float:
        """
        Get current weight for any loss type with unified scheduling.
        
        Args:
            loss_type: Type of loss weight to retrieve
            
        Returns:
            Current weight value
            
        Raises:
            ValueError: If loss_type is not supported
        """
        if loss_type not in self.supported_losses:
            raise ValueError(f"Unsupported loss type: {loss_type}. Supported types: {list(self.supported_losses.keys())}")
        
        if loss_type == 'beta' and self.adaptive_kl:
            # Respect delay: return 0 until delay period has passed
            if self.current_epoch < getattr(self, 'kl_delay_epochs', 0):
                return 0.0
            # Apply floor when cyclical annealing is enabled
            if self.kl_cycle_enabled:
                return max(self.kl_beta_floor, self.current_beta)
            return self.current_beta
        
        if loss_type == 'mse_weight' and self.adaptive_mse_l1:
            return self.current_mse_weight
        
        if loss_type == 'l1_weight' and self.adaptive_mse_l1:
            return self.current_l1_weight
        
        # Base from schedule or stage-based
        if self.stage_based:
            base_value = self._get_stage_based_weight(loss_type)
        else:
            base_value = self._get_scheduled_weight(loss_type)

        # Apply all strategy multipliers from the centralized controller
        return self.strategy_controller.apply_multipliers(loss_type, base_value)
    
    def _get_stage_based_weight(self, loss_type: str) -> float:
        """Get weight value based on current training stage."""
        # Find the current stage
        for stage_name, stage_config in self.stages.items():
            start_epoch, end_epoch = stage_config['epochs']
            if start_epoch <= self.current_epoch < end_epoch:
                return stage_config.get(loss_type, self.supported_losses[loss_type])
        
        # If we're past all stages, use the last stage's values
        if self.stages:
            last_stage = list(self.stages.values())[-1]
            return last_stage.get(loss_type, self.supported_losses[loss_type])
        
        # Fallback to config values
        return self._get_scheduled_weight(loss_type)
    
    def _get_scheduled_weight(self, loss_type: str) -> float:
        """
        Get scheduled weight value using unified scheduling logic.
        
        This method implements the core scheduling logic that works for all weight types:
        - Beta scheduling: beta_schedule, beta_start, beta_end
        - Perceptual scheduling: perceptual_schedule, perceptual_start, perceptual_end, perceptual_warmup_epochs
        - Generation scheduling: generation_schedule, generation_start, generation_end, generation_warmup_epochs
        - Any other weight type can use the same pattern
        """
        # Get the base weight from loss_config first, then config, then default
        base_weight = self.loss_config.get(loss_type, self.config.get(loss_type, self.supported_losses[loss_type]))
        
        # Extract base name from loss_type (e.g., "generation_weight" -> "generation")
        base_name = loss_type.replace('_weight', '')
        
        # Check if this weight type has scheduling enabled
        schedule_key = f"{base_name}_schedule"
        if not self.config.get(schedule_key, False):
            return base_weight
        
        # Get scheduling parameters
        start_key = f"{base_name}_start"
        end_key = f"{base_name}_end"
        warmup_key = f"{base_name}_warmup_epochs"
        
        start_value = self.config.get(start_key, base_weight)
        end_value = self.config.get(end_key, base_weight)
        warmup_epochs = self.config.get(warmup_key, self.max_epochs)
        
        # Calculate scheduled value using linear interpolation with effective epoch
        effective_epoch = self.get_effective_epoch()
        if effective_epoch < warmup_epochs:
            # Linear interpolation during warmup period
            progress = effective_epoch / warmup_epochs
            scheduled_value = start_value + (end_value - start_value) * progress
        else:
            # Use end value after warmup period
            scheduled_value = end_value
        
        return scheduled_value
    
    def get_current_stage_name(self) -> str:
        """Get the name of the current training stage."""
        if not self.stage_based:
            return "standard"
        
        # Find the current stage
        for stage_name, stage_config in self.stages.items():
            start_epoch, end_epoch = stage_config['epochs']
            if start_epoch <= self.current_epoch < end_epoch:
                return stage_name
        
        # If we're past all stages, return the last stage
        if self.stages:
            return list(self.stages.keys())[-1]
        
        return "unknown"
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get all current weights as a dictionary."""
        return {loss_type: self.get_weight(loss_type) for loss_type in self.supported_losses.keys()}
    
    def get_weight_info(self, loss_type: str) -> Dict[str, Any]:
        """
        Get weight information including scheduling type and parameters.
        
        Args:
            loss_type: Type of loss weight
            
        Returns:
            Dictionary with weight value, type, and additional info
        """
        weight = self.get_weight(loss_type)
        
        # Get priority information using the centralized manager
        priority_info = self.priority_manager.get_display_info(self.current_epoch)
        
        if self.stage_based:
            stage_name = self.get_current_stage_name()
            info = {
                'value': weight,
                'type': 'stage-based',
                'stage': stage_name
            }
            # Add priority information if any priority is active
            if priority_info:
                info.update(priority_info)
            return info
        
        # Extract base name from loss_type (e.g., "generation_weight" -> "generation")
        base_name = loss_type.replace('_weight', '')
        
        # Check if this weight type has scheduling enabled
        schedule_key = f"{base_name}_schedule"
        if self.config.get(schedule_key, False):
            start_key = f"{base_name}_start"
            end_key = f"{base_name}_end"
            warmup_key = f"{base_name}_warmup_epochs"
            
            info: Dict[str, Any] = {
                'value': weight,
                'type': 'scheduled',
                'start': self.config.get(start_key, weight),
                'end': self.config.get(end_key, weight),
                'warmup_epochs': self.config.get(warmup_key, self.max_epochs)
            }
            
            # Add MSE priority information if active
            mse_priority_active = bool(self.mse_priority_phase) and (self.current_epoch < int(self.mse_priority_epochs))
            if mse_priority_active:
                info.update({
                    'mse_priority_active': True,
                    'mse_priority_epochs': self.mse_priority_epochs,
                    'mse_priority_multiplier': self.mse_priority_multiplier,
                    'mse_priority_remaining': self.mse_priority_epochs - self.current_epoch
                })

            # Add schedule progress within warmup (0-100%)
            try:
                warmup_epochs = max(1, int(info['warmup_epochs']))
                progress = min(1.0, max(0.0, float(self.get_effective_epoch()) / float(warmup_epochs)))
                info['progress'] = progress
            except Exception:
                pass

            # Special handling for KL cyclical annealing to reflect up/down phases
            if loss_type == 'beta' and self.kl_cycle_enabled:
                period = max(1, int(self.kl_cycle_period))
                cooldown = max(0, int(self.kl_cooldown_epochs))
                cycle_pos = self.current_epoch % period
                phase = 'down' if cycle_pos < cooldown else 'up'
                info['cycle_period'] = period
                info['cycle_pos'] = cycle_pos
                info['phase'] = phase
            
            return info
        else:
            return {
                'value': weight,
                'type': 'fixed'
            }
    
    def get_scheduling_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get scheduling summary for all weight types."""
        return {loss_type: self.get_weight_info(loss_type) for loss_type in self.supported_losses.keys()}
    
    def print_scheduling_info(self) -> None:
        """Print current scheduling information for all weights."""
        print("📊 WEIGHT SCHEDULING SUMMARY:")
        print("=" * 50)
        
        for loss_type in self.supported_losses.keys():
            info = self.get_weight_info(loss_type)
            weight = info['value']
            
            if info['type'] == 'stage-based':
                print(f"  • {loss_type}: {weight:.3f} (stage: {info['stage']})")
            elif info['type'] == 'scheduled':
                start = info['start']
                end = info['end']
                warmup = info['warmup_epochs']
                print(f"  • {loss_type}: {weight:.3f} (scheduled: {start:.3f} → {end:.3f} over {warmup} epochs)")
            else:
                print(f"  • {loss_type}: {weight:.3f} (fixed)")
        
        print("=" * 50)
