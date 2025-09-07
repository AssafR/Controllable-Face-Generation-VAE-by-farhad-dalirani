#!/usr/bin/env python3
"""
Loss Weight Management System
Centralized, extensible loss weight management with stage-based scheduling.
"""

from typing import Dict, Any, List, Optional


class LossWeightManager:
    """Centralized loss weight management with stage-based scheduling."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the loss weight manager.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.current_epoch = 0
        self.stage_based = config.get('stage_based_training', False)
        self.stages = config.get('stages', {})
        self.loss_config = config.get('loss_config', {})
        
        # Define all supported loss types and their default values
        # This is the single source of truth for all loss types
        self.supported_losses = {
            'mse_weight': 0.0,
            'l1_weight': 0.0,
            'perceptual_weight': 0.0,
            'generation_weight': 0.0,
            'lpips_weight': 0.0,
            'beta': 1.0
        }
    
    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for weight calculations."""
        self.current_epoch = epoch
    
    def get_weight(self, loss_type: str) -> float:
        """
        Get current weight for any loss type with stage-based or fixed scheduling.
        
        Args:
            loss_type: Type of loss weight to retrieve
            
        Returns:
            Current weight value
            
        Raises:
            ValueError: If loss_type is not supported
        """
        if loss_type not in self.supported_losses:
            raise ValueError(f"Unsupported loss type: {loss_type}. Supported types: {list(self.supported_losses.keys())}")
        
        if self.stage_based:
            return self._get_stage_based_weight(loss_type)
        else:
            return self._get_fixed_weight(loss_type)
    
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
        return self.loss_config.get(loss_type, self.supported_losses[loss_type])
    
    def _get_fixed_weight(self, loss_type: str) -> float:
        """Get fixed weight from config with fallback to default."""
        if loss_type == 'beta':
            return self.config.get('beta', self.supported_losses[loss_type])
        else:
            return self.loss_config.get(loss_type, self.supported_losses[loss_type])
    
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
        Get weight information including scheduling type.
        
        Args:
            loss_type: Type of loss weight
            
        Returns:
            Dictionary with weight value, type, and additional info
        """
        weight = self.get_weight(loss_type)
        
        if self.stage_based:
            stage_name = self.get_current_stage_name()
            return {
                'value': weight,
                'type': 'stage-based',
                'stage': stage_name
            }
        elif loss_type == 'beta' and self.config.get('beta_schedule', False):
            return {
                'value': weight,
                'type': 'scheduled',
                'start': self.config.get('beta_start', 0.1),
                'end': self.config.get('beta_end', 2.0)
            }
        elif loss_type == 'perceptual_weight' and self.config.get('perceptual_schedule', False):
            return {
                'value': weight,
                'type': 'scheduled',
                'start': self.config.get('perceptual_start', 0.0),
                'end': self.config.get('perceptual_end', 0.6)
            }
        else:
            return {
                'value': weight,
                'type': 'fixed'
            }
    
    def add_loss_type(self, loss_type: str, default_value: float = 0.0) -> None:
        """
        Add a new loss type to the supported losses.
        
        Args:
            loss_type: Name of the new loss type
            default_value: Default value for this loss type
        """
        self.supported_losses[loss_type] = default_value
    
    def get_supported_losses(self) -> List[str]:
        """Get list of all supported loss types."""
        return list(self.supported_losses.keys())
    
    def validate_stage_config(self) -> List[str]:
        """
        Validate stage configuration for completeness.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.stage_based:
            return errors
        
        if not self.stages:
            errors.append("Stage-based training enabled but no stages defined")
            return errors
        
        # Check for overlapping epochs
        stage_ranges = []
        for stage_name, stage_config in self.stages.items():
            if 'epochs' not in stage_config:
                errors.append(f"Stage '{stage_name}' missing 'epochs' field")
                continue
            
            epochs = stage_config['epochs']
            if not isinstance(epochs, list) or len(epochs) != 2:
                errors.append(f"Stage '{stage_name}' epochs must be [start, end] list")
                continue
            
            start, end = epochs
            if start >= end:
                errors.append(f"Stage '{stage_name}' start epoch ({start}) must be less than end epoch ({end})")
                continue
            
            stage_ranges.append((start, end, stage_name))
        
        # Check for overlaps
        stage_ranges.sort()
        for i in range(len(stage_ranges) - 1):
            current_end = stage_ranges[i][1]
            next_start = stage_ranges[i + 1][0]
            if current_end > next_start:
                errors.append(f"Stage '{stage_ranges[i][2]}' overlaps with '{stage_ranges[i + 1][2]}'")
        
        return errors
