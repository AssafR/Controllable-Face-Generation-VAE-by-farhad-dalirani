"""
Loss Priority Manager

Centralized management of loss priority phases to eliminate code duplication
and provide a clean, extensible interface for priority state queries.

This system is designed to be extensible to any loss type, not just MSE.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum


class PriorityPhase(Enum):
    """Enumeration of priority phases."""
    NONE = "none"
    ACTIVE = "active"
    TRANSITION = "transition"


@dataclass
class PriorityState:
    """State information for any priority phase."""
    phase: PriorityPhase
    remaining_epochs: int
    multiplier: float
    transition_remaining: int
    loss_type: str


class LossPriorityManager:
    """
    Centralized manager for loss priority phases.
    
    This class provides a unified, extensible system for managing priority phases
    for any loss type. It eliminates code duplication and provides a clean interface
    for priority state queries and weight modifications.
    """
    
    def __init__(self, loss_manager):
        """
        Initialize loss priority manager.
        
        Args:
            loss_manager: The loss weight manager instance
        """
        self.loss_manager = loss_manager
        self.transition_epochs = 3  # Default transition period
        
        # Priority configurations - easily extensible to other loss types
        self.priority_configs = {
            'mse': {
                'enabled': 'mse_priority_phase',
                'epochs': 'mse_priority_epochs', 
                'multiplier': 'mse_priority_multiplier',
                'suppressed_losses': ['beta', 'perceptual_weight', 'generation_weight', 'lpips_weight']
            },
            # Future: Add other loss types here
            # 'kl': {
            #     'enabled': 'kl_priority_phase',
            #     'epochs': 'kl_priority_epochs',
            #     'multiplier': 'kl_priority_multiplier', 
            #     'suppressed_losses': ['mse_weight', 'perceptual_weight']
            # }
        }
    
    def is_priority_enabled(self, loss_type: str) -> bool:
        """Check if priority phase is enabled for a specific loss type."""
        if loss_type not in self.priority_configs:
            return False
        
        config = self.priority_configs[loss_type]
        return getattr(self.loss_manager, config['enabled'], False)
    
    def get_priority_state(self, loss_type: str, epoch: int) -> PriorityState:
        """
        Get complete priority state for a specific loss type and epoch.
        
        Args:
            loss_type: Type of loss (e.g., 'mse', 'kl')
            epoch: Current training epoch
            
        Returns:
            PriorityState object with all priority information
        """
        if not self.is_priority_enabled(loss_type):
            return PriorityState(
                phase=PriorityPhase.NONE,
                remaining_epochs=0,
                multiplier=1.0,
                transition_remaining=0,
                loss_type=loss_type
            )
        
        config = self.priority_configs[loss_type]
        priority_epochs = getattr(self.loss_manager, config['epochs'], 0)
        multiplier = getattr(self.loss_manager, config['multiplier'], 1.0)
        
        # During priority phase
        if epoch < priority_epochs:
            return PriorityState(
                phase=PriorityPhase.ACTIVE,
                remaining_epochs=priority_epochs - epoch,
                multiplier=multiplier,
                transition_remaining=0,
                loss_type=loss_type
            )
        
        # During transition phase
        elif epoch < priority_epochs + self.transition_epochs:
            return PriorityState(
                phase=PriorityPhase.TRANSITION,
                remaining_epochs=0,
                multiplier=multiplier,
                transition_remaining=(priority_epochs + self.transition_epochs) - epoch,
                loss_type=loss_type
            )
        
        # After both phases
        else:
            return PriorityState(
                phase=PriorityPhase.NONE,
                remaining_epochs=0,
                multiplier=1.0,
                transition_remaining=0,
                loss_type=loss_type
            )
    
    def is_priority_active(self, loss_type: str, epoch: int) -> bool:
        """Check if priority is currently active for a specific loss type."""
        return self.get_priority_state(loss_type, epoch).phase == PriorityPhase.ACTIVE
    
    def is_transition_active(self, loss_type: str, epoch: int) -> bool:
        """Check if transition phase is currently active for a specific loss type."""
        return self.get_priority_state(loss_type, epoch).phase == PriorityPhase.TRANSITION
    
    def get_remaining_epochs(self, loss_type: str, epoch: int) -> int:
        """Get remaining epochs in current phase for a specific loss type."""
        state = self.get_priority_state(loss_type, epoch)
        return state.remaining_epochs if state.phase == PriorityPhase.ACTIVE else state.transition_remaining
    
    def get_multiplier(self, loss_type: str, epoch: int) -> float:
        """Get current multiplier for a specific loss type."""
        return self.get_priority_state(loss_type, epoch).multiplier
    
    def get_context_for_analysis(self, epoch: int) -> Dict[str, Any]:
        """
        Get priority context for loss analysis for all active priority types.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary with priority context for analysis
        """
        context = {}
        
        for loss_type in self.priority_configs.keys():
            if self.is_priority_enabled(loss_type):
                state = self.get_priority_state(loss_type, epoch)
                
                context[f'{loss_type}_priority_active'] = state.phase == PriorityPhase.ACTIVE
                context[f'{loss_type}_priority_transition'] = state.phase == PriorityPhase.TRANSITION
                context[f'{loss_type}_priority_multiplier'] = state.multiplier
                
                if state.phase == PriorityPhase.ACTIVE:
                    context[f'{loss_type}_priority_remaining'] = state.remaining_epochs
                elif state.phase == PriorityPhase.TRANSITION:
                    context[f'{loss_type}_priority_transition_remaining'] = state.transition_remaining
        
        return context
    
    def get_display_info(self, epoch: int) -> Dict[str, Any]:
        """
        Get priority information for display purposes for all active priority types.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary with display information
        """
        display_info = {}
        
        for loss_type in self.priority_configs.keys():
            if self.is_priority_enabled(loss_type):
                state = self.get_priority_state(loss_type, epoch)
                
                info = {
                    'is_active': state.phase == PriorityPhase.ACTIVE,
                    'is_transition': state.phase == PriorityPhase.TRANSITION,
                    'multiplier': state.multiplier,
                    'loss_type': loss_type.upper()
                }
                
                if state.phase == PriorityPhase.ACTIVE:
                    info['remaining'] = state.remaining_epochs
                    info['message'] = f"{loss_type.upper()} PRIORITY PHASE: {state.remaining_epochs} epochs remaining (×{state.multiplier:.1f} multiplier)"
                elif state.phase == PriorityPhase.TRANSITION:
                    info['transition_remaining'] = state.transition_remaining
                    info['message'] = f"{loss_type.upper()} TRANSITION PHASE: {state.transition_remaining} epochs remaining (gradual balance)"
                
                display_info[loss_type] = info
        
        return display_info
    
    def get_weight_modifier(self, loss_type: str, epoch: int) -> float:
        """
        Get weight modifier for a specific loss type during any active priority phase.
        
        Args:
            loss_type: Type of loss weight (e.g., 'mse_weight', 'beta')
            epoch: Current training epoch
            
        Returns:
            Multiplier to apply to the base weight
        """
        # Check each priority configuration
        for priority_loss_type, config in self.priority_configs.items():
            if not self.is_priority_enabled(priority_loss_type):
                continue
            
            state = self.get_priority_state(priority_loss_type, epoch)
            priority_weight_type = f"{priority_loss_type}_weight"
            
            if state.phase == PriorityPhase.ACTIVE:
                if loss_type == priority_weight_type:
                    return state.multiplier
                elif loss_type in config['suppressed_losses']:
                    return 0.0  # Suppressed during priority
            elif state.phase == PriorityPhase.TRANSITION:
                # Gradual transition
                priority_epochs = getattr(self.loss_manager, config['epochs'], 0)
                transition_progress = (epoch - priority_epochs) / self.transition_epochs
                
                if loss_type == priority_weight_type:
                    # Gradually reduce priority multiplier
                    return state.multiplier * (1 - transition_progress) + 1.0 * transition_progress
                elif loss_type in config['suppressed_losses']:
                    # Gradually introduce other objectives
                    return transition_progress
        
        return 1.0
    
    def add_priority_config(self, loss_type: str, config: Dict[str, Any]) -> None:
        """
        Add a new priority configuration for a loss type.
        
        Args:
            loss_type: Type of loss (e.g., 'kl', 'perceptual')
            config: Configuration dictionary with keys:
                - enabled: Attribute name for enabled flag
                - epochs: Attribute name for priority epochs
                - multiplier: Attribute name for multiplier
                - suppressed_losses: List of loss types to suppress during priority
        """
        self.priority_configs[loss_type] = config
    
    def get_active_priorities(self, epoch: int) -> List[str]:
        """Get list of loss types with active priority phases."""
        active = []
        for loss_type in self.priority_configs.keys():
            if self.is_priority_enabled(loss_type) and self.is_priority_active(loss_type, epoch):
                active.append(loss_type)
        return active
