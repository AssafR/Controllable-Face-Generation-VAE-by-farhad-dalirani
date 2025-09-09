#!/usr/bin/env python3
"""
Centralized Strategy Controller

Single source of truth for training strategies that affect loss weights.
Applies an ordered set of modifiers uniformly to any base weight:
  base → schedule/stage (outside) → priority → cycle → adaptive (future) → bounds

Currently implemented:
- MSE priority (delegated to LossPriorityManager provided by caller)
- Alternating cycle training (reconstruction vs variational phases)

This controller is intentionally lightweight and stateless across calls,
with epoch-bound phase cached via set_epoch().
"""

from typing import Dict, Optional


class StrategyController:
    def __init__(self, config: Dict, priority_manager):
        self.config = config
        self.priority_manager = priority_manager
        self.current_epoch: int = 0

        # Alternating-cycle training configuration (defaults are conservative)
        self.cycle_enabled: bool = bool(config.get('cycle_training', False))
        self.cycle_period: int = int(config.get('cycle_period', 4))
        self.cycle_recon_epochs: int = int(config.get('cycle_recon_epochs', 2))
        # Remaining epochs implicitly variational
        self.cycle_variational_epochs: int = int(
            config.get('cycle_variational_epochs', max(0, self.cycle_period - self.cycle_recon_epochs))
        )

        # Phase multipliers allow fine-grained control per loss key
        # Keys should match LossWeightManager.supported_losses
        self.recon_multipliers: Dict[str, float] = config.get('cycle_recon_multipliers', {
            'mse_weight': 1.5,
            'l1_weight': 1.2,
            'perceptual_weight': 1.0,
            'generation_weight': 0.5,
            'beta': 0.5,
        })
        self.variational_multipliers: Dict[str, float] = config.get('cycle_variational_multipliers', {
            'mse_weight': 0.8,
            'l1_weight': 0.8,
            'perceptual_weight': 0.8,
            'generation_weight': 1.5,
            'beta': 1.5,
        })

        self._phase_name: Optional[str] = None
        self._phase_pos: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        if self.cycle_enabled and self.cycle_period > 0:
            pos = epoch % self.cycle_period
            self._phase_pos = pos
            if pos < self.cycle_recon_epochs:
                self._phase_name = 'reconstruction'
            else:
                self._phase_name = 'variational'
        else:
            self._phase_name = None
            self._phase_pos = 0

    def get_priority_multiplier(self, loss_type: str) -> float:
        """Retrieve any priority-based modifier (e.g., MSE priority) from the priority manager."""
        try:
            return float(self.priority_manager.get_weight_modifier(loss_type, self.current_epoch))
        except Exception:
            return 1.0

    def get_cycle_multiplier(self, loss_type: str) -> float:
        """Multiplier based on the current cycle phase, if enabled."""
        if not self.cycle_enabled or self.cycle_period <= 0 or not self._phase_name:
            return 1.0
        if self._phase_name == 'reconstruction':
            return float(self.recon_multipliers.get(loss_type, 1.0))
        else:
            return float(self.variational_multipliers.get(loss_type, 1.0))

    def apply_multipliers(self, loss_type: str, base_value: float) -> float:
        """Apply all strategy multipliers in a deterministic order to a base value."""
        value = float(base_value)

        # 1) Priority (e.g., MSE priority phase)
        value *= self.get_priority_multiplier(loss_type)

        # 2) Cycle phase (reconstruction vs variational)
        value *= self.get_cycle_multiplier(loss_type)

        return value

    def get_multipliers(self, loss_type: str) -> Dict[str, float]:
        """Return individual multipliers and combined for a loss type."""
        priority = self.get_priority_multiplier(loss_type)
        cycle = self.get_cycle_multiplier(loss_type)
        return {
            'priority': float(priority),
            'cycle': float(cycle),
            'combined': float(priority * cycle),
        }

    def get_phase_for_epoch(self, epoch: int) -> Dict:
        """Return phase info for an arbitrary epoch without mutating state."""
        if not self.cycle_enabled or self.cycle_period <= 0:
            return {'enabled': False}
        pos = epoch % self.cycle_period
        name = 'reconstruction' if pos < self.cycle_recon_epochs else 'variational'
        return {
            'enabled': True,
            'phase': name,
            'phase_pos': pos,
            'cycle_period': self.cycle_period,
        }

    def get_multipliers_for_epoch(self, loss_type: str, epoch: int) -> Dict[str, float]:
        """Return multipliers (priority, cycle, combined) for a given epoch without mutating state."""
        # Priority multiplier for that epoch
        priority = 1.0
        try:
            priority = float(self.priority_manager.get_weight_modifier(loss_type, epoch))
        except Exception:
            priority = 1.0
        # Cycle multiplier for that epoch based on phase name
        phase_info = self.get_phase_for_epoch(epoch)
        if not phase_info.get('enabled'):
            cycle = 1.0
        else:
            if phase_info.get('phase') == 'reconstruction':
                cycle = float(self.recon_multipliers.get(loss_type, 1.0))
            else:
                cycle = float(self.variational_multipliers.get(loss_type, 1.0))
        return {
            'priority': float(priority),
            'cycle': float(cycle),
            'combined': float(priority * cycle),
        }

    def get_display_info(self) -> Dict:
        """Provide a compact snapshot for reporting."""
        info: Dict = {
            'cycle_enabled': self.cycle_enabled,
        }
        if self.cycle_enabled:
            info.update({
                'cycle_period': self.cycle_period,
                'cycle_recon_epochs': self.cycle_recon_epochs,
                'cycle_variational_epochs': self.cycle_variational_epochs,
                'phase': self._phase_name or 'none',
                'phase_pos': self._phase_pos,
            })
        # Priority manager display (if available)
        try:
            info.update(self.priority_manager.get_display_info(self.current_epoch))
        except Exception:
            pass
        return info

    def get_recommendations(self) -> Dict:
        """Lightweight, strategy-linked recommendations for reporting."""
        rec: Dict = {
            'messages': []
        }
        if self.cycle_enabled:
            if self._phase_name == 'reconstruction':
                rec['messages'].append('Reconstruction-focused phase: prioritize feature fidelity; monitor KL to avoid collapse')
            elif self._phase_name == 'variational':
                rec['messages'].append('Variational-focused phase: encourage latent utilization; watch reconstruction sharpness')
        # Priority notes
        try:
            pinfo = self.priority_manager.get_display_info(self.current_epoch)
            for key, val in pinfo.items():
                if isinstance(val, dict) and val.get('is_active'):
                    rec['messages'].append(val.get('message', f'{key} priority active'))
        except Exception:
            pass
        return rec


