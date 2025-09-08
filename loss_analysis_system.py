#!/usr/bin/env python3
"""
Unified Extensible Loss Analysis System
Provides comprehensive loss analysis capabilities for VAE training.

This system supports three main analysis approaches:
1. Standard Loss Analysis - Observes if loss is stuck even with weight changes
2. Constant Weight Analysis - Calculates loss based on fixed reference weights
3. Pareto Criterion Analysis - Multi-objective optimization analysis

The system is designed to be extensible, allowing easy addition of new analysis methods.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime


class AnalysisMethod(Enum):
    """Enumeration of available analysis methods."""
    STANDARD = "standard"
    CONSTANT_WEIGHT = "constant_weight"
    PARETO = "pareto"


@dataclass
class LossAnalysisResult:
    """Container for loss analysis results."""
    method: AnalysisMethod
    epoch: int
    analysis_data: Dict[str, Any]
    recommendations: List[str]
    health_score: float  # 0.0 to 1.0, where 1.0 is optimal
    timestamp: datetime


@dataclass
class AnalysisConfig:
    """Configuration for loss analysis methods."""
    # Standard analysis parameters
    stuck_threshold: float = 0.001  # Minimum improvement to not be considered stuck
    stuck_patience: int = 5  # Epochs without improvement before considering stuck
    trend_window: int = 10  # Window for trend analysis
    
    # Constant weight analysis parameters
    reference_weights: Optional[Dict[str, float]] = None
    reference_loss_components: List[str] = None
    
    # Pareto analysis parameters
    pareto_objectives: List[str] = None  # Loss components to optimize
    pareto_weights: Optional[Dict[str, float]] = None  # Weights for Pareto objectives
    pareto_tolerance: float = 0.01  # Tolerance for Pareto optimality
    
    # General parameters
    enable_logging: bool = True
    log_file: str = "loss_analysis.json"
    save_plots: bool = True
    plot_dir: str = "loss_analysis_plots"


class BaseLossAnalyzer(ABC):
    """Abstract base class for loss analysis methods."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.history: List[LossAnalysisResult] = []
    
    @abstractmethod
    def analyze(self, epoch: int, train_metrics: Dict[str, float], 
                val_metrics: Dict[str, float], current_weights: Dict[str, float]) -> LossAnalysisResult:
        """Perform loss analysis for the given epoch."""
        pass
    
    def get_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis data."""
        return []
    
    def calculate_health_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate a health score from 0.0 to 1.0."""
        return 0.5  # Default neutral score
    
    def save_result(self, result: LossAnalysisResult):
        """Save analysis result to history."""
        self.history.append(result)
        
        if self.config.enable_logging:
            self._log_result(result)
    
    def _log_result(self, result: LossAnalysisResult):
        """Log analysis result to file."""
        log_data = {
            'epoch': result.epoch,
            'method': result.method.value,
            'analysis_data': result.analysis_data,
            'recommendations': result.recommendations,
            'health_score': result.health_score,
            'timestamp': result.timestamp.isoformat()
        }
        
        # Append to log file
        log_file = self.config.log_file
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        existing_data.append(log_data)
        
        with open(log_file, 'w') as f:
            json.dump(existing_data, f, indent=2)


class StandardLossAnalyzer(BaseLossAnalyzer):
    """
    Standard loss analysis that observes if loss is stuck even with weight changes.
    
    This analyzer tracks loss trends and detects when training has plateaued,
    regardless of weight adjustments.
    """
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.loss_history: List[float] = []
        self.weight_history: List[Dict[str, float]] = []
        self.stuck_epochs = 0
        self.best_loss = float('inf')
    
    def analyze(self, epoch: int, train_metrics: Dict[str, float], 
                val_metrics: Dict[str, float], current_weights: Dict[str, float]) -> LossAnalysisResult:
        """Analyze loss behavior using standard approach."""
        current_loss = train_metrics.get('loss', 0.0)
        
        # Update history
        self.loss_history.append(current_loss)
        self.weight_history.append(current_weights.copy())
        
        # Keep only recent history
        if len(self.loss_history) > self.config.trend_window:
            self.loss_history = self.loss_history[-self.config.trend_window:]
            self.weight_history = self.weight_history[-self.config.trend_window:]
        
        # Analyze loss behavior
        analysis_data = self._analyze_loss_trends(epoch, current_loss)
        analysis_data.update(self._analyze_weight_impact(epoch))
        
        # Generate recommendations
        recommendations = self.get_recommendations(analysis_data)
        
        # Calculate health score
        health_score = self.calculate_health_score(analysis_data)
        
        result = LossAnalysisResult(
            method=AnalysisMethod.STANDARD,
            epoch=epoch,
            analysis_data=analysis_data,
            recommendations=recommendations,
            health_score=health_score,
            timestamp=datetime.now()
        )
        
        self.save_result(result)
        return result
    
    def analyze_with_context(self, epoch: int, train_metrics: Dict[str, float], 
                           val_metrics: Dict[str, float], current_weights: Dict[str, float],
                           additional_context: Dict[str, Any]) -> LossAnalysisResult:
        """Analyze loss behavior using standard approach with additional context."""
        current_loss = train_metrics.get('loss', 0.0)
        
        # Update history
        self.loss_history.append(current_loss)
        self.weight_history.append(current_weights.copy())
        
        # Keep only recent history
        if len(self.loss_history) > self.config.trend_window:
            self.loss_history = self.loss_history[-self.config.trend_window:]
            self.weight_history = self.weight_history[-self.config.trend_window:]
        
        # Analyze loss behavior
        analysis_data = self._analyze_loss_trends(epoch, current_loss)
        analysis_data.update(self._analyze_weight_impact(epoch))
        
        # Add MSE priority context
        if additional_context:
            analysis_data.update(additional_context)
        
        # Generate recommendations
        recommendations = self.get_recommendations(analysis_data)
        
        # Calculate health score
        health_score = self.calculate_health_score(analysis_data)
        
        result = LossAnalysisResult(
            method=AnalysisMethod.STANDARD,
            epoch=epoch,
            analysis_data=analysis_data,
            recommendations=recommendations,
            health_score=health_score,
            timestamp=datetime.now()
        )
        
        self.save_result(result)
        return result
    
    def _analyze_loss_trends(self, epoch: int, current_loss: float) -> Dict[str, Any]:
        """Analyze loss trends over time."""
        if len(self.loss_history) < 3:
            return {
                'trend': 'insufficient_data',
                'is_stuck': False,
                'improvement_rate': 0.0,
                'volatility': 0.0
            }
        
        # Calculate trend
        recent_losses = self.loss_history[-5:]  # Last 5 epochs
        if len(recent_losses) >= 3:
            # Linear regression to determine trend
            x = np.arange(len(recent_losses))
            y = np.array(recent_losses)
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            
            if slope < -self.config.stuck_threshold:
                trend = 'improving'
            elif slope > self.config.stuck_threshold:
                trend = 'worsening'
            else:
                trend = 'plateaued'
        else:
            trend = 'insufficient_data'
            slope = 0.0
        
        # Check if stuck
        is_stuck = False
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.stuck_epochs = 0
        else:
            self.stuck_epochs += 1
            if self.stuck_epochs >= self.config.stuck_patience:
                is_stuck = True
        
        # Calculate improvement rate
        if len(self.loss_history) >= 2:
            improvement_rate = (self.loss_history[-2] - current_loss) / self.loss_history[-2]
        else:
            improvement_rate = 0.0
        
        # Calculate volatility
        if len(self.loss_history) >= 3:
            volatility = np.std(self.loss_history[-3:])
        else:
            volatility = 0.0
        
        return {
            'trend': trend,
            'is_stuck': is_stuck,
            'improvement_rate': improvement_rate,
            'volatility': volatility,
            'stuck_epochs': self.stuck_epochs,
            'best_loss': self.best_loss,
            'current_loss': current_loss
        }
    
    def _analyze_weight_impact(self, epoch: int) -> Dict[str, Any]:
        """Analyze the impact of weight changes on loss."""
        if len(self.weight_history) < 2:
            return {
                'weight_changes': {},
                'weight_impact': 'insufficient_data'
            }
        
        # Calculate weight changes
        current_weights = self.weight_history[-1]
        previous_weights = self.weight_history[-2]
        
        weight_changes = {}
        for key in current_weights:
            if key in previous_weights:
                change = current_weights[key] - previous_weights[key]
                weight_changes[key] = change
        
        # Analyze if weight changes correlate with loss changes
        if len(self.loss_history) >= 2:
            loss_change = self.loss_history[-1] - self.loss_history[-2]
            
            # Simple correlation analysis
            weight_impact = 'neutral'
            if abs(loss_change) < self.config.stuck_threshold:
                weight_impact = 'no_impact'
            elif loss_change < 0:
                weight_impact = 'positive'
            else:
                weight_impact = 'negative'
        else:
            weight_impact = 'insufficient_data'
        
        return {
            'weight_changes': weight_changes,
            'weight_impact': weight_impact
        }
    
    def get_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on standard analysis."""
        recommendations = []
        
        # Check if MSE priority phase is active
        mse_priority_active = analysis_data.get('mse_priority_active', False)
        
        if analysis_data.get('is_stuck', False):
            if mse_priority_active:
                recommendations.append("Training appears stuck during MSE priority phase - consider extending priority phase or adjusting MSE multiplier")
            else:
                recommendations.append("Training appears stuck - consider adjusting learning rate or loss weights")
        
        if analysis_data.get('trend') == 'worsening':
            if mse_priority_active:
                recommendations.append("Loss worsening during MSE priority phase - this may be normal as network learns reconstruction")
            else:
                recommendations.append("Loss is worsening - consider reducing learning rate or checking for overfitting")
        
        if analysis_data.get('volatility', 0) > 0.1:
            if mse_priority_active:
                recommendations.append("High volatility during MSE priority phase - monitor reconstruction quality")
            else:
                recommendations.append("High loss volatility detected - consider stabilizing training")
        
        if analysis_data.get('weight_impact') == 'no_impact':
            if mse_priority_active:
                recommendations.append("Weight changes not affecting loss during MSE priority - this is expected as other objectives are suppressed")
            else:
                recommendations.append("Weight changes not affecting loss - consider more aggressive weight adjustments")
        
        return recommendations
    
    def calculate_health_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate health score for standard analysis."""
        score = 0.5  # Start with neutral score
        
        # Adjust based on trend
        trend = analysis_data.get('trend', 'insufficient_data')
        if trend == 'improving':
            score += 0.3
        elif trend == 'worsening':
            score -= 0.3
        elif trend == 'plateaued':
            score -= 0.1
        
        # Adjust based on stuck status
        if analysis_data.get('is_stuck', False):
            score -= 0.2
        
        # Adjust based on improvement rate
        improvement_rate = analysis_data.get('improvement_rate', 0)
        if improvement_rate > 0.01:  # 1% improvement
            score += 0.2
        elif improvement_rate < -0.01:  # 1% worsening
            score -= 0.2
        
        return max(0.0, min(1.0, score))


class ConstantWeightAnalyzer(BaseLossAnalyzer):
    """
    Constant weight analysis that calculates loss based on fixed reference weights.
    
    This analyzer provides a stable evaluation metric that doesn't change
    with weight adjustments, useful for comparing different training phases.
    """
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.reference_weights = config.reference_weights or {
            'mse_weight': 1.0,
            'l1_weight': 0.2,
            'perceptual_weight': 0.01,
            'generation_weight': 0.001,
            'beta': 0.0
        }
        self.reference_loss_history: List[float] = []
    
    def analyze(self, epoch: int, train_metrics: Dict[str, float], 
                val_metrics: Dict[str, float], current_weights: Dict[str, float]) -> LossAnalysisResult:
        """Analyze loss using constant reference weights."""
        # Calculate reference loss using constant weights
        reference_loss = self._calculate_reference_loss(train_metrics)
        self.reference_loss_history.append(reference_loss)
        
        # Keep only recent history
        if len(self.reference_loss_history) > self.config.trend_window:
            self.reference_loss_history = self.reference_loss_history[-self.config.trend_window:]
        
        # Analyze reference loss behavior
        analysis_data = self._analyze_reference_loss(epoch, reference_loss)
        analysis_data['reference_weights'] = self.reference_weights.copy()
        analysis_data['current_weights'] = current_weights.copy()
        
        # Generate recommendations
        recommendations = self.get_recommendations(analysis_data)
        
        # Calculate health score
        health_score = self.calculate_health_score(analysis_data)
        
        result = LossAnalysisResult(
            method=AnalysisMethod.CONSTANT_WEIGHT,
            epoch=epoch,
            analysis_data=analysis_data,
            recommendations=recommendations,
            health_score=health_score,
            timestamp=datetime.now()
        )
        
        self.save_result(result)
        return result
    
    def _calculate_reference_loss(self, train_metrics: Dict[str, float]) -> float:
        """Calculate loss using constant reference weights."""
        reference_loss = 0.0
        
        # Map loss components to their weight keys
        loss_mapping = {
            'mse': 'mse_weight',
            'l1': 'l1_weight',
            'perceptual': 'perceptual_weight',
            'generation_quality': 'generation_weight',
            'kl': 'beta'
        }
        
        for loss_key, weight_key in loss_mapping.items():
            if loss_key in train_metrics and weight_key in self.reference_weights:
                loss_value = train_metrics[loss_key]
                weight_value = self.reference_weights[weight_key]
                reference_loss += loss_value * weight_value
        
        return reference_loss
    
    def _analyze_reference_loss(self, epoch: int, reference_loss: float) -> Dict[str, Any]:
        """Analyze reference loss behavior."""
        if len(self.reference_loss_history) < 3:
            return {
                'reference_loss': reference_loss,
                'trend': 'insufficient_data',
                'improvement_rate': 0.0,
                'stability': 0.0
            }
        
        # Calculate trend
        recent_losses = self.reference_loss_history[-5:]
        if len(recent_losses) >= 3:
            x = np.arange(len(recent_losses))
            y = np.array(recent_losses)
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            
            if slope < -self.config.stuck_threshold:
                trend = 'improving'
            elif slope > self.config.stuck_threshold:
                trend = 'worsening'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
            slope = 0.0
        
        # Calculate improvement rate
        if len(self.reference_loss_history) >= 2:
            improvement_rate = (self.reference_loss_history[-2] - reference_loss) / self.reference_loss_history[-2]
        else:
            improvement_rate = 0.0
        
        # Calculate stability (inverse of volatility)
        if len(self.reference_loss_history) >= 3:
            volatility = np.std(self.reference_loss_history[-3:])
            stability = 1.0 / (1.0 + volatility)  # Higher stability for lower volatility
        else:
            stability = 0.5
        
        return {
            'reference_loss': reference_loss,
            'trend': trend,
            'improvement_rate': improvement_rate,
            'stability': stability,
            'volatility': volatility if len(self.reference_loss_history) >= 3 else 0.0
        }
    
    def get_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on constant weight analysis."""
        recommendations = []
        
        trend = analysis_data.get('trend', 'insufficient_data')
        if trend == 'worsening':
            recommendations.append("Reference loss is worsening - consider fundamental training adjustments")
        elif trend == 'stable':
            recommendations.append("Reference loss is stable - training may have converged")
        
        stability = analysis_data.get('stability', 0.5)
        if stability < 0.3:
            recommendations.append("High reference loss volatility - consider stabilizing training")
        
        improvement_rate = analysis_data.get('improvement_rate', 0)
        if improvement_rate < -0.01:  # 1% worsening
            recommendations.append("Reference loss deteriorating - check for overfitting or learning rate issues")
        
        return recommendations
    
    def calculate_health_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate health score for constant weight analysis."""
        score = 0.5  # Start with neutral score
        
        # Adjust based on trend
        trend = analysis_data.get('trend', 'insufficient_data')
        if trend == 'improving':
            score += 0.3
        elif trend == 'worsening':
            score -= 0.3
        elif trend == 'stable':
            score += 0.1
        
        # Adjust based on stability
        stability = analysis_data.get('stability', 0.5)
        score += (stability - 0.5) * 0.4  # Scale stability impact
        
        # Adjust based on improvement rate
        improvement_rate = analysis_data.get('improvement_rate', 0)
        if improvement_rate > 0.01:
            score += 0.2
        elif improvement_rate < -0.01:
            score -= 0.2
        
        return max(0.0, min(1.0, score))


class ParetoAnalyzer(BaseLossAnalyzer):
    """
    Pareto criterion analysis for multi-objective optimization.
    
    This analyzer evaluates the Pareto optimality of different loss components,
    helping to balance multiple objectives in VAE training.
    """
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.pareto_objectives = config.pareto_objectives or ['mse', 'l1', 'perceptual', 'kl']
        self.pareto_weights = config.pareto_weights or {obj: 1.0 for obj in self.pareto_objectives}
        self.pareto_history: List[Dict[str, float]] = []
        self.pareto_front: List[Dict[str, float]] = []
    
    def analyze(self, epoch: int, train_metrics: Dict[str, float], 
                val_metrics: Dict[str, float], current_weights: Dict[str, float]) -> LossAnalysisResult:
        """Analyze loss using Pareto criterion."""
        # Extract objective values
        objectives = self._extract_objectives(train_metrics)
        self.pareto_history.append(objectives.copy())
        
        # Keep only recent history
        if len(self.pareto_history) > self.config.trend_window:
            self.pareto_history = self.pareto_history[-self.config.trend_window:]
        
        # Update Pareto front
        self._update_pareto_front(objectives)
        
        # Analyze Pareto optimality
        analysis_data = self._analyze_pareto_optimality(epoch, objectives)
        analysis_data['objectives'] = objectives.copy()
        analysis_data['pareto_weights'] = self.pareto_weights.copy()
        
        # Generate recommendations
        recommendations = self.get_recommendations(analysis_data)
        
        # Calculate health score
        health_score = self.calculate_health_score(analysis_data)
        
        result = LossAnalysisResult(
            method=AnalysisMethod.PARETO,
            epoch=epoch,
            analysis_data=analysis_data,
            recommendations=recommendations,
            health_score=health_score,
            timestamp=datetime.now()
        )
        
        self.save_result(result)
        return result
    
    def _extract_objectives(self, train_metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract objective values from training metrics."""
        objectives = {}
        for obj in self.pareto_objectives:
            if obj in train_metrics:
                objectives[obj] = train_metrics[obj]
            else:
                objectives[obj] = 0.0
        return objectives
    
    def _update_pareto_front(self, objectives: Dict[str, float]):
        """Update the Pareto front with new objective values."""
        # Check if current point is dominated by any point in the front
        is_dominated = False
        for front_point in self.pareto_front:
            if self._dominates(front_point, objectives):
                is_dominated = True
                break
        
        # If not dominated, add to front and remove dominated points
        if not is_dominated:
            # Remove points dominated by current point
            self.pareto_front = [point for point in self.pareto_front 
                               if not self._dominates(objectives, point)]
            # Add current point
            self.pareto_front.append(objectives.copy())
    
    def _dominates(self, point1: Dict[str, float], point2: Dict[str, float]) -> bool:
        """Check if point1 dominates point2 in Pareto sense."""
        # For minimization, point1 dominates point2 if point1 is better in all objectives
        # and strictly better in at least one
        better_in_all = True
        better_in_some = False
        
        for obj in self.pareto_objectives:
            if obj in point1 and obj in point2:
                if point1[obj] > point2[obj]:  # point2 is better (lower loss)
                    better_in_all = False
                    break
                elif point2[obj] > point1[obj]:  # point1 is better
                    better_in_some = True
        
        return better_in_all and better_in_some
    
    def _analyze_pareto_optimality(self, epoch: int, objectives: Dict[str, float]) -> Dict[str, Any]:
        """Analyze Pareto optimality of current objectives."""
        # Check if current point is on the Pareto front
        is_pareto_optimal = objectives in self.pareto_front
        
        # Calculate weighted sum for comparison
        weighted_sum = sum(objectives.get(obj, 0) * self.pareto_weights.get(obj, 1.0) 
                          for obj in self.pareto_objectives)
        
        # Calculate distance to Pareto front
        if self.pareto_front:
            distances = []
            for front_point in self.pareto_front:
                distance = sum((objectives.get(obj, 0) - front_point.get(obj, 0))**2 
                             for obj in self.pareto_objectives)
                distances.append(distance**0.5)
            min_distance = min(distances)
        else:
            min_distance = 0.0
        
        # Calculate objective balance
        if len(objectives) > 1:
            values = list(objectives.values())
            balance = 1.0 - (np.std(values) / (np.mean(values) + 1e-8))
        else:
            balance = 1.0
        
        return {
            'is_pareto_optimal': is_pareto_optimal,
            'weighted_sum': weighted_sum,
            'distance_to_front': min_distance,
            'objective_balance': balance,
            'pareto_front_size': len(self.pareto_front)
        }
    
    def get_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on Pareto analysis."""
        recommendations = []
        
        if not analysis_data.get('is_pareto_optimal', False):
            recommendations.append("Current point is not Pareto optimal - consider adjusting loss weights")
        
        distance = analysis_data.get('distance_to_front', 0)
        if distance > self.config.pareto_tolerance:
            recommendations.append("Far from Pareto front - consider multi-objective optimization")
        
        balance = analysis_data.get('objective_balance', 1.0)
        if balance < 0.5:
            recommendations.append("Poor objective balance - some objectives dominating others")
        
        weighted_sum = analysis_data.get('weighted_sum', 0)
        if len(self.pareto_history) >= 2:
            prev_weighted_sum = sum(sum(point.get(obj, 0) * self.pareto_weights.get(obj, 1.0) 
                                    for obj in self.pareto_objectives) 
                                  for point in self.pareto_history[-2:]) / 2
            if weighted_sum > prev_weighted_sum:
                recommendations.append("Weighted objective sum increasing - consider rebalancing")
        
        return recommendations
    
    def calculate_health_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate health score for Pareto analysis."""
        score = 0.5  # Start with neutral score
        
        # Adjust based on Pareto optimality
        if analysis_data.get('is_pareto_optimal', False):
            score += 0.3
        
        # Adjust based on distance to front
        distance = analysis_data.get('distance_to_front', 0)
        if distance < self.config.pareto_tolerance:
            score += 0.2
        elif distance > self.config.pareto_tolerance * 2:
            score -= 0.2
        
        # Adjust based on objective balance
        balance = analysis_data.get('objective_balance', 1.0)
        score += (balance - 0.5) * 0.3
        
        return max(0.0, min(1.0, score))


class UnifiedLossAnalysisSystem:
    """
    Unified system that coordinates multiple loss analysis methods.
    
    This is the main interface for the loss analysis system, providing
    a unified way to run different analysis methods and aggregate results.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.analyzers: Dict[AnalysisMethod, BaseLossAnalyzer] = {}
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialize all available analyzers."""
        self.analyzers[AnalysisMethod.STANDARD] = StandardLossAnalyzer(self.config)
        self.analyzers[AnalysisMethod.CONSTANT_WEIGHT] = ConstantWeightAnalyzer(self.config)
        self.analyzers[AnalysisMethod.PARETO] = ParetoAnalyzer(self.config)
    
    def analyze_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                     val_metrics: Dict[str, float], current_weights: Dict[str, float],
                     methods: Optional[List[AnalysisMethod]] = None, 
                     additional_context: Optional[Dict[str, Any]] = None) -> Dict[AnalysisMethod, LossAnalysisResult]:
        """
        Run loss analysis for a given epoch using specified methods.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            current_weights: Current loss weights dictionary
            methods: List of analysis methods to run (None for all)
            
        Returns:
            Dictionary mapping analysis methods to their results
        """
        if methods is None:
            methods = list(AnalysisMethod)
        
        results = {}
        for method in methods:
            if method in self.analyzers:
                analyzer = self.analyzers[method]
                # Pass additional context to analyzers that support it
                if hasattr(analyzer, 'analyze_with_context'):
                    result = analyzer.analyze_with_context(epoch, train_metrics, val_metrics, current_weights, additional_context)
                else:
                    result = analyzer.analyze(epoch, train_metrics, val_metrics, current_weights)
                results[method] = result
        
        return results
    
    def get_combined_recommendations(self, results: Dict[AnalysisMethod, LossAnalysisResult]) -> List[str]:
        """Get combined recommendations from all analysis methods."""
        all_recommendations = []
        for result in results.values():
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def get_user_friendly_summary(self, results: Dict[AnalysisMethod, LossAnalysisResult]) -> Dict[str, Any]:
        """Get a user-friendly summary with prioritized recommendations."""
        if not results:
            return {
                'status': '🟡 No data',
                'message': 'Insufficient data for analysis',
                'priority_actions': [],
                'health_indicators': []
            }
        
        # Get overall health
        overall_health = self.get_overall_health_score(results)
        
        # Determine status and message
        if overall_health >= 0.8:
            status = '🟢 Excellent'
            message = 'Training is progressing very well'
        elif overall_health >= 0.6:
            status = '🟡 Good'
            message = 'Training is progressing well with minor issues'
        elif overall_health >= 0.4:
            status = '🟠 Fair'
            message = 'Training has some issues that need attention'
        else:
            status = '🔴 Poor'
            message = 'Training has significant issues requiring immediate attention'
        
        # Extract priority actions from analysis
        priority_actions = []
        health_indicators = []
        
        # Check for critical issues first
        for method, result in results.items():
            if method == AnalysisMethod.STANDARD:
                analysis_data = result.analysis_data
                
                # Critical issues (high priority)
                if analysis_data.get('is_stuck', False):
                    priority_actions.append({
                        'action': '⚡ Activate acceleration',
                        'reason': 'Training appears stuck',
                        'priority': 'high'
                    })
                
                if analysis_data.get('trend') == 'worsening':
                    priority_actions.append({
                        'action': '🔻 Reduce learning rate',
                        'reason': 'Loss is getting worse',
                        'priority': 'high'
                    })
                
                # Health indicators
                if analysis_data.get('volatility', 0) > 0.1:
                    health_indicators.append('📊 High loss volatility detected')
                
                if analysis_data.get('improvement_rate', 0) < -0.05:
                    health_indicators.append('📉 Loss improvement rate is negative')
            
            elif method == AnalysisMethod.CONSTANT_WEIGHT:
                analysis_data = result.analysis_data
                
                if analysis_data.get('trend') == 'worsening':
                    priority_actions.append({
                        'action': '🔧 Check training fundamentals',
                        'reason': 'Reference loss is deteriorating',
                        'priority': 'medium'
                    })
                
                if analysis_data.get('stability', 0) > 0.8:
                    health_indicators.append('✅ Reference loss is stable')
            
            elif method == AnalysisMethod.PARETO:
                analysis_data = result.analysis_data
                
                if not analysis_data.get('is_pareto_optimal', True):
                    priority_actions.append({
                        'action': '⚖️ Rebalance loss weights',
                        'reason': 'Objectives are not optimally balanced',
                        'priority': 'low'
                    })
                
                balance = analysis_data.get('objective_balance', 1.0)
                if balance < 0.5:
                    health_indicators.append('⚖️ Some objectives dominating others')
        
        # Sort priority actions by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        priority_actions.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return {
            'status': status,
            'message': message,
            'health_score': overall_health,
            'priority_actions': priority_actions[:3],  # Show top 3 actions
            'health_indicators': health_indicators[:3]  # Show top 3 indicators
        }
    
    def get_overall_health_score(self, results: Dict[AnalysisMethod, LossAnalysisResult]) -> float:
        """Get overall health score from all analysis methods."""
        if not results:
            return 0.5
        
        scores = [result.health_score for result in results.values()]
        return sum(scores) / len(scores)
    
    def get_analysis_summary(self, results: Dict[AnalysisMethod, LossAnalysisResult]) -> Dict[str, Any]:
        """Get a summary of all analysis results."""
        summary = {
            'epoch': results[list(results.keys())[0]].epoch if results else 0,
            'overall_health_score': self.get_overall_health_score(results),
            'recommendations': self.get_combined_recommendations(results),
            'method_results': {}
        }
        
        for method, result in results.items():
            summary['method_results'][method.value] = {
                'health_score': result.health_score,
                'recommendations': result.recommendations,
                'analysis_data': result.analysis_data
            }
        
        return summary
    
    def add_custom_analyzer(self, method: AnalysisMethod, analyzer: BaseLossAnalyzer):
        """Add a custom analyzer for a specific method."""
        self.analyzers[method] = analyzer
    
    def get_analyzer_history(self, method: AnalysisMethod) -> List[LossAnalysisResult]:
        """Get analysis history for a specific method."""
        if method in self.analyzers:
            return self.analyzers[method].history
        return []
    
    def clear_history(self):
        """Clear analysis history for all methods."""
        for analyzer in self.analyzers.values():
            analyzer.history.clear()


def create_loss_analysis_system(config: Dict[str, Any]) -> UnifiedLossAnalysisSystem:
    """
    Factory function to create a loss analysis system from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured UnifiedLossAnalysisSystem instance
    """
    analysis_config = AnalysisConfig(
        stuck_threshold=config.get('stuck_threshold', 0.001),
        stuck_patience=config.get('stuck_patience', 5),
        trend_window=config.get('trend_window', 10),
        reference_weights=config.get('reference_weights'),
        reference_loss_components=config.get('reference_loss_components'),
        pareto_objectives=config.get('pareto_objectives'),
        pareto_weights=config.get('pareto_weights'),
        pareto_tolerance=config.get('pareto_tolerance', 0.01),
        enable_logging=config.get('enable_logging', True),
        log_file=config.get('log_file', 'loss_analysis.json'),
        save_plots=config.get('save_plots', True),
        plot_dir=config.get('plot_dir', 'loss_analysis_plots')
    )
    
    return UnifiedLossAnalysisSystem(analysis_config)


if __name__ == "__main__":
    # Example usage
    config = {
        'stuck_threshold': 0.001,
        'stuck_patience': 5,
        'trend_window': 10,
        'reference_weights': {
            'mse_weight': 1.0,
            'l1_weight': 0.2,
            'perceptual_weight': 0.01,
            'generation_weight': 0.001,
            'beta': 0.0
        },
        'pareto_objectives': ['mse', 'l1', 'perceptual', 'kl'],
        'enable_logging': True
    }
    
    system = create_loss_analysis_system(config)
    print("Loss Analysis System initialized successfully!")
