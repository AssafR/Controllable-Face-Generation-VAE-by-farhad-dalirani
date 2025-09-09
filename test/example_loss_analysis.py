#!/usr/bin/env python3
"""
Example script demonstrating the unified loss analysis system.

This script shows how to use the loss analysis system independently
or integrate it with existing training loops.
"""

import torch
import numpy as np
from loss_analysis_system import create_loss_analysis_system, AnalysisMethod
import json


def simulate_training_metrics(epoch: int) -> dict:
    """Simulate training metrics for demonstration purposes."""
    # Simulate some realistic loss behavior
    base_mse = 0.1 * np.exp(-epoch * 0.05) + 0.02
    base_l1 = 0.05 * np.exp(-epoch * 0.03) + 0.01
    base_perceptual = 0.2 * np.exp(-epoch * 0.02) + 0.05
    base_kl = 0.01 * (1 + 0.1 * np.sin(epoch * 0.5))
    
    # Add some noise
    noise_factor = 0.1
    mse = base_mse + np.random.normal(0, base_mse * noise_factor)
    l1 = base_l1 + np.random.normal(0, base_l1 * noise_factor)
    perceptual = base_perceptual + np.random.normal(0, base_perceptual * noise_factor)
    kl = base_kl + np.random.normal(0, base_kl * noise_factor)
    
    # Calculate total loss with some weights
    total_loss = mse * 1.0 + l1 * 0.2 + perceptual * 0.01 + kl * 0.1
    
    return {
        'loss': total_loss,
        'mse': mse,
        'l1': l1,
        'perceptual': perceptual,
        'generation_quality': 0.05 + np.random.normal(0, 0.01),
        'kl': kl
    }


def simulate_weight_changes(epoch: int) -> dict:
    """Simulate weight changes during training."""
    # Simulate some weight scheduling
    mse_weight = 1.0
    l1_weight = 0.2
    perceptual_weight = 0.01 * (1 + 0.5 * np.sin(epoch * 0.1))
    generation_weight = 0.001 * (1 + epoch * 0.01)
    beta = 0.1 * min(1.0, epoch / 10.0)  # Beta annealing
    
    return {
        'mse_weight': mse_weight,
        'l1_weight': l1_weight,
        'perceptual_weight': perceptual_weight,
        'generation_weight': generation_weight,
        'beta': beta
    }


def main():
    """Demonstrate the loss analysis system."""
    print("🔬 Loss Analysis System Demonstration")
    print("=" * 50)
    
    # Configuration for the loss analysis system
    config = {
        'stuck_threshold': 0.001,
        'stuck_patience': 3,
        'trend_window': 8,
        'reference_weights': {
            'mse_weight': 1.0,
            'l1_weight': 0.2,
            'perceptual_weight': 0.01,
            'generation_weight': 0.001,
            'beta': 0.0
        },
        'pareto_objectives': ['mse', 'l1', 'perceptual', 'kl'],
        'pareto_weights': {
            'mse': 1.0,
            'l1': 1.0,
            'perceptual': 1.0,
            'kl': 1.0
        },
        'enable_logging': True,
        'log_file': 'demo_loss_analysis.json'
    }
    
    # Create the loss analysis system
    analysis_system = create_loss_analysis_system(config)
    
    print("✅ Loss analysis system initialized")
    print(f"📊 Analysis methods: {[method.value for method in AnalysisMethod]}")
    print()
    
    # Simulate training for 20 epochs
    num_epochs = 20
    print(f"🎯 Simulating training for {num_epochs} epochs...")
    print()
    
    for epoch in range(num_epochs):
        # Simulate training metrics
        train_metrics = simulate_training_metrics(epoch)
        val_metrics = simulate_training_metrics(epoch)  # Simplified: same as train
        
        # Simulate weight changes
        current_weights = simulate_weight_changes(epoch)
        
        # Run loss analysis
        analysis_results = analysis_system.analyze_epoch(
            epoch, train_metrics, val_metrics, current_weights
        )
        
        # Print summary every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"\n📊 EPOCH {epoch+1} ANALYSIS SUMMARY")
            print("-" * 40)
            
            # Get analysis summary
            summary = analysis_system.get_analysis_summary(analysis_results)
            
            # Print overall health score
            overall_health = summary['overall_health_score']
            health_icon = "🟢" if overall_health > 0.7 else "🟡" if overall_health > 0.4 else "🔴"
            print(f"{health_icon} Overall Health Score: {overall_health:.3f}")
            
            # Print method-specific results
            method_results = summary.get('method_results', {})
            for method_name, result in method_results.items():
                method_icon = "📈" if method_name == "standard" else "⚖️" if method_name == "constant_weight" else "🎯"
                print(f"{method_icon} {method_name.replace('_', ' ').title()}: {result['health_score']:.3f}")
            
            # Print recommendations
            recommendations = summary.get('recommendations', [])
            if recommendations:
                print(f"💡 Top recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                    print(f"  {i}. {rec}")
            
            print()
    
    # Print final analysis
    print("🎉 TRAINING SIMULATION COMPLETE")
    print("=" * 50)
    
    # Get final analysis summary
    final_analysis = analysis_system.get_analysis_summary(analysis_results)
    print(f"Final Overall Health Score: {final_analysis['overall_health_score']:.3f}")
    
    # Show analysis history for each method
    print("\n📈 Analysis History Summary:")
    for method in AnalysisMethod:
        history = analysis_system.get_analyzer_history(method)
        if history:
            health_scores = [result.health_score for result in history]
            avg_health = sum(health_scores) / len(health_scores)
            print(f"  {method.value}: {len(history)} analyses, avg health {avg_health:.3f}")
    
    # Show log file info
    if config.get('enable_logging', False):
        log_file = config.get('log_file', 'loss_analysis.json')
        print(f"\n📝 Analysis results saved to: {log_file}")
        
        # Load and show some log data
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            print(f"   Total logged analyses: {len(log_data)}")
        except FileNotFoundError:
            print("   Log file not found")
    
    print("\n✅ Demonstration complete!")


if __name__ == "__main__":
    main()
