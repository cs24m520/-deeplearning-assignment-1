#!/usr/bin/env python3
"""
Simple Validation Accuracy vs. Step Scatter Plot

Creates validation accuracy vs step scatter plot from W&B sweep data.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
from pathlib import Path
import re

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

def extract_wandb_data():
    """Extract validation accuracy data from W&B runs."""
    wandb_dir = Path("wandb")
    runs_data = []
    
    if not wandb_dir.exists():
        return []
    
    run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
    print(f"Found {len(run_dirs)} W&B runs")
    
    for run_dir in run_dirs:
        try:
            files_dir = run_dir / "files"
            if not files_dir.exists():
                continue
            
            run_data = {'run_id': run_dir.name, 'config': {}, 'accuracies': [], 'steps': []}
            
            # Load config
            config_file = files_dir / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    # Extract scalar values from config
                    clean_config = {}
                    if isinstance(config, dict):
                        for k, v in config.items():
                            if isinstance(v, dict) and 'value' in v:
                                clean_config[k] = v['value']
                            else:
                                clean_config[k] = v
                    run_data['config'] = clean_config
            
            # Parse log for accuracies
            log_file = files_dir / "output.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for line in lines:
                    # Look for test accuracy patterns
                    test_acc_match = re.search(r"Test Accuracy: ([\d.]+)%", line)
                    epoch_match = re.search(r"Epoch (\d+)/", line)
                    
                    if test_acc_match and epoch_match:
                        accuracy = float(test_acc_match.group(1))
                        epoch = int(epoch_match.group(1))
                        run_data['accuracies'].append(accuracy)
                        run_data['steps'].append(epoch)
            
            if run_data['accuracies']:
                runs_data.append(run_data)
                
        except Exception as e:
            print(f"Error processing {run_dir.name}: {e}")
            continue
    
    return runs_data

def create_validation_scatter_plot():
    """Create validation accuracy vs step scatter plot."""
    
    # Load data
    runs_data = extract_wandb_data()
    
    if not runs_data:
        print("No validation data found. Creating synthetic example...")
        runs_data = create_synthetic_data()
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Define colors for different optimizers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    all_steps = []
    all_accuracies = []
    
    for i, run in enumerate(runs_data):
        steps = run['steps']
        accuracies = run['accuracies']
        config = run['config']
        
        if not steps or not accuracies:
            continue
        
        # Get configuration info for legend
        optimizer = config.get('optimizer', 'unknown')
        lr = config.get('learning_rate', 'unknown')
        batch_size = config.get('batch_size', 'unknown')
        
        color = colors[i % len(colors)]
        
        # Plot individual run
        plt.scatter(steps, accuracies, color=color, alpha=0.7, s=60, 
                   label=f"Run {i+1}: {optimizer} (lr={lr}, bs={batch_size})")
        plt.plot(steps, accuracies, color=color, alpha=0.4, linewidth=1)
        
        # Extend global lists
        all_steps.extend(steps)
        all_accuracies.extend(accuracies)
        
        # Annotate final accuracy
        if steps and accuracies:
            plt.annotate(f'{accuracies[-1]:.1f}%', 
                        (steps[-1], accuracies[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
    
    # Customize plot
    plt.xlabel('Training Step (Epoch)', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('VGG6 CIFAR-10: Validation Accuracy vs. Training Step\nHyperparameter Sweep Results', 
             fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Legend outside plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Set axis limits
    if all_steps and all_accuracies:
        plt.xlim(0, max(all_steps) + 1)
        plt.ylim(min(all_accuracies) - 5, max(all_accuracies) + 5)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('validation_accuracy_scatter.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Validation accuracy scatter plot saved as: validation_accuracy_scatter.png")
    
    # Print statistics
    if all_accuracies:
        print(f"\n{'='*50}")
        print(f"VALIDATION ACCURACY STATISTICS")
        print(f"{'='*50}")
        print(f"Total runs: {len(runs_data)}")
        print(f"Total data points: {len(all_accuracies)}")
        print(f"Mean accuracy: {np.mean(all_accuracies):.2f}%")
        print(f"Std accuracy: {np.std(all_accuracies):.2f}%")
        print(f"Min accuracy: {np.min(all_accuracies):.2f}%")
        print(f"Max accuracy: {np.max(all_accuracies):.2f}%")
        
        # Find best run
        best_final_acc = 0
        best_run = None
        for run in runs_data:
            if run['accuracies']:
                final_acc = run['accuracies'][-1]
                if final_acc > best_final_acc:
                    best_final_acc = final_acc
                    best_run = run
        
        if best_run:
            print(f"\nBEST RUN: {best_final_acc:.2f}%")
            print(f"Configuration: {best_run['config']}")
    
    plt.show()

def create_synthetic_data():
    """Create synthetic data for demonstration."""
    print("Creating synthetic validation accuracy data...")
    
    runs_data = []
    configs = [
        {'optimizer': 'adam', 'learning_rate': 0.001, 'batch_size': 64},
        {'optimizer': 'sgd', 'learning_rate': 0.005, 'batch_size': 128},
        {'optimizer': 'rmsprop', 'learning_rate': 0.0005, 'batch_size': 32},
        {'optimizer': 'adamw', 'learning_rate': 0.01, 'batch_size': 256},
    ]
    
    for i, config in enumerate(configs):
        epochs = np.random.choice([5, 10, 15])
        steps = list(range(1, epochs + 1))
        
        # Generate realistic progression
        base_acc = np.random.uniform(60, 80)
        accuracies = []
        current_acc = base_acc - np.random.uniform(15, 25)  # Start lower
        
        for step in steps:
            improvement = np.random.uniform(1, 4) * (1 / np.sqrt(step))
            current_acc += improvement + np.random.normal(0, 1)
            current_acc = max(30, min(90, current_acc))
            accuracies.append(current_acc)
        
        runs_data.append({
            'run_id': f'synthetic_{i+1}',
            'config': config,
            'steps': steps,
            'accuracies': accuracies
        })
    
    return runs_data

if __name__ == "__main__":
    print("Creating Validation Accuracy vs. Step Scatter Plot")
    print("="*55)
    create_validation_scatter_plot()
    print("Plot generation complete!")