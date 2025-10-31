#!/usr/bin/env python3
"""
Validation Accuracy vs. Step Scatter Plot Generator

This script creates scatter plots showing validation accuracy progression
across training steps for the VGG6 hyperparameter sweep experiments.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import json
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_wandb_data():
    """
    Load W&B run data from local wandb directory.
    
    Returns:
        list: List of run data dictionaries
    """
    wandb_dir = Path("wandb")
    runs_data = []
    
    if not wandb_dir.exists():
        print("No wandb directory found. Make sure the sweep has run.")
        return runs_data
    
    # Find all run directories
    run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
    
    print(f"Found {len(run_dirs)} W&B runs")
    
    for run_dir in run_dirs:
        try:
            # Load run metadata
            wandb_metadata = run_dir / "wandb-metadata.json"
            wandb_summary = run_dir / "wandb-summary.json"
            
            run_data = {
                'run_dir': str(run_dir),
                'run_id': run_dir.name.split('-')[-1],
                'validation_accuracies': [],
                'steps': [],
                'config': {}
            }
            
            # Load config and summary if available
            if wandb_metadata.exists():
                with open(wandb_metadata, 'r') as f:
                    metadata = json.load(f)
                    run_data['config'] = metadata.get('config', {})
            
            if wandb_summary.exists():
                with open(wandb_summary, 'r') as f:
                    summary = json.load(f)
                    run_data['final_accuracy'] = summary.get('final_test_accuracy', 0)
                    run_data['best_accuracy'] = summary.get('best_test_accuracy', 0)
            
            # Look for events.out.tfevents files or CSV logs
            events_files = list(run_dir.glob("*.csv"))
            if events_files:
                # If CSV logs exist, load them
                for csv_file in events_files:
                    try:
                        df = pd.read_csv(csv_file)
                        if 'test_accuracy' in df.columns and 'epoch' in df.columns:
                            run_data['validation_accuracies'] = df['test_accuracy'].tolist()
                            run_data['steps'] = df['epoch'].tolist()
                            break
                    except:
                        continue
            
            runs_data.append(run_data)
            
        except Exception as e:
            print(f"Error processing run {run_dir}: {e}")
            continue
    
    return runs_data

def generate_synthetic_data():
    """
    Generate synthetic validation accuracy data for demonstration.
    
    Returns:
        list: List of synthetic run data
    """
    print("Generating synthetic validation accuracy data for demonstration...")
    
    # Simulate different hyperparameter configurations
    configs = [
        {'optimizer': 'adam', 'lr': 0.001, 'batch_size': 64, 'activation': 'relu'},
        {'optimizer': 'sgd', 'lr': 0.005, 'batch_size': 128, 'activation': 'gelu'},
        {'optimizer': 'rmsprop', 'lr': 0.0005, 'batch_size': 32, 'activation': 'silu'},
        {'optimizer': 'adamw', 'lr': 0.01, 'batch_size': 256, 'activation': 'relu'},
        {'optimizer': 'adam', 'lr': 0.0001, 'batch_size': 64, 'activation': 'gelu'},
        {'optimizer': 'sgd', 'lr': 0.001, 'batch_size': 128, 'activation': 'relu'},
    ]
    
    runs_data = []
    
    for i, config in enumerate(configs):
        # Simulate training progression
        epochs = np.random.choice([5, 10, 15])
        steps = list(range(1, epochs + 1))
        
        # Generate realistic validation accuracy progression
        base_accuracy = np.random.uniform(40, 70)  # Starting accuracy
        improvement_rate = np.random.uniform(0.5, 3.0)  # How much it improves
        noise_level = np.random.uniform(0.5, 2.0)  # Noise in progression
        
        accuracies = []
        for step in steps:
            # Logarithmic improvement with noise
            accuracy = base_accuracy + improvement_rate * np.log(step + 1) + np.random.normal(0, noise_level)
            accuracy = max(10, min(95, accuracy))  # Clamp between 10-95%
            accuracies.append(accuracy)
        
        run_data = {
            'run_id': f'synthetic_{i+1:02d}',
            'validation_accuracies': accuracies,
            'steps': steps,
            'config': config,
            'final_accuracy': accuracies[-1],
            'best_accuracy': max(accuracies)
        }
        
        runs_data.append(run_data)
    
    return runs_data

def create_validation_accuracy_plot(runs_data, save_path="validation_accuracy_scatter.png"):
    """
    Create validation accuracy vs. step scatter plot.
    
    Args:
        runs_data (list): List of run data dictionaries
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(14, 10))
    
    # Create subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])
    
    # Color map for different optimizers
    optimizer_colors = {
        'adam': '#1f77b4',
        'sgd': '#ff7f0e', 
        'rmsprop': '#2ca02c',
        'adamw': '#d62728',
        'adagrad': '#9467bd',
        'nadam': '#8c564b'
    }
    
    all_steps = []
    all_accuracies = []
    run_labels = []
    
    # Main scatter plot
    for i, run in enumerate(runs_data):
        if not run['validation_accuracies'] or not run['steps']:
            continue
            
        steps = run['steps']
        accuracies = run['validation_accuracies']
        config = run['config']
        
        # Extend to global lists for statistics
        all_steps.extend(steps)
        all_accuracies.extend(accuracies)
        run_labels.extend([f"Run {i+1}"] * len(steps))
        
        # Get optimizer and color
        optimizer = config.get('optimizer', 'unknown')
        color = optimizer_colors.get(optimizer, '#333333')
        
        # Create label for legend
        lr = config.get('lr', config.get('learning_rate', 'unknown'))
        batch_size = config.get('batch_size', 'unknown')
        activation = config.get('activation', 'unknown')
        
        label = f"{optimizer.upper()} (lr={lr}, bs={batch_size}, act={activation})"
        
        # Plot scatter points
        ax1.scatter(steps, accuracies, 
                   color=color, alpha=0.7, s=60, 
                   label=label, edgecolors='white', linewidth=0.5)
        
        # Connect points with lines for each run
        ax1.plot(steps, accuracies, 
                color=color, alpha=0.3, linewidth=1, linestyle='-')
        
        # Annotate final accuracy
        if steps and accuracies:
            final_step = steps[-1]
            final_acc = accuracies[-1]
            ax1.annotate(f'{final_acc:.1f}%', 
                        (final_step, final_acc),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    # Customize main plot
    ax1.set_xlabel('Training Step (Epoch)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('VGG6 Validation Accuracy vs. Training Step\nHyperparameter Sweep Results', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Set reasonable axis limits
    if all_steps and all_accuracies:
        ax1.set_xlim(0, max(all_steps) + 1)
        ax1.set_ylim(min(all_accuracies) - 5, max(all_accuracies) + 5)
    
    # Statistics subplot
    if all_accuracies:
        # Create accuracy distribution plot
        ax2.hist(all_accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(all_accuracies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_accuracies):.2f}%')
        ax2.axvline(np.median(all_accuracies), color='orange', linestyle='--',
                   label=f'Median: {np.median(all_accuracies):.2f}%')
        
        ax2.set_xlabel('Validation Accuracy (%)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Validation Accuracies', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Validation accuracy scatter plot saved to: {save_path}")
    
    # Display statistics
    if all_accuracies:
        print(f"\n{'='*50}")
        print(f"VALIDATION ACCURACY STATISTICS")
        print(f"{'='*50}")
        print(f"Total data points: {len(all_accuracies)}")
        print(f"Number of runs: {len(runs_data)}")
        print(f"Mean accuracy: {np.mean(all_accuracies):.2f}%")
        print(f"Median accuracy: {np.median(all_accuracies):.2f}%")
        print(f"Standard deviation: {np.std(all_accuracies):.2f}%")
        print(f"Min accuracy: {np.min(all_accuracies):.2f}%")
        print(f"Max accuracy: {np.max(all_accuracies):.2f}%")
        print(f"Accuracy range: {np.max(all_accuracies) - np.min(all_accuracies):.2f}%")
        
        # Best runs
        print(f"\nTOP 3 BEST FINAL ACCURACIES:")
        final_accuracies = [(run['final_accuracy'], run['config'], run['run_id']) 
                           for run in runs_data if 'final_accuracy' in run]
        final_accuracies.sort(reverse=True, key=lambda x: x[0])
        
        for i, (acc, config, run_id) in enumerate(final_accuracies[:3]):
            print(f"{i+1}. {acc:.2f}% - {config} (Run: {run_id})")
    
    plt.show()
    return fig

def main():
    """Main function to generate validation accuracy plot."""
    print("Generating Validation Accuracy vs. Step Scatter Plot")
    print("="*60)
    
    # Try to load actual W&B data first
    runs_data = load_wandb_data()
    
    # If no real data, generate synthetic data for demonstration
    if not runs_data or all(not run.get('validation_accuracies') for run in runs_data):
        print("No validation accuracy data found in W&B logs.")
        print("This could mean:")
        print("1. The sweep is still running")
        print("2. Data logging was not set up correctly")
        print("3. The sweep hasn't completed yet")
        print("\nGenerating synthetic data for demonstration...")
        runs_data = generate_synthetic_data()
    else:
        print(f"Loaded validation data from {len(runs_data)} W&B runs")
    
    # Create the plot
    fig = create_validation_accuracy_plot(runs_data)
    
    print(f"\nPlot generation complete!")
    print(f"File saved as: validation_accuracy_scatter.png")

if __name__ == "__main__":
    main()