#!/usr/bin/env python3
"""
Simple Validation Accuracy vs. Step Scatter Plot

Creates a clean scatter plot showing validation accuracy progression over training steps.
"""

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path
import re

def create_validation_scatter():
    """Create a clean validation accuracy vs step scatter plot."""
    
    # Extract data from W&B runs
    wandb_dir = Path("wandb")
    runs_data = []
    
    run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
    
    print(f"Extracting validation data from {len(run_dirs)} runs...")
    
    for run_dir in run_dirs:
        files_dir = run_dir / "files"
        if not files_dir.exists():
            continue
        
        # Load config for labeling
        config = {}
        config_file = files_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
                for k, v in raw_config.items():
                    if isinstance(v, dict) and 'value' in v:
                        config[k] = v['value']
        
        # Extract validation accuracies from log
        log_file = files_dir / "output.log"
        epochs = []
        val_accs = []
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if "Epoch" in line and "/5:" in line or "/10:" in line or "/15:" in line:
                    epoch_match = re.search(r"Epoch (\d+)/", line)
                    if epoch_match and i+2 < len(lines):
                        test_line = lines[i+2]  # Test accuracy is usually 2 lines after
                        test_match = re.search(r"Test Accuracy: ([\d.]+)%", test_line)
                        if test_match:
                            epochs.append(int(epoch_match.group(1)))
                            val_accs.append(float(test_match.group(1)))
        
        if epochs and val_accs:
            optimizer = config.get('optimizer', 'unknown')
            lr = config.get('learning_rate', 0)
            bs = config.get('batch_size', 0)
            runs_data.append({
                'epochs': epochs,
                'val_accs': val_accs,
                'optimizer': optimizer,
                'lr': lr,
                'bs': bs,
                'run_id': run_dir.name.split('-')[-1][:8]
            })
    
    print(f"Successfully extracted data from {len(runs_data)} runs")
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Color scheme for optimizers
    optimizer_colors = {
        'adam': '#1f77b4',
        'sgd': '#ff7f0e',
        'rmsprop': '#2ca02c',
        'adamw': '#d62728'
    }
    
    all_epochs = []
    all_accs = []
    
    for i, run in enumerate(runs_data):
        color = optimizer_colors.get(run['optimizer'], '#333333')
        
        # Plot points and connect with lines
        plt.scatter(run['epochs'], run['val_accs'], 
                   color=color, s=60, alpha=0.7, 
                   label=f"{run['optimizer'].upper()} (lr={run['lr']}, bs={run['bs']})" if i < 10 else "")
        
        plt.plot(run['epochs'], run['val_accs'], 
                color=color, alpha=0.4, linewidth=1.5)
        
        # Collect all data
        all_epochs.extend(run['epochs'])
        all_accs.extend(run['val_accs'])
        
        # Annotate final accuracy
        if run['epochs'] and run['val_accs']:
            final_epoch = run['epochs'][-1]
            final_acc = run['val_accs'][-1]
            plt.annotate(f'{final_acc:.1f}%', 
                        (final_epoch, final_acc),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    # Customize plot
    plt.xlabel('Training Step (Epoch)', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('VGG6 CIFAR-10: Validation Accuracy vs. Training Step\nHyperparameter Sweep Results', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Set reasonable limits
    if all_epochs and all_accs:
        plt.xlim(0.5, max(all_epochs) + 0.5)
        plt.ylim(min(all_accs) - 5, max(all_accs) + 5)
    
    # Add summary statistics as text
    if all_accs:
        stats_text = f'''Summary Statistics:
Mean Accuracy: {np.mean(all_accs):.1f}%
Best Accuracy: {np.max(all_accs):.1f}%
Runs Analyzed: {len(runs_data)}
Total Points: {len(all_accs)}'''
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig('validation_accuracy_scatter_final.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("Final validation accuracy scatter plot saved as: validation_accuracy_scatter_final.png")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"VALIDATION ACCURACY SCATTER PLOT SUMMARY")
    print(f"{'='*50}")
    if all_accs:
        print(f"Runs analyzed: {len(runs_data)}")
        print(f"Total data points: {len(all_accs)}")
        print(f"Accuracy range: {np.min(all_accs):.1f}% - {np.max(all_accs):.1f}%")
        print(f"Mean accuracy: {np.mean(all_accs):.2f}%")
        print(f"Standard deviation: {np.std(all_accs):.2f}%")
        
        # Find best performing run
        best_final_acc = 0
        best_run = None
        for run in runs_data:
            if run['val_accs']:
                final_acc = run['val_accs'][-1]
                if final_acc > best_final_acc:
                    best_final_acc = final_acc
                    best_run = run
        
        if best_run:
            print(f"\nBest Final Accuracy: {best_final_acc:.1f}%")
            print(f"Configuration: {best_run['optimizer']} optimizer, lr={best_run['lr']}, batch_size={best_run['bs']}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating Validation Accuracy vs. Step Scatter Plot")
    print("="*55)
    create_validation_scatter()
    print("Plot creation complete!")