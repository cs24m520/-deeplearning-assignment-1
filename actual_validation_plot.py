#!/usr/bin/env python3
"""
Actual W&B Validation Accuracy vs. Step Scatter Plot

Extracts and plots real validation accuracy data from completed W&B sweep.
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

def extract_real_wandb_data():
    """Extract actual validation accuracy data from W&B runs."""
    wandb_dir = Path("wandb")
    runs_data = []
    
    if not wandb_dir.exists():
        return []
    
    run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
    print(f"Processing {len(run_dirs)} W&B runs...")
    
    for run_dir in run_dirs:
        try:
            files_dir = run_dir / "files"
            if not files_dir.exists():
                continue
            
            run_data = {
                'run_id': run_dir.name.split('-')[-1],
                'config': {},
                'validation_accuracies': [],
                'train_accuracies': [],
                'epochs': [],
                'losses': []
            }
            
            # Load config
            config_file = files_dir / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    # Extract scalar values from nested config structure
                    clean_config = {}
                    if isinstance(config, dict):
                        for k, v in config.items():
                            if isinstance(v, dict) and 'value' in v:
                                clean_config[k] = v['value']
                            elif not isinstance(v, dict):
                                clean_config[k] = v
                    run_data['config'] = clean_config
            
            # Parse log file for actual training metrics
            log_file = files_dir / "output.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Extract epoch-by-epoch results
                lines = content.split('\n')
                for line in lines:
                    # Look for epoch completion pattern
                    epoch_pattern = r"Epoch (\d+)/\d+:"
                    train_acc_pattern = r"Train Accuracy: ([\d.]+)%"
                    test_acc_pattern = r"Test Accuracy: ([\d.]+)%"
                    loss_pattern = r"Average Loss: ([\d.]+)"
                    
                    epoch_match = re.search(epoch_pattern, line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        
                        # Look for the metrics in subsequent lines
                        line_idx = lines.index(line)
                        if line_idx + 3 < len(lines):
                            train_line = lines[line_idx + 1]
                            test_line = lines[line_idx + 2]
                            loss_line = lines[line_idx + 3]
                            
                            train_match = re.search(train_acc_pattern, train_line)
                            test_match = re.search(test_acc_pattern, test_line)
                            loss_match = re.search(loss_pattern, loss_line)
                            
                            if train_match and test_match and loss_match:
                                run_data['epochs'].append(current_epoch)
                                run_data['train_accuracies'].append(float(train_match.group(1)))
                                run_data['validation_accuracies'].append(float(test_match.group(1)))
                                run_data['losses'].append(float(loss_match.group(1)))
            
            if run_data['validation_accuracies']:
                runs_data.append(run_data)
                config_str = f"{run_data['config'].get('optimizer', '?')}, lr={run_data['config'].get('learning_rate', '?')}, bs={run_data['config'].get('batch_size', '?')}"
                print(f"✓ Run {run_data['run_id']}: {len(run_data['validation_accuracies'])} epochs, {config_str}")
                
        except Exception as e:
            print(f"✗ Error processing {run_dir.name}: {e}")
            continue
    
    return runs_data

def create_actual_validation_plot(runs_data):
    """Create validation accuracy scatter plot from actual data."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VGG6 CIFAR-10: Actual Validation Accuracy Results\nHyperparameter Sweep Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))
    
    all_val_accs = []
    all_epochs = []
    optimizer_results = {}
    lr_results = {}
    bs_results = {}
    
    # Plot 1: Individual validation trajectories
    for i, (run, color) in enumerate(zip(runs_data, colors)):
        epochs = run['epochs']
        val_accs = run['validation_accuracies']
        config = run['config']
        
        optimizer = config.get('optimizer', 'unknown')
        lr = config.get('learning_rate', 0)
        bs = config.get('batch_size', 0)
        
        # Plot trajectory
        ax1.plot(epochs, val_accs, color=color, linewidth=2, alpha=0.7, 
                marker='o', markersize=4, label=f"{optimizer} (lr={lr})")
        
        # Collect data for analysis
        all_val_accs.extend(val_accs)
        all_epochs.extend(epochs)
        
        # Group by optimizer
        if optimizer not in optimizer_results:
            optimizer_results[optimizer] = []
        optimizer_results[optimizer].append(max(val_accs))
        
        # Group by learning rate
        if lr not in lr_results:
            lr_results[lr] = []
        lr_results[lr].append(max(val_accs))
        
        # Group by batch size
        if bs not in bs_results:
            bs_results[bs] = []
        bs_results[bs].append(max(val_accs))
        
        # Annotate final accuracy
        final_acc = val_accs[-1]
        ax1.annotate(f'{final_acc:.1f}%', (epochs[-1], final_acc),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontweight='bold') 
    ax1.set_title('Validation Accuracy Trajectories', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Optimizer comparison
    opt_names = list(optimizer_results.keys())
    opt_means = [np.mean(optimizer_results[opt]) for opt in opt_names]
    opt_stds = [np.std(optimizer_results[opt]) if len(optimizer_results[opt]) > 1 else 0 for opt in opt_names]
    
    bars = ax2.bar(opt_names, opt_means, yerr=opt_stds, capsize=5, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(opt_names)], alpha=0.7)
    ax2.set_ylabel('Best Validation Accuracy (%)', fontweight='bold')
    ax2.set_title('Performance by Optimizer', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, mean in zip(bars, opt_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Learning rate vs performance
    lrs = sorted(lr_results.keys())
    lr_means = [np.mean(lr_results[lr]) for lr in lrs]
    
    ax3.scatter(lrs, lr_means, s=100, alpha=0.7, color='red')
    for lr, mean in zip(lrs, lr_means):
        for acc in lr_results[lr]:
            ax3.scatter(lr, acc, alpha=0.4, s=30, color='blue')
    
    ax3.set_xlabel('Learning Rate', fontweight='bold')
    ax3.set_ylabel('Best Validation Accuracy (%)', fontweight='bold')
    ax3.set_title('Learning Rate vs Performance', fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Batch size vs performance  
    batch_sizes = sorted(bs_results.keys())
    bs_means = [np.mean(bs_results[bs]) for bs in batch_sizes]
    
    ax4.scatter(batch_sizes, bs_means, s=100, alpha=0.7, color='green')
    for bs, mean in zip(batch_sizes, bs_means):
        for acc in bs_results[bs]:
            ax4.scatter(bs, acc, alpha=0.4, s=30, color='orange')
    
    ax4.set_xlabel('Batch Size', fontweight='bold')
    ax4.set_ylabel('Best Validation Accuracy (%)', fontweight='bold')
    ax4.set_title('Batch Size vs Performance', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('actual_validation_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    print("Actual validation accuracy analysis saved as: actual_validation_accuracy_analysis.png")
    
    # Print detailed statistics
    print(f"\n{'='*60}")
    print(f"ACTUAL W&B SWEEP VALIDATION ACCURACY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total runs analyzed: {len(runs_data)}")
    print(f"Total validation measurements: {len(all_val_accs)}")
    print(f"Mean validation accuracy: {np.mean(all_val_accs):.2f}%")
    print(f"Std validation accuracy: {np.std(all_val_accs):.2f}%")
    print(f"Best validation accuracy: {np.max(all_val_accs):.2f}%")
    print(f"Worst validation accuracy: {np.min(all_val_accs):.2f}%")
    
    # Find best configuration
    best_acc = 0
    best_run = None
    for run in runs_data:
        run_best = max(run['validation_accuracies'])
        if run_best > best_acc:
            best_acc = run_best
            best_run = run
    
    if best_run:
        print(f"\nBEST CONFIGURATION: {best_acc:.2f}%")
        for key, value in best_run['config'].items():
            print(f"  {key}: {value}")
    
    # Optimizer summary
    print(f"\nPERFORMANCE BY OPTIMIZER:")
    for opt in sorted(optimizer_results.keys()):
        accs = optimizer_results[opt]
        print(f"  {opt:8}: {np.mean(accs):.2f}% ± {np.std(accs):.2f}% (n={len(accs)})")
    
    return fig

def main():
    """Main function."""
    print("Extracting Actual W&B Validation Accuracy Data")
    print("="*60)
    
    # Extract real data
    runs_data = extract_real_wandb_data()
    
    if not runs_data:
        print("No validation accuracy data found in W&B logs!")
        print("Make sure the sweep has completed and generated output.log files.")
        return
    
    print(f"\nSuccessfully extracted data from {len(runs_data)} runs")
    
    # Create comprehensive analysis plot
    create_actual_validation_plot(runs_data)
    
    print("\nValidation accuracy analysis complete!")
    print("Check the generated PNG file for detailed visualization.")

if __name__ == "__main__":
    main()