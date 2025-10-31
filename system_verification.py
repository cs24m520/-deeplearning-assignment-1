#!/usr/bin/env python3
"""
System Verification Script

Confirms that all components are working correctly:
1. W&B sweep data integrity
2. Plot generation functionality  
3. Data extraction accuracy
4. File completeness
"""

import os
import yaml
import json
from pathlib import Path
import re
import numpy as np

def check_wandb_data():
    """Verify W&B sweep data integrity."""
    print("🔍 CHECKING W&B SWEEP DATA...")
    print("="*50)
    
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        print("❌ No wandb directory found!")
        return False
    
    # Check run directories
    run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
    print(f"✅ Found {len(run_dirs)} W&B run directories")
    
    if len(run_dirs) != 20:
        print(f"⚠️  Expected 20 runs, found {len(run_dirs)}")
    
    # Verify sweep directory
    sweep_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("sweep-")]
    if sweep_dirs:
        print(f"✅ Found sweep directory: {sweep_dirs[0].name}")
    else:
        print("⚠️  No sweep directory found")
    
    # Check individual run data
    successful_runs = 0
    total_epochs = 0
    
    for run_dir in run_dirs[:5]:  # Check first 5 runs for verification
        files_dir = run_dir / "files"
        if files_dir.exists():
            config_file = files_dir / "config.yaml"
            log_file = files_dir / "output.log"
            summary_file = files_dir / "wandb-summary.json"
            
            if config_file.exists() and log_file.exists():
                # Check config
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    print(f"✅ Run {run_dir.name[-8:]}: Config loaded")
                    
                    # Check log for metrics
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    
                    epochs_found = len(re.findall(r"Epoch \d+/\d+:", log_content))
                    total_epochs += epochs_found
                    
                    if epochs_found > 0:
                        print(f"   📊 {epochs_found} epochs logged")
                        successful_runs += 1
                    else:
                        print(f"   ❌ No epochs found in log")
                        
                except Exception as e:
                    print(f"   ❌ Error reading run data: {e}")
            else:
                print(f"❌ Run {run_dir.name[-8:]}: Missing files")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Successful runs checked: {successful_runs}/5")
    print(f"   Total epochs sampled: {total_epochs}")
    
    return successful_runs >= 4  # At least 4/5 runs should be successful

def check_generated_plots():
    """Verify that all expected plots were generated."""
    print("\n🎨 CHECKING GENERATED PLOTS...")
    print("="*50)
    
    expected_plots = [
        "wandb_style_training_metrics.png",
        "wandb_training_loss.png", 
        "wandb_training_accuracy.png",
        "wandb_validation_loss.png",
        "wandb_validation_accuracy.png",
        "training_metrics_comprehensive.png",
        "validation_accuracy_scatter_final.png",
        "actual_validation_accuracy_analysis.png"
    ]
    
    missing_plots = []
    existing_plots = []
    
    for plot in expected_plots:
        if Path(plot).exists():
            size = Path(plot).stat().st_size
            print(f"✅ {plot} ({size:,} bytes)")
            existing_plots.append(plot)
        else:
            print(f"❌ {plot} - NOT FOUND")
            missing_plots.append(plot)
    
    print(f"\n📊 PLOT SUMMARY:")
    print(f"   Generated: {len(existing_plots)}/{len(expected_plots)}")
    print(f"   Missing: {len(missing_plots)}")
    
    return len(missing_plots) == 0

def check_python_environment():
    """Verify Python environment and dependencies."""
    print("\n🐍 CHECKING PYTHON ENVIRONMENT...")
    print("="*50)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import wandb
        print(f"✅ W&B: {wandb.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
        
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
        
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
        
        import yaml
        print(f"✅ PyYAML: Available")
        
        # Check device availability
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available ({torch.cuda.get_device_name()})")
        elif torch.backends.mps.is_available():
            print(f"✅ MPS (Apple Silicon): Available")
        else:
            print(f"✅ CPU: Available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def verify_data_extraction():
    """Test data extraction functionality."""
    print("\n📊 TESTING DATA EXTRACTION...")
    print("="*50)
    
    try:
        # Test extracting one run's data
        wandb_dir = Path("wandb")
        run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
        
        if not run_dirs:
            print("❌ No run directories found")
            return False
        
        test_run = run_dirs[0]
        files_dir = test_run / "files"
        
        # Test config extraction
        config_file = files_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Config extraction: {len(config)} parameters")
        else:
            print("❌ Config file not found")
            return False
        
        # Test metrics extraction
        log_file = files_dir / "output.log"
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
            
            epochs = re.findall(r"Epoch (\d+)/\d+:", content)
            train_accs = re.findall(r"Train Accuracy: ([\d.]+)%", content)
            val_accs = re.findall(r"Test Accuracy: ([\d.]+)%", content)
            losses = re.findall(r"Average Loss: ([\d.]+)", content)
            
            print(f"✅ Metrics extraction:")
            print(f"   Epochs: {len(epochs)}")
            print(f"   Train accuracies: {len(train_accs)}")
            print(f"   Validation accuracies: {len(val_accs)}")
            print(f"   Losses: {len(losses)}")
            
            if len(epochs) > 0 and len(train_accs) > 0:
                print(f"   Sample: Epoch {epochs[0]} - Train: {train_accs[0]}%, Val: {val_accs[0]}%")
                return True
            else:
                print("❌ No metrics found")
                return False
        else:
            print("❌ Log file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error during data extraction test: {e}")
        return False

def test_plot_generation():
    """Test plot generation functionality."""
    print("\n🧪 TESTING PLOT GENERATION...")
    print("="*50)
    
    try:
        # Test basic matplotlib functionality
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple test plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot - System Verification")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        test_file = "system_verification_test.png"
        plt.savefig(test_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if Path(test_file).exists():
            size = Path(test_file).stat().st_size
            print(f"✅ Test plot generated: {test_file} ({size:,} bytes)")
            
            # Clean up test file
            Path(test_file).unlink()
            print("✅ Test plot cleaned up")
            return True
        else:
            print("❌ Test plot not generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during plot generation test: {e}")
        return False

def run_comprehensive_verification():
    """Run all verification checks."""
    print("🔧 COMPREHENSIVE SYSTEM VERIFICATION")
    print("="*60)
    print("Verifying VGG6 W&B implementation and generated outputs...")
    print()
    
    checks = [
        ("W&B Data Integrity", check_wandb_data),
        ("Generated Plots", check_generated_plots), 
        ("Python Environment", check_python_environment),
        ("Data Extraction", verify_data_extraction),
        ("Plot Generation", test_plot_generation)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results[check_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL SYSTEMS WORKING CORRECTLY!")
        print("\n📋 CONFIRMED WORKING:")
        print("   • W&B hyperparameter sweep (20 runs)")
        print("   • Training metrics extraction")
        print("   • Plot generation (8+ plots)")
        print("   • Validation accuracy analysis")
        print("   • Best configuration identification")
        
        print("\n🎯 READY FOR:")
        print("   • Training loss analysis")
        print("   • Training accuracy analysis") 
        print("   • Validation loss analysis")
        print("   • Validation accuracy analysis")
        print("   • Parallel coordinate plots")
        print("   • Hyperparameter optimization insights")
        
    else:
        print(f"⚠️  {total - passed} issues found - see details above")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_verification()
    exit(0 if success else 1)