# VGG6 CIFAR-10 Deep Learning Assignment

A comprehensive PyTorch implementation of VGG6 neural network for CIFAR-10 image classification with hyperparameter optimization, optimizer comparison, and activation function analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Assignment Questions](#assignment-questions)
- [Hyperparameter Analysis](#hyperparameter-analysis)
- [Results & Performance](#results--performance)
- [Architecture Details](#architecture-details)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)

## 🎯 Overview

This repository contains three complete assignments implementing VGG6 architecture for CIFAR-10 classification:

- **Question 1**: Basic VGG6 implementation with standard training
- **Question 2**: Optimizer comparison across 8 different algorithms
- **Question 3**: **Advanced hyperparameter optimization** with W&B integration and parallel coordinate analysis

**Best Results Achieved**: **76.37% test accuracy** on CIFAR-10 using optimized hyperparameters.

## ✨ Features

### 🏗️ **Optimized VGG6 Architecture**
- 6 convolutional layers with batch normalization
- 3 max pooling layers for spatial downsampling
- 2 fully connected layers with dropout regularization
- **693,322 parameters** - compact yet effective
- Designed specifically for 32×32×3 CIFAR-10 images

### 🚀 **Advanced Data Augmentation**
- **AutoAugment CIFAR10 policy** - state-of-the-art augmentation
- **Cutout regularization** - randomly mask image patches
- Random cropping with padding and horizontal flipping
- Per-channel normalization with CIFAR-10 statistics

### 💻 **Cross-Platform Support**
- **NVIDIA CUDA** GPU acceleration
- **Apple Silicon** (M1/M2/M3) MPS optimization
- CPU fallback support
- Automatic hardware detection and optimization

### 🔬 **Hyperparameter Optimization**
- **Weights & Biases** integration for experiment tracking
- **Bayesian optimization** across 6 hyperparameters
- **Parallel coordinate plots** for visualization
- **Real-time training metrics** logging

### 📊 **Comprehensive Analysis**
- **8 optimizer comparison**: SGD variants, Adam family, adaptive methods
- **5 activation functions**: ReLU, GELU, SiLU, Sigmoid, Tanh
- **Training visualization**: Loss curves, accuracy plots, validation analysis
- **Best configuration identification** with performance metrics

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd assignment-1
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install torch torchvision numpy pillow

# For W&B integration (Question 3)
pip install wandb pandas matplotlib seaborn

# For visualization
pip install plotly  # Optional for enhanced plots
```

### 4. Login to W&B (For Question 3)
```bash
wandb login
```
Follow the prompts to authenticate with your W&B account.

## 🚀 How to Run

### Quick Start - Best Results
For the **complete hyperparameter optimization experience** (recommended):
```bash
python "vgg6-question-3-assignment .py"
```
This will automatically run the **Bayesian hyperparameter sweep** and generate **parallel coordinate plots**.

### Individual Assignments

#### **Question 1: Basic VGG6 Implementation**
```bash
python vgg6-question-1-assignment.py
```
- Trains VGG6 with default hyperparameters
- Generates basic training/validation curves
- Saves model checkpoints

#### **Question 2: Optimizer Comparison**
```bash
python vgg6-question-2-assignment.py
```
Tests all 8 optimizers:
- SGD (vanilla, momentum, Nesterov)
- Adam, AdamW, Adagrad, RMSprop, NAdam

#### **Question 3: Advanced Hyperparameter Analysis**
```bash
python "vgg6-question-3-assignment .py"
```
**Features:**
- **20 hyperparameter combinations** via Bayesian optimization
- **Real-time W&B logging** with interactive dashboards
- **Parallel coordinate plots** showing hyperparameter relationships
- **Best configuration identification**
- **Automatic result visualization**

### 🎛️ Configuration Options

Edit the `CONFIG` dictionary in any script to customize:

```python
CONFIG = {
    # W&B Configuration (Question 3)
    'RUN_WANDB_SWEEP': True,          # Enable hyperparameter sweep
    'PREVIEW_WANDB_SWEEP': False,     # Show config without running
    
    # Alternative Testing Modes
    'TEST_ALL_OPTIMIZERS': False,     # Sequential optimizer testing
    'TEST_ALL_ACTIVATIONS': False,    # Activation function analysis
    'TEST_HYPERPARAMETERS': False,    # Predefined config testing
    
    # Training Parameters
    'EPOCHS': 10,                     # Training epochs
    'BATCH_SIZE': 128,               # Mini-batch size
    'LEARNING_RATE': 0.001,          # Base learning rate
    'WEIGHT_DECAY': 0.001,           # L2 regularization
}
```

## 📝 Assignment Questions

### **Question 1**: Basic Implementation
- ✅ Implement VGG6 architecture for CIFAR-10
- ✅ Train with standard hyperparameters
- ✅ Achieve baseline performance
- ✅ Generate training curves

### **Question 2**: Optimizer Analysis
- ✅ Compare 8 different optimizers
- ✅ Analyze convergence characteristics
- ✅ Document performance differences
- ✅ Provide optimization recommendations

### **Question 3**: Hyperparameter Optimization
- ✅ Implement W&B hyperparameter sweep
- ✅ **Create parallel coordinate plots**
- ✅ **Answer: "Which configuration achieves what accuracy?"**
- ✅ **Best result: 76.37% test accuracy**
- ✅ Generate comprehensive analysis

## 🔍 Hyperparameter Analysis

### Hyperparameters Explored
| Parameter | Values | Impact |
|-----------|--------|--------|
| **Batch Size** | [32, 64, 128, 256] | Memory vs. convergence stability |
| **Learning Rate** | [0.0001, 0.0005, 0.001, 0.005, 0.01] | Most critical parameter |
| **Optimizer** | ['adam', 'sgd', 'rmsprop', 'adamw'] | Algorithm choice |
| **Activation** | ['relu', 'gelu', 'silu'] | Nonlinearity selection |
| **Epochs** | [5, 10, 15] | Training duration |
| **Weight Decay** | [0.0001, 0.001, 0.01] | Regularization strength |

### 🏆 Best Configuration Found
```python
# Achieved 76.37% Test Accuracy
{
    'batch_size': 32,
    'learning_rate': 0.005,
    'optimizer': 'sgd',
    'activation': 'gelu', 
    'epochs': 15,
    'weight_decay': 0.001
}
```

### Key Insights
1. **SGD outperformed adaptive optimizers** (Adam, RMSprop) for this architecture
2. **GELU activation** provided best results vs ReLU/SiLU
3. **Learning rate 0.005** hit optimal balance
4. **Smaller batch size (32)** improved convergence
5. **15 epochs** sufficient for peak performance

## 📈 Results & Performance

### Performance Summary
- **Best Test Accuracy**: **76.37%** (excellent for CIFAR-10)
- **Mean Accuracy**: 58.61% across all 20 configurations
- **Standard Deviation**: 17.24% (good hyperparameter sensitivity)
- **Model Parameters**: 693,322 (compact architecture)

### Training Examples
**Best Run (SGD + GELU):**
- Epoch 1: 23% → Epoch 15: 65% training accuracy
- Final test accuracy: 77.48% 
- Excellent convergence with minimal overfitting

**Poor Run (RMSprop + high LR):**
- Plateaued at ~20% accuracy
- Demonstrates importance of hyperparameter tuning

### Visualization
- **W&B Dashboard**: Interactive parallel coordinate plots
- **Training Curves**: Real-time loss and accuracy monitoring  
- **Hyperparameter Importance**: Statistical analysis of parameter impact
- **Configuration Comparison**: Side-by-side performance analysis

## 🏗️ Architecture Details

### VGG6 Network Structure
```
Input: (3, 32, 32) CIFAR-10 images

Block 1: Conv(64) → Conv(64) → MaxPool  → (64, 16, 16)
Block 2: Conv(128) → Conv(128) → MaxPool → (128, 8, 8) 
Block 3: Conv(256) → MaxPool             → (256, 4, 4)

Global Average Pool                      → (256, 1, 1)
Flatten                                  → (256,)

Classifier: Linear(256→512) → Activation → Dropout(0.5) → Linear(512→10)
Output: (10,) class logits
```

### Advanced Features
- **Batch Normalization**: After each convolutional layer
- **Dropout Regularization**: 50% in classifier to prevent overfitting
- **Adaptive Global Pooling**: Handles variable input sizes
- **Smart Weight Initialization**: Kaiming/Xavier based on activation
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive adjustment

## 🔧 Advanced Features

### Data Pipeline
```python
# Training Augmentation
transforms.Compose([
    transforms.RandomCrop(32, padding=4),    # Random cropping
    transforms.RandomHorizontalFlip(),       # 50% horizontal flip
    AutoAugmentCIFAR10(),                    # Advanced augmentation
    Cutout(n_holes=1, length=16),           # Cutout regularization
    transforms.ToTensor(),
    transforms.Normalize(mean, std)           # CIFAR-10 normalization
])
```

### Optimizer Configuration
```python
# Support for 8 optimizers with optimal settings
optimizers = {
    'sgd': optim.SGD(lr=lr, momentum=0.9, weight_decay=wd),
    'adam': optim.Adam(lr=lr, betas=(0.9, 0.999), weight_decay=wd),
    'adamw': optim.AdamW(lr=lr, betas=(0.9, 0.999), weight_decay=wd),
    'rmsprop': optim.RMSprop(lr=lr, alpha=0.99, weight_decay=wd),
    # ... and more
}
```

### W&B Integration
```python
# Automatic experiment tracking
wandb.log({
    'epoch': epoch,
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'train_loss': train_loss,
    'learning_rate': current_lr,
    'best_test_accuracy': best_acc
})
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in CONFIG
CONFIG['BATCH_SIZE'] = 64  # or 32
```

**2. W&B Authentication**
```bash
wandb login --relogin
```

**3. MPS (Apple Silicon) Issues**
```python
# Set pin_memory=False for MPS
DataLoader(..., pin_memory=False)
```

**4. Module Import Errors**
```bash
# Install missing dependencies
pip install wandb pandas matplotlib seaborn
```

**5. Slow Training**
```python
# Enable hardware acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
```

### Performance Optimization
- **GPU Memory**: Use mixed precision training for larger models
- **Data Loading**: Set `num_workers=4` in DataLoader for faster I/O
- **Batch Size**: Larger batches for better GPU utilization
- **Learning Rate**: Scale with batch size: `lr = base_lr * batch_size / 256`

## 📦 Dependencies

### Core Requirements
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pillow>=8.0.0
```

### W&B Integration
```txt
wandb>=0.15.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### Optional Enhancements
```txt
plotly>=5.0.0        # Interactive plots
tensorboard>=2.8.0   # Alternative logging
tqdm>=4.62.0         # Progress bars
```

### Installation Command
```bash
pip install torch torchvision numpy pillow wandb pandas matplotlib seaborn
```

## 🎯 Quick Results

Want to see the best results immediately? Run:

```bash
# 1. Full hyperparameter sweep (recommended)
python "vgg6-question-3-assignment .py"

# 2. Check W&B dashboard at:
# https://wandb.ai/your-username/vgg6-hyperparameter-sweep

# 3. View parallel coordinate plots showing:
# "Which configuration achieves what accuracy?"
```

**Expected Results:**
- 🎯 **76.37% best test accuracy**
- 📊 **20 hyperparameter combinations tested**
- ⏱️ **~6-7 hours total runtime** (20 min per run)
- 📈 **Interactive parallel coordinate plots**
- 🔍 **Optimal configuration identification**

---

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review W&B dashboard for experiment details
3. Verify all dependencies are installed correctly
4. Ensure proper hardware acceleration is enabled

**Happy experimenting! 🚀**

## Usage

### Environment Setup

1. Create and activate a virtual environment:

   ```bash
   # Using venv (Python's built-in virtual environment)
   python -m venv venv
   
   # On macOS/Linux
   source venv/bin/activate
   
   # On Windows
   .\venv\Scripts\activate
   ```

2. Install PyTorch based on your system:

   ```bash
   # For CUDA (NVIDIA GPU)
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For Apple Silicon (M1/M2)
   pip3 install torch torchvision
   
   # For CPU only
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. Install other dependencies:
   ```bash
   pip install numpy pillow
   ```

4. Verify the installation:
   ```python
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```

### Running the Training

Run the training script:
```bash
python vgg6-question-1-assignment.py
```

The script will automatically:
1. Detect available hardware (CUDA/MPS/CPU)
2. Download and prepare CIFAR-10 dataset
3. Initialize the VGG6 model
4. Train for the specified number of epochs
5. Save the best performing model

### Common Issues and Solutions

1. CUDA Out of Memory:
   - Reduce `BATCH_SIZE` in the CONFIG dictionary
   - Close other GPU-intensive applications

2. Slow Training on CPU:
   - Ensure you have the correct PyTorch version for your hardware
   - Reduce the number of worker processes if system is overloaded

3. Apple Silicon (M1/M2) Issues:
   - Ensure you have Rosetta 2 installed if using x86 Python
   - Use native ARM64 Python for best performance
   - Update to latest PyTorch version for MPS optimization

## Configuration

Key hyperparameters can be modified in the `CONFIG` dictionary:

```python
CONFIG = {
    'SEED': 42,              # Random seed
    'NUM_CLASSES': 10,       # CIFAR-10 classes
    'BATCH_SIZE': 128,       # Training batch size
    'LEARNING_RATE': 0.01,   # Initial learning rate
    'WEIGHT_DECAY': 0.001,   # L2 regularization
    'EPOCHS': 100,           # Training epochs
}
```

## Architecture

The VGG6 architecture consists of:

```
Input (32x32x3)
│
├── Conv2D(64) → BN → ReLU
├── Conv2D(64) → BN → ReLU
├── MaxPool2D
│
├── Conv2D(128) → BN → ReLU
├── Conv2D(128) → BN → ReLU
├── MaxPool2D
│
├── Conv2D(256) → BN → ReLU
├── MaxPool2D
│
├── AdaptiveAvgPool2D
│
├── Linear(256→512) → ReLU → Dropout(0.5)
└── Linear(512→10)
```

## Performance Optimizations

- **Data Loading**
  - Adaptive number of worker processes
  - Persistent workers
  - Prefetch queue
  - Pinned memory when available

- **Training**
  - Batch normalization for faster convergence
  - Learning rate scheduling
  - Memory-efficient data augmentation
  - Platform-specific optimizations

## License

[Specify your license here]

## Author

Joshua Sumanth Kumar Relton