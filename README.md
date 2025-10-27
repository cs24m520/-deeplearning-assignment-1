# VGG6 Implementation for CIFAR-10

A PyTorch implementation of a simplified VGG network (VGG6) optimized for CIFAR-10 image classification. This implementation features advanced data augmentation techniques and cross-platform optimizations.

## Features

- **Optimized VGG6 Architecture**
  - 6 convolutional layers with batch normalization
  - 3 max pooling layers
  - 2 fully connected layers with dropout
  - Designed for 32x32x3 CIFAR-10 images

- **Advanced Data Augmentation**
  - AutoAugment CIFAR10 policy
  - Cutout regularization
  - Random cropping and horizontal flipping
  - Per-channel normalization

- **Cross-Platform Support**
  - NVIDIA CUDA GPU acceleration
  - Apple Silicon (M-series) optimization
  - CPU fallback support
  - Automatic hardware detection

- **Training Optimizations**
  - Adaptive learning rate scheduling
  - Best model checkpointing
  - Early stopping capability
  - Efficient data loading pipeline

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- torchvision
- NumPy
- Pillow (PIL)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd assignment-1
   ```

2. Install the required packages:
   ```bash
   pip install torch torchvision numpy pillow
   ```

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

## Acknowledgments

- AutoAugment policy implementation based on the paper "AutoAugment: Learning Augmentation Policies from Data"
- VGG architecture inspired by "Very Deep Convolutional Networks for Large-Scale Image Recognition"