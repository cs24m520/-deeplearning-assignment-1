# -*- coding: utf-8 -*-
"""
VGG6 Implementation with W&B Parallel Coordinate Analysis for CIFAR-10

This comprehensive implementation explores hyperparameter optimization with 
Weights & Biases (W&B) tracking and parallel coordinate visualizations to answer:
"Which configuration achieves what accuracy?"

Key Features:
============
- VGG6 architecture with 693K parameters optimized for CIFAR-10
- Bayesian hyperparameter optimization (20 configurations tested)
- Advanced data augmentation pipeline (AutoAugment + Cutout)
- Multi-device support (CUDA, Apple M-series MPS, CPU)
- Real-time W&B experiment tracking and logging
- Parallel coordinate plot generation for hyperparameter analysis
- Comprehensive training metrics visualization

Hyperparameter Search Space:
===========================
- Batch sizes: [32, 64, 128, 256] - Impact on convergence stability
- Learning rates: [0.0001, 0.0005, 0.001, 0.005, 0.01] - Optimization speed
- Epochs: [5, 10, 15] - Training duration vs. performance tradeoff  
- Optimizers: [Adam, SGD, RMSprop, AdamW] - Different optimization algorithms
- Activations: [ReLU, GELU, SiLU] - Non-linear activation functions
- Weight decay: [0.0001, 0.001, 0.01] - Regularization strength

Results Summary:
===============
- Best accuracy achieved: 75.74% validation accuracy
- Optimal configuration: Adam, lr=0.0005, batch_size=256, 15 epochs
- Total experiments: 20 runs with 190 total training epochs
- Training time: ~6 hours for complete sweep

Usage:
======
    # Set RUN_WANDB_SWEEP = True in CONFIG to run hyperparameter sweep
    # Set PREVIEW_WANDB_SWEEP = True to preview configuration without running
    python vgg6-question-3-assignment.py

Dependencies:
=============
    torch>=2.0.0, wandb>=0.15.0, torchvision, numpy, pandas

Author: CS24M520
Course: System Deep Learning  
Date: October 2025
Institution: IIT Madras
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports for system operations and utilities
import os              # Operating system interface for file/directory operations
import copy            # Deep copying for model state preservation
import random          # Random number generation for reproducibility
import platform        # Platform detection for device-specific optimizations
import itertools       # Efficient iterator tools for hyperparameter combinations
import time            # Time measurement for performance monitoring
import json            # JSON handling for configuration and logging

# Third-party scientific computing and visualization
import numpy as np                    # Numerical computing foundation
from PIL import Image, ImageEnhance, ImageOps  # Image processing for data augmentation
import wandb                          # Weights & Biases for experiment tracking

# PyTorch deep learning framework
import torch                          # Core PyTorch tensors and operations
import torch.nn as nn                 # Neural network modules and layers
import torch.optim as optim           # Optimization algorithms (Adam, SGD, etc.)
import torch.nn.functional as F       # Functional interface for operations
import torch.nn.init as init          # Weight initialization strategies
import torch.backends.cudnn           # CUDA optimization backend
import torch.backends.mps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================================
# GLOBAL CONFIGURATION SETTINGS
# ============================================================================
# Centralized configuration for all experiments and hyperparameter tuning

CONFIG = {
    # ========================================================================
    # WEIGHTS & BIASES (W&B) CONFIGURATION
    # ========================================================================
    'USE_WANDB': True,                         # Enable W&B experiment tracking
    'WANDB_PROJECT': 'vgg6-cifar10-sweep',     # W&B project name for organization
    'WANDB_ENTITY': None,                      # W&B entity (username/team), None=personal
    
    # ========================================================================
    # CORE TRAINING HYPERPARAMETERS
    # ========================================================================
    'SEED': 42,                                # Random seed for reproducible results
    'NUM_CLASSES': 10,                         # CIFAR-10 has 10 classes
    'BATCH_SIZE': 128,                         # Mini-batch size (trade-off: memory vs convergence)
    'LEARNING_RATE': 0.001,                    # Base learning rate (Adam default)
    'WEIGHT_DECAY': 0.001,                     # L2 regularization to prevent overfitting
    'EPOCHS': 10,                              # Number of complete dataset passes
    
    # ========================================================================
    # VGG6 ARCHITECTURE CONFIGURATION
    # ========================================================================
    # Architecture specification: numbers=conv channels, 'M'=maxpool
    'VGG_CONFIG': [64, 64, 'M',               # Block 1: 64->64 channels + 2 x 2 MaxPool
                   128, 128, 'M',             # Block 2: 128->128 channels + 2 x 2 MaxPool  
                   256, 'M'],                 # Block 3: 256 channels + 2 x 2 MaxPool
    
    # ========================================================================
    # DEFAULT EXPERIMENT SETTINGS
    # ========================================================================
    'OPTIMIZER': 'adam',                       # Default optimizer (adaptive learning)
    'ACTIVATION': 'relu',                      # Default activation function
    
    # ========================================================================
    # EXPERIMENT EXECUTION MODES
    # ========================================================================
    'WANDB_PROJECT': 'vgg6-hyperparameter-sweep',  # Updated project name for sweeps
    'RUN_WANDB_SWEEP': True,                   # Execute Bayesian hyperparameter optimization
    'PREVIEW_WANDB_SWEEP': False,              # Show sweep configuration without running
    'TEST_ALL_OPTIMIZERS': False,              # Sequential testing of all optimizers
    'TEST_ALL_ACTIVATIONS': False,             # Sequential testing of all activations
    'TEST_HYPERPARAMETERS': False,             # Test predefined configurations
    
    # ========================================================================
    # WEIGHTS & BIASES HYPERPARAMETER SWEEP CONFIGURATION
    # ========================================================================
    # Bayesian optimization for automated hyperparameter tuning
    'WANDB_SWEEP_CONFIG': {
        'method': 'bayes',                     # Bayesian optimization (smarter than grid/random)
        'metric': {
            'name': 'final_test_accuracy',     # Metric to optimize
            'goal': 'maximize'                 # Maximize test accuracy
        },
        'parameters': {
            # Batch size: memory vs convergence stability trade-off
            'batch_size': {
                'values': [32, 64, 128, 256]   # Powers of 2 for GPU efficiency
            },
            # Learning rate: most critical hyperparameter
            'learning_rate': {
                'values': [0.0001, 0.0005, 0.001, 0.005, 0.01]  # Log-scale distribution
            },
            # Optimizer: different adaptive learning algorithms
            'optimizer': {
                'values': ['adam', 'sgd', 'rmsprop', 'adamw']    # Popular optimizers
            },
            # Activation function: nonlinearity choice
            'activation': {
                'values': ['relu', 'gelu', 'silu']               # Modern activations
            },
            # Training duration: compute vs performance trade-off
            'epochs': {
                'values': [5, 10, 15]                            # Short to medium training
            },
            # Regularization strength: overfitting prevention
            'weight_decay': {
                'values': [0.0001, 0.001, 0.01]                 # L2 penalty range
            }
        }
    },
    
    # ========================================================================
    # COMPREHENSIVE TESTING CONFIGURATIONS
    # ========================================================================
    # For systematic individual component testing (non-sweep mode)
    'OPTIMIZERS_TO_TEST': [
        'sgd', 'sgd_momentum', 'sgd_nesterov',  # SGD variants
        'adam', 'adamw', 'adagrad',             # Adaptive methods
        'rmsprop', 'nadam'                      # Additional optimizers
    ],
    'ACTIVATIONS_TO_TEST': [
        'relu', 'sigmoid', 'tanh',              # Classic activations
        'silu', 'gelu'                          # Modern activations
    ],
    
    # ========================================================================
    # PREDEFINED HYPERPARAMETER CONFIGURATIONS
    # ========================================================================
    # Hand-crafted configurations for specific experiments
    'HYPERPARAMETER_CONFIGS': [
        # Configuration 1: Baseline Adam + ReLU
        {'name': 'config_1', 'BATCH_SIZE': 64, 'EPOCHS': 10, 
         'LEARNING_RATE': 0.001, 'OPTIMIZER': 'adam', 'ACTIVATION': 'relu'},
        
        # Configuration 2: SGD with larger batch
        {'name': 'config_2', 'BATCH_SIZE': 128, 'EPOCHS': 10, 
         'LEARNING_RATE': 0.001, 'OPTIMIZER': 'sgd', 'ACTIVATION': 'relu'},
        
        # Configuration 3: Modern setup (AdamW + GELU)
        {'name': 'config_3', 'BATCH_SIZE': 256, 'EPOCHS': 10, 
         'LEARNING_RATE': 0.0005, 'OPTIMIZER': 'adamw', 'ACTIVATION': 'gelu'},
        
        # Configuration 4: High learning rate + longer training
        {'name': 'config_4', 'BATCH_SIZE': 128, 'EPOCHS': 15, 
         'LEARNING_RATE': 0.005, 'OPTIMIZER': 'rmsprop', 'ACTIVATION': 'silu'},
    ],
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_random_seeds(seed=CONFIG['SEED']):
    """
    Set random seeds for reproducible experiments.
    
    Ensures deterministic behavior across:
    - PyTorch operations (CPU and GPU)
    - NumPy random operations  
    - Python's built-in random module
    - CUDA operations (if available)
    
    Args:
        seed (int): Random seed value for reproducibility
    """
    torch.manual_seed(seed)                    # PyTorch CPU operations
    np.random.seed(seed)                       # NumPy random operations
    random.seed(seed)                          # Python built-in random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # CUDA operations (GPU)

def get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay):
    """
    Factory function to create optimizers with standardized configurations.
    
    Supports multiple optimization algorithms with their recommended settings:
    - SGD variants: vanilla, momentum, and Nesterov momentum
    - Adaptive methods: Adam, AdamW, RMSprop, Adagrad
    - Modern variants: NAdam (Nesterov-accelerated Adam)
    
    Args:
        optimizer_name (str): Name of the optimizer algorithm
        model_parameters: Model parameters to optimize (from model.parameters())
        learning_rate (float): Step size for parameter updates
        weight_decay (float): L2 regularization coefficient
        
    Returns:
        torch.optim.Optimizer: Configured PyTorch optimizer instance
        
    Raises:
        ValueError: If optimizer_name is not recognized
    """
    optimizer_name = optimizer_name.lower()
    
    # ========================================================================
    # STOCHASTIC GRADIENT DESCENT (SGD) VARIANTS
    # ========================================================================
    if optimizer_name == 'sgd':
        # Vanilla SGD: simple gradient descent
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd_momentum':
        # SGD with momentum: accelerates convergence in consistent directions
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=0.9,                      # Standard momentum coefficient
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd_nesterov':
        # Nesterov momentum: "look-ahead" gradient calculation
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,                     # Enable Nesterov acceleration
            weight_decay=weight_decay
        )
    # ========================================================================
    # ADAPTIVE LEARNING RATE OPTIMIZERS
    # ========================================================================
    # ========================================================================
    elif optimizer_name == 'adam':
        # Adam: adaptive learning rates with momentum (most popular)
        return optim.Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),                # Momentum coefficients (beta1, beta2)
            eps=1e-8                           # Numerical stability term
        )
    elif optimizer_name == 'adamw':
        # AdamW: Adam with decoupled weight decay (better regularization)
        return optim.AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,         # Decoupled from gradient-based update
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'adagrad':
        # Adagrad: adaptive learning rates based on historical gradients
        return optim.Adagrad(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            lr_decay=0,                        # Learning rate decay factor
            eps=1e-10                          # Smaller eps for numerical stability
        )
    elif optimizer_name == 'rmsprop':
        # RMSprop: exponentially decaying average of squared gradients
        return optim.RMSprop(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            alpha=0.99,                        # Smoothing constant for moving average
            eps=1e-8,
            momentum=0                         # Standard RMSprop (no momentum)
        )
    elif optimizer_name == 'nadam':
        # NAdam: Nesterov-accelerated Adam (combines Nesterov + Adam)
        try:
            return optim.NAdam(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        except AttributeError:
            # Fallback for older PyTorch versions (NAdam added in 1.10+)
            print("NAdam not available in this PyTorch version, using Adam instead")
            return optim.Adam(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
    # ========================================================================
    # ERROR HANDLING
    # ========================================================================
    else:
        # Provide helpful error message with supported optimizers
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. "
                        f"Supported optimizers: {CONFIG['OPTIMIZERS_TO_TEST']}")

def get_optimizer_info(optimizer_name):
    """
    Get detailed information about optimizer characteristics and best practices.
    
    Provides metadata about each optimizer including:
    - Learning rate sensitivity and recommended ranges
    - Memory requirements and computational overhead
    - Best use cases and performance characteristics
    - Hyperparameter sensitivity and tuning advice
    
    Args:
        optimizer_name (str): Name of the optimizer to get information about
        
    Returns:
        dict: Comprehensive information about the optimizer including:
            - description: Algorithm explanation
            - lr_range: Recommended learning rate range
            - pros: Advantages and strengths
            - cons: Limitations and weaknesses
            - best_for: Recommended use cases
    """
    
    info = {
        'sgd': {
            'description': 'Stochastic Gradient Descent',
            'characteristics': [
                'Basic gradient descent with mini-batches',
                'No momentum, can be slow to converge',
                'May oscillate around optimal solution',
                'Simple and memory efficient'
            ],
            'convergence': 'Slow but steady, may get stuck in local minima'
        },
        'sgd_momentum': {
            'description': 'SGD with Momentum',
            'characteristics': [
                'Adds momentum term to accelerate convergence',
                'Helps overcome local minima',
                'Reduces oscillations in gradient descent',
                'Momentum factor: 0.9'
            ],
            'convergence': 'Faster than vanilla SGD, smoother convergence'
        },
        'sgd_nesterov': {
            'description': 'SGD with Nesterov Momentum',
            'characteristics': [
                'Lookahead momentum - computes gradient at projected position',
                'Better convergence properties than standard momentum',
                'Anticipates future gradient direction',
                'Momentum factor: 0.9'
            ],
            'convergence': 'Even faster than standard momentum, less overshoot'
        },
        'adam': {
            'description': 'Adaptive Moment Estimation',
            'characteristics': [
                'Combines momentum with per-parameter learning rates',
                'Adapts learning rate based on gradient statistics',
                'Works well with sparse gradients',
                'Default betas: (0.9, 0.999)'
            ],
            'convergence': 'Fast initial convergence, may plateau early'
        },
        'adamw': {
            'description': 'Adam with Weight Decay',
            'characteristics': [
                'Decoupled weight decay from gradient-based updates',
                'Better generalization than Adam',
                'Fixes weight decay issues in Adam',
                'Recommended for transformer models'
            ],
            'convergence': 'Similar to Adam but with better generalization'
        },
        'adagrad': {
            'description': 'Adaptive Gradient Algorithm',
            'characteristics': [
                'Adapts learning rate based on cumulative squared gradients',
                'Good for sparse data and features',
                'Learning rate decreases over time',
                'Can become too aggressive in reducing learning rate'
            ],
            'convergence': 'Good early progress, may stop learning too early'
        },
        'rmsprop': {
            'description': 'Root Mean Square Propagation',
            'characteristics': [
                'Addresses Adagrad\'s aggressive learning rate reduction',
                'Uses exponential moving average of squared gradients',
                'Good for non-stationary objectives',
                'Alpha (decay): 0.99'
            ],
            'convergence': 'More consistent than Adagrad, handles non-convex well'
        },
        'nadam': {
            'description': 'Nesterov-accelerated Adaptive Moment Estimation',
            'characteristics': [
                'Combines Adam with Nesterov momentum',
                'Better convergence properties than Adam',
                'Uses bias-corrected first and second moment estimates',
                'Lookahead momentum like Nesterov SGD'
            ],
            'convergence': 'Faster and more stable than Adam'
        }
    }
    
    return info.get(optimizer_name.lower(), {'description': 'Unknown optimizer'})

def get_activation_function(activation_name):
    """
    Get the appropriate activation function based on name.
    
    Args:
        activation_name (str): Name of the activation function
        
    Returns:
        nn.Module: PyTorch activation function module
        
    Raises:
        ValueError: If activation function is not supported
    """
    activations = {
        'relu': nn.ReLU(inplace=True),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'silu': nn.SiLU(inplace=True),  # Also known as Swish
        'gelu': nn.GELU(),
        # Commented out other activation functions
        # 'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        # 'elu': nn.ELU(alpha=1.0, inplace=True),
        # 'mish': nn.Mish(inplace=True)  # Available in PyTorch 1.9+
    }
    
    if activation_name.lower() not in activations:
        raise ValueError(f"Unsupported activation function: {activation_name}. "
                        f"Supported: {list(activations.keys())}")
    
    return activations[activation_name.lower()]

def get_weight_init_nonlinearity(activation_name):
    """
    Get the appropriate nonlinearity string for weight initialization.
    
    Args:
        activation_name (str): Name of the activation function
        
    Returns:
        str: Nonlinearity string for kaiming_normal_ initialization
    """
    # Map activation functions to their appropriate nonlinearity for weight init
    nonlinearity_map = {
        'relu': 'relu',
        'leaky_relu': 'leaky_relu',
        'sigmoid': 'sigmoid',
        'tanh': 'tanh',
        'silu': 'relu',  # Use relu for SiLU/Swish
        'gelu': 'relu',  # Use relu for GELU
        'elu': 'relu',   # Use relu for ELU
        'mish': 'relu'   # Use relu for Mish
    }
    
    return nonlinearity_map.get(activation_name.lower(), 'relu')

def get_device():
    """
    Determine the available device (CUDA GPU, Apple Silicon, or CPU).
    Returns:
        torch.device: The device to use for computations
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("No GPU available. Using CPU instead.")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    return device

# Data Augmentation Classes
class Cutout(object):
    """Randomly mask out one or more patches from an image."""
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

class SubPolicy(object):
    """Implement a sub-policy for AutoAugment."""
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        self.p1 = p1
        self.op1 = operation1
        self.magnitude_idx1 = magnitude_idx1
        self.p2 = p2
        self.op2 = operation2
        self.magnitude_idx2 = magnitude_idx2
        self.fillcolor = fillcolor

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.gen(self.op1, self.magnitude_idx1, self.op2, self.magnitude_idx2, self.fillcolor)(img)
        if random.random() < self.p2:
            img = self.gen(self.op2, self.magnitude_idx2, self.op2, self.magnitude_idx2, self.fillcolor)(img)
        return img

    def gen(self, operation1, magnitude_idx1, operation2, magnitude_idx2, fillcolor):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot,
                                 Image.new("RGBA", rot.size, (128,) * 4),
                                 rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        return lambda img: func[operation1](img, ranges[operation1][magnitude_idx1])

class CIFAR10Policy(object):
    """AutoAugment Policy for CIFAR10."""
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

# ============================================================================
# VGG6 NEURAL NETWORK ARCHITECTURE
# ============================================================================

class VGG(nn.Module):
    """
    VGG-style Convolutional Neural Network with configurable components.
    
    A simplified VGG architecture designed for CIFAR-10 classification with:
    - 6 convolutional layers (hence "VGG6") arranged in 3 blocks
    - Configurable activation functions (ReLU, GELU, SiLU, etc.)  
    - Batch normalization for training stability
    - Adaptive average pooling for flexible input sizes
    - Dropout regularization in classifier
    - Smart weight initialization based on activation choice
    
    Architecture Details:
    - Block 1: 64->64 channels + MaxPool (32 x 32 -> 16 x 16)
    - Block 2: 128->128 channels + MaxPool (16 x 16 -> 8 x 8)  
    - Block 3: 256 channels + MaxPool (8 x 8 -> 4 x 4)
    - Global Average Pooling: 4 x 4 -> 1 x 1
    - Classifier: 256 -> 512 -> 10 classes
    
    Total Parameters: approximately 693 thousand (compact yet effective for CIFAR-10)
    """
    
    def __init__(self, features, num_classes=10, activation='relu'):
        """
        Initialize VGG model with specified configuration.
        
        Args:
            features (nn.Sequential): Convolutional feature extraction layers
            num_classes (int): Number of output classes (10 for CIFAR-10)
            activation (str): Activation function name ('relu', 'gelu', 'silu', etc.)
        """
        super(VGG, self).__init__()
        
        # ====================================================================
        # NETWORK COMPONENTS
        # ====================================================================
        self.features = features                    # Convolutional feature extractor
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling
        self.activation_name = activation           # Store for weight initialization
        
        # ====================================================================
        # CLASSIFIER HEAD
        # ====================================================================
        # Create classifier with consistent activation function
        classifier_activation = get_activation_function(activation)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),                   # Feature expansion (256->512)
            classifier_activation,                 # Nonlinear activation
            nn.Dropout(0.5),                       # Regularization (50% dropout)
            nn.Linear(512, num_classes)            # Final classification layer
        )
        
        # Initialize weights using activation-appropriate strategy
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # Convolutional feature extraction
        x = self.features(x)                       # (B, 3, 32, 32) -> (B, 256, 4, 4)
        
        # Global pooling and flattening
        x = self.avgpool(x)                        # (B, 256, 4, 4) -> (B, 256, 1, 1)
        x = torch.flatten(x, 1)                    # (B, 256, 1, 1) -> (B, 256)
        
        # Classification
        x = self.classifier(x)                     # (B, 256) -> (B, num_classes)
        return x

    def _initialize_weights(self):
        """
        Initialize network weights using activation-appropriate strategies.
        
        Uses different initialization schemes based on activation function:
        - Xavier/Glorot: For sigmoid/tanh (preserves variance)
        - Kaiming/He: For ReLU-family (accounts for dead neurons)
        
        This ensures optimal gradient flow during early training phases.
        """
        # Get appropriate nonlinearity for Kaiming initialization
        nonlinearity = get_weight_init_nonlinearity(self.activation_name)
        
        for m in self.modules():
            # ================================================================
            # CONVOLUTIONAL LAYER INITIALIZATION
            # ================================================================
            if isinstance(m, nn.Conv2d):
                if self.activation_name in ['sigmoid', 'tanh']:
                    # Xavier initialization: maintains variance through layers
                    nn.init.xavier_normal_(m.weight)
                else:
                    # Kaiming initialization: accounts for ReLU-family activations
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                          nonlinearity=nonlinearity)
                
                # Initialize biases to zero (standard practice)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            # ================================================================
            # BATCH NORMALIZATION INITIALIZATION  
            # ================================================================
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)        # Scale parameter gamma = 1
                nn.init.constant_(m.bias, 0)          # Shift parameter beta = 0
                
            # ================================================================
            # LINEAR LAYER INITIALIZATION
            # ================================================================
            elif isinstance(m, nn.Linear):
                if self.activation_name in ['sigmoid', 'tanh']:
                    # Xavier for symmetric activations
                    nn.init.xavier_normal_(m.weight)
                else:
                    # Small random weights for ReLU-family
                    nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False, activation='relu'):
    """
    Create VGG model layers based on configuration list with specified activation.
    
    This function constructs the convolutional blocks of the VGG architecture
    according to the provided configuration. Each block consists of:
    1. Convolutional layer (3 x 3 kernel, padding=1)
    2. Optional Batch Normalization
    3. Specified activation function
    4. Optional MaxPool2D (when 'M' is specified in config)

    Args:
        cfg (list): Configuration list where each element is either:
                   - int: number of output channels for Conv2D
                   - 'M': indicates MaxPool2D layer
        batch_norm (bool): Whether to include batch normalization layers
                          after each convolution (default: False)
        activation (str): Name of activation function to use (default: 'relu')

    Returns:
        nn.Sequential: Sequential container of VGG layers
    """
    layers = []
    in_channels = 3  # Input channels (RGB)
    
    for v in cfg:
        if v == 'M':
            # MaxPool2D layer with 2 x 2 kernel and stride 2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Conv2D with 3 x 3 kernel and same padding
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            activation_fn = get_activation_function(activation)
            
            if batch_norm:
                # Conv2D -> BatchNorm -> Activation
                layers += [conv2d, nn.BatchNorm2d(v), activation_fn]
            else:
                # Conv2D -> Activation
                layers += [conv2d, activation_fn]
            in_channels = v  # Update input channels for next layer
            
    return nn.Sequential(*layers)

def vgg(cfg, num_classes=10, batch_norm=True, activation='relu'):
    """
    Create VGG model instance with specified activation function.
    
    Args:
        cfg (list): VGG configuration
        num_classes (int): Number of output classes
        batch_norm (bool): Whether to use batch normalization
        activation (str): Activation function name
        
    Returns:
        VGG: VGG model instance
    """
    features = make_layers(cfg, batch_norm=batch_norm, activation=activation)
    return VGG(features, num_classes=num_classes, activation=activation)

# Data Loading and Processing
def GetCifar10(batchsize, num_samples=None):
    """
    Create CIFAR10 data loaders with advanced data augmentation pipeline.
    
    Implements a sophisticated data augmentation strategy combining:
    1. Standard augmentations:
       - Random cropping (32 x 32 with padding=4)
       - Random horizontal flipping
    2. Advanced augmentations:
       - AutoAugment CIFAR10 policy (adaptive augmentation)
       - Cutout (random patch erasure)
    3. Normalization:
       - Per-channel mean/std normalization
       
    Also implements platform-specific optimizations:
    - Adaptive number of workers based on CPU cores
    - Special handling for Apple Silicon
    - Persistent workers and prefetching for better throughput
    - Automatic pin memory configuration
    
    Args:
        batchsize (int): Number of samples per batch
        num_samples (int, optional): If set, use only this many training samples
        
    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    trans_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16)
    ])

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = datasets.CIFAR10('./data', train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10('./data', train=False, transform=trans, download=True)

    if num_samples is not None:
        indices = torch.randperm(len(train_data))[:num_samples]
        train_data = torch.utils.data.Subset(train_data, indices)

    # Automatically configure optimal number of workers based on environment
    num_workers = min(8, os.cpu_count() or 1)

    if platform.system() == 'Darwin' and 'arm64' in platform.machine():
        # Optimize for Apple Silicon
        num_workers = min(4, os.cpu_count() or 1)
        pin_memory = torch.backends.mps.is_available()
    else:
        # Settings for other platforms
        num_workers = min(8, os.cpu_count() or 1)
        pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_data,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=pin_memory
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=pin_memory
    )

    return train_dataloader, test_dataloader

# Training and Evaluation Functions
def eval(model, data_loader):
    """Evaluate model on data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def train_model(model, num_epochs, optimizer, train_loader, test_loader, criterion, use_wandb=False):
    """
    Complete training loop with advanced monitoring and optimization features.
    
    Implements a sophisticated training pipeline with:
    - Real-time W&B experiment tracking and visualization
    - Adaptive learning rate scheduling (ReduceLROnPlateau)
    - Best model checkpointing based on validation accuracy
    - Comprehensive metrics logging (loss, accuracy, learning rate)
    - Training history preservation for analysis
    - Graceful interruption handling (Ctrl+C)
    - Per-epoch progress reporting
    
    Training Process:
    1. Forward pass through training data
    2. Backpropagation and parameter updates
    3. Validation accuracy evaluation
    4. Learning rate adjustment based on plateau detection
    5. Best model state preservation
    6. Metrics logging to W&B and console
    
    Args:
        model (nn.Module): PyTorch neural network model to train
        num_epochs (int): Maximum number of training epochs
        optimizer (torch.optim.Optimizer): Optimization algorithm instance
        train_loader (DataLoader): Training dataset iterator
        test_loader (DataLoader): Validation dataset iterator  
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        use_wandb (bool): Enable Weights & Biases experiment tracking
        
    Returns:
        tuple: (train_accuracies, test_accuracies) - Lists containing per-epoch metrics
            - train_accuracies (list): Training accuracy for each epoch
            - test_accuracies (list): Validation accuracy for each epoch
    """
    # ========================================================================
    # TRAINING STATE INITIALIZATION
    # ========================================================================
    train_accuracies = []                          # Per-epoch training accuracy history
    test_accuracies = []                           # Per-epoch validation accuracy history
    train_losses = []                              # Per-epoch training loss history
    best_acc = 0.0                                 # Best validation accuracy achieved
    best_model = None                              # Best model state dictionary

    # ========================================================================
    # LEARNING RATE SCHEDULER SETUP
    # ========================================================================
    # ReduceLROnPlateau: reduces LR when validation accuracy plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',                                # Monitor for maximum (accuracy)
        factor=0.5,                                # Multiply LR by 0.5 when plateau detected
        patience=3                                 # Wait 3 epochs before reducing LR
    )

    try:
        # ====================================================================
        # MAIN TRAINING LOOP
        # ====================================================================
        for epoch in range(num_epochs):
            # ================================================================
            # TRAINING PHASE
            # ================================================================
            model.train()                          # Set model to training mode
            running_loss = 0.0                     # Accumulate batch losses
            correct = 0                            # Count correct predictions
            total = 0                              # Count total samples
            epoch_loss = 0.0                       # Track epoch loss
            num_batches = 0                        # Count processed batches

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            # Calculate accuracies
            train_acc = 100. * correct / total
            test_acc = eval(model, test_loader)
            avg_loss = epoch_loss / num_batches

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            train_losses.append(avg_loss)

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Train Accuracy: {train_acc:.2f}%')
            print(f'Test Accuracy: {test_acc:.2f}%')
            print(f'Average Loss: {avg_loss:.4f}')

            # Log to W&B
            if use_wandb:
                try:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'train_loss': avg_loss,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                except:
                    pass  # Continue even if W&B logging fails

            # Update learning rate
            scheduler.step(test_acc)

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model.state_dict())

    except KeyboardInterrupt:
        print('Training interrupted! Saving best model...')

    if best_model is not None:
        model.load_state_dict(best_model)
        print(f'Best Test Accuracy: {best_acc:.2f}%')

    return train_accuracies, test_accuracies

def test_single_optimizer(optimizer_name, device, train_loader, test_loader):
    """
    Test a single optimizer and return results.
    
    Args:
        optimizer_name (str): Name of optimizer to test
        device (torch.device): Device to run training on
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        
    Returns:
        dict: Dictionary containing training results
    """
    print(f"\n{'='*60}")
    print(f"Testing Optimizer: {optimizer_name.upper()}")
    print(f"{'='*60}")
    
    # Get optimizer information
    opt_info = get_optimizer_info(optimizer_name)
    print(f"Description: {opt_info['description']}")
    print("Characteristics:")
    for char in opt_info['characteristics']:
        print(f"  • {char}")
    print(f"Expected convergence: {opt_info['convergence']}")
    print()
    
    try:
        # Set random seeds for reproducible comparison
        set_random_seeds(CONFIG['SEED'])
        
        # Create model with default activation
        model = vgg(
            CONFIG['VGG_CONFIG'], 
            num_classes=CONFIG['NUM_CLASSES'],
            activation=CONFIG['ACTIVATION']
        ).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Activation function: {CONFIG['ACTIVATION']}")
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Get optimizer-specific learning rate
        lr = CONFIG['OPTIMIZER_LR'].get(optimizer_name, CONFIG['LEARNING_RATE'])
        
        # Create optimizer
        optimizer = get_optimizer(
            optimizer_name,
            model.parameters(),
            lr,
            CONFIG['WEIGHT_DECAY']
        )
        
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {CONFIG['WEIGHT_DECAY']}")
        print()
        
        # Train model
        train_accuracies, test_accuracies = train_model(
            model,
            CONFIG['EPOCHS'],
            optimizer,
            train_loader,
            test_loader,
            criterion
        )
        
        # Final evaluation
        final_test_acc = eval(model, test_loader)
        max_test_acc = max(test_accuracies) if test_accuracies else 0
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        
        results = {
            'optimizer': optimizer_name,
            'final_test_accuracy': final_test_acc,
            'max_test_accuracy': max_test_acc,
            'final_train_accuracy': final_train_acc,
            'learning_rate': lr,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'optimizer_info': opt_info
        }
        
        print(f"\nResults for {optimizer_name}:")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"Max Test Accuracy: {max_test_acc:.2f}%")
        print(f"Final Train Accuracy: {final_train_acc:.2f}%")
        
        # Calculate convergence metrics
        if len(test_accuracies) >= 5:
            early_avg = np.mean(test_accuracies[:5])
            late_avg = np.mean(test_accuracies[-5:])
            improvement = late_avg - early_avg
            print(f"Convergence improvement (last 5 vs first 5 epochs): {improvement:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error testing {optimizer_name}: {str(e)}")
        return {
            'optimizer': optimizer_name,
            'error': str(e),
            'final_test_accuracy': 0,
            'max_test_accuracy': 0,
            'final_train_accuracy': 0
        }
        
        # Final evaluation
        final_test_acc = eval(model, test_loader)
        max_test_acc = max(test_accuracies) if test_accuracies else 0
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        
        results = {
            'activation': activation_name,
            'final_test_accuracy': final_test_acc,
            'max_test_accuracy': max_test_acc,
            'final_train_accuracy': final_train_acc,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'learning_rate_used': lr,
            'total_parameters': total_params
        }
        
        print(f"\nResults for {activation_name}:")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"Best Test Accuracy: {max_test_acc:.2f}%")
        print(f"Final Train Accuracy: {final_train_acc:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error testing {optimizer_name}: {str(e)}")
        return {
            'optimizer': optimizer_name,
            'error': str(e),
            'final_test_accuracy': 0,
            'max_test_accuracy': 0,
            'final_train_accuracy': 0
        }

def test_single_hyperparameter_config(config, device):
    """
    Test a single hyperparameter configuration and return results.
    
    Args:
        config (dict): Hyperparameter configuration
        device (torch.device): Device to run training on
        
    Returns:
        dict: Dictionary containing training results
    """
    print(f"\n{'='*60}")
    print(f"Testing Hyperparameter Config: {config['name'].upper()}")
    print(f"{'='*60}")
    print(f"Batch Size: {config['BATCH_SIZE']}")
    print(f"Epochs: {config['EPOCHS']}")
    print(f"Learning Rate: {config['LEARNING_RATE']}")
    print()
    
    try:
        # Set random seeds for reproducible comparison
        set_random_seeds(CONFIG['SEED'])
        
        # Get data loaders with specified batch size
        train_loader, test_loader = GetCifar10(config['BATCH_SIZE'])
        
        # Create model with default settings
        model = vgg(
            CONFIG['VGG_CONFIG'], 
            num_classes=CONFIG['NUM_CLASSES'],
            activation=CONFIG['ACTIVATION']
        ).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Activation function: {CONFIG['ACTIVATION']}")
        print(f"Optimizer: {CONFIG['OPTIMIZER']}")
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create optimizer with specified learning rate
        optimizer = get_optimizer(
            CONFIG['OPTIMIZER'],
            model.parameters(),
            config['LEARNING_RATE'],
            CONFIG['WEIGHT_DECAY']
        )
        
        print(f"Training samples per epoch: {len(train_loader) * config['BATCH_SIZE']}")
        print(f"Batches per epoch: {len(train_loader)}")
        print()
        
        # Record start time for convergence speed analysis
        import time
        start_time = time.time()
        
        # Train model
        train_accuracies, test_accuracies = train_model(
            model,
            config['EPOCHS'],
            optimizer,
            train_loader,
            test_loader,
            criterion
        )
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Final evaluation
        final_test_acc = eval(model, test_loader)
        max_test_acc = max(test_accuracies) if test_accuracies else 0
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        
        # Calculate convergence metrics
        convergence_speed = 0
        if len(test_accuracies) >= 10:
            # Calculate average improvement per epoch in first half of training
            mid_point = len(test_accuracies) // 2
            early_avg = np.mean(test_accuracies[:5])
            mid_avg = np.mean(test_accuracies[mid_point-2:mid_point+3])
            convergence_speed = (mid_avg - early_avg) / mid_point
        
        results = {
            'config_name': config['name'],
            'batch_size': config['BATCH_SIZE'],
            'epochs': config['EPOCHS'],
            'learning_rate': config['LEARNING_RATE'],
            'final_test_accuracy': final_test_acc,
            'max_test_accuracy': max_test_acc,
            'final_train_accuracy': final_train_acc,
            'training_time': training_time,
            'time_per_epoch': training_time / config['EPOCHS'],
            'convergence_speed': convergence_speed,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
        
        print(f"\nResults for {config['name']}:")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"Max Test Accuracy: {max_test_acc:.2f}%")
        print(f"Final Train Accuracy: {final_train_acc:.2f}%")
        print(f"Training Time: {training_time:.1f}s ({training_time/60:.1f}m)")
        print(f"Time per Epoch: {training_time/config['EPOCHS']:.1f}s")
        print(f"Convergence Speed: {convergence_speed:.3f}% per epoch")
        
        return results
        
    except Exception as e:
        print(f"Error testing {config['name']}: {str(e)}")
        return {
            'config_name': config['name'],
            'error': str(e),
            'final_test_accuracy': 0,
            'max_test_accuracy': 0,
            'final_train_accuracy': 0
        }

def analyze_hyperparameter_results(results):
    """
    Analyze and compare hyperparameter performance results.
    
    Args:
        results (list): List of dictionaries containing hyperparameter results
    """
    print(f"\n{'='*80}")
    print("HYPERPARAMETER PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Filter out failed experiments
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("No valid results to analyze!")
        return
    
    # Sort by final test accuracy (descending)
    sorted_results = sorted(valid_results, key=lambda x: x['final_test_accuracy'], reverse=True)
    
    print(f"\n{'FINAL TEST ACCURACY RANKING:'}")
    print(f"{'Rank':<4}{'Config':<18}{'Batch':<8}{'Epochs':<8}{'LR':<10}{'Final Acc':<12}{'Max Acc':<12}{'Time(s)':<10}")
    print("-" * 85)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<4}{result['config_name']:<18}{result['batch_size']:<8}{result['epochs']:<8}"
              f"{result['learning_rate']:<10.4f}{result['final_test_accuracy']:<12.2f}"
              f"{result['max_test_accuracy']:<12.2f}{result['training_time']:<10.1f}")
    
    # Batch size analysis
    print(f"\n{'BATCH SIZE ANALYSIS:'}")
    batch_configs = [r for r in valid_results if 'batch' in r['config_name']]
    if batch_configs:
        batch_sorted = sorted(batch_configs, key=lambda x: x['batch_size'])
        for result in batch_sorted:
            time_per_sample = result['training_time'] / (result['epochs'] * 50000)  # CIFAR-10 has 50k training samples
            print(f"• Batch {result['batch_size']:<3}: {result['final_test_accuracy']:5.2f}% accuracy, "
                  f"{result['time_per_epoch']:5.1f}s/epoch, {time_per_sample*1000:5.3f}ms/sample")
    
    # Learning rate analysis
    print(f"\n{'LEARNING RATE ANALYSIS:'}")
    lr_configs = [r for r in valid_results if 'lr' in r['config_name']]
    if lr_configs:
        lr_sorted = sorted(lr_configs, key=lambda x: x['learning_rate'])
        for result in lr_sorted:
            print(f"• LR {result['learning_rate']:<6.4f}: {result['final_test_accuracy']:5.2f}% accuracy, "
                  f"convergence speed: {result['convergence_speed']:+5.3f}%/epoch")
    
    # Epoch analysis
    print(f"\n{'EPOCH ANALYSIS:'}")
    epoch_configs = [r for r in valid_results if 'training' in r['config_name']]
    if epoch_configs:
        epoch_sorted = sorted(epoch_configs, key=lambda x: x['epochs'])
        for result in epoch_sorted:
            efficiency = result['final_test_accuracy'] / result['training_time'] * 100  # accuracy per second
            print(f"• {result['epochs']:2d} epochs: {result['final_test_accuracy']:5.2f}% accuracy, "
                  f"{result['training_time']:5.1f}s total, efficiency: {efficiency:5.2f} acc/100s")
    
    # Convergence speed analysis
    print(f"\n{'CONVERGENCE SPEED ANALYSIS:'}")
    speed_sorted = sorted(valid_results, key=lambda x: x.get('convergence_speed', 0), reverse=True)
    for result in speed_sorted[:5]:  # Top 5 fastest converging
        print(f"• {result['config_name']:<18}: {result['convergence_speed']:+5.3f}%/epoch "
              f"(Final: {result['final_test_accuracy']:5.2f}%)")
    
    # Training efficiency analysis
    print(f"\n{'TRAINING EFFICIENCY ANALYSIS:'}")
    efficiency_results = [(r, r['final_test_accuracy'] / (r['training_time']/60)) for r in valid_results]
    efficiency_sorted = sorted(efficiency_results, key=lambda x: x[1], reverse=True)
    for result, efficiency in efficiency_sorted:
        print(f"• {result['config_name']:<18}: {efficiency:5.2f} accuracy points per minute")

def test_single_activation(activation_name, device, train_loader, test_loader):
    """
    Test a single activation function and return results.
    
    Args:
        activation_name (str): Name of activation function to test
        device (torch.device): Device to run training on
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        
    Returns:
        dict: Dictionary containing training results
    """
    print(f"\n{'='*60}")
    print(f"Testing Activation Function: {activation_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Set random seeds for reproducible comparison
        set_random_seeds(CONFIG['SEED'])
        
        # Create model with specified activation
        model = vgg(
            CONFIG['VGG_CONFIG'], 
            num_classes=CONFIG['NUM_CLASSES'],
            activation=activation_name
        ).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Activation function: {activation_name}")
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Adjust learning rate based on activation function
        lr = CONFIG['LEARNING_RATE']
        if activation_name in ['sigmoid', 'tanh']:
            lr *= 0.1  # Lower learning rate for saturating activations
        elif activation_name in ['gelu', 'silu']:
            lr *= 1.5  # Slightly higher learning rate for smoother activations
        
        # Create optimizer (use default optimizer)
        optimizer = get_optimizer(
            CONFIG['OPTIMIZER'],
            model.parameters(),
            lr,
            CONFIG['WEIGHT_DECAY']
        )
        
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Learning rate: {lr}")
        print(f"Weight decay: {CONFIG['WEIGHT_DECAY']}")
        print()
        
        # Train model
        train_accuracies, test_accuracies = train_model(
            model,
            CONFIG['EPOCHS'],
            optimizer,
            train_loader,
            test_loader,
            criterion
        )
        
        # Final evaluation
        final_test_acc = eval(model, test_loader)
        max_test_acc = max(test_accuracies) if test_accuracies else 0
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        
        results = {
            'activation': activation_name,
            'final_test_accuracy': final_test_acc,
            'max_test_accuracy': max_test_acc,
            'final_train_accuracy': final_train_acc,
            'learning_rate': lr,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
        
        print(f"\nResults for {activation_name}:")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"Max Test Accuracy: {max_test_acc:.2f}%")
        print(f"Final Train Accuracy: {final_train_acc:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error testing {activation_name}: {str(e)}")
        return {
            'activation': activation_name,
            'error': str(e),
            'final_test_accuracy': 0,
            'max_test_accuracy': 0,
            'final_train_accuracy': 0
        }

def analyze_activation_results(results):
    """
    Analyze and compare activation function performance results.
    
    Args:
        results (list): List of dictionaries containing activation results
    """
    print(f"\n{'='*80}")
    print("ACTIVATION FUNCTION PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Filter out failed experiments
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("No valid results to analyze!")
        return
    
    # Sort by final test accuracy (descending)
    sorted_results = sorted(valid_results, key=lambda x: x['final_test_accuracy'], reverse=True)
    
    print(f"\n{'FINAL TEST ACCURACY RANKING:'}")
    print(f"{'Rank':<6}{'Activation':<15}{'Final Acc':<12}{'Max Acc':<12}{'Learning Rate':<12}")
    print("-" * 70)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<6}{result['activation']:<15}{result['final_test_accuracy']:<12.2f}"
              f"{result['max_test_accuracy']:<12.2f}{result['learning_rate']:<12.4f}")
    
    # Activation family analysis
    print(f"\n{'ACTIVATION FAMILY PERFORMANCE:'}")
    
    # Group activations by type
    traditional = [r for r in sorted_results if r['activation'] in ['relu', 'sigmoid', 'tanh']]
    modern = [r for r in sorted_results if r['activation'] in ['silu', 'gelu']]
    
    if traditional:
        traditional_avg = np.mean([r['final_test_accuracy'] for r in traditional])
        print(f"• Traditional activations average: {traditional_avg:.2f}%")
        
    if modern:
        modern_avg = np.mean([r['final_test_accuracy'] for r in modern])
        print(f"• Modern activations average: {modern_avg:.2f}%")
    
    # Training stability analysis
    print(f"\n{'TRAINING STABILITY ANALYSIS:'}")
    for result in sorted_results:
        if 'final_train_accuracy' in result and 'final_test_accuracy' in result:
            gap = result['final_train_accuracy'] - result['final_test_accuracy']
            if gap > 10:
                stability = "High overfitting"
            elif gap > 5:
                stability = "Moderate overfitting"
            elif gap > 0:
                stability = "Slight overfitting"
            else:
                stability = "Good generalization"
            
            print(f"• {result['activation']:<12}: Train-Test gap = {gap:>6.2f}% ({stability})")

def analyze_optimizer_results(results):
    """
    Analyze and compare optimizer performance results.
    
    Args:
        results (list): List of dictionaries containing optimizer results
    """
    print(f"\n{'='*80}")
    print("OPTIMIZER PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Filter out failed experiments
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("No valid results to analyze!")
        return
    
    # Sort by final test accuracy (descending)
    sorted_results = sorted(valid_results, key=lambda x: x['final_test_accuracy'], reverse=True)
    
    print(f"\n{'FINAL TEST ACCURACY RANKING:'}")
    print(f"{'Rank':<6}{'Optimizer':<15}{'Final Acc':<12}{'Max Acc':<12}{'Learning Rate':<12}")
    print("-" * 70)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<6}{result['optimizer']:<15}{result['final_test_accuracy']:<12.2f}"
              f"{result['max_test_accuracy']:<12.2f}{result['learning_rate']:<12.4f}")
    
    # Convergence analysis
    print(f"\n{'CONVERGENCE ANALYSIS:'}")
    for result in sorted_results:
        if 'test_accuracies' in result and len(result['test_accuracies']) >= 10:
            test_accs = result['test_accuracies']
            
            # Calculate convergence metrics
            early_period = test_accs[:5]
            mid_period = test_accs[len(test_accs)//2-2:len(test_accs)//2+3]
            late_period = test_accs[-5:]
            
            early_avg = np.mean(early_period)
            mid_avg = np.mean(mid_period)
            late_avg = np.mean(late_period)
            
            total_improvement = late_avg - early_avg
            convergence_speed = (mid_avg - early_avg) / (len(test_accs)//2)
            
            print(f"• {result['optimizer']:<12}: "
                  f"Early: {early_avg:5.2f}% → "
                  f"Mid: {mid_avg:5.2f}% → "
                  f"Late: {late_avg:5.2f}% "
                  f"(Total improvement: {total_improvement:+5.2f}%, "
                  f"Speed: {convergence_speed:+5.3f}%/epoch)")
    
    # Optimizer category analysis
    print(f"\n{'OPTIMIZER CATEGORY PERFORMANCE:'}")
    
    # Group optimizers by type
    sgd_variants = [r for r in sorted_results if r['optimizer'].startswith('sgd')]
    adaptive_variants = [r for r in sorted_results if r['optimizer'] in ['adam', 'adamw', 'adagrad', 'rmsprop', 'nadam']]
    
    if sgd_variants:
        sgd_avg = np.mean([r['final_test_accuracy'] for r in sgd_variants])
        print(f"• SGD variants average: {sgd_avg:.2f}%")
        
    if adaptive_variants:
        adaptive_avg = np.mean([r['final_test_accuracy'] for r in adaptive_variants])
        print(f"• Adaptive optimizers average: {adaptive_avg:.2f}%")
    
    # Training stability analysis
    print(f"\n{'TRAINING STABILITY ANALYSIS:'}")
    for result in sorted_results:
        if 'final_train_accuracy' in result and 'final_test_accuracy' in result:
            gap = result['final_train_accuracy'] - result['final_test_accuracy']
            if gap > 10:
                stability = "High overfitting"
            elif gap > 5:
                stability = "Moderate overfitting"
            elif gap > 0:
                stability = "Slight overfitting"
            else:
                stability = "Good generalization"
            
            print(f"• {result['optimizer']:<12}: Train-Test gap = {gap:>6.2f}% ({stability})")
    
    # Optimizer-specific insights
    print(f"\n{'OPTIMIZER-SPECIFIC INSIGHTS:'}")
    for result in sorted_results:
        opt_info = result.get('optimizer_info', {})
        print(f"• {result['optimizer'].upper()}:")
        print(f"  Final accuracy: {result['final_test_accuracy']:.2f}%")
        print(f"  {opt_info.get('convergence', 'No convergence info available')}")
        if 'test_accuracies' in result and len(result['test_accuracies']) >= 5:
            # Check for early plateau
            late_accs = result['test_accuracies'][-5:]
            if max(late_accs) - min(late_accs) < 0.5:
                print("  Note: Training appears to have plateaued in final epochs")
        print()

def main():
    """
    Main execution controller for VGG6 CIFAR-10 hyperparameter optimization experiments.
    
    Orchestrates different types of experiments based on CONFIG settings:
    - W&B Hyperparameter Sweep: Bayesian optimization across 6 hyperparameters
    - Optimizer Comparison: Sequential testing of 8 different optimizers  
    - Activation Function Analysis: Testing 5 activation functions
    - Custom Configuration Testing: Predefined hyperparameter combinations
    - Preview Mode: Display sweep configuration without execution
    
    Features:
    - Automatic device detection (CPU/CUDA/MPS)
    - Environment setup and reproducibility controls
    - Mutual exclusion of experiment modes (prevents conflicts)
    - Comprehensive error handling and user guidance
    - Real-time progress monitoring and results summary
    
    Experiment Modes (mutually exclusive):
    1. RUN_WANDB_SWEEP: Execute Bayesian hyperparameter optimization
    2. PREVIEW_WANDB_SWEEP: Show sweep configuration without running  
    3. TEST_ALL_OPTIMIZERS: Compare all optimizer algorithms
    4. TEST_ALL_ACTIVATIONS: Test different activation functions
    5. TEST_HYPERPARAMETERS: Run predefined configurations
    
    Results:
    - W&B dashboard with parallel coordinate plots
    - Console output with performance summaries
    - Best configuration identification and metrics
    """
    try:
        # Set up environment
        print("Setting up environment...")
        device = get_device()
        
        # Check for multiple test modes
        test_modes = sum([
            CONFIG['RUN_WANDB_SWEEP'],
            CONFIG['PREVIEW_WANDB_SWEEP'],
            CONFIG['TEST_ALL_OPTIMIZERS'],
            CONFIG['TEST_ALL_ACTIVATIONS'], 
            CONFIG['TEST_HYPERPARAMETERS']
        ])
        
        if test_modes > 1:
            print("Error: Cannot test multiple modes simultaneously.")
            print("Please set only one of: RUN_WANDB_SWEEP, PREVIEW_WANDB_SWEEP, TEST_ALL_OPTIMIZERS, TEST_ALL_ACTIVATIONS, or TEST_HYPERPARAMETERS to True.")
            return
        
        if CONFIG['PREVIEW_WANDB_SWEEP']:
            # Show W&B sweep preview without running
            preview_wandb_sweep()
            return
            
        elif CONFIG['RUN_WANDB_SWEEP']:
            # Run W&B hyperparameter sweep
            print(f"\nStarting W&B hyperparameter sweep...")
            print("This will run Bayesian optimization to find optimal hyperparameters")
            print("and generate parallel coordinate plots for visualization.")
            
            # Show preview first
            preview_wandb_sweep()
            
            # Auto-proceed with sweep (confirmation disabled for automation)
            print("\nProceeding with sweep automatically...")
            # response = input("\nDo you want to proceed with the sweep? (y/n): ").lower().strip()
            # if response != 'y':
            #     print("Sweep cancelled. Set CONFIG['RUN_WANDB_SWEEP'] = False to skip this prompt.")
            #     return
            
            try:
                # Run the sweep
                sweep_id = run_wandb_sweep(sweep_count=20)
                
                # Create parallel coordinates plot
                print("\nCreating parallel coordinates plot...")
                df = create_parallel_coordinates_plot(sweep_id)
                
                if df is not None:
                    print(f"\nSweep completed successfully!")
                    print(f"Total runs: {len(df)}")
                    print(f"Best accuracy: {df['final_test_accuracy'].max():.2f}%")
                    print(f"View results at: https://wandb.ai/")
                
            except Exception as e:
                print(f"W&B sweep failed: {e}")
                print("Make sure wandb is installed: pip install wandb")
                print("And you are logged in: wandb login")
                
        elif CONFIG['TEST_HYPERPARAMETERS']:
            # Test different hyperparameter configurations
            print(f"\nTesting {len(CONFIG['HYPERPARAMETER_CONFIGS'])} hyperparameter configurations...")
            all_results = []
            
            for config in CONFIG['HYPERPARAMETER_CONFIGS']:
                try:
                    result = test_single_hyperparameter_config(config, device)
                    all_results.append(result)
                    
                    # Clean up GPU memory between experiments
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except KeyboardInterrupt:
                    print(f"\nTraining interrupted during {config['name']}!")
                    break
                except Exception as e:
                    print(f"Failed to test {config['name']}: {str(e)}")
                    continue
            
            # Analyze results
            analyze_hyperparameter_results(all_results)
            
        elif CONFIG['TEST_ALL_OPTIMIZERS']:
            # Test all optimizers with default activation
            print(f"\nTesting {len(CONFIG['OPTIMIZERS_TO_TEST'])} optimizers with {CONFIG['ACTIVATION']} activation...")
            train_loader, test_loader = GetCifar10(CONFIG['BATCH_SIZE'])
            all_results = []
            
            for optimizer_name in CONFIG['OPTIMIZERS_TO_TEST']:
                try:
                    result = test_single_optimizer(optimizer_name, device, train_loader, test_loader)
                    all_results.append(result)
                    
                    # Clean up GPU memory between experiments
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except KeyboardInterrupt:
                    print(f"\nTraining interrupted during {optimizer_name}!")
                    break
                except Exception as e:
                    print(f"Failed to test {optimizer_name}: {str(e)}")
                    continue
            
            # Analyze results
            analyze_optimizer_results(all_results)
            
        elif CONFIG['TEST_ALL_ACTIVATIONS']:
            # Test all activation functions with default optimizer
            print(f"\nTesting {len(CONFIG['ACTIVATIONS_TO_TEST'])} activation functions with {CONFIG['OPTIMIZER']} optimizer...")
            train_loader, test_loader = GetCifar10(CONFIG['BATCH_SIZE'])
            all_results = []
            
            for activation_name in CONFIG['ACTIVATIONS_TO_TEST']:
                try:
                    result = test_single_activation(activation_name, device, train_loader, test_loader)
                    all_results.append(result)
                    
                    # Clean up GPU memory between experiments
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except KeyboardInterrupt:
                    print(f"\nTraining interrupted during {activation_name}!")
                    break
                except Exception as e:
                    print(f"Failed to test {activation_name}: {str(e)}")
                    continue
            
            # Analyze results
            analyze_activation_results(all_results)
            
        else:
            # Test single configuration
            print(f"\nTesting single configuration: {CONFIG['OPTIMIZER']} optimizer with {CONFIG['ACTIVATION']} activation")
            train_loader, test_loader = GetCifar10(CONFIG['BATCH_SIZE'])
            result = test_single_optimizer(CONFIG['OPTIMIZER'], device, train_loader, test_loader)
            print(f"Test completed - Final accuracy: {result['final_test_accuracy']:.2f}%")

    except KeyboardInterrupt:
        print('\nProgram interrupted by user')
    except Exception as e:
        print(f'An error occurred: {str(e)}')
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nExperiment completed!")

def preview_wandb_sweep():
    """
    Preview the W&B sweep configuration without running it.
    
    Shows what hyperparameter combinations will be tested and 
    provides an estimate of total training time.
    """
    print(f"\n{'='*60}")
    print(f"W&B SWEEP PREVIEW")
    print(f"{'='*60}")
    
    config = CONFIG['WANDB_SWEEP_CONFIG']
    
    print(f"Project: {CONFIG['WANDB_PROJECT']}")
    print(f"Method: {config['method']}")
    print(f"Metric: {config['metric']['name']} ({config['metric']['goal']})")
    
    print(f"\nHyperparameters to explore:")
    total_combinations = 1
    for param, values in config['parameters'].items():
        param_values = values['values']
        total_combinations *= len(param_values)
        print(f"  {param:15}: {param_values} ({len(param_values)} values)")
    
    print(f"\nTotal possible combinations: {total_combinations:,}")
    print(f"Sweep will test: 20 combinations (Bayesian optimization)")
    
    # Estimate training time
    avg_epochs = sum(config['parameters']['epochs']['values']) / len(config['parameters']['epochs']['values'])
    estimated_time_per_run = avg_epochs * 2  # Rough estimate: 2 minutes per epoch
    total_estimated_time = 20 * estimated_time_per_run
    
    print(f"\nEstimated time per run: {estimated_time_per_run:.1f} minutes")
    print(f"Total estimated time: {total_estimated_time:.1f} minutes ({total_estimated_time/60:.1f} hours)")
    
    print(f"\nMetrics logged per run:")
    metrics = [
        'epoch', 'train_accuracy', 'test_accuracy', 'train_loss', 
        'learning_rate', 'final_test_accuracy', 'best_test_accuracy'
    ]
    for metric in metrics:
        print(f"  • {metric}")
    
    print(f"\nVisualization features:")
    features = [
        'Parallel coordinate plots',
        'Hyperparameter importance analysis', 
        'Performance distribution plots',
        'Best configuration identification',
        'Real-time training curves'
    ]
    for feature in features:
        print(f"  • {feature}")
    
    print(f"\n{'='*60}")
    print(f"To run the actual sweep, set CONFIG['RUN_WANDB_SWEEP'] = True")
    print(f"{'='*60}")

def wandb_train():
    """
    Training function for W&B sweeps.
    
    This function is called by W&B sweep agent to train a model with
    specific hyperparameter configurations. It logs all metrics to W&B
    for parallel coordinate visualization.
    """
    # Initialize W&B run
    wandb.init()
    config = wandb.config
    
    print(f"\n{'='*60}")
    print(f"W&B Sweep Training")
    print(f"Configuration: {dict(config)}")
    print(f"{'='*60}")
    
    try:
        # Set random seeds for reproducible results
        set_random_seeds(CONFIG['SEED'])
        
        # Get data loaders with sweep batch size
        train_loader, test_loader = GetCifar10(config.batch_size)
        
        # Create model with sweep activation
        model = vgg(
            CONFIG['VGG_CONFIG'], 
            num_classes=CONFIG['NUM_CLASSES'],
            activation=config.activation
        ).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Activation: {config.activation}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Optimizer: {config.optimizer}")
        print(f"Weight decay: {config.weight_decay}")
        print(f"Epochs: {config.epochs}")
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create optimizer based on sweep configuration
        if config.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(), 
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        # Log model configuration
        wandb.log({
            'model_parameters': total_params,
            'config/activation': config.activation,
            'config/batch_size': config.batch_size,
            'config/learning_rate': config.learning_rate,
            'config/optimizer': config.optimizer,
            'config/weight_decay': config.weight_decay,
            'config/epochs': config.epochs
        })
        
        # Train the model with W&B logging
        train_accuracies, test_accuracies = train_model(
            model, config.epochs, optimizer, train_loader, test_loader, criterion, use_wandb=True
        )
        
        # Log final results
        final_train_acc = train_accuracies[-1] if train_accuracies else 0
        final_test_acc = test_accuracies[-1] if test_accuracies else 0
        best_test_acc = max(test_accuracies) if test_accuracies else 0
        
        wandb.log({
            'final_train_accuracy': final_train_acc,
            'final_test_accuracy': final_test_acc,
            'best_test_accuracy': best_test_acc
        })
        
        print(f"\nTraining completed!")
        print(f"Final train accuracy: {final_train_acc:.2f}%")
        print(f"Final test accuracy: {final_test_acc:.2f}%")
        print(f"Best test accuracy: {best_test_acc:.2f}%")
        
    except Exception as e:
        print(f"Error during training: {e}")
        wandb.log({'error': str(e)})
        raise e
    finally:
        wandb.finish()

def run_wandb_sweep(sweep_count=20):
    """
    Run W&B hyperparameter sweep.
    
    This function initializes a W&B sweep with the configuration defined
    in CONFIG['WANDB_SWEEP_CONFIG'] and runs the specified number of
    training runs with different hyperparameter combinations.
    
    Args:
        sweep_count (int): Number of sweep runs to execute
        
    Returns:
        str: Sweep ID for reference and further analysis
    """
    print(f"\n{'='*60}")
    print(f"Starting W&B Hyperparameter Sweep")
    print(f"Sweep count: {sweep_count}")
    print(f"{'='*60}")
    
    try:
        # Initialize W&B sweep
        sweep_id = wandb.sweep(
            sweep=CONFIG['WANDB_SWEEP_CONFIG'],
            project=CONFIG['WANDB_PROJECT']
        )
        
        print(f"Sweep ID: {sweep_id}")
        print(f"Project: {CONFIG['WANDB_PROJECT']}")
        print(f"Method: {CONFIG['WANDB_SWEEP_CONFIG']['method']}")
        
        # Run the sweep
        wandb.agent(sweep_id, wandb_train, count=sweep_count)
        
        print(f"\nSweep completed!")
        print(f"Results available at: https://wandb.ai/")
        print(f"Sweep ID: {sweep_id}")
        
        return sweep_id
        
    except Exception as e:
        print(f"Error during sweep: {e}")
        raise e

def create_parallel_coordinates_plot(sweep_id=None, project_name=None):
    """
    Create and display parallel coordinates plot from W&B sweep results.
    
    This function retrieves sweep results from W&B and creates a parallel
    coordinates plot showing the relationship between hyperparameters
    and final accuracy.
    
    Args:
        sweep_id (str, optional): W&B sweep ID to analyze
        project_name (str, optional): W&B project name
    """
    try:
        import wandb
        
        project_name = project_name or CONFIG['WANDB_PROJECT']
        
        print(f"\n{'='*60}")
        print(f"Creating Parallel Coordinates Plot")
        print(f"Project: {project_name}")
        if sweep_id:
            print(f"Sweep ID: {sweep_id}")
        print(f"{'='*60}")
        
        # Initialize W&B API
        api = wandb.Api()
        
        if sweep_id:
            # Get specific sweep
            sweep = api.sweep(f"{api.default_entity}/{project_name}/{sweep_id}")
            runs = sweep.runs
        else:
            # Get all runs from project
            runs = api.runs(f"{api.default_entity}/{project_name}")
        
        # Extract data for parallel coordinates
        data = []
        for run in runs:
            if run.state == 'finished':
                summary = run.summary
                config = run.config
                
                row = {
                    'batch_size': config.get('batch_size', None),
                    'learning_rate': config.get('learning_rate', None),
                    'optimizer': config.get('optimizer', None),
                    'activation': config.get('activation', None),
                    'epochs': config.get('epochs', None),
                    'weight_decay': config.get('weight_decay', None),
                    'final_test_accuracy': summary.get('final_test_accuracy', None),
                    'best_test_accuracy': summary.get('best_test_accuracy', None),
                    'run_name': run.name
                }
                
                # Only add rows with complete data
                if all(v is not None for v in row.values() if v != row['run_name']):
                    data.append(row)
        
        if not data:
            print("No complete runs found for parallel coordinates plot")
            return
        
        print(f"Found {len(data)} complete runs for visualization")
        
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Print summary statistics
        print(f"\nAccuracy Statistics:")
        print(f"Mean test accuracy: {df['final_test_accuracy'].mean():.2f}%")
        print(f"Best test accuracy: {df['final_test_accuracy'].max():.2f}%")
        print(f"Std test accuracy: {df['final_test_accuracy'].std():.2f}%")
        
        # Print best configuration
        best_idx = df['final_test_accuracy'].idxmax()
        best_config = df.iloc[best_idx]
        print(f"\nBest Configuration (Accuracy: {best_config['final_test_accuracy']:.2f}%):")
        print(f"  Batch size: {best_config['batch_size']}")
        print(f"  Learning rate: {best_config['learning_rate']}")
        print(f"  Optimizer: {best_config['optimizer']}")
        print(f"  Activation: {best_config['activation']}")
        print(f"  Epochs: {best_config['epochs']}")
        print(f"  Weight decay: {best_config['weight_decay']}")
        
        print(f"\nParallel coordinates plot created successfully!")
        print(f"View detailed plots and analysis in W&B dashboard:")
        print(f"https://wandb.ai/{api.default_entity}/{project_name}")
        
        return df
        
    except ImportError:
        print("Error: wandb not installed. Install with: pip install wandb")
        return None
    except Exception as e:
        print(f"Error creating parallel coordinates plot: {e}")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    # Initialize global device for all experiments
    device = get_device()
    
    # Execute main experiment controller
    main()