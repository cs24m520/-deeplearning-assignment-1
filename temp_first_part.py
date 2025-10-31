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
    # ========================================================================
    # ADAPTIVE LEARNING RATE OPTIMIZERS
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
    
    Args:
        optimizer_name (str): Name of the optimizer
        
    Returns:
        dict: Information about the optimizer
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
