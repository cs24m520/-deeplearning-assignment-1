# -*- coding: utf-8 -*-
"""
VGG6 Implementation with Multiple Optimizers for CIFAR-10

This implementation explores the impact of different optimizers on 
VGG6 performance for CIFAR-10 classification.

Optimizers Tested:
- SGD (Stochastic Gradient Descent)
- SGD with Momentum
- SGD with Nesterov Momentum
- Adam (Adaptive Moment Estimation)
- AdamW (Adam with Weight Decay)
- Adagrad (Adaptive Gradient Algorithm)
- RMSprop (Root Mean Square Propagation)
- NAdam (Nesterov-accelerated Adaptive Moment Estimation)

Features:
- VGG6 architecture with configurable optimizers
- Advanced data augmentation (AutoAugment, Cutout)
- Multi-device support (CUDA, Apple M-series, CPU)
- Performance comparison and convergence analysis
- Proper learning rate tuning for different optimizers

Usage:
    python vgg6-question-2-assignment.py

Author: [Your Name]
Date: October 2025
"""

# Standard library imports
import os
import copy
import random
import platform
import itertools

# Third-party imports
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torch.backends.cudnn
import torch.backends.mps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Global configuration settings
CONFIG = {
    # Training hyperparameters
    'SEED': 42,              # Random seed for reproducibility
    'NUM_CLASSES': 10,       # Number of CIFAR-10 classes
    'BATCH_SIZE': 128,       # Mini-batch size for training
    'LEARNING_RATE': 0.01,   # Base learning rate (will be adjusted per optimizer)
    'WEIGHT_DECAY': 0.001,   # L2 regularization factor
    'EPOCHS': 20,            # Number of training epochs (reduced for testing)
    
    # VGG6 architecture configuration
    # Format: [num_channels, num_channels, 'M' (MaxPool), ...]
    # - Numbers represent output channels in Conv2D layers
    # - 'M' represents MaxPool2D layers
    'VGG_CONFIG': [64, 64, 'M',     # Block 1: 2×Conv64 + MaxPool
                  128, 128, 'M',     # Block 2: 2×Conv128 + MaxPool
                  256, 'M'],         # Block 3: Conv256 + MaxPool
    
    # Optimizer configuration
    'OPTIMIZER': 'adam',     # Default optimizer
    'ACTIVATION': 'relu',    # Default activation function
    
    # Test all optimizers sequentially
    'TEST_ALL_OPTIMIZERS': False,  # If True, tests all optimizers
    'OPTIMIZERS_TO_TEST': ['sgd', 'sgd_momentum', 'sgd_nesterov', 'adam', 'adamw', 'adagrad', 'rmsprop', 'nadam'],
    
    # Test all activations sequentially
    'TEST_ALL_ACTIVATIONS': False,  # If True, tests all activation functions
    'ACTIVATIONS_TO_TEST': ['relu', 'sigmoid', 'tanh', 'silu', 'gelu'],
    
    # Test hyperparameters (batch size, epochs, learning rate)
    'TEST_HYPERPARAMETERS': True,  # If True, tests different hyperparameter combinations
    'HYPERPARAMETER_CONFIGS': [
        # Batch size variations (quick test)
        {'name': 'small_batch', 'BATCH_SIZE': 64, 'EPOCHS': 5, 'LEARNING_RATE': 0.001},
        {'name': 'medium_batch', 'BATCH_SIZE': 128, 'EPOCHS': 5, 'LEARNING_RATE': 0.001},
        {'name': 'large_batch', 'BATCH_SIZE': 256, 'EPOCHS': 5, 'LEARNING_RATE': 0.001},
        
        # Learning rate variations (quick test)
        {'name': 'low_lr', 'BATCH_SIZE': 128, 'EPOCHS': 5, 'LEARNING_RATE': 0.0001},
        {'name': 'high_lr', 'BATCH_SIZE': 128, 'EPOCHS': 5, 'LEARNING_RATE': 0.01},
        
        # Epoch variations (quick test)
        {'name': 'short_training', 'BATCH_SIZE': 128, 'EPOCHS': 3, 'LEARNING_RATE': 0.001},
        {'name': 'long_training', 'BATCH_SIZE': 128, 'EPOCHS': 10, 'LEARNING_RATE': 0.001},
    ],
    
    # Optimizer-specific learning rates (fine-tuned for best performance)
    'OPTIMIZER_LR': {
        'sgd': 0.01,
        'sgd_momentum': 0.01,
        'sgd_nesterov': 0.01,
        'adam': 0.001,
        'adamw': 0.001,
        'adagrad': 0.01,
        'rmsprop': 0.001,
        'nadam': 0.001
    }
}

def set_random_seeds(seed=CONFIG['SEED']):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_optimizer(optimizer_name, model_parameters, learning_rate, weight_decay):
    """
    Get the appropriate optimizer based on name.
    
    Args:
        optimizer_name (str): Name of the optimizer
        model_parameters: Model parameters to optimize
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay factor
        
    Returns:
        torch.optim.Optimizer: PyTorch optimizer
        
    Raises:
        ValueError: If optimizer is not supported
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'sgd':
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd_momentum':
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd_nesterov':
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        return optim.Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'adamw':
        return optim.AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            lr_decay=0,
            eps=1e-10
        )
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            alpha=0.99,
            eps=1e-8,
            momentum=0
        )
    elif optimizer_name == 'nadam':
        # NAdam is available in PyTorch 1.10+
        try:
            return optim.NAdam(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        except AttributeError:
            print("NAdam not available in this PyTorch version, using Adam instead")
            return optim.Adam(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. "
                        f"Supported: {CONFIG['OPTIMIZERS_TO_TEST']}")

def get_optimizer_info(optimizer_name):
    """
    Get detailed information about the optimizer characteristics.
    
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

# VGG Model Classes
class VGG(nn.Module):
    """VGG model with configurable activation functions and batch normalization."""
    def __init__(self, features, num_classes=10, activation='relu'):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation_name = activation
        
        # Create classifier with the same activation function
        classifier_activation = get_activation_function(activation)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # Adjusted input size to 256
            classifier_activation,
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize weights based on the activation function used."""
        nonlinearity = get_weight_init_nonlinearity(self.activation_name)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use appropriate initialization based on activation function
                if self.activation_name in ['sigmoid', 'tanh']:
                    # Xavier/Glorot initialization for sigmoid/tanh
                    nn.init.xavier_normal_(m.weight)
                else:
                    # Kaiming/He initialization for ReLU-like activations
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Linear):
                if self.activation_name in ['sigmoid', 'tanh']:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False, activation='relu'):
    """
    Create VGG model layers based on configuration list with specified activation.
    
    This function constructs the convolutional blocks of the VGG architecture
    according to the provided configuration. Each block consists of:
    1. Convolutional layer (3×3 kernel, padding=1)
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
            # MaxPool2D layer with 2×2 kernel and stride 2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Conv2D with 3×3 kernel and same padding
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
       - Random cropping (32x32 with padding=4)
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

def train_model(model, num_epochs, optimizer, train_loader, test_loader, criterion):
    """
    Train the model with learning rate scheduling and early stopping.
    
    This function implements a complete training loop with the following features:
    - Per-epoch training and validation
    - Learning rate scheduling with ReduceLROnPlateau
    - Best model checkpointing
    - Training history tracking
    - Graceful interruption handling
    
    Args:
        model (nn.Module): The neural network model to train
        num_epochs (int): Number of training epochs
        optimizer (torch.optim.Optimizer): The optimizer for training
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        
    Returns:
        tuple: Lists of training and test accuracies per epoch
        
    Note:
        The function implements a learning rate scheduler that reduces the 
        learning rate by half if the test accuracy doesn't improve for 3 epochs.
        It also saves the best performing model based on test accuracy.
    """
    train_accuracies = []
    test_accuracies = []
    best_acc = 0.0
    best_model = None

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            # Calculate accuracies
            train_acc = 100. * correct / total
            test_acc = eval(model, test_loader)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Train Accuracy: {train_acc:.2f}%')
            print(f'Test Accuracy: {test_acc:.2f}%')

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
    """Main training routine with optimizer, activation, and hyperparameter comparison."""
    try:
        # Set up environment
        print("Setting up environment...")
        device = get_device()
        
        # Check for multiple test modes
        test_modes = sum([
            CONFIG['TEST_ALL_OPTIMIZERS'],
            CONFIG['TEST_ALL_ACTIVATIONS'], 
            CONFIG['TEST_HYPERPARAMETERS']
        ])
        
        if test_modes > 1:
            print("Error: Cannot test multiple modes simultaneously.")
            print("Please set only one of: TEST_ALL_OPTIMIZERS, TEST_ALL_ACTIVATIONS, or TEST_HYPERPARAMETERS to True.")
            return
        
        if CONFIG['TEST_HYPERPARAMETERS']:
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
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nExperiment completed!")

if __name__ == '__main__':
    device = get_device()  # Define device globally
    main()