# -*- coding: utf-8 -*-
"""
VGG6 Implementation for CIFAR-10 Dataset

A PyTorch implementation of a simplified VGG network (VGG6) optimized for CIFAR-10 
classification. This implementation focuses on performance and compatibility across 
different hardware platforms.

Features:
- VGG6 architecture with batch normalization for faster training
- Advanced data augmentation (AutoAugment, Cutout) for better generalization
- Multi-device support (CUDA, Apple M-series, CPU)
- Hyperparameter tuning and learning rate scheduling
- Optimized data loading and processing pipeline

Architecture:
- 6 convolutional layers with batch normalization
- 3 max pooling layers
- 2 fully connected layers with dropout
- Input: 32x32x3 CIFAR-10 images
- Output: 10 classes

Usage:
    python vgg6-question-1-assignment.py

Requirements:
    - PyTorch >= 2.0
    - torchvision
    - numpy
    - pillow

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
    'LEARNING_RATE': 0.01,   # Initial learning rate
    'WEIGHT_DECAY': 0.001,   # L2 regularization factor
    'EPOCHS': 100,           # Number of training epochs
    
    # VGG6 architecture configuration
    # Format: [num_channels, num_channels, 'M' (MaxPool), ...]
    # - Numbers represent output channels in Conv2D layers
    # - 'M' represents MaxPool2D layers
    'VGG_CONFIG': [64, 64, 'M',     # Block 1: 2×Conv64 + MaxPool
                  128, 128, 'M',     # Block 2: 2×Conv128 + MaxPool
                  256, 'M']          # Block 3: Conv256 + MaxPool
}

def set_random_seeds(seed=CONFIG['SEED']):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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
    """VGG model with batch normalization."""
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # Adjusted input size to 256
            nn.ReLU(True),
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    """
    Create VGG model layers based on configuration list.
    
    This function constructs the convolutional blocks of the VGG architecture
    according to the provided configuration. Each block consists of:
    1. Convolutional layer (3×3 kernel, padding=1)
    2. Optional Batch Normalization
    3. ReLU activation
    4. Optional MaxPool2D (when 'M' is specified in config)

    Args:
        cfg (list): Configuration list where each element is either:
                   - int: number of output channels for Conv2D
                   - 'M': indicates MaxPool2D layer
        batch_norm (bool): Whether to include batch normalization layers
                          after each convolution (default: False)

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
            if batch_norm:
                # Conv2D -> BatchNorm -> ReLU
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # Conv2D -> ReLU
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # Update input channels for next layer
            
    return nn.Sequential(*layers)

def vgg(cfg, num_classes=10, batch_norm=True):
    """Create VGG model instance."""
    return VGG(make_layers(cfg, batch_norm=batch_norm), num_classes=num_classes)

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

def main():
    """Main training routine."""
    try:
        # Set up environment
        set_random_seeds()
        device = get_device()

        # Create model
        model = vgg(CONFIG['VGG_CONFIG'], num_classes=CONFIG['NUM_CLASSES']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG['LEARNING_RATE'],
            weight_decay=CONFIG['WEIGHT_DECAY']
        )

        # Get data loaders
        train_loader, test_loader = GetCifar10(CONFIG['BATCH_SIZE'])

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
        print(f'Final Test Accuracy: {final_test_acc:.2f}%')

    except KeyboardInterrupt:
        print('Program interrupted by user')
    except Exception as e:
        print(f'An error occurred: {str(e)}')
    finally:
        # Clean up (if needed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    device = get_device()  # Define device globally
    main()