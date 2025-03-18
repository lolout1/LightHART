#!/usr/bin/env python3
"""
Fall Detection Training Pipeline with Kalman Filter Variants

This script implements a comprehensive training and evaluation pipeline for fall detection
using various sensor fusion approaches, including Madgwick, Complementary, Kalman, EKF, and UKF filters.
It supports k-fold cross-validation with specified fold assignments and is optimized for multi-GPU training.

Features:
- K-fold cross-validation with consistent fold assignments
- Multi-GPU training support optimized for A100 GPUs
- Comprehensive logging and performance visualization
- Support for multiple fusion filter types
- Early stopping and model checkpointing
- Detailed performance metrics and confusion matrices
"""

import traceback
from typing import List, Dict, Tuple, Union, Optional, Any
import random
import sys
import os
import time
import shutil
import argparse
import yaml
from copy import deepcopy
import json
import itertools
import logging
from collections import defaultdict

# Environmental imports
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Local imports
from utils.dataset import prepare_smartfallmm, split_by_subjects
from utils.imu_fusion import cleanup_resources, update_thread_configuration
from utils.filter_comparison import (
    compare_filter_accuracy,
    visualize_filter_comparison,
    compare_filter_features,
    visualize_feature_comparison
)

# Register cleanup function to ensure proper resource management on exit
import atexit
def cleanup_on_exit():
    """Clean up resources on script exit."""
    cleanup_resources()
atexit.register(cleanup_on_exit)

# Configure logging
logger = logging.getLogger('fall_detection')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def str2bool(v):
    '''
    Function to parse boolean values from command line arguments

    Args:
        v: Value to convert to boolean

    Returns:
        Boolean conversion of the input

    Raises:
        ArgumentTypeError: If the input cannot be converted to a boolean
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_args():
    '''
    Function to build the argument parser for command line options.
    This is the ONLY place where arguments should be defined.

    Returns:
        Configured ArgumentParser object
    '''
    parser = argparse.ArgumentParser(description='Fall Detection and Human Activity Recognition')
    parser.add_argument('--config', default='./config/smartfallmm/fusion_ekf.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--dataset', type=str, default='smartfallmm',
                        help='Dataset name to use')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='Input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='Input batch size for testing (default: 16)')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N',
                        help='Input batch size for validation (default: 16)')
    parser.add_argument('--num-epoch', type=int, default=60, metavar='N',
                        help='Number of epochs to train (default: 60)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch number (default: 0)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping (default: 15)')

    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='Optimizer to use (adamw, adam, sgd)')
    parser.add_argument('--base-lr', type=float, default=0.0005, metavar='LR',
                        help='Base learning rate (default: 0.0005)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='Weight decay factor (default: 0.001)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        help='Learning rate scheduler (plateau, cosine)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0)')

    # Model parameters
    parser.add_argument('--model', default=None,
                        help='Model class path to load')
    parser.add_argument('--device', nargs='+', default=[0, 1], type=int,
                        help='CUDA device IDs to use')
    parser.add_argument('--model-args', default=None,
                        help='Dictionary of model arguments')
    parser.add_argument('--weights', type=str,
                        help='Path to pretrained weights file')
    parser.add_argument('--model-saved-name', type=str, default='model',
                        help='Name for saving the trained model')

    # Loss function
    parser.add_argument('--loss', default='torch.nn.BCEWithLogitsLoss',
                        help='Loss function class path')
    parser.add_argument('--loss-args', default="{}", type=str,
                        help='Dictionary of loss function arguments')

    # Dataset configuration
    parser.add_argument('--dataset-args', default=None,
                        help='Arguments for the dataset')
    parser.add_argument('--subjects', nargs='+', type=int,
                        help='Subject IDs to include')
    parser.add_argument('--feeder', default=None,
                        help='DataLoader class path')
    parser.add_argument('--train-feeder-args', default=None,
                        help='Arguments for training data loader')
    parser.add_argument('--val-feeder-args', default=None,
                        help='Arguments for validation data loader')
    parser.add_argument('--test-feeder-args', default=None,
                        help='Arguments for test data loader')
    parser.add_argument('--include-val', type=str2bool, default=True,
                        help='Whether to include validation set')

    # Initialization
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')

    # Output and logging
    parser.add_argument('--work-dir', type=str, default='work_dir',
                        help='Working directory for outputs')
    parser.add_argument('--print-log', type=str2bool, default=True,
                        help='Whether to print and save logs')
    parser.add_argument('--phase', type=str, default='train',
                        help='Phase: train or test')
    parser.add_argument('--num-worker', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--result-file', type=str,
                        help='File to save results to')
    
    # Multi-GPU and parallel processing
    parser.add_argument('--multi-gpu', type=str2bool, default=True,
                        help='Whether to use multiple GPUs when available')
    parser.add_argument('--parallel-threads', type=int, default=48,
                        help='Number of parallel threads for preprocessing')
    
    # Cross-validation
    parser.add_argument('--kfold', type=str2bool, default=True,
                        help='Whether to use k-fold cross-validation')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    
    # Filter comparison
    parser.add_argument('--compare-filters', type=str2bool, default=False,
                        help='Whether to compare different filter types')
    parser.add_argument('--filter-types', nargs='+', default=['madgwick', 'ekf'], 
                        help='Filter types to compare')

    return parser

def init_seed(seed):
    '''
    Initialize random seeds for reproducibility

    Args:
        seed: Seed value for random number generators
    '''
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Use deterministic algorithms when possible
    torch.backends.cudnn.deterministic = False
    # Enable cudnn benchmarking for performance
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    '''
    Dynamically imports a class from its string name

    Args:
        import_str: String with the full import path of the class

    Returns:
        The imported class

    Raises:
        ImportError: If the class cannot be found
    '''
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found ({traceback.format_exception(*sys.exc_info())})')

def setup_gpu_environment(args):
    """
    Configure GPUs for processing.
    
    Args:
        args: Command line arguments with device specifications
    
    Returns:
        Tuple of (list of available GPU devices, bool indicating if AMP should be used)
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPUs")
        
        if isinstance(args.device, list) and len(args.device) > 0:
            # Use specified GPUs
            devices = args.device
            logger.info(f"Using specified GPUs: {devices}")
            
            # Check GPU models
            for i in devices:
                if i < num_gpus:
                    gpu_name = torch.cuda.get_device_name(i)
                    logger.info(f"  GPU {i}: {gpu_name}")
        elif num_gpus >= 2 and args.multi_gpu:
            # Default to using both GPUs if multi-GPU is enabled
            devices = [0, 1]
            logger.info(f"Using both GPUs: {devices}")
            
            # Log GPU models
            for i in devices:
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {gpu_name}")
        elif num_gpus == 1:
            # Use the single available GPU
            devices = [0]
            logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            # No GPUs available
            devices = []
            logger.warning("No GPUs found, using CPU")
            
        # Set visible devices
        gpu_list = ",".join(map(str, devices))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_list}")
        
        # Enable AMP if A100 GPUs are detected
        for i in devices:
            if i < num_gpus and "A100" in torch.cuda.get_device_name(i):
                logger.info("A100 GPU detected - enabling Automatic Mixed Precision")
                return devices, True
        
        return devices, False
    else:
        logger.warning("PyTorch CUDA not available, using CPU")
        return [], False

def configure_parallel_processing(args):
    """
    Configure parallel processing based on command-line arguments.
    
    Args:
        args: Command-line arguments with parallel-threads parameter
    """
    if hasattr(args, 'parallel_threads') and args.parallel_threads > 0:
        # Set up optimal thread allocation
        new_total = args.parallel_threads
        
        if new_total < 4:
            # Minimum configuration
            max_files = 1
            threads_per_file = new_total
        else:
            # Calculate optimal distribution
            # Prioritize number of files over threads per file
            max_files = min(12, new_total // 4)
            threads_per_file = min(4, new_total // max_files)
        
        # Update the thread configuration
        update_thread_configuration(max_files, threads_per_file)
        
        logger.info(f"Using {max_files} parallel files with {threads_per_file} threads per file")
        logger.info(f"Total parallel threads: {max_files * threads_per_file}")

class Trainer:
    '''
    Main trainer class that handles the complete training and evaluation workflow

    This class manages the training process, including data loading, model training,
    validation, testing, and result visualization. Optimized with multithreading
    and multi-GPU support.
    '''
    def __init__(self, arg):
        '''
        Initialize the trainer with command line arguments

        Args:
            arg: Parsed command line arguments
        '''
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_f1 = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0
        self.test_accuracy = 0
        self.test_f1 = 0
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.optimizer = None
        self.scheduler = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = dict()
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}.pt'
        self.patience_counter = 0
        self.scaler = None  # For mixed precision training
        self.fold_metrics = []
        
        # Get sensor modalities for fusion
        self.inertial_modality = [modality for modality in arg.dataset_args['modalities']
                                 if modality != 'skeleton']
        self.has_gyro = 'gyroscope' in self.inertial_modality
        self.has_fusion = len(self.inertial_modality) > 1 or (
            'fusion_options' in arg.dataset_args and
            arg.dataset_args['fusion_options'].get('enabled', False)
        )
        self.fuse = self.has_fusion

        # Create working directory
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            if arg.config:
                self.save_config(arg.config, arg.work_dir)

        # Set up GPU environment
        self.available_gpus, self.use_amp = setup_gpu_environment(arg)
        arg.device = self.available_gpus if self.available_gpus else arg.device
        self.output_device = arg.device[0] if type(arg.device) is list and len(arg.device) > 0 else arg.device
        
        # Create gradient scaler for AMP if needed
        if self.use_amp:
            self.scaler = torch.amp.GradScaler() if torch.__version__ >= '1.6.0' else None
            self.print_log("Using Automatic Mixed Precision (AMP) training")
        
        # Load model
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            self.model = self.load_weights()

        # Set up multi-GPU if available and requested
        if len(self.available_gpus) > 1 and arg.multi_gpu:
            self.print_log(f"Using {len(self.available_gpus)} GPUs with DataParallel")
            self.model = nn.DataParallel(
                self.model, 
                device_ids=self.available_gpus
            )

        # Load loss function
        self.load_loss()

        # Set validation option
        self.include_val = arg.include_val

        # Log model parameters
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params:,}')
        self.print_log(f'Model size: {num_params / (1024 ** 2):.2f} MB')

        # Log sensor configuration
        self.print_log(f'Sensor modalities: {self.inertial_modality}')
        self.print_log(f'Using fusion: {self.fuse}')
        
        # Log GPU configuration
        if self.available_gpus:
            self.print_log(f'Using GPUs: {self.available_gpus}')
        else:
            self.print_log('Using CPU for computation')
            
        # Initialize k-fold cross-validation
        if hasattr(self.arg, 'kfold') and self.arg.kfold:
            self.use_kfold = True
            self.num_folds = self.arg.num_folds
            self.fold_metrics = []
            self.print_log(f'Using {self.num_folds}-fold cross-validation')
        else:
            self.use_kfold = False
            self.print_log('Using single train/val/test split')

    def save_config(self, src_path: str, desc_path: str) -> None:
        '''
        Save the configuration file to the working directory

        Args:
            src_path: Source path of the configuration file
            desc_path: Destination path to save the configuration
        '''
        config_file = src_path.rpartition("/")[-1]
        self.print_log(f'Saving config to {desc_path}/{config_file}')
        shutil.copy(src_path, f'{desc_path}/{config_file}')

    def count_parameters(self, model):
        '''
        Count the number of trainable parameters in the model

        Args:
            model: PyTorch model

        Returns:
            Total number of parameters
        '''
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def load_model(self, model_path, model_args):
        '''
        Load the model class and instantiate it with provided arguments

        Args:
            model_path: Model class path
            model_args: Dictionary of model arguments

        Returns:
            Instantiated model on the appropriate device
        '''
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'

        Model = import_class(model_path)
        self.print_log(f"Loading model: {model_path}")
        self.print_log(f"Model arguments: {model_args}")
        
        # Ensure feature_dim is set correctly to avoid dimension mismatches
        if 'feature_dim' not in model_args and 'embed_dim' in model_args:
            if model_args.get('fusion_type', 'concat') == 'concat':
                model_args['feature_dim'] = model_args['embed_dim'] * 2
            else:
                model_args['feature_dim'] = model_args['embed_dim']
            self.print_log(f"Auto-setting feature_dim to {model_args['feature_dim']}")
        
        # Ensure num_heads divides feature_dim evenly
        if 'num_heads' in model_args and 'feature_dim' in model_args:
            if model_args['feature_dim'] % model_args['num_heads'] != 0:
                old_heads = model_args['num_heads']
                # Find closest divisor
                for heads in [old_heads-1, old_heads-2, old_heads+1, old_heads+2]:
                    if heads > 0 and model_args['feature_dim'] % heads == 0:
                        model_args['num_heads'] = heads
                        self.print_log(f"Adjusted num_heads from {old_heads} to {heads} to ensure divisibility with feature_dim={model_args['feature_dim']}")
                        break
        
        try:
            model = Model(**model_args).to(device)
        except Exception as e:
            self.print_log(f"Error instantiating model: {e}")
            self.print_log(f"Attempting to fix dimension issue and retry...")
            
            # Last resort: Adjust feature_dim to be divisible by num_heads
            if 'feature_dim' in model_args and 'num_heads' in model_args:
                heads = model_args['num_heads']
                feature_dim = model_args['feature_dim']
                new_feature_dim = (feature_dim // heads) * heads
                if new_feature_dim != feature_dim:
                    model_args['feature_dim'] = new_feature_dim
                    self.print_log(f"Adjusted feature_dim from {feature_dim} to {new_feature_dim} to ensure divisibility with num_heads={heads}")
            
            # Try again with adjusted parameters
            model = Model(**model_args).to(device)
        
        return model

    def load_weights(self):
        '''
        Load a model from saved weights file

        Returns:
            Loaded model
        '''
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        
        if not self.arg.weights:
            raise ValueError("No weights file specified for testing")
            
        self.print_log(f"Loading weights from: {self.arg.weights}")
        
        try:
            # Try loading complete model
            model = torch.load(self.arg.weights, map_location=device)
            self.print_log("Loaded complete model")
            return model
        except Exception as e:
            # If that fails, try loading as state dict
            try:
                Model = import_class(self.arg.model)
                model = Model(**self.arg.model_args).to(device)
                state_dict = torch.load(self.arg.weights, map_location=device)
                
                # Handle different state dict formats
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    # Load from checkpoint dict
                    model.load_state_dict(state_dict['model_state_dict'])
                    self.print_log("Loaded model weights from checkpoint dictionary")
                else:
                    # Load direct state dict
                    model.load_state_dict(state_dict)
                    self.print_log("Loaded model weights from state dictionary")
                
                return model
            except Exception as load_err:
                raise ValueError(f"Failed to load weights: {str(load_err)}\nOriginal error: {str(e)}")

    def load_loss(self):
        '''
        Load the loss function for training
        '''
        try:
            # Handle torch.nn losses directly
            if self.arg.loss.startswith('torch.nn.'):
                loss_class = getattr(torch.nn, self.arg.loss.split('.')[-1])
                loss_args = eval(self.arg.loss_args) if isinstance(self.arg.loss_args, str) else self.arg.loss_args
                self.criterion = loss_class(**loss_args)
            else:
                # For custom losses
                Loss = import_class(self.arg.loss)
                self.criterion = Loss(**eval(self.arg.loss_args))
                
            self.print_log(f"Using loss function: {self.arg.loss}")
            
        except Exception as e:
            self.print_log(f"Error loading loss function: {e}")
            # Fallback to BCEWithLogitsLoss for binary classification
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.print_log("Fallback to default BCEWithLogitsLoss")

    def load_optimizer(self) -> None:
        '''
        Configure and load the optimizer based on command line arguments
        '''
        optimizer_name = self.arg.optimizer.lower()

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
            self.print_log(f"Using Adam optimizer with lr={self.arg.base_lr}, weight_decay={self.arg.weight_decay}")
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
            self.print_log(f"Using AdamW optimizer with lr={self.arg.base_lr}, weight_decay={self.arg.weight_decay}")
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay
            )
            self.print_log(f"Using SGD optimizer with lr={self.arg.base_lr}, momentum=0.9, weight_decay={self.arg.weight_decay}")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Load scheduler
        self.load_scheduler()

    def load_scheduler(self) -> None:
        '''
        Configure and load the learning rate scheduler
        '''
        scheduler_name = getattr(self.arg, 'scheduler', 'plateau')
        
        if scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=10,
                verbose=True
            )
            self.print_log("Using ReduceLROnPlateau scheduler")
        elif scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.arg.num_epoch,
                eta_min=1e-6
            )
            self.print_log("Using CosineAnnealingLR scheduler")
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
            self.print_log("Using StepLR scheduler")
        else:
            self.scheduler = None
            self.print_log("No learning rate scheduler")

    def create_folds(self, subjects, num_folds=5):
        """
        Create folds for cross-validation with fixed fold assignments

        Args:
            subjects: List of subject IDs
            num_folds: Number of folds (default: 5)

        Returns:
            List of (train_subjects, val_subjects) tuples for each fold
        """
        # Try to get fold assignments from config file
        if (hasattr(self.arg, 'kfold') and isinstance(self.arg.kfold, dict) and 
            'fold_assignments' in self.arg.kfold):
            fold_assignments = self.arg.kfold.get('fold_assignments', [])
            
            if len(fold_assignments) == num_folds:
                self.print_log(f"Using {num_folds} fold assignments from config file")
                
                # Create folds with configured assignments
                folds = []
                for i, val_subjects in enumerate(fold_assignments, 1):
                    # Training subjects are all subjects not in validation
                    train_subjects = [s for s in subjects if s not in val_subjects]
                    folds.append((train_subjects, val_subjects))
                    
                    # Log fold information
                    self.print_log(f"\nFold {i} assignments:")
                    self.print_log(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
                    self.print_log(f"Training subjects ({len(train_subjects)}): {train_subjects}")
                
                return folds
        
        # Default fixed fold assignments if not in config
        fold_assignments = [
            ([43, 35, 36], "Fold 1: 38.3% falls"),
            ([44, 34, 32], "Fold 2: 39.7% falls"),
            ([45, 37, 38], "Fold 3: 44.8% falls"),
            ([46, 29, 31], "Fold 4: 41.4% falls"),
            ([30, 39], "Fold 5: 43.3% falls")
        ]
        
        # Create folds with fixed assignments
        folds = []
        for val_subjects, fold_desc in fold_assignments:
            # Filter validation subjects to only include those in the requested subjects
            valid_val_subjects = [s for s in val_subjects if s in subjects]
            
            # Skip this fold if no valid validation subjects
            if not valid_val_subjects:
                continue
                
            # Training subjects are all subjects not in validation
            train_subjects = [s for s in subjects if s not in valid_val_subjects]
            folds.append((train_subjects, valid_val_subjects))
            
            # Log fold information
            fold_num = len(folds)
            self.print_log(f"\nCreated {fold_desc}")
            self.print_log(f"Validation subjects ({len(valid_val_subjects)}): {valid_val_subjects}")
            self.print_log(f"Training subjects ({len(train_subjects)}): {train_subjects}")
    
        return folds

    def distribution_viz(self, labels: np.array, work_dir: str, mode: str) -> None:
        '''
        Visualize the distribution of class labels in the dataset

        Args:
            labels: Array of class labels
            work_dir: Directory to save the visualization
            mode: Mode indicator (train, val, test)
        '''
        values, count = np.unique(labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(values, count)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title(f'{mode.capitalize()} Label Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{work_dir}/{mode}_label_distribution.png')
        plt.close()
        self.print_log(f"Created {mode} label distribution visualization with classes {values}")
        
        # Print distribution percentages
        total = np.sum(count)
        percentages = (count / total) * 100
        for i, (val, cnt, pct) in enumerate(zip(values, count, percentages)):
            self.print_log(f"  Class {val}: {cnt} samples ({pct:.1f}%)")

    def load_data(self, train_subjects=None, val_subjects=None):
        '''
        Load and prepare datasets for training, validation, and testing

        Args:
            train_subjects: List of subject IDs for training
            val_subjects: List of subject IDs for validation

        Returns:
            True if all data was loaded successfully, False otherwise
        '''
        # Import the data feeder class
        Feeder = import_class(self.arg.feeder)
        self.print_log(f"Using data feeder: {self.arg.feeder}")

        if self.arg.phase == 'train':
            # Prepare dataset with progress updates
            self.print_log("Preparing SmartFallMM dataset...")
            builder = prepare_smartfallmm(self.arg)
            self.print_log("Dataset preparation complete")

            # Use provided subjects or all subjects
            train_subjects = train_subjects or self.train_subjects or self.arg.subjects
            val_subjects = val_subjects or self.val_subject or self.arg.subjects
            
            self.print_log(f"Splitting data for subjects: train={train_subjects}, val={val_subjects}")
            
            try:
                self.norm_train = split_by_subjects(builder, train_subjects, self.fuse)
                self.norm_val = split_by_subjects(builder, val_subjects, self.fuse)

                # Check if data is valid
                if not self.norm_train or not self.norm_val:
                    self.print_log("ERROR: Split produced empty datasets")
                    return False
                
                # Check for empty modalities
                for dataset_name, dataset in [("training", self.norm_train), ("validation", self.norm_val)]:
                    for key, value in dataset.items():
                        if key != 'labels' and (len(value) == 0 or value is None):
                            self.print_log(f"WARNING: Empty {key} in {dataset_name} data")
            except Exception as e:
                self.print_log(f"ERROR: Failed to split data: {e}")
                self.print_log(traceback.format_exc())
                return False

            # Check if validation data is valid
            if not self.norm_val or all(len(v) == 0 for k, v in self.norm_val.items() if k != 'labels'):
                self.print_log("ERROR: Validation data has empty values")
                return False

            # Create training data loader
            try:
                self.data_loader['train'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.train_feeder_args, dataset=self.norm_train),
                    batch_size=self.arg.batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker,
                    pin_memory=True,
                    drop_last=self.arg.train_feeder_args.get('drop_last', False),
                    persistent_workers=True if self.arg.num_worker > 0 else False
                )
                self.print_log(f"Training data loaded with {len(self.data_loader['train'])} batches")
            except Exception as e:
                self.print_log(f"ERROR: Failed to create training data loader: {e}")
                self.print_log(traceback.format_exc())
                return False

            # Visualize training data distribution
            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')

            # Create validation data loader
            try:
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.val_feeder_args, dataset=self.norm_val),
                    batch_size=self.arg.batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    pin_memory=True,
                    drop_last=self.arg.val_feeder_args.get('drop_last', False)
                )
                self.print_log(f"Validation data loaded with {len(self.data_loader['val'])} batches")
            except Exception as e:
                self.print_log(f"ERROR: Failed to create validation data loader: {e}")
                self.print_log(traceback.format_exc())
                return False

            # Visualize validation data distribution
            self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')

            # Prepare test data using validation subjects
            self.print_log(f"Preparing test data for subjects {val_subjects}")
            try:
                self.norm_test = split_by_subjects(builder, val_subjects, self.fuse)

                # Check if test data is valid
                if not self.norm_test or all(len(v) == 0 for k, v in self.norm_test.items() if k != 'labels'):
                    self.print_log("ERROR: Test data has empty values")
                    return False
            except Exception as e:
                self.print_log(f"ERROR: Failed to prepare test data: {e}")
                self.print_log(traceback.format_exc())
                return False

            # Create test data loader
            try:
                self.data_loader['test'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    pin_memory=True,
                    drop_last=self.arg.test_feeder_args.get('drop_last', False)
                )
                self.print_log(f"Test data loaded with {len(self.data_loader['test'])} batches")
            except Exception as e:
                self.print_log(f"ERROR: Failed to create test data loader: {e}")
                self.print_log(traceback.format_exc())
                return False

            # Log dataset sizes and modalities
            self.print_log(f"Training data modalities: {list(self.norm_train.keys())}")
            self.print_log(f"Validation data modalities: {list(self.norm_val.keys())}")
            self.print_log(f"Test data modalities: {list(self.norm_test.keys())}")

            return True
        else:
            # Testing phase
            self.print_log(f"Preparing test data for subjects {self.arg.subjects}")
            try:
                builder = prepare_smartfallmm(self.arg)
                self.norm_test = split_by_subjects(builder, self.arg.subjects, self.fuse)

                if not self.norm_test or all(len(v) == 0 for k, v in self.norm_test.items() if k != 'labels'):
                    self.print_log("ERROR: Test data has empty values")
                    return False
                
                self.data_loader['test'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    pin_memory=True,
                    drop_last=self.arg.test_feeder_args.get('drop_last', False)
                )
                self.print_log(f"Test data loaded with {len(self.data_loader['test'])} batches")
                
                # Visualize test data distribution
                self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, 'test')

                return True
            except Exception as e:
                self.print_log(f"ERROR: Failed to load test data: {e}")
                self.print_log(traceback.format_exc())
                return False

    def record_time(self):
        '''
        Record the current time for measuring durations

        Returns:
            Current time in seconds
        '''
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        '''
        Calculate the time since the last record_time() call

        Returns:
            Elapsed time in seconds
        '''
        split_time_val = time.time() - self.cur_time
        self.record_time()
        return split_time_val

    def print_log(self, string: str, print_time=True) -> None:
        '''
        Print a message to the console and save to the log file

        Args:
            string: Message to log
            print_time: Whether to include a timestamp
        '''
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            string = f"[ {localtime} ] {string}"
            
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)

    def loss_viz(self, train_loss: List[float], val_loss: List[float], fold=None):
        '''
        Visualize training and validation loss curves

        Args:
            train_loss: List of training loss values
            val_loss: List of validation loss values
            fold: Optional fold number for k-fold cross-validation
        '''
        if not train_loss or not val_loss:
            self.print_log("WARNING: Missing loss data for visualization")
            return
            
        epochs = range(len(train_loss))
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_loss, 'b-', label="Training Loss", marker='o', markersize=4, linewidth=2)
        plt.plot(epochs, val_loss, 'r-', label="Validation Loss", marker='s', markersize=4, linewidth=2)
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Add min/max values as annotations
        min_train_idx = np.argmin(train_loss)
        min_val_idx = np.argmin(val_loss)
        plt.annotate(f'Min: {train_loss[min_train_idx]:.4f}', 
                    xy=(min_train_idx, train_loss[min_train_idx]),
                    xytext=(min_train_idx+0.5, train_loss[min_train_idx]*1.1),
                    arrowprops=dict(facecolor='blue', shrink=0.05),
                    color='blue')
        plt.annotate(f'Min: {val_loss[min_val_idx]:.4f}', 
                    xy=(min_val_idx, val_loss[min_val_idx]),
                    xytext=(min_val_idx+0.5, val_loss[min_val_idx]*1.1),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    color='red')
        
        save_path = f'{self.arg.work_dir}/train_vs_val_loss.png'
        if fold is not None:
            fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            save_path = f'{fold_dir}/train_vs_val_loss.png'
            
        plt.savefig(save_path)
        plt.close()
        self.print_log(f"Created loss curve visualization at {save_path}")

    def metrics_viz(self, train_metrics, val_metrics, fold=None):
        '''
        Visualize training and validation metrics

        Args:
            train_metrics: List of training metrics dictionaries
            val_metrics: List of validation metrics dictionaries
            fold: Optional fold number for k-fold cross-validation
        '''
        if not train_metrics or not val_metrics:
            return
            
        epochs = range(len(train_metrics))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            
            train_values = [m.get(metric, 0) for m in train_metrics]
            val_values = [m.get(metric, 0) for m in val_metrics]
            
            plt.plot(epochs, train_values, 'b-', label=f'Train {metric}', marker='o', markersize=3)
            plt.plot(epochs, val_values, 'r-', label=f'Val {metric}', marker='s', markersize=3)
            plt.title(f'{metric.capitalize()} vs Epochs')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add max value annotations
            max_train_idx = np.argmax(train_values)
            max_val_idx = np.argmax(val_values)
            plt.annotate(f'Max: {train_values[max_train_idx]:.4f}', 
                        xy=(max_train_idx, train_values[max_train_idx]),
                        xytext=(max_train_idx+0.5, min(train_values[max_train_idx]*1.05, 1.0)),
                        arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.7),
                        color='blue')
            plt.annotate(f'Max: {val_values[max_val_idx]:.4f}', 
                        xy=(max_val_idx, val_values[max_val_idx]),
                        xytext=(max_val_idx+0.5, min(val_values[max_val_idx]*1.05, 1.0)),
                        arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.7),
                        color='red')
        
        plt.tight_layout()
        
        save_path = f'{self.arg.work_dir}/metrics.png'
        if fold is not None:
            fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            save_path = f'{fold_dir}/metrics.png'
            
        plt.savefig(save_path)
        plt.close()
        self.print_log(f"Created metrics visualization at {save_path}")

    def cm_viz(self, y_pred: List[int], y_true: List[int], fold=None):
        '''
        Create and save a confusion matrix visualization

        Args:
            y_pred: Predicted class labels
            y_true: True class labels
            fold: Optional fold number for k-fold cross-validation
        '''
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()

            # Set axis labels and title
            class_labels = ["Non-Fall", "Fall"]
            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels)
            plt.yticks(tick_marks, class_labels)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")

            # Add text annotations
            thresh = cm.max() / 2
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            
            save_path = f'{self.arg.work_dir}/confusion_matrix.png'
            if fold is not None:
                fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
                os.makedirs(fold_dir, exist_ok=True)
                save_path = f'{fold_dir}/confusion_matrix.png'
                
            plt.savefig(save_path)
            plt.close()
            self.print_log(f"Created confusion matrix visualization at {save_path}")
            
            # Calculate additional metrics
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            balanced_accuracy = (sensitivity + specificity) / 2
            
            self.print_log(f"Confusion Matrix Statistics:")
            self.print_log(f"  True Negatives: {tn}")
            self.print_log(f"  False Positives: {fp}")
            self.print_log(f"  False Negatives: {fn}")
            self.print_log(f"  True Positives: {tp}")
            self.print_log(f"  Specificity: {specificity:.4f}")
            self.print_log(f"  Sensitivity: {sensitivity:.4f}")
            self.print_log(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
        except Exception as e:
            self.print_log(f"ERROR: Failed to create confusion matrix: {e}")
            self.print_log(traceback.format_exc())

    def compute_metrics(self, outputs, targets):
        """
        Compute comprehensive metrics for fall detection

        Args:
            outputs: Model outputs
            targets: Ground truth labels

        Returns:
            Dictionary of computed metrics
        """
        # Convert logits to binary predictions
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().float()
            targets = targets.detach().cpu().float()
            
            # Apply sigmoid if needed
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                # Multi-class case
                predictions = torch.argmax(outputs, dim=1)
            else:
                # Binary classification case
                predictions = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
        else:
            # Handle numpy inputs
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                predictions = np.argmax(outputs, axis=1)
            else:
                predictions = (1 / (1 + np.exp(-outputs.reshape(-1))) > 0.5).astype(float)
        
        # Convert to numpy for sklearn metrics
        predictions_np = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        targets_np = targets.numpy() if isinstance(targets, torch.Tensor) else targets
        
        # Compute basic metrics
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
            
            accuracy = np.mean(predictions_np == targets_np)
            
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets_np, predictions_np, average='binary', zero_division=0
                )
            except Exception as e:
                self.print_log(f"Warning: Error computing precision_recall_fscore: {e}")
                precision = recall = f1 = 0.0
                
        # Compute confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(targets_np, predictions_np, labels=[0, 1]).ravel()
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            balanced_accuracy = (sensitivity + specificity) / 2
        except Exception as e:
            self.print_log(f"Warning: Error computing confusion matrix: {e}")
            tn = fp = fn = tp = 0
            false_alarm_rate = miss_rate = specificity = sensitivity = balanced_accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'false_alarm_rate': false_alarm_rate,
            'miss_rate': miss_rate,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy
        }

    def train(self, epoch, fold=None):
        '''
        Train the model for one epoch

        Args:
            epoch: Current epoch number
            fold: Optional fold number for k-fold cross-validation

        Returns:
            Validation loss after training
        '''
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'

        # Set model to training mode
        self.model.train()
        self.record_time()

        # Get data loader and initialize tracking variables
        loader = self.data_loader['train']
        timer = dict(dataloader=0.001, model=0.001, stats=0.001)
        train_metrics = defaultdict(float)
        total_samples = 0
        batch_count = 0

        # Training loop with progress bar
        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch} (Train)")
        for batch_idx, (inputs, targets, idx) in enumerate(process):
            batch_count += 1
            batch_size = targets.size(0)
            total_samples += batch_size
            
            # Move data to device
            with torch.no_grad():
                # Get the appropriate data modalities
                data_dict = {}
                if 'accelerometer' in inputs:
                    data_dict['accelerometer'] = inputs['accelerometer'].to(device).float()
                for modality in ['gyroscope', 'skeleton', 'quaternion', 'linear_acceleration', 'fusion_features']:
                    if modality in inputs:
                        data_dict[modality] = inputs[modality].to(device).float()
                targets = targets.to(device).float()

            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(data_dict)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                if hasattr(self.arg, 'grad_clip') and self.arg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data_dict)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if hasattr(self.arg, 'grad_clip') and self.arg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip)
                self.optimizer.step()

            timer['model'] += self.split_time()

            batch_metrics = self.compute_metrics(outputs, targets)
            for k, v in batch_metrics.items():
                train_metrics[k] += v * batch_size
            train_metrics['loss'] += loss.item() * batch_size

            process.set_postfix({
                'loss': f"{train_metrics['loss']/total_samples:.4f}",
                'acc': f"{100.0*train_metrics['accuracy']/total_samples:.2f}%",
                'f1': f"{train_metrics['f1']/batch_count:.4f}"
            })
            
            timer['stats'] += self.split_time()
            
        for k in train_metrics:
            train_metrics[k] /= total_samples

        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        self.print_log(
            f'Epoch {epoch+1}/{self.arg.num_epoch} - '
            f"Training - Loss: {train_metrics['loss']:.4f}, "
            f"Accuracy: {train_metrics['accuracy']:.4f}, "
            f"Precision: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}"
        )
        self.print_log(f'Time consumption: [Data]{proportion["dataloader"]}, '
                      f'[Network]{proportion["model"]}, [Stats]{proportion["stats"]}')

        self.train_metrics = train_metrics
        self.train_metrics_history.append(train_metrics.copy())
        self.train_loss_summary.append(train_metrics['loss'])

        val_metrics = self.eval(epoch, loader_name='val', fold=fold)
        self.val_metrics_history.append(val_metrics.copy())
        self.val_loss_summary.append(val_metrics['loss'])

        return val_metrics['loss']

    def eval(self, epoch, loader_name='val', fold=None, result_file=None):
        '''
        Evaluate the model on validation or test data

        Args:
            epoch: Current epoch number
            loader_name: Which data loader to use ('val' or 'test')
            fold: Optional fold number for k-fold cross-validation
            result_file: Optional file to save detailed results

        Returns:
            Dictionary of average metrics for the evaluation set
        '''
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'

        if result_file is not None:
            f_r = open(result_file, 'w', encoding='utf-8')

        self.model.eval()
        self.print_log(f'Evaluating on {loader_name} set (Epoch {epoch+1})')

        val_metrics = defaultdict(float)
        total_samples = 0
        batch_count = 0
        all_predictions = []
        all_targets = []
        all_outputs = []

        process = tqdm(self.data_loader[loader_name], desc=f"Epoch {epoch+1} ({loader_name.capitalize()})")
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                batch_count += 1
                batch_size = targets.size(0)
                total_samples += batch_size
                
                data_dict = {}
                if 'accelerometer' in inputs:
                    data_dict['accelerometer'] = inputs['accelerometer'].to(device).float()
                for modality in ['gyroscope', 'skeleton', 'quaternion', 'linear_acceleration', 'fusion_features']:
                    if modality in inputs:
                        data_dict[modality] = inputs[modality].to(device).float()
                targets = targets.to(device).float()

                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data_dict)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(data_dict)
                    loss = self.criterion(outputs, targets)

                batch_metrics = self.compute_metrics(outputs, targets)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v * batch_size
                
                val_metrics['loss'] += loss.item() * batch_size
                
                predictions = torch.sigmoid(outputs.view(-1)) > 0.5 if outputs.numel() == targets.numel() else torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.view(-1).cpu().numpy())
                
                process.set_postfix({
                    'loss': f"{val_metrics['loss']/total_samples:.4f}",
                    'acc': f"{100.0*val_metrics['accuracy']/total_samples:.2f}%",
                    'f1': f"{val_metrics['f1']/batch_count:.4f}"
                })
                
                if result_file is not None and f_r is not None:
                    for i, (pred, target) in enumerate(zip(predictions.cpu(), targets.cpu())):
                        f_r.write(f"{int(pred.item())} ==> {int(target.item())}\n")

        if result_file is not None and f_r is not None:
            f_r.close()

        for k in val_metrics:
            val_metrics[k] /= total_samples
            
        if all_predictions and all_targets:
            self.cm_viz(all_predictions, all_targets, fold)

        self.print_log(
            f'{loader_name.capitalize()} metrics: Loss={val_metrics["loss"]:.4f}, '
            f'Accuracy={val_metrics["accuracy"]:.4f}, '
            f'F1={val_metrics["f1"]:.4f}, '
            f'Precision={val_metrics["precision"]:.4f}, '
            f'Recall={val_metrics["recall"]:.4f}, '
            f'FAR={val_metrics["false_alarm_rate"]:.4f}, '
            f'MR={val_metrics["miss_rate"]:.4f}, '
            f'Balanced Acc={val_metrics["balanced_accuracy"]:.4f}'
        )

        if loader_name in ['val', 'test']:
            metrics_file = f'{self.arg.work_dir}/{loader_name}_result.txt'
            if fold is not None:
                fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
                os.makedirs(fold_dir, exist_ok=True)
                metrics_file = f'{fold_dir}/{loader_name}_result.txt'
                
            with open(metrics_file, 'w') as f:
                f.write(f"accuracy {val_metrics['accuracy']:.6f}\n")
                f.write(f"f1_score {val_metrics['f1']:.6f}\n")
                f.write(f"precision {val_metrics['precision']:.6f}\n")
                f.write(f"recall {val_metrics['recall']:.6f}\n")
                f.write(f"false_alarm_rate {val_metrics['false_alarm_rate']:.6f}\n")
                f.write(f"miss_rate {val_metrics['miss_rate']:.6f}\n")
                f.write(f"balanced_accuracy {val_metrics['balanced_accuracy']:.6f}\n")

        if loader_name == 'val':
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.best_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_best_model(epoch, val_metrics, fold)
                self.print_log(f'New best model saved: F1={val_metrics["f1"]:.4f}, Acc={val_metrics["accuracy"]:.4f}')
            else:
                self.patience_counter += 1
                self.print_log(f'No improvement for {self.patience_counter} epochs (patience: {self.arg.patience})')
                
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
        else:
            self.test_accuracy = val_metrics['accuracy']
            self.test_f1 = val_metrics['f1']

        self.val_metrics = val_metrics
        
        return val_metrics

    def save_best_model(self, epoch, metrics, fold=None):
        """
        Save the best model based on validation metrics

        Args:
            epoch: Current epoch number
            metrics: Dictionary of validation metrics
            fold: Optional fold number for k-fold cross-validation
        """
        if fold is not None:
            fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            save_path = os.path.join(fold_dir, f"{self.arg.model_saved_name}.pt")
        else:
            save_path = self.model_path
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if not isinstance(self.model, nn.DataParallel) 
                               else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_f1': self.best_f1,
            'best_accuracy': self.best_accuracy,
            'metrics': metrics
        }
        
        try:
            torch.save(state_dict, save_path)
            self.print_log(f"Best model saved to {save_path}")
            model_copy_path = os.path.join(
                os.path.dirname(save_path),
                f"{os.path.basename(save_path).split('.')[0]}_f1_{metrics['f1']:.4f}_acc_{metrics['accuracy']:.4f}_epoch_{epoch}.pt"
            )
            torch.save(state_dict, model_copy_path)
        except Exception as e:
            self.print_log(f"Error saving model: {e}")
            self.print_log(f"Trying alternative approach...")
            try:
                model_state = self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else self.model.module.state_dict()
                torch.save(model_state, save_path)
                self.print_log(f"Model state dict saved to {save_path}")
            except Exception as e2:
                self.print_log(f"Failed to save model: {e2}")

    def train_fold(self, fold_idx, train_subjects, val_subjects):
        """
        Train the model for a specific fold in cross-validation

        Args:
            fold_idx: Index of the current fold
            train_subjects: List of subject IDs for training
            val_subjects: List of subject IDs for validation

        Returns:
            Dictionary of best metrics for this fold
        """
        fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        
        self.print_log(f"\n{'='*20} Training Fold {fold_idx} {'='*20}")
        self.print_log(f"Training subjects: {train_subjects}")
        self.print_log(f"Validation subjects: {val_subjects}")
        
        self.model = self.load_model(self.arg.model, self.arg.model_args)
        
        if len(self.available_gpus) > 1 and self.arg.multi_gpu:
            self.model = nn.DataParallel(
                self.model, 
                device_ids=self.available_gpus
            )
            
        self.load_optimizer()
        
        self.best_f1 = 0
        self.best_accuracy = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        loaded = self.load_data(train_subjects, val_subjects)
        if not loaded:
            self.print_log(f"Error loading data for fold {fold_idx}, skipping")
            return {'fold': fold_idx, 'f1': 0, 'accuracy': 0, 'epochs_trained': 0, 'error': True}
        
        for epoch in range(self.arg.num_epoch):
            val_loss = self.train(epoch, fold=fold_idx)
            
            if self.patience_counter >= self.arg.patience:
                self.print_log(f"Early stopping triggered at epoch {epoch}")
                break
                
        self.loss_viz(self.train_loss_summary, self.val_loss_summary, fold=fold_idx)
        self.metrics_viz(self.train_metrics_history, self.val_metrics_history, fold=fold_idx)
        
        test_metrics = self.eval(0, loader_name='test', fold=fold_idx, 
                               result_file=os.path.join(fold_dir, "test_predictions.txt"))
        
        best_metrics = {
            'fold': fold_idx,
            'f1': self.best_f1,
            'accuracy': self.best_accuracy,
            'balanced_accuracy': test_metrics['balanced_accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'false_alarm_rate': test_metrics['false_alarm_rate'],
            'miss_rate': test_metrics['miss_rate'],
            'epochs_trained': len(self.train_loss_summary),
            'test_f1': test_metrics['f1']
        }
        
        self.print_log(f"Fold {fold_idx} completed with best validation F1: {self.best_f1:.4f}")
        self.print_log(f"Test set F1: {test_metrics['f1']:.4f}")
        
        return best_metrics

    def compare_filters(self, fold=None):
        """
        Compare performance of EKF vs Madgwick filters on validation data
        
        Args:
            fold: Optional fold number for cross-validation
        """
        self.print_log("\nComparing Extended Kalman Filter (EKF) and Madgwick Filter Performance")
        
        results = {}
        
        if hasattr(self.arg, 'filter_types') and self.arg.filter_types:
            filter_types = self.arg.filter_types
        else:
            filter_types = ['madgwick', 'ekf']
            
        self.print_log(f"Comparing filter types: {filter_types}")
        
        if hasattr(self.data_loader, 'test') and self.data_loader['test'] is not None:
            try:
                for batch_idx, (inputs, targets, idx) in enumerate(self.data_loader['test']):
                    if 'accelerometer' in inputs and 'gyroscope' in inputs:
                        acc_sample = inputs['accelerometer'][0].numpy()
                        gyro_sample = inputs['gyroscope'][0].numpy()
                        
                        filter_accuracy = compare_filter_accuracy(acc_sample, gyro_sample)
                        
                        filter_vis_dir = os.path.join(self.arg.work_dir, "filter_comparison")
                        os.makedirs(filter_vis_dir, exist_ok=True)
                        visualize_filter_comparison(filter_accuracy, filter_vis_dir)
                        
                        feature_results = compare_filter_features(acc_sample, gyro_sample)
                        visualize_feature_comparison(feature_results, filter_vis_dir)
                        
                        self.print_log(f"Direct filter comparison visualizations saved to {filter_vis_dir}")
                        break
            except Exception as e:
                self.print_log(f"Error in direct filter comparison: {e}")
        
        for filter_type in filter_types:
            if 'fusion_options' not in self.arg.dataset_args:
                self.arg.dataset_args['fusion_options'] = {}
            self.arg.dataset_args['fusion_options']['enabled'] = True
            self.arg.dataset_args['fusion_options']['filter_type'] = filter_type
            
            self.print_log(f"\nTesting with {filter_type} filter")
            self.load_data()
            
            metrics = self.eval(0, loader_name='test', fold=fold)
            
            results[filter_type] = {
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'false_alarm_rate': metrics['false_alarm_rate'],
                'miss_rate': metrics['miss_rate'],
                'balanced_accuracy': metrics.get('balanced_accuracy', (metrics['precision'] + metrics['recall'])/2)
            }
        
        self.print_log("\nFilter Performance Comparison:")
        self.print_log(f"{'Metric':<15} " + " ".join([f"{ft:<10}" for ft in filter_types]))
        self.print_log("-" * (15 + 10 * len(filter_types)))
        
        best_filter = max(results.items(), key=lambda x: x[1]['f1'])[0]
        
        for metric in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']:
            metric_values = [results[ft][metric] for ft in filter_types]
            best_idx = np.argmax(metric_values)
            
            metric_str = f"{metric.capitalize():<15} "
            for i, val in enumerate(metric_values):
                if i == best_idx:
                    metric_str += f"\033[1m{val:.4f}\033[0m    "
                else:
                    metric_str += f"{val:.4f}    "
            
            self.print_log(metric_str)
        
        plt.figure(figsize=(12, 8))
        metrics_list = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        x = np.arange(len(metrics_list))
        width = 0.8 / len(filter_types)
        
        for i, filter_type in enumerate(filter_types):
            values = [results[filter_type][m] for m in metrics_list]
            offset = width * i - width * (len(filter_types) - 1) / 2
            plt.bar(x + offset, values, width, label=f'{filter_type.upper()} Filter')
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Performance Comparison by Filter Type')
        plt.xticks(x, [m.capitalize() for m in metrics_list])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        filter_dir = os.path.join(self.arg.work_dir, "filter_comparison")
        os.makedirs(filter_dir, exist_ok=True)
        plt.savefig(os.path.join(filter_dir, "filter_performance.png"))
        plt.close()
        
        best_f1_improvement = 0
        second_best = ""
        
        if len(filter_types) > 1:
            f1_values = [results[ft]['f1'] for ft in filter_types]
            sorted_indices = np.argsort(f1_values)[::-1]
            
            best_filter = filter_types[sorted_indices[0]]
            second_best = filter_types[sorted_indices[1]]
            best_f1_improvement = (results[best_filter]['f1'] - results[second_best]['f1']) / results[second_best]['f1'] * 100
            
            self.print_log(f"\n{best_filter.upper()} Filter performed better with {best_f1_improvement:.2f}% higher F1 score than {second_best.upper()}")
        
        with open(os.path.join(filter_dir, "comparison_results.txt"), 'w') as f:
            f.write(f"Filter Comparison Results\n")
            f.write(f"=======================\n\n")
            f.write(f"{'Metric':<15} " + " ".join([f"{ft:<10}" for ft in filter_types]) + "\n")
            f.write("-" * (15 + 10 * len(filter_types)) + "\n")
            for metric in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy', 'false_alarm_rate', 'miss_rate']:
                metric_values = [results[ft][metric] for ft in filter_types]
                metric_str = f"{metric.capitalize():<15} " + " ".join([f"{val:.4f}    " for val in metric_values])
                f.write(metric_str + "\n")
            if len(filter_types) > 1:
                f.write(f"\nBest filter: {best_filter.upper()} with {best_f1_improvement:.2f}% higher F1 score than {second_best.upper()}")
            else:
                f.write(f"\nOnly tested {filter_types[0].upper()} filter")
        
        return results

    def kfold_cross_validation(self):
        """
        Perform k-fold cross-validation using the specified fold assignments

        Returns:
            Dictionary with average metrics across all folds
        """
        self.print_log(f"\n{'='*20} K-Fold Cross-Validation {'='*20}")
        
        folds = self.create_folds(self.arg.subjects, self.num_folds)
        all_fold_metrics = []
        
        for fold_idx, (train_subjects, val_subjects) in enumerate(folds, 1):
            fold_metrics = self.train_fold(fold_idx, train_subjects, val_subjects)
            all_fold_metrics.append(fold_metrics)
            
        avg_metrics = {}
        metric_keys = ['f1', 'accuracy', 'precision', 'recall', 'false_alarm_rate', 
                       'miss_rate', 'balanced_accuracy', 'test_f1', 'epochs_trained']
        
        for key in metric_keys:
            values = [m.get(key, 0) for m in all_fold_metrics if not m.get('error', False)]
            if values:
                avg_metrics[key] = sum(values) / len(values)
                std_metrics = np.std(values)
                avg_metrics[f'{key}_std'] = std_metrics
            else:
                avg_metrics[key] = 0
                avg_metrics[f'{key}_std'] = 0
        
        self.print_log(f"\n{'='*20} Cross-Validation Results {'='*20}")
        for key in ['accuracy', 'f1', 'balanced_accuracy', 'precision', 'recall']:
            if key in avg_metrics:
                self.print_log(f"Average {key.capitalize()}: {avg_metrics[key]:.4f} ± {avg_metrics[f'{key}_std']:.4f}")
        
        self.print_log(f"Average Epochs: {avg_metrics['epochs_trained']:.1f}")
        
        summary = {
            'fold_metrics': all_fold_metrics,
            'average_metrics': avg_metrics
        }
        
        with open(os.path.join(self.arg.work_dir, 'cv_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        self.create_cv_summary_plot(all_fold_metrics)
        
        if hasattr(self.arg, 'compare_filters') and self.arg.compare_filters:
            self.compare_filters()
        
        return summary

    def create_cv_summary_plot(self, fold_metrics):
        """
        Create a summary plot of the cross-validation results

        Args:
            fold_metrics: List of metrics dictionaries for each fold
        """
        valid_metrics = [m for m in fold_metrics if not m.get('error', False)]
        
        if not valid_metrics:
            self.print_log("No valid fold metrics available for visualization")
            return
            
        plt.figure(figsize=(12, 8))
        
        folds = [m['fold'] for m in valid_metrics]
        f1_scores = [m['f1'] for m in valid_metrics]
        accuracies = [m['accuracy'] for m in valid_metrics]
        balanced_accs = [m.get('balanced_accuracy', 0) for m in valid_metrics]
        test_f1s = [m.get('test_f1', 0) for m in valid_metrics]
        
        x = np.arange(len(folds))
        width = 0.2
        
        ax = plt.subplot(111)
        bars1 = ax.bar(x - width*1.5, f1_scores, width, label='Val F1 Score')
        bars2 = ax.bar(x - width/2, test_f1s, width, label='Test F1 Score')
        bars3 = ax.bar(x + width/2, accuracies, width, label='Accuracy')
        bars4 = ax.bar(x + width*1.5, balanced_accs, width, label='Balanced Acc')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation Results by Fold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {f}' for f in folds])
        ax.legend()
        plt.grid(axis='y', alpha=0.3)
        
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=8)
                
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        add_labels(bars4)
        
        avg_f1 = np.mean(f1_scores)
        avg_test_f1 = np.mean(test_f1s)
        avg_acc = np.mean(accuracies)
        avg_balanced = np.mean(balanced_accs)
        
        ax.axhline(y=avg_f1, color='C0', linestyle='--', alpha=0.7, label=f'Avg Val F1: {avg_f1:.4f}')
        ax.axhline(y=avg_test_f1, color='C1', linestyle='--', alpha=0.7, label=f'Avg Test F1: {avg_test_f1:.4f}')
        ax.axhline(y=avg_acc, color='C2', linestyle='--', alpha=0.7, label=f'Avg Acc: {avg_acc:.4f}')
        ax.axhline(y=avg_balanced, color='C3', linestyle='--', alpha=0.7, label=f'Avg Bal Acc: {avg_balanced:.4f}')
        
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.arg.work_dir, 'cv_summary.png'), bbox_inches='tight')
        plt.close()
        
        self.print_log(f"Created cross-validation summary plot")

    def train_final_model(self, best_fold_idx=None):
        """
        Train the final model using all subjects based on cross-validation results

        Args:
            best_fold_idx: Index of the best performing fold to use as reference
        """
        self.print_log(f"\n{'='*20} Training Final Model {'='*20}")
        
        if best_fold_idx:
            self.print_log(f"Using configuration from fold {best_fold_idx} as reference")
            fold_dir = os.path.join(self.arg.work_dir, f'fold_{best_fold_idx}')
            best_model_path = os.path.join(fold_dir, f"{self.arg.model_saved_name}.pt")
            
            if os.path.exists(best_model_path):
                best_state = torch.load(best_model_path)
                best_epoch = best_state['epoch']
                self.print_log(f"Best fold reached optimal performance at epoch {best_epoch}")
                num_epochs = best_epoch + 1
            else:
                self.print_log(f"Best fold model not found, using default epochs")
                num_epochs = self.arg.num_epoch
        else:
            num_epochs = self.arg.num_epoch
        
        self.print_log(f"Training final model for {num_epochs} epochs")
        
        self.model = self.load_model(self.arg.model, self.arg.model_args)
        
        if len(self.available_gpus) > 1 and self.arg.multi_gpu:
            self.model = nn.DataParallel(
                self.model, 
                device_ids=self.available_gpus
            )
            
        self.load_optimizer()
        
        self.best_f1 = 0
        self.best_accuracy = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        all_subjects = self.arg.subjects
        self.load_data(all_subjects, all_subjects)
        
        for epoch in range(num_epochs):
            val_loss = self.train(epoch)
            if self.patience_counter >= self.arg.patience:
                self.print_log(f"Early stopping triggered at epoch {epoch}")
                break
        
        self.loss_viz(self.train_loss_summary, self.val_loss_summary)
        self.metrics_viz(self.train_metrics_history, self.val_metrics_history)
            
        final_state = {
            'epoch': num_epochs - 1,
            'model_state_dict': self.model.state_dict() if not isinstance(self.model, nn.DataParallel) 
                               else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': self.val_metrics
        }
        
        final_path = os.path.join(self.arg.work_dir, "final_model.pt")
        torch.save(final_state, final_path)
        
        self.print_log(f"Final model saved to {final_path}")
        
        test_metrics = self.eval(num_epochs - 1, loader_name='test')
        
        self.print_log(f"\n{'='*20} Final Model Results {'='*20}")
        self.print_log(f"Test F1 Score: {test_metrics['f1']:.4f}")
        self.print_log(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.print_log(f"Test Precision: {test_metrics['precision']:.4f}")
        self.print_log(f"Test Recall: {test_metrics['recall']:.4f}")
        self.print_log(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")

    def create_df(self, columns=['test_subject', 'train_subjects', 'accuracy', 'f1_score']):
        '''
        Create an empty DataFrame for storing results

        Args:
            columns: Column names for the DataFrame

        Returns:
            Empty DataFrame with the specified columns
        '''
        return pd.DataFrame(columns=columns)

    def start(self):
        '''
        Start the training or testing process
        
        This method implements the complete workflow for model training,
        validation, and testing using a leave-one-subject-out cross-validation
        approach for robust evaluation.
        '''
        if self.arg.phase == 'train':
            try:
                configure_parallel_processing(self.arg)
                self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
                
                if self.use_kfold:
                    cv_results = self.kfold_cross_validation()
                    
                    valid_folds = [m for m in cv_results['fold_metrics'] if not m.get('error', False)]
                    if valid_folds:
                        best_fold = max(valid_folds, key=lambda x: x['f1'])
                        best_fold_idx = best_fold['fold']
                        
                        self.print_log(f"Best performing fold: {best_fold_idx} with F1 score: {best_fold['f1']:.4f}")
                        
                        self.train_final_model(best_fold_idx)
                    else:
                        self.print_log("No valid folds completed, cannot train final model")
                else:
                    self.train_subjects = self.arg.subjects
                    self.val_subject = self.arg.subjects
                    
                    if not self.load_data():
                        self.print_log("Error loading data, aborting training")
                        return
                    
                    self.load_optimizer()
                    
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        val_loss = self.train(epoch)
                        if self.patience_counter >= self.arg.patience:
                            self.print_log(f"Early stopping triggered at epoch {epoch}")
                            break
                    
                    if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                        self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                    
                    if len(self.train_metrics_history) > 0 and len(self.val_metrics_history) > 0:
                        self.metrics_viz(self.train_metrics_history, self.val_metrics_history)
                    
                    self.print_log(f"\n{'='*20} Final Evaluation {'='*20}")
                    self.eval(0, loader_name='test')

                if hasattr(self.arg, 'compare_filters') and self.arg.compare_filters:
                    self.compare_filters()

            except Exception as e:
                self.print_log(f"Error during training: {e}")
                self.print_log(traceback.format_exc())
                cleanup_resources()
        
        else:
            try:
                if not self.load_data():
                    self.print_log("Error loading test data")
                    return
                    
                self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)
                self.print_log(f'Test accuracy: {self.test_accuracy:.4f}')
                self.print_log(f'Test F1 score: {self.test_f1:.4f}')
            
            except Exception as e:
                self.print_log(f"Error during testing: {e}")
                self.print_log(traceback.format_exc())
            finally:
                cleanup_resources()

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)

        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WARNING: Unrecognized configuration parameter: {k}')

        parser.set_defaults(**default_arg)
        args = parser.parse_args()

    init_seed(args.seed)
    trainer = Trainer(args)
    try:
        trainer.start()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        cleanup_resources()

