#!/usr/bin/env python3
"""
Fall Detection Training and Evaluation System

This script provides a complete pipeline for training and evaluating fall detection
models using inertial sensor data with Madgwick filter-based sensor fusion. It supports:

1. Multi-modal sensor inputs (accelerometer, gyroscope, quaternion)
2. Cross-validation for robust performance assessment
3. Detailed performance metrics and visualizations
4. Multiple fusion approaches with Madgwick filter as the default
5. Robust error handling and resource management

Author: Claude
Date: March 2025
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
import logging
from collections import defaultdict
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
from utils.dataset import prepare_smartfallmm, split_by_subjects
from utils.imu_fusion import cleanup_resources, update_thread_configuration

# Register cleanup function to ensure resources are released
import atexit
def cleanup_on_exit():
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
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fall Detection and Human Activity Recognition')
    
    # Basic configuration
    parser.add_argument('--config', default='./config/smartfallmm/madgwick_fusion.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--dataset', type=str, default='smartfallmm',
                        help='Dataset name to use')
    parser.add_argument('--phase', type=str, default='train',
                        help='Phase: train or test')
    parser.add_argument('--work-dir', type=str, default='work_dir',
                        help='Working directory for outputs')
    
    # Model configuration
    parser.add_argument('--model', default=None,
                        help='Model class path to load')
    parser.add_argument('--model-args', default=None,
                        help='Dictionary of model arguments')
    parser.add_argument('--weights', type=str,
                        help='Path to pretrained weights file')
    parser.add_argument('--model-saved-name', type=str, default='model',
                        help='Name for saving the trained model')
    
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
    
    # Optimizer settings
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
    
    # Loss function
    parser.add_argument('--loss', default='torch.nn.CrossEntropyLoss',
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
    
    # Hardware and environment
    parser.add_argument('--device', nargs='+', default=[0, 1], type=int,
                        help='CUDA device IDs to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num-worker', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--multi-gpu', type=str2bool, default=True,
                        help='Whether to use multiple GPUs when available')
    parser.add_argument('--parallel-threads', type=int, default=48,
                        help='Number of parallel threads for preprocessing')
    
    # Evaluation and validation
    parser.add_argument('--include-val', type=str2bool, default=True,
                        help='Whether to include validation set')
    parser.add_argument('--result-file', type=str,
                        help='File to save results to')
    
    # Cross-validation
    parser.add_argument('--kfold', type=str2bool, default=True,
                        help='Whether to use k-fold cross-validation')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    parser.add_argument('--print-log', type=str2bool, default=True,
                        help='Whether to print and save logs')
    
    return parser

def init_seed(seed):
    """Initialize random seeds for reproducibility"""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Use deterministic algorithms where possible
    torch.backends.cudnn.deterministic = False
    # Enable benchmark mode for faster training
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    """Dynamically import a class from a string path"""
    try:
        mod_str, _sep, class_str = import_str.rpartition('.')
        __import__(mod_str)
        return getattr(sys.modules[mod_str], class_str)
    except (ImportError, AttributeError) as e:
        raise ImportError(f'Class {class_str} cannot be found ({str(e)})')

def setup_gpu_environment(args):
    """
    Configure GPU environment based on available resources.
    
    Args:
        args: Command line arguments with device preferences
        
    Returns:
        List of available device IDs and whether to use AMP
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPUs")
        
        # Select devices based on arguments and availability
        if isinstance(args.device, list) and len(args.device) > 0:
            devices = [i for i in args.device if i < num_gpus]
            if not devices:
                devices = [0] if num_gpus > 0 else []
        elif num_gpus >= 2 and args.multi_gpu:
            devices = list(range(min(2, num_gpus)))
        elif num_gpus > 0:
            devices = [0]
        else:
            devices = []
            
        if devices:
            # Set visible devices for CUDA
            gpu_list = ",".join(map(str, devices))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
            
            # Check if AMP (Automatic Mixed Precision) is supported
            use_amp = False
            for i in devices:
                if i < num_gpus:
                    gpu_name = torch.cuda.get_device_name(i)
                    # Enable AMP for newer GPUs
                    if any(x in gpu_name for x in ["A100", "H100", "A10", "A40"]):
                        use_amp = True
                        break
            
            return devices, use_amp
        
    # No GPU available
    return [], False

def configure_parallel_processing(args):
    """
    Configure thread pools for parallel processing.
    
    Args:
        args: Command line arguments with thread configuration
    """
    if hasattr(args, 'parallel_threads') and args.parallel_threads > 0:
        new_total = args.parallel_threads
        
        # Distribute threads between file processing and per-file processing
        if new_total < 4:
            max_files = 1
            threads_per_file = new_total
        else:
            max_files = min(12, new_total // 4)
            threads_per_file = min(4, new_total // max_files)
        
        # Update thread pool configuration in imu_fusion module
        update_thread_configuration(max_files, threads_per_file)
        logger.info(f"Configured parallel processing: {max_files} files × {threads_per_file} threads = {new_total} total")

class Trainer:
    """
    Main trainer class that handles the training, validation, and testing pipeline.
    
    This class orchestrates the entire training process, including data loading,
    model initialization, optimization, and evaluation. It supports cross-validation
    and different filter types for IMU fusion.
    """
    def __init__(self, arg):
        self.arg = arg
        
        # Initialize tracking variables
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_f1 = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0
        self.test_accuracy = 0
        self.test_f1 = 0
        self.patience_counter = 0
        self.fold_metrics = []
        
        # Subject information
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        
        # Resources
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision training
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = dict()
        
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}.pt'
        
        # Determine modalities and fusion settings
        self.inertial_modality = [modality for modality in arg.dataset_args['modalities']
                                 if modality != 'skeleton']
        self.has_gyro = 'gyroscope' in self.inertial_modality
        self.has_fusion = len(self.inertial_modality) > 1 or (
            'fusion_options' in arg.dataset_args and
            arg.dataset_args['fusion_options'].get('enabled', False)
        )
        self.fuse = self.has_fusion
        self.filter_type = arg.dataset_args.get('fusion_options', {}).get('filter_type', 'madgwick')

        # Setup work directory
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            if arg.config:
                self.save_config(arg.config, arg.work_dir)

        # Configure GPU environment
        self.available_gpus, self.use_amp = setup_gpu_environment(arg)
        arg.device = self.available_gpus if self.available_gpus else arg.device
        self.output_device = arg.device[0] if isinstance(arg.device, list) and len(arg.device) > 0 else arg.device
        
        # Configure mixed precision
        if self.use_amp:
            self.scaler = torch.amp.GradScaler() if torch.__version__ >= '1.6.0' else None
            self.print_log("Using Automatic Mixed Precision (AMP) training")
        
        # Load model
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            self.model = self.load_weights()

        # Configure multi-GPU
        if len(self.available_gpus) > 1 and arg.multi_gpu:
            self.model = nn.DataParallel(
                self.model, 
                device_ids=self.available_gpus
            )

        # Load loss function
        self.load_loss()
        self.include_val = arg.include_val

        # Log model info
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params:,}')
        self.print_log(f'Model size: {num_params / (1024 ** 2):.2f} MB')
        self.print_log(f'Sensor modalities: {self.inertial_modality}')
        self.print_log(f'Using fusion: {self.fuse} (filter: {self.filter_type})')
        
        if self.available_gpus:
            self.print_log(f'Using GPUs: {self.available_gpus}')
        else:
            self.print_log('Using CPU for computation')
            
        # Configure cross-validation
        if hasattr(self.arg, 'kfold') and self.arg.kfold:
            self.use_kfold = True
            self.num_folds = self.arg.num_folds
            self.fold_metrics = []
            self.print_log(f'Using {self.num_folds}-fold cross-validation')
        else:
            self.use_kfold = False
            self.print_log('Using single train/val/test split')

    def save_config(self, src_path, desc_path):
        """
        Save a copy of the configuration file to the working directory.
        
        Args:
            src_path: Source configuration file path
            desc_path: Destination directory
        """
        config_file = src_path.rpartition("/")[-1]
        self.print_log(f'Saving config to {desc_path}/{config_file}')
        shutil.copy(src_path, f'{desc_path}/{config_file}')

    def count_parameters(self, model):
        """
        Count the number of trainable parameters in a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def load_model(self, model_path, model_args):
        """
        Load model from the specified class path with given arguments.
        
        Args:
            model_path: Path to model class
            model_args: Arguments to pass to model constructor
            
        Returns:
            Initialized model
        """
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'

        Model = import_class(model_path)
        self.print_log(f"Loading model: {model_path}")
        
        # Fix feature dimension for fusion models
        if 'feature_dim' not in model_args and 'embed_dim' in model_args:
            if model_args.get('fusion_type', 'concat') == 'concat':
                model_args['feature_dim'] = model_args['embed_dim'] * 3
            else:
                model_args['feature_dim'] = model_args['embed_dim']
            self.print_log(f"Auto-setting feature_dim to {model_args['feature_dim']}")
        
        # Ensure num_heads divides feature_dim evenly
        if 'num_heads' in model_args and 'feature_dim' in model_args:
            if model_args['feature_dim'] % model_args['num_heads'] != 0:
                old_heads = model_args['num_heads']
                for heads in [old_heads-1, old_heads-2, old_heads+1, old_heads+2]:
                    if heads > 0 and model_args['feature_dim'] % heads == 0:
                        model_args['num_heads'] = heads
                        self.print_log(f"Adjusted num_heads from {old_heads} to {heads} to ensure divisibility with feature_dim={model_args['feature_dim']}")
                        break
        
        try:
            model = Model(**model_args).to(device)
        except Exception as e:
            self.print_log(f"Error instantiating model: {str(e)}")
            self.print_log(f"Attempting to fix dimension issue and retry...")
            
            if 'feature_dim' in model_args and 'num_heads' in model_args:
                heads = model_args['num_heads']
                feature_dim = model_args['feature_dim']
                new_feature_dim = (feature_dim // heads) * heads
                if new_feature_dim != feature_dim:
                    model_args['feature_dim'] = new_feature_dim
                    self.print_log(f"Adjusted feature_dim from {feature_dim} to {new_feature_dim} to ensure divisibility with num_heads={heads}")
            
            model = Model(**model_args).to(device)
        
        return model

    def load_weights(self):
        """
        Load model weights from file.
        
        Returns:
            Model with loaded weights
        """
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        
        if not self.arg.weights:
            raise ValueError("No weights file specified for testing")
            
        self.print_log(f"Loading weights from: {self.arg.weights}")
        
        try:
            # First try loading the entire model
            model = torch.load(self.arg.weights, map_location=device)
            self.print_log("Loaded complete model")
            return model
        except Exception as e:
            try:
                # If that fails, load just the state dict
                Model = import_class(self.arg.model)
                model = Model(**self.arg.model_args).to(device)
                
                state_dict = torch.load(self.arg.weights, map_location=device)
                
                if isinstance(state_dict, dict):
                    if 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                    elif any(k.startswith('module.') for k in state_dict.keys()):
                        # Handle DataParallel models
                        from collections import OrderedDict
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            name = k[7:] if k.startswith('module.') else k
                            new_state_dict[name] = v
                        model.load_state_dict(new_state_dict)
                    else:
                        model.load_state_dict(state_dict)
                else:
                    raise ValueError(f"Unexpected state_dict type: {type(state_dict)}")
                
                return model
            except Exception as load_err:
                raise ValueError(f"Failed to load weights: {str(load_err)}\nOriginal error: {str(e)}")

    def load_loss(self):
        """Load the loss function based on configuration"""
        try:
            if self.arg.loss.startswith('torch.nn.'):
                loss_class_name = self.arg.loss.split('.')[-1]
                
                if loss_class_name == 'BCEWithLogitsLoss':
                    self.print_log("Replacing BCEWithLogitsLoss with CrossEntropyLoss for multi-class outputs")
                    self.criterion = torch.nn.CrossEntropyLoss()
                else:
                    loss_class = getattr(torch.nn, loss_class_name)
                    loss_args = eval(self.arg.loss_args) if isinstance(self.arg.loss_args, str) else self.arg.loss_args
                    self.criterion = loss_class(**(loss_args or {}))
            else:
                Loss = import_class(self.arg.loss)
                self.criterion = Loss(**(eval(self.arg.loss_args) if isinstance(self.arg.loss_args, str) else {}))
                
            self.print_log(f"Using loss function: {self.criterion.__class__.__name__}")
            
        except Exception as e:
            self.print_log(f"Error loading loss function: {str(e)}")
            self.criterion = torch.nn.CrossEntropyLoss()
            self.print_log("Fallback to CrossEntropyLoss")

    def load_optimizer(self):
        """Initialize optimizer and scheduler based on configuration"""
        optimizer_name = self.arg.optimizer.lower()

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        self.load_scheduler()

    def load_scheduler(self):
        """Initialize learning rate scheduler based on configuration"""
        scheduler_name = getattr(self.arg, 'scheduler', 'plateau')
        
        if scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=10,
                verbose=True
            )
        elif scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.arg.num_epoch,
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
        else:
            self.scheduler = None

    def create_folds(self, subjects, num_folds=5):
        """
        Create cross-validation folds from subject IDs.
        
        Args:
            subjects: List of all subject IDs
            num_folds: Number of folds to create
            
        Returns:
            List of (train_subjects, val_subjects) tuples for each fold
        """
        # Use fold assignments from config if available
        if (hasattr(self.arg, 'kfold') and isinstance(self.arg.kfold, dict) and 
            'fold_assignments' in self.arg.kfold):
            fold_assignments = self.arg.kfold.get('fold_assignments', [])
            
            if len(fold_assignments) == num_folds:
                self.print_log(f"Using {num_folds} fold assignments from config file")
                
                folds = []
                for i, val_subjects in enumerate(fold_assignments, 1):
                    val_subjects = [s for s in val_subjects if s in subjects]
                    train_subjects = [s for s in subjects if s not in val_subjects]
                    folds.append((train_subjects, val_subjects))
                    
                    self.print_log(f"\nFold {i} assignments:")
                    self.print_log(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
                    self.print_log(f"Training subjects ({len(train_subjects)}): {train_subjects}")
                
                return folds
        
        # Default fold assignments if not specified in config
        fold_assignments = [
            ([43, 35, 36], "Fold 1: 38.3% falls"),
            ([44, 34, 32], "Fold 2: 39.7% falls"),
            ([45, 37, 38], "Fold 3: 44.8% falls"),
            ([46, 29, 31], "Fold 4: 41.4% falls"),
            ([30, 39], "Fold 5: 43.3% falls")
        ]
        
        folds = []
        for val_subjects, fold_desc in fold_assignments:
            valid_val_subjects = [s for s in val_subjects if s in subjects]
            
            if not valid_val_subjects:
                continue
                
            train_subjects = [s for s in subjects if s not in valid_val_subjects]
            folds.append((train_subjects, valid_val_subjects))
            
            fold_num = len(folds)
            self.print_log(f"\nCreated {fold_desc}")
            self.print_log(f"Validation subjects ({len(valid_val_subjects)}): {valid_val_subjects}")
            self.print_log(f"Training subjects ({len(train_subjects)}): {train_subjects}")
    
        return folds

    def distribution_viz(self, labels, work_dir, mode):
        """
        Create visualization of label distribution.
        
        Args:
            labels: Array of class labels
            work_dir: Directory to save visualization
            mode: Dataset split (train, val, test)
        """
        try:
            values, count = np.unique(labels, return_counts=True)
            plt.figure(figsize=(10, 6))
            plt.bar(values, count)
            plt.xlabel('Classes')
            plt.ylabel('Count')
            plt.title(f'{mode.capitalize()} Label Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{work_dir}/{mode}_label_distribution.png')
            plt.close()
            
            total = np.sum(count)
            percentages = (count / total) * 100
            self.print_log(f"Class distribution in {mode} set:")
            for i, (val, cnt, pct) in enumerate(zip(values, count, percentages)):
                self.print_log(f"  Class {val}: {cnt} samples ({pct:.1f}%)")
        except Exception as e:
            self.print_log(f"Error creating distribution visualization: {str(e)}")

    def _prepare_input_data(self, inputs):
        """
        Prepare input data by moving tensors to the correct device.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of device-placed tensors
        """
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        
        data_dict = {}
        
        # Handle required modalities
        if 'accelerometer' in inputs:
            data_dict['accelerometer'] = inputs['accelerometer'].to(device).float()
        
        if 'gyroscope' in inputs:
            data_dict['gyroscope'] = inputs['gyroscope'].to(device).float()
        
        # Handle optional modalities
        for modality in ['quaternion', 'linear_acceleration', 'fusion_features', 'skeleton']:
            if modality in inputs:
                data_dict[modality] = inputs[modality].to(device).float()
        
        return data_dict

    def load_data(self, train_subjects=None, val_subjects=None):
        """
        Load and prepare datasets for training, validation, and testing.
        
        Args:
            train_subjects: List of subject IDs for training
            val_subjects: List of subject IDs for validation
            
        Returns:
            Boolean indicating success
        """
        try:
            Feeder = import_class(self.arg.feeder)
            self.print_log(f"Using data feeder: {self.arg.feeder}")

            if self.arg.phase == 'train':
                self.print_log("Preparing SmartFallMM dataset...")
                builder = prepare_smartfallmm(self.arg)
                self.print_log("Dataset preparation complete")

                train_subjects = train_subjects or self.train_subjects or self.arg.subjects
                val_subjects = val_subjects or self.val_subject or self.arg.subjects
                
                self.print_log(f"Splitting data for subjects: train={train_subjects}, val={val_subjects}")
                
                try:
                    # Split data by subjects
                    self.norm_train = split_by_subjects(builder, train_subjects, self.fuse)
                    self.norm_val = split_by_subjects(builder, val_subjects, self.fuse)

                    if not self.norm_train or not self.norm_val:
                        self.print_log("ERROR: Split produced empty datasets")
                        return False
                    
                    # Check that required modalities are present
                    required_keys = ['accelerometer', 'labels']
                    
                    for dataset_name, dataset in [("training", self.norm_train), ("validation", self.norm_val)]:
                        self.print_log(f"Checking {dataset_name} dataset...")
                        
                        missing_required = [key for key in required_keys if key not in dataset or len(dataset[key]) == 0]
                        if missing_required:
                            self.print_log(f"ERROR: Missing required keys in {dataset_name} data: {missing_required}")
                            return False
                        
                        present_keys = [key for key in dataset.keys() if key != 'labels']
                        self.print_log(f"{dataset_name.capitalize()} modalities: {present_keys}")
                        
                        # Log shape information
                        for key, value in dataset.items():
                            self.print_log(f"  {key}: {value.shape if hasattr(value, 'shape') else len(value)}")
                except Exception as e:
                    self.print_log(f"ERROR: Failed to split data: {str(e)}")
                    self.print_log(traceback.format_exc())
                    return False

                try:
                    # Create training data loader
                    train_dataset = Feeder(dataset=self.norm_train, batch_size=self.arg.batch_size)
                    
                    drop_last = True
                    train_feeder_args = getattr(self.arg, 'train_feeder_args', {}) or {}
                    if isinstance(train_feeder_args, dict) and 'drop_last' in train_feeder_args:
                        drop_last = train_feeder_args.get('drop_last')
                    
                    self.data_loader['train'] = torch.utils.data.DataLoader(
                        dataset=train_dataset,
                        batch_size=self.arg.batch_size,
                        shuffle=True,
                        num_workers=self.arg.num_worker,
                        pin_memory=True,
                        drop_last=drop_last,
                        persistent_workers=self.arg.num_worker > 0,
                        collate_fn=getattr(Feeder, 'custom_collate_fn', None)
                    )
                    
                    # Create validation data loader
                    val_dataset = Feeder(dataset=self.norm_val, batch_size=self.arg.val_batch_size)
                    
                    val_feeder_args = getattr(self.arg, 'val_feeder_args', {}) or {}
                    val_drop_last = False
                    if isinstance(val_feeder_args, dict) and 'drop_last' in val_feeder_args:
                        val_drop_last = val_feeder_args.get('drop_last')
                    
                    self.data_loader['val'] = torch.utils.data.DataLoader(
                        dataset=val_dataset,
                        batch_size=self.arg.val_batch_size,
                        shuffle=False,
                        num_workers=self.arg.num_worker,
                        pin_memory=True,
                        drop_last=val_drop_last,
                        collate_fn=getattr(Feeder, 'custom_collate_fn', None)
                    )
                except Exception as e:
                    self.print_log(f"ERROR: Failed to create data loaders: {str(e)}")
                    self.print_log(traceback.format_exc())
                    return False

                # Visualize label distributions
                self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
                self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')

                # Prepare test data (using val subjects for testing)
                self.print_log(f"Preparing test data for subjects {val_subjects}")
                try:
                    self.norm_test = split_by_subjects(builder, val_subjects, self.fuse)
                    
                    test_dataset = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size)
                    
                    test_feeder_args = getattr(self.arg, 'test_feeder_args', {}) or {}
                    test_drop_last = False
                    if isinstance(test_feeder_args, dict) and 'drop_last' in test_feeder_args:
                        test_drop_last = test_feeder_args.get('drop_last')
                    
                    self.data_loader['test'] = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=self.arg.test_batch_size,
                        shuffle=False,
                        num_workers=self.arg.num_worker,
                        pin_memory=True,
                        drop_last=test_drop_last,
                        collate_fn=getattr(Feeder, 'custom_collate_fn', None)
                    )
                except Exception as e:
                    self.print_log(f"ERROR: Failed to prepare test data: {str(e)}")
                    self.print_log(traceback.format_exc())
                    return False

                return True
            
            else:
                # Testing phase - only load test data
                self.print_log(f"Preparing test data for subjects {self.arg.subjects}")
                try:
                    builder = prepare_smartfallmm(self.arg)
                    self.norm_test = split_by_subjects(builder, self.arg.subjects, self.fuse)
                    
                    test_dataset = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size)
                    
                    test_feeder_args = getattr(self.arg, 'test_feeder_args', {}) or {}
                    test_drop_last = False
                    if isinstance(test_feeder_args, dict) and 'drop_last' in test_feeder_args:
                        test_drop_last = test_feeder_args.get('drop_last')
                    
                    self.data_loader['test'] = torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=self.arg.test_batch_size,
                        shuffle=False,
                        num_workers=self.arg.num_worker,
                        pin_memory=True,
                        drop_last=test_drop_last,
                        collate_fn=getattr(Feeder, 'custom_collate_fn', None)
                    )
                    
                    self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, 'test')

                    return True
                except Exception as e:
                    self.print_log(f"ERROR: Failed to load test data: {str(e)}")
                    self.print_log(traceback.format_exc())
                    return False
        except Exception as e:
            self.print_log(f"ERROR: Failed to load data: {str(e)}")
            self.print_log(traceback.format_exc())
            return False

    def record_time(self):
        """Record current time for timing operations"""
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        """Calculate elapsed time since last record_time call"""
        split_time_val = time.time() - self.cur_time
        self.record_time()
        return split_time_val

    def print_log(self, string, print_time=True):
        """
        Print and log a message.
        
        Args:
            string: Message to print/log
            print_time: Whether to include timestamp
        """
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            string = f"[ {localtime} ] {string}"
            
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)

    def loss_viz(self, train_loss, val_loss, fold=None):
        """
        Create visualization of training and validation loss curves.
        
        Args:
            train_loss: List of training loss values
            val_loss: List of validation loss values
            fold: Optional fold number for cross-validation
        """
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
        
        # Annotate minimum loss points
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
        
        # Save to appropriate location
        save_path = f'{self.arg.work_dir}/train_vs_val_loss.png'
        if fold is not None:
            fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            save_path = f'{fold_dir}/train_vs_val_loss.png'
            
        plt.savefig(save_path)
        plt.close()

    def metrics_viz(self, train_metrics, val_metrics, fold=None):
        """
        Create visualization of performance metrics during training.
        
        Args:
            train_metrics: List of training metrics dictionaries
            val_metrics: List of validation metrics dictionaries
            fold: Optional fold number for cross-validation
        """
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
            
            # Annotate maximum metric points
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
        
        # Save to appropriate location
        save_path = f'{self.arg.work_dir}/metrics.png'
        if fold is not None:
            fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            save_path = f'{fold_dir}/metrics.png'
            
        plt.savefig(save_path)
        plt.close()

    def cm_viz(self, y_pred, y_true, fold=None):
        """
        Create confusion matrix visualization.
        
        Args:
            y_pred: Predicted labels
            y_true: True labels
            fold: Optional fold number for cross-validation
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()

            class_labels = ["Non-Fall", "Fall"]
            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels)
            plt.yticks(tick_marks, class_labels)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")

            # Add text annotations to confusion matrix cells
            thresh = cm.max() / 2
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            
            # Save to appropriate location
            save_path = f'{self.arg.work_dir}/confusion_matrix.png'
            if fold is not None:
                fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
                os.makedirs(fold_dir, exist_ok=True)
                save_path = f'{fold_dir}/confusion_matrix.png'
                
            plt.savefig(save_path)
            plt.close()
            
            # Calculate additional metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            balanced_accuracy = (sensitivity + specificity) / 2
            
            # Log confusion matrix statistics
            self.print_log(f"Confusion Matrix Statistics:")
            self.print_log(f"  True Negatives: {tn}")
            self.print_log(f"  False Positives: {fp}")
            self.print_log(f"  False Negatives: {fn}")
            self.print_log(f"  True Positives: {tp}")
            self.print_log(f"  Specificity: {specificity:.4f}")
            self.print_log(f"  Sensitivity: {sensitivity:.4f}")
            self.print_log(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
        except Exception as e:
            self.print_log(f"ERROR: Failed to create confusion matrix: {str(e)}")

    def compute_metrics(self, outputs, targets):
        """
        Compute evaluation metrics from model outputs and targets.
        
        Args:
            outputs: Model output logits
            targets: Ground truth labels
            
        Returns:
            Dictionary of metrics (accuracy, precision, recall, f1, etc.)
        """
        try:
            # Convert tensors to numpy if needed
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
            
            # Get class predictions from logits
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                predictions = np.argmax(outputs, axis=1)
            else:
                predictions = (1 / (1 + np.exp(-outputs.reshape(-1))) > 0.5).astype(float)
            
            # Calculate confusion matrix elements
            tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
            
            # Calculate accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            # Calculate precision, recall, F1 with handling for division by zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = recall  # Same as recall
            balanced_accuracy = (specificity + sensitivity) / 2
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Return all metrics as a dictionary
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,  # This is the correct F1 score between 0 and 1
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
        except Exception as e:
            self.print_log(f"Error computing metrics: {str(e)}")
            self.print_log(traceback.format_exc())
            
            # Return default values on error
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0,
                'false_alarm_rate': 0.0,
                'miss_rate': 0.0,
                'specificity': 0.0,
                'sensitivity': 0.0,
                'balanced_accuracy': 0.0
            }

    def train(self, epoch, fold=None):
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            fold: Optional fold number for cross-validation
            
        Returns:
            Validation loss for the epoch
        """
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        device_type = 'cuda' if use_cuda else 'cpu'

        # Set model to training mode
        self.model.train()
        self.record_time()

        # Get data loader
        loader = self.data_loader['train']
        
        # Initialize tracking variables
        timer = dict(dataloader=0.001, model=0.001, stats=0.001)
        train_metrics = defaultdict(float)
        per_class_metrics = defaultdict(lambda: defaultdict(int))
        total_samples = 0
        batch_count = 0

        # Process each batch with progress bar
        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch} (Train)")
        for batch_idx, (inputs, targets, idx) in enumerate(process):
            batch_count += 1
            batch_size = targets.size(0)
            total_samples += batch_size
            
            # Prepare data
            data_dict = self._prepare_input_data(inputs)
            targets = targets.to(device).long()

            timer['dataloader'] += self.split_time()

            # Forward pass and backward pass
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                # Use mixed precision training if available
                with torch.amp.autocast(device_type=device_type):
                    outputs = self.model(data_dict)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                # Apply gradient clipping if configured
                if hasattr(self.arg, 'grad_clip') and self.arg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(data_dict)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Apply gradient clipping if configured
                if hasattr(self.arg, 'grad_clip') and self.arg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip)
                
                self.optimizer.step()

            timer['model'] += self.split_time()

            # Calculate metrics for this batch
            batch_metrics = self.compute_metrics(outputs, targets)
            for k, v in batch_metrics.items():
                train_metrics[k] += v * batch_size
            train_metrics['loss'] += loss.item() * batch_size

            # Track per-class accuracy
            predictions = torch.argmax(outputs, dim=1) if outputs.shape[1] > 1 else (torch.sigmoid(outputs) > 0.5).long()
            for pred, target in zip(predictions.cpu(), targets.cpu()):
                pred, target = pred.item(), target.item()
                per_class_metrics[target]['total'] += 1
                if pred == target:
                    per_class_metrics[target]['correct'] += 1

            # Update progress bar with correctly formatted metrics
            process.set_postfix({
                'loss': f"{train_metrics['loss']/total_samples:.4f}",
                'acc': f"{100.0*train_metrics['accuracy']/total_samples:.2f}%",
                'f1': f"{train_metrics['f1']/batch_count:.4f}"  # Correct format for F1
            })
            
            timer['stats'] += self.split_time()
            
        # Calculate final metrics
        for k in train_metrics:
            train_metrics[k] /= total_samples

        # Calculate time proportions
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        # Log epoch summary with correctly formatted metrics
        self.print_log(
            f'Epoch {epoch+1}/{self.arg.num_epoch} - '
            f"Training - Loss: {train_metrics['loss']:.4f}, "
            f"Accuracy: {train_metrics['accuracy']:.4f}, "
            f"Precision: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}"  # Correct format for F1
        )
        
        # Log per-class metrics
        for class_idx, metrics in per_class_metrics.items():
            class_acc = metrics["correct"] / max(1, metrics["total"])
            self.print_log(f"  Class {class_idx}: {metrics['correct']}/{metrics['total']} = {class_acc:.4f}")

        # Store metrics for visualization
        self.train_metrics = train_metrics
        self.train_metrics_history.append(train_metrics)
        self.train_loss_summary.append(train_metrics['loss'])

        # Run evaluation on validation set
        val_metrics = self.eval(epoch, loader_name='val', fold=fold)
        self.val_metrics_history.append(val_metrics)
        self.val_loss_summary.append(val_metrics['loss'])

        return val_metrics['loss']

    def eval(self, epoch, loader_name='val', fold=None, result_file=None):
        """
        Evaluate the model on validation or test data.
        
        Args:
            epoch: Current epoch number
            loader_name: Which data loader to use ('val' or 'test')
            fold: Optional fold number for cross-validation
            result_file: Optional file to save prediction results
            
        Returns:
            Dictionary of evaluation metrics
        """
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        device_type = 'cuda' if use_cuda else 'cpu'

        # Open result file if specified
        if result_file is not None:
            f_r = open(result_file, 'w', encoding='utf-8')

        # Set model to evaluation mode
        self.model.eval()
        self.print_log(f'Evaluating on {loader_name} set (Epoch {epoch+1})')

        # Initialize tracking variables
        val_metrics = defaultdict(float)
        total_samples = 0
        batch_count = 0
        all_predictions = []
        all_targets = []

        # Process each batch with progress bar
        process = tqdm(self.data_loader[loader_name], desc=f"Epoch {epoch+1} ({loader_name.capitalize()})")
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                batch_count += 1
                batch_size = targets.size(0)
                total_samples += batch_size
                
                # Prepare data
                data_dict = self._prepare_input_data(inputs)
                targets = targets.to(device).long()

                # Forward pass
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast(device_type=device_type):
                        outputs = self.model(data_dict)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(data_dict)
                    loss = self.criterion(outputs, targets)

                # Calculate metrics for this batch
                batch_metrics = self.compute_metrics(outputs, targets)
                for k, v in batch_metrics.items():
                    val_metrics[k] += v * batch_size
                
                val_metrics['loss'] += loss.item() * batch_size
                
                # Collect predictions for confusion matrix
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar with correctly formatted metrics
                process.set_postfix({
                    'loss': f"{val_metrics['loss']/total_samples:.4f}",
                    'acc': f"{100.0*val_metrics['accuracy']/total_samples:.2f}%",
                    'f1': f"{val_metrics['f1']/batch_count:.4f}"  # Correct format for F1
                })
                
                # Write predictions to result file if specified
                if result_file is not None and f_r is not None:
                    for i, (pred, target) in enumerate(zip(predictions.cpu(), targets.cpu())):
                        f_r.write(f"{int(pred.item())} ==> {int(target.item())}\n")

        # Close result file if open
        if result_file is not None and f_r is not None:
            f_r.close()

        # Calculate final metrics
        for k in val_metrics:
            val_metrics[k] /= total_samples
            
        # Create confusion matrix visualization
        if all_predictions and all_targets:
            self.cm_viz(all_predictions, all_targets, fold)

        # Log evaluation summary with correctly formatted metrics
        self.print_log(
            f'{loader_name.capitalize()} metrics: Loss={val_metrics["loss"]:.4f}, '
            f'Accuracy={val_metrics["accuracy"]:.4f}, '
            f'F1={val_metrics["f1"]:.4f}, '  # Correct format for F1
            f'Precision={val_metrics["precision"]:.4f}, '
            f'Recall={val_metrics["recall"]:.4f}, '
            f'BAcc={val_metrics["balanced_accuracy"]:.4f}'
        )

        # Save metrics to file
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

        # Update best model if validation performance improved
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
                
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
        else:
            # Update test metrics
            self.test_accuracy = val_metrics['accuracy']
            self.test_f1 = val_metrics['f1']

        self.val_metrics = val_metrics
        
        return val_metrics

    def save_best_model(self, epoch, metrics, fold=None):
        """
        Save the best model and its metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of performance metrics
            fold: Optional fold number for cross-validation
        """
        if fold is not None:
            fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            save_path = os.path.join(fold_dir, f"{self.arg.model_saved_name}.pt")
        else:
            save_path = self.model_path
        
        # Prepare model state dictionary
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
        
        # Save model
        try:
            torch.save(state_dict, save_path)
        except Exception as e:
            self.print_log(f"Error saving model: {str(e)}")
            try:
                # Fallback to save just the model state
                model_state = self.model.state_dict() if not isinstance(self.model, nn.DataParallel) else self.model.module.state_dict()
                torch.save(model_state, save_path)
            except Exception as e2:
                self.print_log(f"Failed to save model: {str(e2)}")

    def train_fold(self, fold_idx, train_subjects, val_subjects):
        """
        Train model for one cross-validation fold.
        
        Args:
            fold_idx: Current fold index
            train_subjects: List of subject IDs for training
            val_subjects: List of subject IDs for validation
            
        Returns:
            Dictionary with fold metrics
        """
        fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        
        self.print_log(f"\n{'='*20} Training Fold {fold_idx} {'='*20}")
        self.print_log(f"Training subjects: {train_subjects}")
        self.print_log(f"Validation subjects: {val_subjects}")
        
        # Initialize new model for this fold
        self.model = self.load_model(self.arg.model, self.arg.model_args)
        
        if len(self.available_gpus) > 1 and self.arg.multi_gpu:
            self.model = nn.DataParallel(
                self.model, 
                device_ids=self.available_gpus
            )
            
        # Initialize optimizer and scheduler
        self.load_optimizer()
        
        # Reset tracking variables
        self.best_f1 = 0
        self.best_accuracy = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        # Load data for this fold
        loaded = self.load_data(train_subjects, val_subjects)
        if not loaded:
            self.print_log(f"Error loading data for fold {fold_idx}, skipping")
            return {'fold': fold_idx, 'f1': 0, 'accuracy': 0, 'epochs_trained': 0, 'error': True}
        
        # Train for specified number of epochs
        for epoch in range(self.arg.num_epoch):
            try:
                val_loss = self.train(epoch, fold=fold_idx)
                
                # Check early stopping
                if self.patience_counter >= self.arg.patience:
                    self.print_log(f"Early stopping triggered at epoch {epoch}")
                    break
            except Exception as e:
                self.print_log(f"Error during training for fold {fold_idx}, epoch {epoch}: {str(e)}")
                self.print_log(traceback.format_exc())
                break
                
        # Create visualizations
        self.loss_viz(self.train_loss_summary, self.val_loss_summary, fold=fold_idx)
        self.metrics_viz(self.train_metrics_history, self.val_metrics_history, fold=fold_idx)
        
        # Evaluate on test set
        test_metrics = self.eval(0, loader_name='test', fold=fold_idx, 
                               result_file=os.path.join(fold_dir, "test_predictions.txt"))
        
        # Collect metrics
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

    def kfold_cross_validation(self):
        """
        Perform k-fold cross-validation.
        
        Returns:
            Dictionary with cross-validation summary
        """
        self.print_log(f"\n{'='*20} K-Fold Cross-Validation {'='*20}")
        
        # Create folds
        folds = self.create_folds(self.arg.subjects, self.num_folds)
        all_fold_metrics = []
        
        # Train and evaluate each fold
        for fold_idx, (train_subjects, val_subjects) in enumerate(folds, 1):
            fold_metrics = self.train_fold(fold_idx, train_subjects, val_subjects)
            all_fold_metrics.append(fold_metrics)
            
        # Calculate average metrics across folds
        avg_metrics = {}
        metric_keys = ['f1', 'accuracy', 'precision', 'recall', 'false_alarm_rate', 
                       'miss_rate', 'balanced_accuracy', 'test_f1', 'epochs_trained']
        
        for key in metric_keys:
            valid_values = [m.get(key, 0) for m in all_fold_metrics if not m.get('error', False)]
            if valid_values:
                avg_metrics[key] = sum(valid_values) / len(valid_values)
                std_metrics = np.std(valid_values)
                avg_metrics[f'{key}_std'] = std_metrics
            else:
                avg_metrics[key] = 0
                avg_metrics[f'{key}_std'] = 0
        
        # Log summary
        self.print_log(f"\n{'='*20} Cross-Validation Results {'='*20}")
        for key in ['accuracy', 'f1', 'balanced_accuracy', 'precision', 'recall']:
            if key in avg_metrics:
                self.print_log(f"Average {key.capitalize()}: {avg_metrics[key]:.4f} ± {avg_metrics[f'{key}_std']:.4f}")
        
        # Save summary to file
        summary = {
            'fold_metrics': all_fold_metrics,
            'average_metrics': avg_metrics
        }
        
        with open(os.path.join(self.arg.work_dir, 'cv_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Create summary visualization
        self.create_cv_summary_plot(all_fold_metrics)
        
        return summary

    def create_cv_summary_plot(self, fold_metrics):
        """
        Create visualization of cross-validation results.
        
        Args:
            fold_metrics: List of metrics dictionaries for each fold
        """
        valid_metrics = [m for m in fold_metrics if not m.get('error', False)]
        
        if not valid_metrics:
            self.print_log("No valid fold metrics available for visualization")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Extract metrics for visualization
        folds = [m['fold'] for m in valid_metrics]
        f1_scores = [m['f1'] for m in valid_metrics]
        accuracies = [m['accuracy'] for m in valid_metrics]
        balanced_accs = [m.get('balanced_accuracy', 0) for m in valid_metrics]
        test_f1s = [m.get('test_f1', 0) for m in valid_metrics]
        
        # Set up bar positions
        x = np.arange(len(folds))
        width = 0.2
        
        # Create grouped bar chart
        ax = plt.subplot(111)
        bars1 = ax.bar(x - width*1.5, f1_scores, width, label='Val F1 Score')
        bars2 = ax.bar(x - width/2, test_f1s, width, label='Test F1 Score')
        bars3 = ax.bar(x + width/2, accuracies, width, label='Accuracy')
        bars4 = ax.bar(x + width*1.5, balanced_accs, width, label='Balanced Acc')
        
        # Add labels and formatting
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation Results by Fold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {f}' for f in folds])
        ax.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
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
        
        # Add average lines
        avg_f1 = np.mean(f1_scores)
        avg_test_f1 = np.mean(test_f1s)
        avg_acc = np.mean(accuracies)
        avg_balanced = np.mean(balanced_accs)
        
        ax.axhline(y=avg_f1, color='C0', linestyle='--', alpha=0.7)
        ax.axhline(y=avg_test_f1, color='C1', linestyle='--', alpha=0.7)
        ax.axhline(y=avg_acc, color='C2', linestyle='--', alpha=0.7)
        ax.axhline(y=avg_balanced, color='C3', linestyle='--', alpha=0.7)
        
        # Adjust legend position
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.arg.work_dir, 'cv_summary.png'), bbox_inches='tight')
        plt.close()

    def train_final_model(self, best_fold_idx=None):
        """
        Train final model using all data after cross-validation.
        
        Args:
            best_fold_idx: Optional best fold index to use as reference
        """
        self.print_log(f"\n{'='*20} Training Final Model {'='*20}")
        
        # Determine number of epochs based on best fold
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
                num_epochs = self.arg.num_epoch
        else:
            num_epochs = self.arg.num_epoch
        
        self.print_log(f"Training final model for {num_epochs} epochs")
        
        # Initialize model
        self.model = self.load_model(self.arg.model, self.arg.model_args)
        
        if len(self.available_gpus) > 1 and self.arg.multi_gpu:
            self.model = nn.DataParallel(
                self.model, 
                device_ids=self.available_gpus
            )
            
        # Initialize optimizer
        self.load_optimizer()
        
        # Reset tracking variables
        self.best_f1 = 0
        self.best_accuracy = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        # Load all data
        all_subjects = self.arg.subjects
        self.load_data(all_subjects, all_subjects)
        
        # Train for specified number of epochs
        for epoch in range(num_epochs):
            val_loss = self.train(epoch)
            if self.patience_counter >= self.arg.patience:
                self.print_log(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Create visualizations
        self.loss_viz(self.train_loss_summary, self.val_loss_summary)
        self.metrics_viz(self.train_metrics_history, self.val_metrics_history)
            
        # Save final model
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
        
        # Evaluate on test set
        test_metrics = self.eval(num_epochs - 1, loader_name='test')
        
        # Log final results
        self.print_log(f"\n{'='*20} Final Model Results {'='*20}")
        self.print_log(f"Test F1 Score: {test_metrics['f1']:.4f}")
        self.print_log(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.print_log(f"Test Precision: {test_metrics['precision']:.4f}")
        self.print_log(f"Test Recall: {test_metrics['recall']:.4f}")
        self.print_log(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")

    def start(self):
        """
        Start the training or testing process.
        """
        if self.arg.phase == 'train':
            try:
                # Configure parallel processing
                configure_parallel_processing(self.arg)
                self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
                
                if self.use_kfold:
                    # Perform cross-validation
                    cv_results = self.kfold_cross_validation()
                    
                    # Train final model using best fold
                    valid_folds = [m for m in cv_results['fold_metrics'] if not m.get('error', False)]
                    if valid_folds:
                        best_fold = max(valid_folds, key=lambda x: x['f1'])
                        best_fold_idx = best_fold['fold']
                        
                        self.print_log(f"Best performing fold: {best_fold_idx} with F1 score: {best_fold['f1']:.4f}")
                        
                        self.train_final_model(best_fold_idx)
                    else:
                        self.print_log("No valid folds completed, cannot train final model")
                else:
                    # Regular training without cross-validation
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

            except Exception as e:
                self.print_log(f"Error during training: {str(e)}")
                self.print_log(traceback.format_exc())
                cleanup_resources()
        
        else:
            try:
                # Testing phase
                if not self.load_data():
                    self.print_log("Error loading test data")
                    return
                    
                self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)
                self.print_log(f'Test accuracy: {self.test_accuracy:.4f}')
                self.print_log(f'Test F1 score: {self.test_f1:.4f}')
            
            except Exception as e:
                self.print_log(f"Error during testing: {str(e)}")
                self.print_log(traceback.format_exc())
            finally:
                cleanup_resources()


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    # Load configuration file if provided
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)

        # Check for unrecognized parameters
        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WARNING: Unrecognized configuration parameter: {k}')

        # Set defaults from config file
        parser.set_defaults(**default_arg)
        args = parser.parse_args()

    # Initialize random seed for reproducibility
    init_seed(args.seed)
    
    # Create and start trainer
    trainer = Trainer(args)
    try:
        trainer.start()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        cleanup_resources()
