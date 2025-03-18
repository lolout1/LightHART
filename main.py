'''
Script to train the models for Fall Detection and Activity Recognition

This script handles the training, validation, and testing workflows for
SmartFallMM dataset models, supporting both accelerometer-only and
fusion-based approaches. It implements leave-one-subject-out cross-validation
to ensure robust evaluation of model performance with multithreading and
dual GPU support.
'''
import traceback
from typing import List, Dict, Tuple, Union, Optional
import random
import sys
import os
import time
import shutil
import argparse
import yaml
from copy import deepcopy

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

# Local imports
from utils.dataset import prepare_smartfallmm, split_by_subjects
from utils.imu_fusion import cleanup_resources

# Register cleanup function to ensure proper resource management on exit
import atexit
def cleanup_on_exit():
    """Clean up resources on script exit."""
    cleanup_resources()
atexit.register(cleanup_on_exit)

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
    parser.add_argument('--config', default='./config/smartfallmm/fusion_madgwick.yaml',
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

    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use (Adam, SGD, etc.)')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR',
                        help='Base learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0004,
                        help='Weight decay factor (default: 0.0004)')

    # Model parameters
    parser.add_argument('--model', default=None,
                        help='Model class path to load')
    parser.add_argument('--device', nargs='+', default=[0], type=int,
                        help='CUDA device IDs to use')
    parser.add_argument('--model-args', default=None,
                        help='Dictionary of model arguments')
    parser.add_argument('--weights', type=str,
                        help='Path to pretrained weights file')
    parser.add_argument('--model-saved-name', type=str, default='model',
                        help='Name for saving the trained model')

    # Loss function
    parser.add_argument('--loss', default='loss.BCE',
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
    parser.add_argument('--test_feeder_args', default=None,
                        help='Arguments for test data loader')
    parser.add_argument('--include-val', type=str2bool, default=True,
                        help='Whether to include validation set')

    # Initialization
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed (default: 2)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')

    # Output and logging
    parser.add_argument('--work-dir', type=str, default='work_dir',
                        help='Working directory for outputs')
    parser.add_argument('--print-log', type=str2bool, default=True,
                        help='Whether to print and save logs')
    parser.add_argument('--phase', type=str, default='train',
                        help='Phase: train or test')
    parser.add_argument('--num-worker', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--result-file', type=str,
                        help='File to save results to')
    
    # Multi-GPU and parallel processing
    parser.add_argument('--multi-gpu', type=str2bool, default=True,
                        help='Whether to use multiple GPUs when available')
    parser.add_argument('--parallel-threads', type=int, default=48,
                        help='Number of parallel threads for preprocessing')

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
        List of available GPU devices
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs")
        
        if isinstance(args.device, list) and len(args.device) > 0:
            # Use specified GPUs
            devices = args.device
            print(f"Using specified GPUs: {devices}")
        elif num_gpus >= 2 and args.multi_gpu:
            # Default to using both GPUs if multi-GPU is enabled
            devices = [0, 1]
            print(f"Using both GPUs: {devices}")
        elif num_gpus == 1:
            # Use the single available GPU
            devices = [0]
            print(f"Using single GPU: {devices}")
        else:
            # No GPUs available
            devices = []
            print("No GPUs found, using CPU")
            
        # Set visible devices
        gpu_list = ",".join(map(str, devices))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"Set CUDA_VISIBLE_DEVICES={gpu_list}")
        
        return devices
    else:
        print("PyTorch CUDA not available, using CPU")
        return []


def configure_parallel_processing(args):
    """
    Configure parallel processing based on command-line arguments.
    
    Args:
        args: Command-line arguments with parallel-threads parameter
    """
    if hasattr(args, 'parallel_threads') and args.parallel_threads > 0:
        # Import the necessary function to update thread configuration
        from utils.imu_fusion import update_thread_configuration
        
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
        
        print(f"Using {max_files} parallel files with {threads_per_file} threads per file")
        print(f"Total parallel threads: {max_files * threads_per_file}")


class Trainer:
    # ... [rest of Trainer class implementation stays the same] ...
    '''
    Main trainer class that handles the complete training and evaluation workflow

    This class manages the training process, including data loading, model training,
    validation, testing, and result visualization. Optimized with multithreading
    and multi-GPU support.
    '''
    # ... [implementation as provided in previous message] ...


if __name__ == "__main__":
    # Parse command line arguments - ONLY parse once
    parser = get_args()
    args = parser.parse_args()

    # Load configuration from file if provided
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)

        # Validate config keys
        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WARNING: Unrecognized configuration parameter: {k}')

        # Update args with values from config file
        for k, v in default_arg.items():
            if k in key:
                setattr(args, k, v)

    # Configure parallel processing based on the final arguments
    if hasattr(args, 'parallel_threads') and args.parallel_threads > 0:
        configure_parallel_processing(args)

    # Initialize random seeds
    init_seed(args.seed)

    # Create and start trainer
    trainer = Trainer(args)
    try:
        trainer.start()
    finally:
        # Ensure cleanup even on exceptions
        cleanup_resources()
