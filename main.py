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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from utils.dataset import prepare_smartfallmm, split_by_subjects

# Global thread pool for parallel processing
MAX_THREADS = 4
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)

def get_args():
    '''
    Function to build the argument parser for command line options

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
    parser.add_argument('--weights-only', type=str2bool, default=False,
                        help='Whether to load only weights (not full model)')
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use (Adam, SGD, etc.)')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR',
                        help='Base learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0004,
                        help='Weight decay factor (default: 0.0004)')
        # Cross-validation parameters (new)
    parser.add_argument('--kfold', type=str2bool, default=False,
                        help='Whether to use k-fold cross validation')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
                        
    # Early stopping parameters (new)
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--monitor', type=str, default='val_loss',
                        help='Metric to monitor for early stopping')
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
    parser.add_argument('--parallel-threads', type=int, default=4,
                        help='Number of parallel threads for preprocessing')

    return parser


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
        self.best_f1 = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0
        self.test_accuracy = 0
        self.test_f1 = 0
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = dict()
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}.pt'
        
        # Configure thread pool for parallel processing
        self.max_threads = min(arg.parallel_threads, MAX_THREADS)
        print(f"Using {self.max_threads} threads for parallel processing")

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
            self.save_config(arg.config, arg.work_dir)

        # Set up GPU environment
        self.available_gpus = setup_gpu_environment(arg)
        arg.device = self.available_gpus if self.available_gpus else arg.device
        self.output_device = arg.device[0] if type(arg.device) is list and len(arg.device) > 0 else arg.device
        
        # Load model
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            use_cuda = torch.cuda.is_available()
            self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
            self.model = torch.load(self.arg.weights)

        # Set up multi-GPU if available and requested
        if len(self.available_gpus) > 1 and arg.multi_gpu:
            print(f"Using {len(self.available_gpus)} GPUs with DataParallel")
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
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size: {num_params / (1024 ** 2):.2f} MB')

        # Log sensor configuration
        self.print_log(f'Sensor modalities: {self.inertial_modality}')
        self.print_log(f'Using fusion: {self.fuse}')
        
        # Log GPU configuration
        if self.available_gpus:
            self.print_log(f'Using GPUs: {self.available_gpus}')
        else:
            self.print_log('Using CPU for computation')

    def save_config(self, src_path: str, desc_path: str) -> None:
        '''
        Save the configuration file to the working directory

        Args:
            src_path: Source path of the configuration file
            desc_path: Destination path to save the configuration
        '''
        config_file = src_path.rpartition("/")[-1]
        print(f'Saving config to {desc_path}/{config_file}')
        shutil.copy(src_path, f'{desc_path}/{config_file}')

    def count_parameters(self, model):
        '''
        Count the number of trainable parameters in the model

        Args:
            model: PyTorch model

        Returns:
            Total size of parameters and buffers in bytes
        '''
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.nelement() * buffer.element_size()
        return total_size

    def has_empty_value(self, *lists):
        '''
        Check if any of the provided lists are empty

        Args:
            *lists: Variable number of lists to check

        Returns:
            True if any list is empty, False otherwise
        '''
        return any(len(lst) == 0 for lst in lists)

    def load_model(self, model, model_args):
        '''
        Load the model class and instantiate it with provided arguments

        Args:
            model: Model class path
            model_args: Dictionary of model arguments

        Returns:
            Instantiated model on the appropriate device
        '''
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'

        Model = import_class(model)
        self.print_log(f"Loading model: {model}")
        self.print_log(f"Model arguments: {model_args}")
        model = Model(**model_args).to(device)
        return model
    def early_stopping(self, current_val_loss, epoch):
        """
        Implement early stopping to prevent overfitting
        
        Args:
            current_val_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            Boolean indicating whether to stop training
        """
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
            self.patience_counter = 0
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.arg.patience:
                self.print_log(f"Early stopping triggered after {epoch+1} epochs (no improvement for {self.arg.patience} epochs)")
                return True
            return False
    def load_loss(self):
        '''
        Load the loss function for training
        '''
        self.criterion = torch.nn.CrossEntropyLoss()
        self.print_log("Using CrossEntropyLoss for training")

    def load_weights(self):
        '''
        Load model weights from the saved model path
        '''
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(self.model_path))
        self.print_log(f"Loaded model weights from {self.model_path}")

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
                weight_decay=self.arg.weight_decay
            )
            self.print_log(f"Using SGD optimizer with lr={self.arg.base_lr}, weight_decay={self.arg.weight_decay}")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

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
        plt.savefig(f'{work_dir}/{mode}_label_distribution.png')
        plt.close()
        self.print_log(f"Created {mode} label distribution visualization with classes {values}")

    def load_data(self):
        '''
        Load and prepare datasets for training, validation, and testing

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

            # Split data for training and validation
            self.print_log(f"Splitting data for subjects: train={self.train_subjects}, val={self.val_subject}")
            self.norm_train = split_by_subjects(builder, self.train_subjects, self.fuse)
            self.norm_val = split_by_subjects(builder, self.val_subject, self.fuse)

            # Check if validation data is valid
            if self.has_empty_value(list(self.norm_val.values())):
                self.print_log("ERROR: Validation data has empty values")
                return False

            # Create training data loader
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args, dataset=self.norm_train),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            )
            self.print_log(f"Training data loaded with {len(self.data_loader['train'])} batches")

            # Visualize training data distribution
            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')

            # Create validation data loader
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args, dataset=self.norm_val),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            )
            self.print_log(f"Validation data loaded with {len(self.data_loader['val'])} batches")

            # Visualize validation data distribution
            self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')

            # Prepare test data
            self.print_log(f"Preparing test data for subject {self.test_subject}")
            self.norm_test = split_by_subjects(builder, self.test_subject, self.fuse)

            # Check if test data is valid
            if self.has_empty_value(list(self.norm_test.values())):
                self.print_log("ERROR: Test data has empty values")
                return False

            # Create test data loader
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            )
            self.print_log(f"Test data loaded with {len(self.data_loader['test'])} batches")

            # Log dataset sizes and modalities
            self.print_log(f"Training data modalities: {list(self.norm_train.keys())}")
            self.print_log(f"Validation data modalities: {list(self.norm_val.keys())}")
            self.print_log(f"Test data modalities: {list(self.norm_test.keys())}")

            return True
        else:
            # Testing phase
            self.print_log(f"Preparing test data for subject {self.test_subject}")
            builder = prepare_smartfallmm(self.arg)
            self.norm_test = split_by_subjects(builder, self.test_subject, self.fuse)

            if self.has_empty_value(list(self.norm_test.values())):
                self.print_log("ERROR: Test data has empty values")
                return False

            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            )
            self.print_log(f"Test data loaded with {len(self.data_loader['test'])} batches")

            return True

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
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string: str, print_time=True) -> None:
        '''
        Print a message to the console and save to the log file

        Args:
            string: Message to log
            print_time: Whether to include a timestamp (not implemented)
        '''
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)

    def loss_viz(self, train_loss: List[float], val_loss: List[float]):
        '''
        Visualize training and validation loss curves

        Args:
            train_loss: List of training loss values
            val_loss: List of validation loss values
        '''
        epochs = range(len(train_loss))
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_loss, 'b-', label="Training Loss")
        plt.plot(epochs, val_loss, 'r-', label="Validation Loss")
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'{self.arg.work_dir}/train_vs_val_loss.png')
        plt.close()
        self.print_log("Created loss curve visualization")

    def cm_viz(self, y_pred: List[int], y_true: List[int]):
        '''
        Create and save a confusion matrix visualization

        Args:
            y_pred: Predicted class labels
            y_true: True class labels
        '''
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()

        # Set axis labels and title
        class_labels = np.unique(y_true)
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
        plt.savefig(f'{self.arg.work_dir}/confusion_matrix.png')
        plt.close()
        self.print_log("Created confusion matrix visualization")

    def create_df(self, columns=['test_subject', 'train_subjects', 'accuracy', 'f1_score']) -> pd.DataFrame:
        '''
        Create an empty DataFrame for storing results

        Args:
            columns: Column names for the DataFrame

        Returns:
            Empty DataFrame with the specified columns
        '''
        return pd.DataFrame(columns=columns)

    def train(self, epoch):
        '''
        Train the model for one epoch

        Args:
            epoch: Current epoch number

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
        acc_value = []
        accuracy = 0
        cnt = 0
        train_loss = 0

        # Training loop with progress bar
        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch} (Train)")
        for batch_idx, (inputs, targets, idx) in enumerate(process):
            # Move data to device
            with torch.no_grad():
                # Always get accelerometer data
                acc_data = inputs['accelerometer'].to(device)
                targets = targets.to(device)

                # Get gyroscope data if available
                gyro_data = None
                if 'gyroscope' in inputs:
                    gyro_data = inputs['gyroscope'].to(device)

                # Get fusion features if available
                fusion_features = None
                if 'fusion_features' in inputs:
                    fusion_features = inputs['fusion_features'].to(device)

                # Get quaternion data if available
                quaternion = None
                if 'quaternion' in inputs:
                    quaternion = inputs['quaternion'].to(device)

            # Record data loading time
            timer['dataloader'] += self.split_time()

            # Forward pass and loss calculation
            self.optimizer.zero_grad()

            # Forward pass depends on available inputs
            if hasattr(self.model, 'forward_fusion') and fusion_features is not None:
                # Model has specialized fusion handling
                logits = self.model.forward_fusion(acc_data.float(), fusion_features.float())
            elif hasattr(self.model, 'forward_quaternion') and quaternion is not None:
                # Model has specialized quaternion handling
                logits = self.model.forward_quaternion(acc_data.float(), quaternion.float())
            elif gyro_data is not None and hasattr(self.model, 'forward_multi_sensor'):
                # Model can handle multiple sensor inputs
                logits = self.model.forward_multi_sensor(acc_data.float(), gyro_data.float())
            else:
                # Default to just accelerometer data
                logits = self.model(acc_data.float())

            # Calculate loss, do backward pass, and update weights
            loss = self.criterion(logits, targets)
            loss.mean().backward()
            self.optimizer.step()

            # Record model training time
            timer['model'] += self.split_time()

            # Track loss and accuracy
            with torch.no_grad():
                train_loss += loss.mean().item()
                accuracy += (torch.argmax(F.log_softmax(logits, dim=1), 1) == targets).sum().item()

            cnt += len(targets)
            timer['stats'] += self.split_time()
            
            # Update progress bar with current loss and accuracy
            process.set_postfix({
                'loss': f"{train_loss/(batch_idx+1):.4f}",
                'acc': f"{100.0*accuracy/cnt:.2f}%"
            })

        # Compute final metrics
        train_loss /= cnt
        accuracy *= 100. / cnt

        # Record results
        self.train_loss_summary.append(train_loss)
        acc_value.append(accuracy)

        # Calculate timing proportions
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        # Log results
        self.print_log(
            f'Epoch {epoch+1}/{self.arg.num_epoch} - '
            f'Training Loss: {train_loss:.4f}, Training Acc: {accuracy:.2f}%'
        )
        self.print_log(f'Time consumption: [Data]{proportion["dataloader"]}, '
                      f'[Network]{proportion["model"]}, [Stats]{proportion["stats"]}')

        # Evaluate on validation set
        val_loss = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
        self.val_loss_summary.append(val_loss)

        return val_loss

    def eval(self, epoch, loader_name='val', result_file=None):
        '''
        Evaluate the model on validation or test data

        Args:
            epoch: Current epoch number
            loader_name: Which data loader to use ('val' or 'test')
            result_file: Optional file to save detailed results

        Returns:
            Average loss for the evaluation set
        '''
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'

        # Open results file if specified
        if result_file is not None:
            f_r = open(result_file, 'w', encoding='utf-8')

        # Set model to evaluation mode
        self.model.eval()
        self.print_log(f'Evaluating on {loader_name} set (Epoch {epoch+1})')

        # Initialize tracking variables
        loss = 0
        cnt = 0
        accuracy = 0
        label_list = []
        pred_list = []

        # Evaluation loop with progress bar
        process = tqdm(self.data_loader[loader_name], desc=f"Epoch {epoch+1} ({loader_name.capitalize()})")
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                # Move data to device
                acc_data = inputs['accelerometer'].to(device)
                targets = targets.to(device)

                # Get gyroscope data if available
                gyro_data = None
                if 'gyroscope' in inputs:
                    gyro_data = inputs['gyroscope'].to(device)

                # Get fusion features if available
                fusion_features = None
                if 'fusion_features' in inputs:
                    fusion_features = inputs['fusion_features'].to(device)

                # Get quaternion data if available
                quaternion = None
                if 'quaternion' in inputs:
                    quaternion = inputs['quaternion'].to(device)

                # Forward pass depends on available inputs
                if hasattr(self.model, 'forward_fusion') and fusion_features is not None:
                    # Model has specialized fusion handling
                    logits = self.model.forward_fusion(acc_data.float(), fusion_features.float())
                elif hasattr(self.model, 'forward_quaternion') and quaternion is not None:
                    # Model has specialized quaternion handling
                    logits = self.model.forward_quaternion(acc_data.float(), quaternion.float())
                elif gyro_data is not None and hasattr(self.model, 'forward_multi_sensor'):
                    # Model can handle multiple sensor inputs
                    logits = self.model.forward_multi_sensor(acc_data.float(), gyro_data.float())
                else:
                    # Default to just accelerometer data
                    logits = self.model(acc_data.float())

                # Calculate loss and track metrics
                batch_loss = self.criterion(logits, targets)
                loss += batch_loss.sum().item()

                # Get predictions and calculate accuracy
                predictions = torch.argmax(F.log_softmax(logits, dim=1), 1)
                accuracy += (predictions == targets).sum().item()

                # Save predictions and labels for metrics
                label_list.extend(targets.cpu().tolist())
                pred_list.extend(predictions.cpu().tolist())
                cnt += len(targets)
                
                # Update progress bar
                process.set_postfix({
                    'loss': f"{loss/cnt:.4f}",
                    'acc': f"{100.0*accuracy/cnt:.2f}%"
                })

            # Calculate final metrics
            loss /= cnt
            target = np.array(label_list)
            y_pred = np.array(pred_list)

            # Calculate F1 score and other metrics
            f1 = f1_score(target, y_pred, average='macro') * 100
            precision, recall, _, _ = precision_recall_fscore_support(
                target, y_pred, average='macro')
            accuracy *= 100. / cnt

            # Write detailed results to file if provided
            if result_file is not None:
                predict = pred_list
                true = label_list
                for i, x in enumerate(predict):
                    f_r.write(f"{x} ==> {true[i]}\n")
                f_r.close()

        # Log results
        self.print_log(
            f'{loader_name.capitalize()} metrics: Loss={loss:.4f}, '
            f'Accuracy={accuracy:.2f}%, F1={f1:.2f}, '
            f'Precision={precision*100:.2f}%, Recall={recall*100:.2f}%'
        )

        # Save best model if in validation phase
        if loader_name == 'val':
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_accuracy = accuracy
                self.best_f1 = f1

                # Save the model weights - handle DataParallel
                if isinstance(self.model, nn.DataParallel):
                    torch.save(deepcopy(self.model.module.state_dict()), self.model_path)
                else:
                    torch.save(deepcopy(self.model.state_dict()), self.model_path)
                    
                self.print_log('Best model saved: improved validation loss, accuracy, and F1 score')

                # Visualize confusion matrix for best model
                if len(np.unique(target)) > 1:  # Only create CM if multiple classes
                    self.cm_viz(y_pred, target)
        else:
            # For test set, store the results
            self.test_accuracy = accuracy
            self.test_f1 = f1

        return loss

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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from utils.dataset import prepare_smartfallmm, split_by_subjects

# Global thread pool for parallel processing
MAX_THREADS = 4
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)

def get_args():
    '''
    Function to build the argument parser for command line options

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
    parser.add_argument('--weights-only', type=str2bool, default=False,
                        help='Whether to load only weights (not full model)')
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer to use (Adam, SGD, etc.)')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR',
                        help='Base learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0004,
                        help='Weight decay factor (default: 0.0004)')
        # Cross-validation parameters (new)
    parser.add_argument('--kfold', type=str2bool, default=False,
                        help='Whether to use k-fold cross validation')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of folds for cross-validation')
                        
    # Early stopping parameters (new)
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--monitor', type=str, default='val_loss',
                        help='Metric to monitor for early stopping')
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
    parser.add_argument('--parallel-threads', type=int, default=4,
                        help='Number of parallel threads for preprocessing')

    return parser


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
        self.best_f1 = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0
        self.test_accuracy = 0
        self.test_f1 = 0
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = dict()
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}.pt'
        
        # Configure thread pool for parallel processing
        self.max_threads = min(arg.parallel_threads, MAX_THREADS)
        print(f"Using {self.max_threads} threads for parallel processing")

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
            self.save_config(arg.config, arg.work_dir)

        # Set up GPU environment
        self.available_gpus = setup_gpu_environment(arg)
        arg.device = self.available_gpus if self.available_gpus else arg.device
        self.output_device = arg.device[0] if type(arg.device) is list and len(arg.device) > 0 else arg.device
        
        # Load model
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            use_cuda = torch.cuda.is_available()
            self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
            self.model = torch.load(self.arg.weights)

        # Set up multi-GPU if available and requested
        if len(self.available_gpus) > 1 and arg.multi_gpu:
            print(f"Using {len(self.available_gpus)} GPUs with DataParallel")
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
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size: {num_params / (1024 ** 2):.2f} MB')

        # Log sensor configuration
        self.print_log(f'Sensor modalities: {self.inertial_modality}')
        self.print_log(f'Using fusion: {self.fuse}')
        
        # Log GPU configuration
        if self.available_gpus:
            self.print_log(f'Using GPUs: {self.available_gpus}')
        else:
            self.print_log('Using CPU for computation')

    def save_config(self, src_path: str, desc_path: str) -> None:
        '''
        Save the configuration file to the working directory

        Args:
            src_path: Source path of the configuration file
            desc_path: Destination path to save the configuration
        '''
        config_file = src_path.rpartition("/")[-1]
        print(f'Saving config to {desc_path}/{config_file}')
        shutil.copy(src_path, f'{desc_path}/{config_file}')

    def count_parameters(self, model):
        '''
        Count the number of trainable parameters in the model

        Args:
            model: PyTorch model

        Returns:
            Total size of parameters and buffers in bytes
        '''
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.nelement() * buffer.element_size()
        return total_size

    def has_empty_value(self, *lists):
        '''
        Check if any of the provided lists are empty

        Args:
            *lists: Variable number of lists to check

        Returns:
            True if any list is empty, False otherwise
        '''
        return any(len(lst) == 0 for lst in lists)

    def load_model(self, model, model_args):
        '''
        Load the model class and instantiate it with provided arguments

        Args:
            model: Model class path
            model_args: Dictionary of model arguments

        Returns:
            Instantiated model on the appropriate device
        '''
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'

        Model = import_class(model)
        self.print_log(f"Loading model: {model}")
        self.print_log(f"Model arguments: {model_args}")
        model = Model(**model_args).to(device)
        return model
    def early_stopping(self, current_val_loss, epoch):
        """
        Implement early stopping to prevent overfitting
        
        Args:
            current_val_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            Boolean indicating whether to stop training
        """
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
            self.patience_counter = 0
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.arg.patience:
                self.print_log(f"Early stopping triggered after {epoch+1} epochs (no improvement for {self.arg.patience} epochs)")
                return True
            return False
    def load_loss(self):
        '''
        Load the loss function for training
        '''
        self.criterion = torch.nn.CrossEntropyLoss()
        self.print_log("Using CrossEntropyLoss for training")

    def load_weights(self):
        '''
        Load model weights from the saved model path
        '''
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(self.model_path))
        self.print_log(f"Loaded model weights from {self.model_path}")

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
                weight_decay=self.arg.weight_decay
            )
            self.print_log(f"Using SGD optimizer with lr={self.arg.base_lr}, weight_decay={self.arg.weight_decay}")
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

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
        plt.savefig(f'{work_dir}/{mode}_label_distribution.png')
        plt.close()
        self.print_log(f"Created {mode} label distribution visualization with classes {values}")

    def load_data(self):
        '''
        Load and prepare datasets for training, validation, and testing

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

            # Split data for training and validation
            self.print_log(f"Splitting data for subjects: train={self.train_subjects}, val={self.val_subject}")
            self.norm_train = split_by_subjects(builder, self.train_subjects, self.fuse)
            self.norm_val = split_by_subjects(builder, self.val_subject, self.fuse)

            # Check if validation data is valid
            if self.has_empty_value(list(self.norm_val.values())):
                self.print_log("ERROR: Validation data has empty values")
                return False

            # Create training data loader
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args, dataset=self.norm_train),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            )
            self.print_log(f"Training data loaded with {len(self.data_loader['train'])} batches")

            # Visualize training data distribution
            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')

            # Create validation data loader
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args, dataset=self.norm_val),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            )
            self.print_log(f"Validation data loaded with {len(self.data_loader['val'])} batches")

            # Visualize validation data distribution
            self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')

            # Prepare test data
            self.print_log(f"Preparing test data for subject {self.test_subject}")
            self.norm_test = split_by_subjects(builder, self.test_subject, self.fuse)

            # Check if test data is valid
            if self.has_empty_value(list(self.norm_test.values())):
                self.print_log("ERROR: Test data has empty values")
                return False

            # Create test data loader
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            )
            self.print_log(f"Test data loaded with {len(self.data_loader['test'])} batches")

            # Log dataset sizes and modalities
            self.print_log(f"Training data modalities: {list(self.norm_train.keys())}")
            self.print_log(f"Validation data modalities: {list(self.norm_val.keys())}")
            self.print_log(f"Test data modalities: {list(self.norm_test.keys())}")

            return True
        else:
            # Testing phase
            self.print_log(f"Preparing test data for subject {self.test_subject}")
            builder = prepare_smartfallmm(self.arg)
            self.norm_test = split_by_subjects(builder, self.test_subject, self.fuse)

            if self.has_empty_value(list(self.norm_test.values())):
                self.print_log("ERROR: Test data has empty values")
                return False

            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            )
            self.print_log(f"Test data loaded with {len(self.data_loader['test'])} batches")

            return True

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
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string: str, print_time=True) -> None:
        '''
        Print a message to the console and save to the log file

        Args:
            string: Message to log
            print_time: Whether to include a timestamp (not implemented)
        '''
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)

    def loss_viz(self, train_loss: List[float], val_loss: List[float]):
        '''
        Visualize training and validation loss curves

        Args:
            train_loss: List of training loss values
            val_loss: List of validation loss values
        '''
        epochs = range(len(train_loss))
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_loss, 'b-', label="Training Loss")
        plt.plot(epochs, val_loss, 'r-', label="Validation Loss")
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'{self.arg.work_dir}/train_vs_val_loss.png')
        plt.close()
        self.print_log("Created loss curve visualization")

    def cm_viz(self, y_pred: List[int], y_true: List[int]):
        '''
        Create and save a confusion matrix visualization

        Args:
            y_pred: Predicted class labels
            y_true: True class labels
        '''
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()

        # Set axis labels and title
        class_labels = np.unique(y_true)
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
        plt.savefig(f'{self.arg.work_dir}/confusion_matrix.png')
        plt.close()
        self.print_log("Created confusion matrix visualization")

    def create_df(self, columns=['test_subject', 'train_subjects', 'accuracy', 'f1_score']) -> pd.DataFrame:
        '''
        Create an empty DataFrame for storing results

        Args:
            columns: Column names for the DataFrame

        Returns:
            Empty DataFrame with the specified columns
        '''
        return pd.DataFrame(columns=columns)

    def train(self, epoch):
        '''
        Train the model for one epoch

        Args:
            epoch: Current epoch number

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
        acc_value = []
        accuracy = 0
        cnt = 0
        train_loss = 0

        # Training loop with progress bar
        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch} (Train)")
        for batch_idx, (inputs, targets, idx) in enumerate(process):
            # Move data to device
            with torch.no_grad():
                # Always get accelerometer data
                acc_data = inputs['accelerometer'].to(device)
                targets = targets.to(device)

                # Get gyroscope data if available
                gyro_data = None
                if 'gyroscope' in inputs:
                    gyro_data = inputs['gyroscope'].to(device)

                # Get fusion features if available
                fusion_features = None
                if 'fusion_features' in inputs:
                    fusion_features = inputs['fusion_features'].to(device)

                # Get quaternion data if available
                quaternion = None
                if 'quaternion' in inputs:
                    quaternion = inputs['quaternion'].to(device)

            # Record data loading time
            timer['dataloader'] += self.split_time()

            # Forward pass and loss calculation
            self.optimizer.zero_grad()

            # Forward pass depends on available inputs
            if hasattr(self.model, 'forward_fusion') and fusion_features is not None:
                # Model has specialized fusion handling
                logits = self.model.forward_fusion(acc_data.float(), fusion_features.float())
            elif hasattr(self.model, 'forward_quaternion') and quaternion is not None:
                # Model has specialized quaternion handling
                logits = self.model.forward_quaternion(acc_data.float(), quaternion.float())
            elif gyro_data is not None and hasattr(self.model, 'forward_multi_sensor'):
                # Model can handle multiple sensor inputs
                logits = self.model.forward_multi_sensor(acc_data.float(), gyro_data.float())
            else:
                # Default to just accelerometer data
                logits = self.model(acc_data.float())

            # Calculate loss, do backward pass, and update weights
            loss = self.criterion(logits, targets)
            loss.mean().backward()
            self.optimizer.step()

            # Record model training time
            timer['model'] += self.split_time()

            # Track loss and accuracy
            with torch.no_grad():
                train_loss += loss.mean().item()
                accuracy += (torch.argmax(F.log_softmax(logits, dim=1), 1) == targets).sum().item()

            cnt += len(targets)
            timer['stats'] += self.split_time()
            
            # Update progress bar with current loss and accuracy
            process.set_postfix({
                'loss': f"{train_loss/(batch_idx+1):.4f}",
                'acc': f"{100.0*accuracy/cnt:.2f}%"
            })

        # Compute final metrics
        train_loss /= cnt
        accuracy *= 100. / cnt

        # Record results
        self.train_loss_summary.append(train_loss)
        acc_value.append(accuracy)

        # Calculate timing proportions
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        # Log results
        self.print_log(
            f'Epoch {epoch+1}/{self.arg.num_epoch} - '
            f'Training Loss: {train_loss:.4f}, Training Acc: {accuracy:.2f}%'
        )
        self.print_log(f'Time consumption: [Data]{proportion["dataloader"]}, '
                      f'[Network]{proportion["model"]}, [Stats]{proportion["stats"]}')

        # Evaluate on validation set
        val_loss = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
        self.val_loss_summary.append(val_loss)

        return val_loss

    def eval(self, epoch, loader_name='val', result_file=None):
        '''
        Evaluate the model on validation or test data

        Args:
            epoch: Current epoch number
            loader_name: Which data loader to use ('val' or 'test')
            result_file: Optional file to save detailed results

        Returns:
            Average loss for the evaluation set
        '''
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'

        # Open results file if specified
        if result_file is not None:
            f_r = open(result_file, 'w', encoding='utf-8')

        # Set model to evaluation mode
        self.model.eval()
        self.print_log(f'Evaluating on {loader_name} set (Epoch {epoch+1})')

        # Initialize tracking variables
        loss = 0
        cnt = 0
        accuracy = 0
        label_list = []
        pred_list = []

        # Evaluation loop with progress bar
        process = tqdm(self.data_loader[loader_name], desc=f"Epoch {epoch+1} ({loader_name.capitalize()})")
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                # Move data to device
                acc_data = inputs['accelerometer'].to(device)
                targets = targets.to(device)

                # Get gyroscope data if available
                gyro_data = None
                if 'gyroscope' in inputs:
                    gyro_data = inputs['gyroscope'].to(device)

                # Get fusion features if available
                fusion_features = None
                if 'fusion_features' in inputs:
                    fusion_features = inputs['fusion_features'].to(device)

                # Get quaternion data if available
                quaternion = None
                if 'quaternion' in inputs:
                    quaternion = inputs['quaternion'].to(device)

                # Forward pass depends on available inputs
                if hasattr(self.model, 'forward_fusion') and fusion_features is not None:
                    # Model has specialized fusion handling
                    logits = self.model.forward_fusion(acc_data.float(), fusion_features.float())
                elif hasattr(self.model, 'forward_quaternion') and quaternion is not None:
                    # Model has specialized quaternion handling
                    logits = self.model.forward_quaternion(acc_data.float(), quaternion.float())
                elif gyro_data is not None and hasattr(self.model, 'forward_multi_sensor'):
                    # Model can handle multiple sensor inputs
                    logits = self.model.forward_multi_sensor(acc_data.float(), gyro_data.float())
                else:
                    # Default to just accelerometer data
                    logits = self.model(acc_data.float())

                # Calculate loss and track metrics
                batch_loss = self.criterion(logits, targets)
                loss += batch_loss.sum().item()

                # Get predictions and calculate accuracy
                predictions = torch.argmax(F.log_softmax(logits, dim=1), 1)
                accuracy += (predictions == targets).sum().item()

                # Save predictions and labels for metrics
                label_list.extend(targets.cpu().tolist())
                pred_list.extend(predictions.cpu().tolist())
                cnt += len(targets)
                
                # Update progress bar
                process.set_postfix({
                    'loss': f"{loss/cnt:.4f}",
                    'acc': f"{100.0*accuracy/cnt:.2f}%"
                })

            # Calculate final metrics
            loss /= cnt
            target = np.array(label_list)
            y_pred = np.array(pred_list)

            # Calculate F1 score and other metrics
            f1 = f1_score(target, y_pred, average='macro') * 100
            precision, recall, _, _ = precision_recall_fscore_support(
                target, y_pred, average='macro')
            accuracy *= 100. / cnt

            # Write detailed results to file if provided
            if result_file is not None:
                predict = pred_list
                true = label_list
                for i, x in enumerate(predict):
                    f_r.write(f"{x} ==> {true[i]}\n")
                f_r.close()

        # Log results
        self.print_log(
            f'{loader_name.capitalize()} metrics: Loss={loss:.4f}, '
            f'Accuracy={accuracy:.2f}%, F1={f1:.2f}, '
            f'Precision={precision*100:.2f}%, Recall={recall*100:.2f}%'
        )

        # Save best model if in validation phase
        if loader_name == 'val':
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_accuracy = accuracy
                self.best_f1 = f1

                # Save the model weights - handle DataParallel
                if isinstance(self.model, nn.DataParallel):
                    torch.save(deepcopy(self.model.module.state_dict()), self.model_path)
                else:
                    torch.save(deepcopy(self.model.state_dict()), self.model_path)
                    
                self.print_log('Best model saved: improved validation loss, accuracy, and F1 score')

                # Visualize confusion matrix for best model
                if len(np.unique(target)) > 1:  # Only create CM if multiple classes
                    self.cm_viz(y_pred, target)
        else:
            # For test set, store the results
            self.test_accuracy = accuracy
            self.test_f1 = f1

        return loss

    def start(self):
        '''
        Start the training and evaluation workflow.
        
        This method implements the complete workflow for model training,
        validation, and testing using cross-validation for comprehensive
        performance evaluation across different subjects and filter types.
        '''
        try:
            if self.arg.phase == 'train':
                # Initialize training metrics
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_accuracy = 0
                self.best_f1 = 0
                self.best_loss = float('inf')
                self.patience_counter = 0

                # Log the parameters for this run
                self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
                self.print_log(f'Starting training with {self.arg.optimizer} optimizer, LR={self.arg.base_lr}')
                self.print_log(f'Using filter type: {self.arg.dataset_args["fusion_options"]["filter_type"]}')

                # Create results dataframe to store metrics
                results = self.create_df(columns=['fold', 'test_subject', 'train_subjects', 'accuracy', 'f1_score', 'precision', 'recall'])
                
                # Check if kfold setting exists in config
                use_kfold = False
                fold_assignments = []
                
                if hasattr(self.arg, 'kfold') and self.arg.kfold:
                    use_kfold = True
                    if hasattr(self.arg, 'kfold_assignments'):
                        fold_assignments = self.arg.kfold_assignments
                        self.print_log(f"Using provided fold assignments with {len(fold_assignments)} folds")
                elif hasattr(self.arg, 'kfold') and isinstance(self.arg.kfold, dict):
                    use_kfold = self.arg.kfold.get('enabled', False)
                    if use_kfold and 'fold_assignments' in self.arg.kfold:
                        fold_assignments = self.arg.kfold.get('fold_assignments', [])
                        self.print_log(f"Using provided fold assignments with {len(fold_assignments)} folds")
                
                if use_kfold and not fold_assignments:
                    # Create automatic fold assignments if none provided
                    import numpy as np
                    all_subjects = self.arg.subjects.copy()
                    num_folds = getattr(self.arg, 'num_folds', 5)
                    if hasattr(self.arg, 'kfold') and isinstance(self.arg.kfold, dict):
                        num_folds = self.arg.kfold.get('num_folds', 5)
                        
                    np.random.seed(self.arg.seed)
                    np.random.shuffle(all_subjects)
                    
                    # Create approximately equal sized folds
                    fold_size = len(all_subjects) // num_folds
                    for i in range(num_folds):
                        start_idx = i * fold_size
                        end_idx = start_idx + fold_size if i < num_folds - 1 else len(all_subjects)
                        fold_assignments.append(all_subjects[start_idx:end_idx])
                    self.print_log(f"Created {num_folds} automatic fold assignments")
                
                if use_kfold:
                    # Perform k-fold cross validation
                    self.print_log(f"Starting {len(fold_assignments)}-fold cross-validation")
                    fold_metrics = []
                    
                    all_subjects = self.arg.subjects.copy()
                    
                    for fold_idx, test_subjects in enumerate(fold_assignments):
                        self.print_log(f"\n{'='*20} Starting Fold {fold_idx+1}/{len(fold_assignments)} {'='*20}")
                        
                        # Reset best values for each fold
                        self.best_loss = float('inf')
                        self.best_accuracy = 0
                        self.best_f1 = 0
                        self.patience_counter = 0
                        
                        # Set up test, validation, and training subjects
                        self.test_subject = test_subjects
                        
                        # Use next fold for validation
                        next_fold_idx = (fold_idx + 1) % len(fold_assignments)
                        self.val_subject = fold_assignments[next_fold_idx]
                        
                        # Use all remaining folds for training
                        self.train_subjects = []
                        for i, fold in enumerate(fold_assignments):
                            if i != fold_idx and i != next_fold_idx:
                                self.train_subjects.extend(fold)
                        
                        self.print_log(f'Fold {fold_idx+1}: Test subjects={self.test_subject}')
                        self.print_log(f'Validation subjects={self.val_subject}')
                        self.print_log(f'Training subjects={self.train_subjects}')
                        
                        # Create a new model instance for this fold
                        self.model = self.load_model(self.arg.model, self.arg.model_args)
                        
                        # Set up multi-GPU if available and requested
                        if len(self.available_gpus) > 1 and self.arg.multi_gpu:
                            self.model = nn.DataParallel(
                                self.model, 
                                device_ids=self.available_gpus
                            )
                        
                        # Load data for this fold
                        self.print_log(f"Loading data for fold {fold_idx+1}...")
                        if not self.load_data():
                            self.print_log(f"ERROR: Failed to load data for fold {fold_idx+1}")
                            continue
                        self.print_log(f"Data loaded successfully: {len(self.data_loader['train'])} training batches, " +
                                    f"{len(self.data_loader['val'])} validation batches, " +
                                    f"{len(self.data_loader['test'])} test batches")
                        
                        # Set up optimizer
                        self.load_optimizer()
                        
                        # Training loop
                        self.print_log(f"Starting training for fold {fold_idx+1}...")
                        self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
                        
                        # Get early stopping settings
                        patience = 10  # Default patience
                        if hasattr(self.arg, 'early_stopping') and isinstance(self.arg.early_stopping, dict):
                            patience = self.arg.early_stopping.get('patience', 10)
                        elif hasattr(self.arg, 'patience'):
                            patience = self.arg.patience
                        
                        # Train for specified number of epochs with early stopping
                        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                            # Train one epoch and get validation loss
                            val_loss = self.train(epoch)
                            
                            # Check for early stopping
                            if val_loss < self.best_loss:
                                self.best_loss = val_loss
                                self.patience_counter = 0
                                self.print_log(f"New best validation loss: {val_loss:.4f}")
                            else:
                                self.patience_counter += 1
                                self.print_log(f"No improvement for {self.patience_counter}/{patience} epochs")
                                
                                if self.patience_counter >= patience:
                                    self.print_log(f"Early stopping triggered after {epoch+1} epochs")
                                    break
                        
                        # Create loss visualization for this fold
                        if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                            self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                            self.print_log(f"Loss curves saved to {self.arg.work_dir}/train_vs_val_loss.png")
                        
                        # Evaluate on test set using best model
                        self.print_log(f'Training complete for fold {fold_idx+1}, loading best model for testing')
                        test_model = self.load_model(self.arg.model, self.arg.model_args)
                        
                        # Set up multi-GPU for test model if needed
                        if len(self.available_gpus) > 1 and self.arg.multi_gpu:
                            test_model = nn.DataParallel(
                                test_model, 
                                device_ids=self.available_gpus
                            )
                        
                        # Load best model weights
                        try:
                            if isinstance(test_model, nn.DataParallel):
                                test_model.module.load_state_dict(torch.load(self.model_path))
                            else:
                                test_model.load_state_dict(torch.load(self.model_path))
                            self.print_log(f"Successfully loaded best model from {self.model_path}")
                        except Exception as e:
                            self.print_log(f"WARNING: Could not load best model: {str(e)}")
                            self.print_log("Using current model state for testing")
                            test_model = self.model
                        
                        self.model = test_model
                        self.model.eval()
                        
                        # Test on this fold's test set
                        self.print_log(f'------ Testing on subjects {self.test_subject} ------')
                        test_loss = self.eval(epoch=0, loader_name='test')
                        
                        # Store fold metrics
                        fold_result = {
                            'fold': fold_idx + 1,
                            'test_subjects': self.test_subject,
                            'accuracy': self.test_accuracy,
                            'f1': self.test_f1,
                            'precision': self.test_precision,
                            'recall': self.test_recall,
                            'balanced_accuracy': self.test_balanced_accuracy
                        }
                        fold_metrics.append(fold_result)
                        
                        # Add to dataframe
                        subject_result = pd.Series({
                            'fold': fold_idx + 1,
                            'test_subject': str(self.test_subject),
                            'train_subjects': str(self.train_subjects),
                            'accuracy': round(self.test_accuracy, 2),
                            'f1_score': round(self.test_f1, 2),
                            'precision': round(self.test_precision, 2),
                            'recall': round(self.test_recall, 2)
                        })
                        results.loc[len(results)] = subject_result
                        
                        # Save fold-specific results
                        fold_dir = os.path.join(self.arg.work_dir, f"fold_{fold_idx+1}")
                        os.makedirs(fold_dir, exist_ok=True)
                        
                        # Save fold visualization
                        if hasattr(self, 'cm_viz'):
                            try:
                                import numpy as np
                                self.cm_viz(np.array(self.test_pred), np.array(self.test_true))
                                fold_cm_path = os.path.join(fold_dir, "confusion_matrix.png")
                                shutil.copy(os.path.join(self.arg.work_dir, "confusion_matrix.png"), fold_cm_path)
                                self.print_log(f"Saved fold-specific confusion matrix to {fold_cm_path}")
                            except Exception as e:
                                self.print_log(f"Error saving confusion matrix: {str(e)}")
                        
                        # Reset for next fold
                        self.train_loss_summary = []
                        self.val_loss_summary = []
                    
                    # Calculate and save cross-validation summary with average metrics
                    if fold_metrics:
                        import numpy as np
                        avg_metrics = {
                            'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
                            'accuracy_std': np.std([m['accuracy'] for m in fold_metrics]),
                            'f1': np.mean([m['f1'] for m in fold_metrics]),
                            'f1_std': np.std([m['f1'] for m in fold_metrics]),
                            'precision': np.mean([m['precision'] for m in fold_metrics]),
                            'precision_std': np.std([m['precision'] for m in fold_metrics]),
                            'recall': np.mean([m['recall'] for m in fold_metrics]),
                            'recall_std': np.std([m['recall'] for m in fold_metrics]),
                            'balanced_accuracy': np.mean([m['balanced_accuracy'] for m in fold_metrics]),
                            'balanced_accuracy_std': np.std([m['balanced_accuracy'] for m in fold_metrics])
                        }
                        
                        # Create and save summary
                        cv_summary = {
                            'fold_metrics': fold_metrics,
                            'average_metrics': avg_metrics,
                            'filter_type': self.arg.dataset_args['fusion_options']['filter_type']
                        }
                        
                        with open(os.path.join(self.arg.work_dir, 'cv_summary.json'), 'w') as f:
                            import json
                            json.dump(cv_summary, f, indent=2)
                        
                        self.print_log(f"Cross-validation summary saved to {self.arg.work_dir}/cv_summary.json")
                        
                        # Print average performance metrics
                        self.print_log(f'\n===== Cross-Validation Results =====')
                        self.print_log(f'Mean accuracy: {avg_metrics["accuracy"]:.2f}% ± {avg_metrics["accuracy_std"]:.2f}%')
                        self.print_log(f'Mean F1 score: {avg_metrics["f1"]:.2f} ± {avg_metrics["f1_std"]:.2f}')
                        self.print_log(f'Mean precision: {avg_metrics["precision"]:.2f}% ± {avg_metrics["precision_std"]:.2f}%')
                        self.print_log(f'Mean recall: {avg_metrics["recall"]:.2f}% ± {avg_metrics["recall_std"]:.2f}%')
                        self.print_log(f'Mean balanced accuracy: {avg_metrics["balanced_accuracy"]:.2f}% ± {avg_metrics["balanced_accuracy_std"]:.2f}%')
                    
                    # Save all results
                    results.to_csv(os.path.join(self.arg.work_dir, 'fold_scores.csv'), index=False)
                    self.print_log(f"Fold-specific scores saved to {self.arg.work_dir}/fold_scores.csv")
                    
                else:
                    # Regular training without cross-validation
                    self.print_log("Starting standard train/val/test split training (no cross-validation)")
                    
                    # Set up test, validation, and training subjects
                    total_subjects = len(self.arg.subjects)
                    test_idx = total_subjects // 5  # Use ~20% for testing
                    val_idx = test_idx * 2  # Use ~20% for validation
                    
                    self.test_subject = [self.arg.subjects[0:test_idx]]
                    self.val_subject = [self.arg.subjects[test_idx:val_idx]]
                    self.train_subjects = self.arg.subjects[val_idx:]
                    
                    self.print_log(f'Test subjects: {self.test_subject}')
                    self.print_log(f'Validation subjects: {self.val_subject}')
                    self.print_log(f'Training subjects: {self.train_subjects}')
                    
                    # Load data
                    if not self.load_data():
                        self.print_log("ERROR: Failed to load data")
                        return
                    
                    # Set up optimizer
                    self.load_optimizer()
                    
                    # Train for specified number of epochs
                    self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
                    
                    # Get early stopping settings
                    patience = 10  # Default patience
                    if hasattr(self.arg, 'early_stopping') and isinstance(self.arg.early_stopping, dict):
                        patience = self.arg.early_stopping.get('patience', 10)
                    elif hasattr(self.arg, 'patience'):
                        patience = self.arg.patience
                    
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        # Train one epoch
                        val_loss = self.train(epoch)
                        
                        # Check for early stopping
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.patience_counter = 0
                            self.print_log(f"New best validation loss: {val_loss:.4f}")
                        else:
                            self.patience_counter += 1
                            self.print_log(f"No improvement for {self.patience_counter}/{patience} epochs")
                            
                            if self.patience_counter >= patience:
                                self.print_log(f"Early stopping triggered after {epoch+1} epochs")
                                break
                    
                    # Create loss visualization
                    if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                        self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                    
                    # Evaluate on test set using best model
                    self.print_log('Training complete, loading best model for testing')
                    test_model = self.load_model(self.arg.model, self.arg.model_args)
                    
                    if len(self.available_gpus) > 1 and self.arg.multi_gpu:
                        test_model = nn.DataParallel(
                            test_model, 
                            device_ids=self.available_gpus
                        )
                    
                    # Load best model weights
                    try:
                        if isinstance(test_model, nn.DataParallel):
                            test_model.module.load_state_dict(torch.load(self.model_path))
                        else:
                            test_model.load_state_dict(torch.load(self.model_path))
                    except Exception as e:
                        self.print_log(f"WARNING: Could not load best model: {str(e)}")
                        self.print_log("Using current model state for testing")
                        test_model = self.model
                    
                    self.model = test_model
                    self.model.eval()
                    
                    # Test on the test set
                    self.print_log(f'------ Testing on subjects {self.test_subject} ------')
                    self.eval(epoch=0, loader_name='test')
                    
                    # Save test results
                    test_result = pd.Series({
                        'test_subject': str(self.test_subject),
                        'train_subjects': str(self.train_subjects),
                        'accuracy': round(self.test_accuracy, 2),
                        'f1_score': round(self.test_f1, 2),
                        'precision': round(self.test_precision, 2),
                        'recall': round(self.test_recall, 2)
                    })
                    results.loc[len(results)] = test_result
                    results.to_csv(os.path.join(self.arg.work_dir, 'test_scores.csv'), index=False)
            
            else:
                # Testing phase only
                self.print_log('Testing mode - evaluating pre-trained model')
                
                # Set test subject if not already set
                if not hasattr(self, 'test_subject') or not self.test_subject:
                    self.test_subject = self.arg.subjects
                
                # Load data
                if not self.load_data():
                    self.print_log("ERROR: Failed to load test data")
                    return
                
                # Evaluate on test set
                self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)
                
                self.print_log(f'Test results: Accuracy={self.test_accuracy:.2f}%, F1={self.test_f1:.4f}, '
                            f'Precision={self.test_precision:.4f}, Recall={self.test_recall:.4f}')
                
                # Save test results
                test_results = {
                    'filter_type': self.arg.dataset_args['fusion_options']['filter_type'],
                    'accuracy': self.test_accuracy,
                    'f1': self.test_f1,
                    'precision': self.test_precision,
                    'recall': self.test_recall,
                    'balanced_accuracy': self.test_balanced_accuracy
                }
                
                with open(os.path.join(self.arg.work_dir, 'test_results.json'), 'w') as f:
                    import json
                    json.dump(test_results, f, indent=2)
                
                self.print_log(f"Test results saved to {self.arg.work_dir}/test_results.json")
                
                # Create confusion matrix visualization if available
                if hasattr(self, 'cm_viz') and hasattr(self, 'test_pred') and hasattr(self, 'test_true'):
                    try:
                        import numpy as np
                        self.cm_viz(np.array(self.test_pred), np.array(self.test_true))
                        self.print_log(f"Confusion matrix saved to {self.arg.work_dir}/confusion_matrix.png")
                    except Exception as e:
                        self.print_log(f"Error creating confusion matrix: {str(e)}")
        
        except Exception as e:
            self.print_log(f"ERROR in training/testing workflow: {str(e)}")
            import traceback
            self.print_log(traceback.format_exc())
            
            # Try to save current state even in case of error
            if hasattr(self, 'model') and self.arg.phase == 'train':
                emergency_path = os.path.join(self.arg.work_dir, 'emergency_checkpoint.pt')
                try:
                    if isinstance(self.model, nn.DataParallel):
                        torch.save(self.model.module.state_dict(), emergency_path)
                    else:
                        torch.save(self.model.state_dict(), emergency_path)
                    self.print_log(f"Saved emergency checkpoint to {emergency_path}")
                except Exception as save_error:
                    self.print_log(f"Could not save emergency checkpoint: {str(save_error)}")
