'''
Script to train models for Fall Detection and Activity Recognition
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
import json  # Added json import for cross-validation summary
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, balanced_accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from utils.dataset import prepare_smartfallmm, split_by_subjects

MAX_THREADS = 40
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)

def get_args():
    parser = argparse.ArgumentParser(description='Fall Detection and Human Activity Recognition')
    parser.add_argument('--config', default='./config/smartfallmm/fusion_madgwick.yaml', help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='smartfallmm', help='Dataset name to use')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='Input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N', help='Input batch size for testing')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N', help='Input batch size for validation')
    parser.add_argument('--num-epoch', type=int, default=60, metavar='N', help='Number of epochs to train')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch number')
    parser.add_argument('--weights-only', type=str2bool, default=False, help='Whether to load only weights (not full model)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (Adam, SGD, etc.)')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR', help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0004, help='Weight decay factor')
    parser.add_argument('--kfold', type=str2bool, default=False, help='Whether to use k-fold cross validation')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--monitor', type=str, default='val_loss', help='Metric to monitor for early stopping')
    parser.add_argument('--model', default=None, help='Model class path to load')
    parser.add_argument('--device', nargs='+', default=[0], type=int, help='CUDA device IDs to use')
    parser.add_argument('--model-args', default=None, help='Dictionary of model arguments')
    parser.add_argument('--weights', type=str, help='Path to pretrained weights file')
    parser.add_argument('--model-saved-name', type=str, default='model', help='Name for saving the trained model')
    parser.add_argument('--loss', default='loss.BCE', help='Loss function class path')
    parser.add_argument('--loss-args', default="{}", type=str, help='Dictionary of loss function arguments')
    parser.add_argument('--dataset-args', default=None, help='Arguments for the dataset')
    parser.add_argument('--subjects', nargs='+', type=int, help='Subject IDs to include')
    parser.add_argument('--feeder', default=None, help='DataLoader class path')
    parser.add_argument('--train-feeder-args', default=None, help='Arguments for training data loader')
    parser.add_argument('--val-feeder-args', default=None, help='Arguments for validation data loader')
    parser.add_argument('--test-feeder-args', default=None, help='Arguments for test data loader')
    parser.add_argument('--include-val', type=str2bool, default=True, help='Whether to include validation set')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='How many batches to wait before logging')
    parser.add_argument('--work-dir', type=str, default='work_dir', help='Working directory for outputs')
    parser.add_argument('--print-log', type=str2bool, default=True, help='Whether to print and save logs')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--num-worker', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--result-file', type=str, help='File to save results to')
    parser.add_argument('--multi-gpu', type=str2bool, default=True, help='Whether to use multiple GPUs')
    parser.add_argument('--parallel-threads', type=int, default=4, help='Number of parallel threads for preprocessing')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Whether to print verbose debugging info')
    parser.add_argument('--run-comparison', type=str2bool, default=False, help='Whether to run filter comparison analysis')
    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found ({traceback.format_exception(*sys.exc_info())})')

def setup_gpu_environment(args):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs")
        if isinstance(args.device, list) and len(args.device) > 0:
            devices = args.device
            print(f"Using specified GPUs: {devices}")
        elif num_gpus >= 2 and args.multi_gpu:
            devices = [0, 1]
            print(f"Using both GPUs: {devices}")
        elif num_gpus == 1:
            devices = [0]
            print(f"Using single GPU: {devices}")
        else:
            devices = []
            print("No GPUs found, using CPU")
        gpu_list = ",".join(map(str, devices))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"Set CUDA_VISIBLE_DEVICES={gpu_list}")
        return devices
    else:
        print("PyTorch CUDA not available, using CPU")
        return []

class Trainer:
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_f1 = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0
        self.test_accuracy = 0
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_balanced_accuracy = 0
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = dict()
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}.pt'
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
        
        # Get the filter type if specified
        self.filter_type = "madgwick"  # Default filter
        if 'fusion_options' in arg.dataset_args and arg.dataset_args['fusion_options'].get('enabled', False):
            self.filter_type = arg.dataset_args['fusion_options'].get('filter_type', 'madgwick')
            
        # Create working directory
        os.makedirs(self.arg.work_dir, exist_ok=True)
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
            self.model = nn.DataParallel(self.model, device_ids=self.available_gpus)
        
        # Load loss function
        self.load_loss()
        
        # Set validation option
        self.include_val = arg.include_val
        
        # Initialize patience counter for early stopping
        self.patience_counter = 0
        
        # Log model parameters
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size: {num_params / (1024 ** 2):.2f} MB')
        self.print_log(f'Sensor modalities: {self.inertial_modality}')
        self.print_log(f'Using fusion: {self.fuse} with filter type: {self.filter_type}')
        
        # Log GPU configuration
        if self.available_gpus:
            self.print_log(f'Using GPUs: {self.available_gpus}')
        else:
            self.print_log('Using CPU for computation')

    def save_config(self, src_path, desc_path):
        config_file = src_path.rpartition("/")[-1]
        print(f'Saving config to {desc_path}/{config_file}')
        shutil.copy(src_path, f'{desc_path}/{config_file}')

    def count_parameters(self, model):
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.nelement() * buffer.element_size()
        return total_size

    def has_empty_value(self, *lists):
        return any(len(lst) == 0 for lst in lists)

    def load_model(self, model, model_args):
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        Model = import_class(model)
        self.print_log(f"Loading model: {model}")
        self.print_log(f"Model arguments: {model_args}")
        model = Model(**model_args).to(device)
        return model

    def early_stopping(self, current_val_loss, current_val_f1, epoch):
        """
        Enhanced early stopping that monitors both validation loss and F1 score.
        
        Args:
            current_val_loss: Current validation loss
            current_val_f1: Current validation F1 score
            epoch: Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Check if there's improvement in either metric
        improved = False
        
        # Check for improvement in validation loss
        if current_val_loss < self.best_loss:
            self.best_loss = current_val_loss
            improved = True
            self.print_log(f"Validation loss improved to {current_val_loss:.4f}")
        
        # Check for improvement in F1 score
        if current_val_f1 > self.best_f1:
            self.best_f1 = current_val_f1
            improved = True
            self.print_log(f"Validation F1 improved to {current_val_f1:.2f}")
        
        # Reset or increment patience counter
        if improved:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            self.print_log(f"No improvement for {self.patience_counter} epochs (patience: {self.arg.patience})")
            
            if self.patience_counter >= self.arg.patience:
                self.print_log(f"Early stopping triggered after {epoch+1} epochs")
                return True
            return False

    def load_loss(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.print_log("Using CrossEntropyLoss for training")

    def load_weights(self):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(self.model_path))
        self.print_log(f"Loaded model weights from {self.model_path}")

    def load_optimizer(self):
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

    def distribution_viz(self, labels, work_dir, mode):
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
        # Import the data feeder class
        Feeder = import_class(self.arg.feeder)
        self.print_log(f"Using data feeder: {self.arg.feeder}")

        if self.arg.phase == 'train':
            # Prepare dataset
            self.print_log("Preparing SmartFallMM dataset...")
            builder = prepare_smartfallmm(self.arg)
            self.print_log("Dataset preparation complete")

            # Split data for training, validation, and testing
            self.print_log(f"Splitting data for subjects: train={self.train_subjects}, val={self.val_subject}")
            
            # Load training data
            self.norm_train = split_by_subjects(builder, self.train_subjects, self.fuse)
            
            # Check if training data is valid
            if self.has_empty_value(list(self.norm_train.values())):
                self.print_log("WARNING: Training data has some empty values")
                
                # Ensure required modalities are present
                if 'accelerometer' not in self.norm_train or len(self.norm_train['accelerometer']) == 0:
                    self.print_log("ERROR: No accelerometer data available for training")
                    return False
                
                if 'labels' not in self.norm_train or len(self.norm_train['labels']) == 0:
                    self.print_log("ERROR: No labels available for training")
                    return False
            
            # Create training data loader
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args, dataset=self.norm_train),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            )
            self.print_log(f"Training data loaded with {len(self.data_loader['train'])} batches")
            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
            
            # Only proceed with validation if include_val is True
            if self.include_val and self.val_subject:
                # Load validation data
                try:
                    self.norm_val = split_by_subjects(builder, self.val_subject, self.fuse)
                    
                    # Validate that we have sufficient validation data
                    if 'accelerometer' not in self.norm_val or len(self.norm_val['accelerometer']) == 0:
                        self.print_log("WARNING: No accelerometer data available for validation")
                        self.print_log("Skipping validation, using training data for model selection")
                        self.include_val = False
                    elif 'labels' not in self.norm_val or len(self.norm_val['labels']) == 0:
                        self.print_log("WARNING: No labels available for validation")
                        self.print_log("Skipping validation, using training data for model selection")
                        self.include_val = False
                    else:
                        # Create validation data loader
                        self.data_loader['val'] = torch.utils.data.DataLoader(
                            dataset=Feeder(**self.arg.val_feeder_args, dataset=self.norm_val),
                            batch_size=self.arg.batch_size,
                            shuffle=True,
                            num_workers=self.arg.num_worker
                        )
                        self.print_log(f"Validation data loaded with {len(self.data_loader['val'])} batches")
                        self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')
                except Exception as e:
                    self.print_log(f"WARNING: Error loading validation data: {str(e)}")
                    self.print_log("Skipping validation, using training data for model selection")
                    self.include_val = False
            else:
                self.print_log("Validation not included in this run")
                self.include_val = False
            
            # Load test data if test subjects are specified
            if self.test_subject:
                self.print_log(f"Preparing test data for subject {self.test_subject}")
                try:
                    self.norm_test = split_by_subjects(builder, self.test_subject, self.fuse)
                    
                    # Validate that we have sufficient test data
                    if 'accelerometer' not in self.norm_test or len(self.norm_test['accelerometer']) == 0:
                        self.print_log("WARNING: No accelerometer data available for testing")
                    elif 'labels' not in self.norm_test or len(self.norm_test['labels']) == 0:
                        self.print_log("WARNING: No labels available for testing")
                    else:
                        # Create test data loader
                        self.data_loader['test'] = torch.utils.data.DataLoader(
                            dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                            batch_size=self.arg.test_batch_size,
                            shuffle=True,
                            num_workers=self.arg.num_worker
                        )
                        self.print_log(f"Test data loaded with {len(self.data_loader['test'])} batches")
                except Exception as e:
                    self.print_log(f"WARNING: Error loading test data: {str(e)}")
                    self.print_log("Testing will be skipped")
            
            # Log dataset sizes and modalities
            if 'train' in self.data_loader:
                self.print_log(f"Training data modalities: {list(self.norm_train.keys())}")
            if 'val' in self.data_loader:
                self.print_log(f"Validation data modalities: {list(self.norm_val.keys())}")
            if 'test' in self.data_loader:
                self.print_log(f"Test data modalities: {list(self.norm_test.keys())}")
            
            return True
        else:
            # Testing phase only
            self.print_log(f"Preparing test data for subject {self.test_subject}")
            builder = prepare_smartfallmm(self.arg)
            
            try:
                self.norm_test = split_by_subjects(builder, self.test_subject, self.fuse)
                
                # Validate that we have sufficient test data
                if 'accelerometer' not in self.norm_test or len(self.norm_test['accelerometer']) == 0:
                    self.print_log("ERROR: No accelerometer data available for testing")
                    return False
                if 'labels' not in self.norm_test or len(self.norm_test['labels']) == 0:
                    self.print_log("ERROR: No labels available for testing")
                    return False
                
                # Create test data loader
                self.data_loader['test'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                    batch_size=self.arg.test_batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker
                )
                self.print_log(f"Test data loaded with {len(self.data_loader['test'])} batches")
                return True
            except Exception as e:
                self.print_log(f"ERROR: Error loading test data: {str(e)}")
                self.print_log(traceback.format_exc())
                return False

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string, print_time=True):
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)

    def loss_viz(self, train_loss, val_loss):
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

    def cm_viz(self, y_pred, y_true):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        class_labels = np.unique(y_true)
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels)
        plt.yticks(tick_marks, class_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(f'{self.arg.work_dir}/confusion_matrix.png')
        plt.close()
        self.print_log("Created confusion matrix visualization")

    def create_df(self, columns=['test_subject', 'train_subjects', 'accuracy', 'f1_score']):
        return pd.DataFrame(columns=columns)

    def train(self, epoch):
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        self.model.train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader=0.001, model=0.001, stats=0.001)
        acc_value = []
        accuracy = 0
        cnt = 0
        train_loss = 0
        
        # Additional metrics for training
        y_true = []
        y_pred = []
        
        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch} (Train)")
        for batch_idx, (inputs, targets, idx) in enumerate(process):
            with torch.no_grad():
                # Load data to device
                acc_data = inputs['accelerometer'].to(device)
                targets = targets.to(device)
                gyro_data = None
                if 'gyroscope' in inputs:
                    gyro_data = inputs['gyroscope'].to(device)
                fusion_features = None
                if 'fusion_features' in inputs:
                    fusion_features = inputs['fusion_features'].to(device)
                quaternion = None
                if 'quaternion' in inputs:
                    quaternion = inputs['quaternion'].to(device)
            
            timer['dataloader'] += self.split_time()
            
            # Forward pass with appropriate model function based on available data
            self.optimizer.zero_grad()
            if hasattr(self.model, 'forward_fusion') and fusion_features is not None:
                logits = self.model.forward_fusion(acc_data.float(), fusion_features.float())
            elif hasattr(self.model, 'forward_quaternion') and quaternion is not None:
                logits = self.model.forward_quaternion(acc_data.float(), quaternion.float())
            elif gyro_data is not None and hasattr(self.model, 'forward_multi_sensor'):
                logits = self.model.forward_multi_sensor(acc_data.float(), gyro_data.float())
            else:
                logits = self.model(acc_data.float())
            
            # Calculate loss and backpropagate
            loss = self.criterion(logits, targets)
            loss.mean().backward()
            self.optimizer.step()
            timer['model'] += self.split_time()
            
            # Calculate metrics
            with torch.no_grad():
                train_loss += loss.mean().item()
                predictions = torch.argmax(F.log_softmax(logits, dim=1), 1)
                accuracy += (predictions == targets).sum().item()
                
                # Collect predictions and targets for F1 calculation
                y_true.extend(targets.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())
            
            cnt += len(targets)
            timer['stats'] += self.split_time()
            
            # Update progress bar
            process.set_postfix({
                'loss': f"{train_loss/(batch_idx+1):.4f}",
                'acc': f"{100.0*accuracy/cnt:.2f}%"
            })
        
        # Calculate final metrics
        train_loss /= len(loader)
        accuracy *= 100. / cnt
        
        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision *= 100
        recall *= 100
        balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
        
        # Store training metrics
        self.train_loss_summary.append(train_loss)
        acc_value.append(accuracy)
        
        # Store detailed metrics
        train_metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'balanced_accuracy': balanced_acc
        }
        self.train_metrics.append(train_metrics)
        
        # Log time consumption
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }
        self.print_log(
            f'Epoch {epoch+1}/{self.arg.num_epoch} - '
            f'Training Loss: {train_loss:.4f}, Training Acc: {accuracy:.2f}%, F1: {f1:.2f}%'
        )
        self.print_log(f'Time consumption: [Data]{proportion["dataloader"]}, '
                     f'[Network]{proportion["model"]}, [Stats]{proportion["stats"]}')
        
        # Run validation if available
        val_loss = train_loss  # Default to using training loss if no validation
        val_f1 = f1  # Default to using training F1 if no validation
        
        if self.include_val and 'val' in self.data_loader:
            val_metrics = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
            val_loss = val_metrics['loss']
            val_f1 = val_metrics['f1']
            self.val_loss_summary.append(val_loss)
            self.val_metrics.append(val_metrics)
        else:
            # No validation, use training metrics for model selection
            self.val_loss_summary.append(train_loss)
            self.val_metrics.append(train_metrics)
            self.print_log("No validation data - using training metrics for model selection")
            
            # Save model if training metrics improve
            save_model = False
            if f1 > self.best_f1:
                self.best_f1 = f1
                save_model = True
                self.print_log(f"New best model saved: improved training F1 to {f1:.2f}")
            elif train_loss < self.best_loss:
                self.best_loss = train_loss
                save_model = True
                self.print_log(f"New best model saved: improved training loss to {train_loss:.4f}")
            
            if save_model:
                self.best_accuracy = accuracy
                if isinstance(self.model, nn.DataParallel):
                    torch.save(deepcopy(self.model.module.state_dict()), self.model_path)
                else:
                    torch.save(deepcopy(self.model.state_dict()), self.model_path)
        
        return {'loss': val_loss, 'f1': val_f1}

    def eval(self, epoch, loader_name='val', result_file=None):
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        if result_file is not None:
            f_r = open(result_file, 'w', encoding='utf-8')
        self.model.eval()
        self.print_log(f'Evaluating on {loader_name} set (Epoch {epoch+1})')
        loss = 0
        cnt = 0
        accuracy = 0
        label_list = []
        pred_list = []
        process = tqdm(self.data_loader[loader_name], desc=f"Epoch {epoch+1} ({loader_name.capitalize()})")
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                acc_data = inputs['accelerometer'].to(device)
                targets = targets.to(device)
                gyro_data = None
                if 'gyroscope' in inputs:
                    gyro_data = inputs['gyroscope'].to(device)
                fusion_features = None
                if 'fusion_features' in inputs:
                    fusion_features = inputs['fusion_features'].to(device)
                quaternion = None
                if 'quaternion' in inputs:
                    quaternion = inputs['quaternion'].to(device)
                
                # Forward pass - select appropriate method based on available data
                if hasattr(self.model, 'forward_fusion') and fusion_features is not None:
                    logits = self.model.forward_fusion(acc_data.float(), fusion_features.float())
                elif hasattr(self.model, 'forward_quaternion') and quaternion is not None:
                    logits = self.model.forward_quaternion(acc_data.float(), quaternion.float())
                elif gyro_data is not None and hasattr(self.model, 'forward_multi_sensor'):
                    logits = self.model.forward_multi_sensor(acc_data.float(), gyro_data.float())
                else:
                    logits = self.model(acc_data.float())
                
                batch_loss = self.criterion(logits, targets)
                loss += batch_loss.sum().item()
                predictions = torch.argmax(F.log_softmax(logits, dim=1), 1)
                accuracy += (predictions == targets).sum().item()
                label_list.extend(targets.cpu().tolist())
                pred_list.extend(predictions.cpu().tolist())
                cnt += len(targets)
                process.set_postfix({
                    'loss': f"{loss/cnt:.4f}",
                    'acc': f"{100.0*accuracy/cnt:.2f}%"
                })
            
            # Calculate final metrics
            loss /= cnt
            target = np.array(label_list)
            y_pred = np.array(pred_list)
            f1 = f1_score(target, y_pred, average='macro') * 100
            precision, recall, _, _ = precision_recall_fscore_support(target, y_pred, average='macro')
            precision *= 100
            recall *= 100
            balanced_acc = balanced_accuracy_score(target, y_pred) * 100
            accuracy *= 100. / cnt
            
            # Write prediction results to file if requested
            if result_file is not None:
                predict = pred_list
                true = label_list
                for i, x in enumerate(predict):
                    f_r.write(f"{x} ==> {true[i]}\n")
                f_r.close()
        
        # Log evaluation results
        self.print_log(
            f'{loader_name.capitalize()} metrics: Loss={loss:.4f}, '
            f'Accuracy={accuracy:.2f}%, F1={f1:.2f}, '
            f'Precision={precision:.2f}%, Recall={recall:.2f}%, '
            f'Balanced Accuracy={balanced_acc:.2f}%'
        )
        
        # Store metrics
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'balanced_accuracy': balanced_acc,
            'false_alarm_rate': 100 - precision,
            'miss_rate': 100 - recall
        }
        
        if loader_name == 'val':
            # Check if this is the best model so far
            save_model = False
            if f1 > self.best_f1:
                self.best_f1 = f1
                save_model = True
                self.print_log(f"New best model saved: improved validation F1 to {f1:.2f}")
            elif loss < self.best_loss:
                self.best_loss = loss
                save_model = True
                self.print_log(f"New best model saved: improved validation loss to {loss:.4f}")
            
            # Save model if improved
            if save_model:
                self.best_accuracy = accuracy
                try:
                    if isinstance(self.model, nn.DataParallel):
                        torch.save(deepcopy(self.model.module.state_dict()), self.model_path)
                    else:
                        torch.save(deepcopy(self.model.state_dict()), self.model_path)
                    self.print_log(f"Successfully saved model to {self.model_path}")
                except Exception as e:
                    self.print_log(f"Error saving model: {str(e)}")
                
                # Create confusion matrix for best model
                if len(np.unique(target)) > 1:
                    try:
                        self.cm_viz(y_pred, target)
                    except Exception as e:
                        self.print_log(f"Error creating confusion matrix: {str(e)}")
        else:
            # Store test results
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_precision = precision
            self.test_recall = recall
            self.test_balanced_accuracy = balanced_acc
            self.test_true = label_list
            self.test_pred = pred_list
            
            # Create confusion matrix for test set
            try:
                self.cm_viz(y_pred, target)
            except Exception as e:
                self.print_log(f"Error creating confusion matrix: {str(e)}")
        
        return metrics

    def generate_filter_comparison(self):
        """Generate comprehensive filter performance comparison analysis."""
        comparison_dir = os.path.join(self.arg.work_dir, 'filter_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Get the cross-validation summary for this filter
        cv_summary_path = os.path.join(self.arg.work_dir, 'cv_summary.json')
        if not os.path.exists(cv_summary_path):
            self.print_log(f"No cross-validation summary found at {cv_summary_path}")
            return
        
        try:
            # Save filter-specific results
            with open(cv_summary_path, 'r') as f:
                cv_summary = json.load(f)
            
            # Add filter type to summary if not present
            if 'filter_type' not in cv_summary:
                cv_summary['filter_type'] = self.filter_type
            
            # Save the summary with filter name
            filter_summary_path = os.path.join(comparison_dir, f'{self.filter_type}_summary.json')
            with open(filter_summary_path, 'w') as f:
                json.dump(cv_summary, f, indent=2)
            
            self.print_log(f"Saved filter summary to {filter_summary_path}")
        except Exception as e:
            self.print_log(f"Error saving filter comparison data: {str(e)}")
    
    def start(self):
        try:
            if self.arg.phase == 'train':
                # Initialize training metrics
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.train_metrics = []
                self.val_metrics = []
                self.best_accuracy = 0
                self.best_f1 = 0
                self.best_loss = float('inf')
                
                # Log training parameters
                self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
                self.print_log(f'Starting training with {self.arg.optimizer} optimizer, LR={self.arg.base_lr}')
                self.print_log(f'Using fusion with filter type: {self.filter_type}')
                
                # Create DataFrame for results
                results = self.create_df(columns=['fold', 'test_subject', 'train_subjects', 'accuracy', 'f1_score', 'precision', 'recall'])
                
                # Determine if we're using k-fold cross-validation
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
                
                # If k-fold is enabled but no assignments provided, create automatic assignments
                if use_kfold and not fold_assignments:
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
                
                # Perform k-fold cross-validation if enabled
                if use_kfold:
                    self.print_log(f"Starting {len(fold_assignments)}-fold cross-validation")
                    fold_metrics = []
                    all_subjects = self.arg.subjects.copy()
                    
                    for fold_idx, test_subjects in enumerate(fold_assignments):
                        # Reset metrics for this fold
                        self.print_log(f"\n{'='*20} Starting Fold {fold_idx+1}/{len(fold_assignments)} {'='*20}")
                        self.best_loss = float('inf')
                        self.best_accuracy = 0
                        self.best_f1 = 0
                        self.patience_counter = 0
                        
                        # Set up test, validation, and training subjects
                        next_fold_idx = (fold_idx + 1) % len(fold_assignments)
                        self.val_subject = fold_assignments[next_fold_idx]
                        self.test_subject = test_subjects
                        self.train_subjects = []
                        for i, fold in enumerate(fold_assignments):
                            if i != fold_idx and i != next_fold_idx:
                                self.train_subjects.extend(fold)
                        
                        self.print_log(f'Fold {fold_idx+1}: Test subjects={self.test_subject}')
                        self.print_log(f'Validation subjects={self.val_subject}')
                        self.print_log(f'Training subjects={self.train_subjects}')
                        
                        # Create a new model instance for this fold
                        self.model = self.load_model(self.arg.model, self.arg.model_args)
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
                        
                        # Log data loading results
                        train_batches = len(self.data_loader.get('train', [])) if 'train' in self.data_loader else 0
                        val_batches = len(self.data_loader.get('val', [])) if 'val' in self.data_loader else 0
                        test_batches = len(self.data_loader.get('test', [])) if 'test' in self.data_loader else 0
                        
                        self.print_log(f"Data loaded: {train_batches} training batches, "
                                    f"{val_batches} validation batches, "
                                    f"{test_batches} test batches")
                        
                        # Initialize optimizer
                        self.load_optimizer()
                        
                        # Training loop
                        self.print_log(f"Starting training for fold {fold_idx+1}...")
                        self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
                        
                        patience = self.arg.patience
                        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                            val_metrics = self.train(epoch)
                            val_loss = val_metrics['loss']
                            val_f1 = val_metrics['f1']
                            
                            if self.early_stopping(val_loss, val_f1, epoch):
                                self.print_log(f"Early stopping triggered after {epoch+1} epochs")
                                break
                        
                        # Create loss visualization for this fold
                        if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                            try:
                                self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                                self.print_log(f"Loss curves saved to {self.arg.work_dir}/train_vs_val_loss.png")
                            except Exception as e:
                                self.print_log(f"Error creating loss visualization: {str(e)}")
                        
                        # Evaluate on test set using best model
                        self.print_log(f'Training complete for fold {fold_idx+1}, loading best model for testing')
                        if os.path.exists(self.model_path):
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
                                self.print_log(f"Successfully loaded best model from {self.model_path}")
                                self.model = test_model
                            except Exception as e:
                                self.print_log(f"WARNING: Could not load best model: {str(e)}")
                                self.print_log("Using current model state for testing")
                        else:
                            self.print_log(f"WARNING: No saved model found at {self.model_path}")
                            self.print_log("Using current model state for testing")
                        
                        # Set model to evaluation mode
                        self.model.eval()
                        
                        # Test on this fold's test set (if test data loader exists)
                        if 'test' in self.data_loader:
                            self.print_log(f'------ Testing on subjects {self.test_subject} ------')
                            test_metrics = self.eval(epoch=0, loader_name='test')
                            
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
                            
                            # Add to results DataFrame
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
                            
                            # Save fold-specific test results
                            test_results = {
                                'accuracy': self.test_accuracy,
                                'f1': self.test_f1,
                                'precision': self.test_precision,
                                'recall': self.test_recall,
                                'balanced_accuracy': self.test_balanced_accuracy
                            }
                            with open(os.path.join(fold_dir, 'test_results.json'), 'w') as f:
                                json.dump(test_results, f, indent=2)
                            
                            # Save fold visualization
                            if hasattr(self, 'cm_viz') and hasattr(self, 'test_pred') and hasattr(self, 'test_true'):
                                try:
                                    self.cm_viz(np.array(self.test_pred), np.array(self.test_true))
                                    fold_cm_path = os.path.join(fold_dir, "confusion_matrix.png")
                                    shutil.copy(os.path.join(self.arg.work_dir, "confusion_matrix.png"), fold_cm_path)
                                    self.print_log(f"Saved fold-specific confusion matrix to {fold_cm_path}")
                                except Exception as e:
                                    self.print_log(f"Error saving confusion matrix: {str(e)}")
                        else:
                            self.print_log(f"WARNING: No test data loader available for fold {fold_idx+1}")
                        
                        # Reset for next fold
                        self.train_loss_summary = []
                        self.val_loss_summary = []
                    
                    # After training all folds, create cross-validation summary
                    if use_kfold and fold_metrics:
                        try:
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
                                'filter_type': self.filter_type
                            }
                            
                            # Save CV summary
                            summary_path = os.path.join(self.arg.work_dir, 'cv_summary.json')
                            with open(summary_path, 'w') as f:
                                json.dump(cv_summary, f, indent=2)
                                
                            self.print_log(f"Cross-validation summary saved to {summary_path}")
                            
                            # Print average performance metrics
                            self.print_log(f'\n===== Cross-Validation Results =====')
                            self.print_log(f'Mean accuracy: {avg_metrics["accuracy"]:.2f}% ± {avg_metrics["accuracy_std"]:.2f}%')
                            self.print_log(f'Mean F1 score: {avg_metrics["f1"]:.2f} ± {avg_metrics["f1_std"]:.2f}')
                            self.print_log(f'Mean precision: {avg_metrics["precision"]:.2f}% ± {avg_metrics["precision_std"]:.2f}%')
                            self.print_log(f'Mean recall: {avg_metrics["recall"]:.2f}% ± {avg_metrics["recall_std"]:.2f}%')
                            self.print_log(f'Mean balanced accuracy: {avg_metrics["balanced_accuracy"]:.2f}% ± {avg_metrics["balanced_accuracy_std"]:.2f}%')
                            
                            # Generate filter comparison if requested
                            if hasattr(self.arg, 'run_comparison') and self.arg.run_comparison:
                                self.generate_filter_comparison()
                        except Exception as e:
                            self.print_log(f"Error creating cross-validation summary: {str(e)}")
                            self.print_log(traceback.format_exc())
                    
                    # Save all results
                    try:
                        results.to_csv(os.path.join(self.arg.work_dir, 'fold_scores.csv'), index=False)
                        self.print_log(f"Fold-specific scores saved to {self.arg.work_dir}/fold_scores.csv")
                    except Exception as e:
                        self.print_log(f"Error saving fold scores: {str(e)}")
                    
                else:
                    # Regular training without cross-validation
                    self.print_log("Starting standard train/val/test split training (no cross-validation)")
                    
                    # Set up test, validation, and training subjects if not already defined
                    if not self.train_subjects and not self.val_subject and not self.test_subject:
                        total_subjects = len(self.arg.subjects)
                        test_idx = max(1, total_subjects // 5)  # Use ~20% for testing
                        val_idx = test_idx * 2  # Use ~20% for validation
                        
                        self.test_subject = self.arg.subjects[0:test_idx]
                        self.val_subject = self.arg.subjects[test_idx:val_idx]
                        self.train_subjects = self.arg.subjects[val_idx:]
                    
                    self.print_log(f'Test subjects: {self.test_subject}')
                    self.print_log(f'Validation subjects: {self.val_subject}')
                    self.print_log(f'Training subjects: {self.train_subjects}')
                    
                    # Load data
                    if not self.load_data():
                        self.print_log("WARNING: Data loading issues encountered - will attempt to continue with available data")
                    
                    # Set up optimizer
                    self.load_optimizer()
                    
                    # Train for specified number of epochs
                    self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
                    
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        # Train one epoch
                        val_metrics = self.train(epoch)
                        val_loss = val_metrics['loss']
                        val_f1 = val_metrics['f1']
                        
                        # Check for early stopping
                        if self.early_stopping(val_loss, val_f1, epoch):
                            self.print_log(f"Early stopping triggered after {epoch+1} epochs")
                            break
                    
                    # Create loss visualization
                    if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                        try:
                            self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                        except Exception as e:
                            self.print_log(f"Error creating loss visualization: {str(e)}")
                    
                    # Evaluate on test set using best model
                    self.print_log('Training complete, loading best model for testing')
                    if os.path.exists(self.model_path):
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
                            self.model = test_model
                            self.print_log("Successfully loaded best model weights")
                        except Exception as e:
                            self.print_log(f"WARNING: Could not load best model: {str(e)}")
                            self.print_log("Using current model state for testing")
                    else:
                        self.print_log(f"WARNING: No saved model found at {self.model_path}")
                        self.print_log("Using current model state for testing")
                    
                    self.model.eval()
                    
                    # Test on the test set (if test data loader exists)
                    if 'test' in self.data_loader:
                        self.print_log(f'------ Testing on subjects {self.test_subject} ------')
                        test_metrics = self.eval(epoch=0, loader_name='test')
                        
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
                        
                        try:
                            results.to_csv(os.path.join(self.arg.work_dir, 'test_scores.csv'), index=False)
                        except Exception as e:
                            self.print_log(f"Error saving test scores: {str(e)}")
            
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
                
                # Log results
                self.print_log(f'Test results: Accuracy={self.test_accuracy:.2f}%, F1={self.test_f1:.4f}, '
                            f'Precision={self.test_precision:.4f}, Recall={self.test_recall:.4f}')
                
                # Save test results
                try:
                    test_results = {
                        'filter_type': self.filter_type,
                        'accuracy': self.test_accuracy,
                        'f1': self.test_f1,
                        'precision': self.test_precision,
                        'recall': self.test_recall,
                        'balanced_accuracy': self.test_balanced_accuracy
                    }
                    
                    with open(os.path.join(self.arg.work_dir, 'test_results.json'), 'w') as f:
                        json.dump(test_results, f, indent=2)
                    
                    self.print_log(f"Test results saved to {self.arg.work_dir}/test_results.json")
                except Exception as e:
                    self.print_log(f"Error saving test results: {str(e)}")
                
                # Create confusion matrix visualization if available
                if hasattr(self, 'cm_viz') and hasattr(self, 'test_pred') and hasattr(self, 'test_true'):
                    try:
                        self.cm_viz(np.array(self.test_pred), np.array(self.test_true))
                        self.print_log(f"Confusion matrix saved to {self.arg.work_dir}/confusion_matrix.png")
                    except Exception as e:
                        self.print_log(f"Error creating confusion matrix: {str(e)}")
        
        except Exception as e:
            self.print_log(f"ERROR in training/testing workflow: {str(e)}")
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

def main():
    parser = get_args()
    arg = parser.parse_args()
    
    if arg.config is not None:
        with open(arg.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(arg).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
        parser.set_defaults(**default_arg)
        arg = parser.parse_args()
    
    init_seed(arg.seed)
    trainer = Trainer(arg)

    if arg.phase == 'train':
        trainer.start()
    elif arg.phase == 'test':
        if arg.weights is None:
            raise ValueError('Please appoint --weights.')
        trainer.test_subject = arg.subjects
        trainer.start()
    else:
        raise ValueError('Unknown phase: ' + arg.phase)

if __name__ == '__main__':
    main()
