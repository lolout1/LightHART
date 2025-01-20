import traceback
from typing import List, Dict, Any
import random
import sys
import os
import time
import shutil
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning
from utils.dataset import prepare_smartfallmm, filter_subjects
import warnings
import json
import itertools
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description='Fall Detection Training')
    parser.add_argument('--config', default='./config/smartfallmm/teacher.yaml')
    parser.add_argument('--dataset', type=str, default='utd')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--val-batch-size', type=int, default=32)
    parser.add_argument('--num-epoch', type=int, default=200)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--epoch', type=int, default=None, help='if specified, skip LOOCV and train for this many epochs')

    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--base-lr', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    # Model parameters
    parser.add_argument('--model', default=None)
    parser.add_argument('--model-args', default=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--model-saved-name', type=str, default='fall_detection_model')
    parser.add_argument('--device', nargs='+', default=[0], type=int)

    # Dataset parameters
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--dataset-args', default=str)
    
    # Training components
    parser.add_argument('--loss', default='loss.FallDetectionLoss')
    parser.add_argument('--loss-args', default="{}")
    parser.add_argument('--scheduler', type=str, default='plateau')
    parser.add_argument('--scheduler-args', default="{}")
    
    # Data loading
    parser.add_argument('--feeder', default=None)
    parser.add_argument('--train-feeder-args', default=str)
    parser.add_argument('--val-feeder-args', default=str)
    parser.add_argument('--test-feeder-args', default=str)
    parser.add_argument('--include-val', type=str2bool, default=True)
    parser.add_argument('--num-worker', type=int, default=4)
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--work-dir', type=str, default='work_dir')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='train')
    
    return parser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class FallDetectionTrainer:
    def __init__(self, arg):
        self.arg = arg
        self.early_stopping_patience = 20  # Add early stopping patience
        self.setup_environment()
        self.setup_components()
        self.setup_metrics()
        
    def setup_environment(self):
        """Initialize training environment and directories"""
        # Ensure work_dir exists
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            self.save_config()
            
        # Setup device
        self.device = (f'cuda:{self.arg.device[0]}' if isinstance(self.arg.device, list) 
                      else f'cuda:{self.arg.device}' if torch.cuda.is_available() else 'cpu')
        
        self.global_step = 0
        self.best_metrics = {'precision': 0, 'f1': 0, 'recall': 0, 'accuracy': 0}
        self.best_val_loss = float('inf')
        
        # Early stopping parameters
        self.patience = 16  # Number of epochs to wait before early stopping
        self.patience_counter = 0
        self.min_delta = 1e-4  # Minimum change in monitored value to qualify as an improvement
        
    def setup_components(self):
        """Initialize model, optimizer, and related components"""
        # Initialize model
        self.model = self.load_model()
        self.model = self.model.to(self.device)
        
        # Training components
        if self.arg.phase == 'train':
            self.criterion = self.load_loss()
            self.optimizer = self.load_optimizer()
            self.scheduler = self.load_scheduler()
            
        # Initialize data trackers
        self.train_metrics = []
        self.val_metrics = []
        
    def load_data(self, train_subjects, test_subjects):
        """Prepare data loaders for training, validation, and testing."""
        try:
            Feeder = self.import_class(self.arg.feeder)  # Dynamically import the Feeder class
            
            # Prepare dataset builder
            builder = prepare_smartfallmm(self.arg)  # Function to build dataset (ensure it's defined)
            
            # Filter subjects for training and testing
            train_data = filter_subjects(builder, train_subjects)
            test_data = filter_subjects(builder, test_subjects)
            
            # Create DataLoader for training
            self.data_loader = {}
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(
                    **self.arg.train_feeder_args,
                    dataset=train_data
                ),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                pin_memory=True
            )
            
            # Create DataLoader for validation, if applicable
            if self.arg.include_val:
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder(
                        **self.arg.val_feeder_args,
                        dataset=test_data
                    ),
                    batch_size=self.arg.val_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    pin_memory=True
                )
            
            # Create DataLoader for testing
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(
                    **self.arg.test_feeder_args,
                    dataset=test_data
                ),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                pin_memory=True
            )
            
        except Exception as e:
            raise ValueError(f"Error in loading data: {e}")
        
    def setup_metrics(self):
        """Initialize training metrics"""
        self.train_metrics = []
        self.val_metrics = []
        self.train_losses = []
        self.val_losses = []
        self.best_metrics = {'loss': float('inf'), 'precision': 0, 'f1': 0, 'recall': 0, 'accuracy': 0}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def load_model(self):
        """Load model architecture"""
        Model = self.import_class(self.arg.model)
        model = Model(**self.arg.model_args)
        
        if self.arg.weights:
            model.load_state_dict(torch.load(self.arg.weights))
            
        return model
    
    def load_optimizer(self):
        """Initialize optimizer with parameters"""
        if self.arg.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")
            
    def load_scheduler(self):
        """Initialize learning rate scheduler"""
        if self.arg.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=10,
                verbose=True
            )
        elif self.arg.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.arg.num_epoch,
                eta_min=1e-6
            )
        return None
    
    def load_loss(self):
        """Load loss function dynamically"""
        try:
            # Handle torch.nn losses directly
            if self.arg.loss.startswith('torch.nn.'):
                loss_class = getattr(torch.nn, self.arg.loss.split('.')[-1])
            else:
                # For custom losses, use the previous import method
                module_name, _sep, class_str = self.arg.loss.rsplit('.', 1)
                loss_module = __import__(module_name, fromlist=[class_str])
                loss_class = getattr(loss_module, class_str)
            
            loss_args = eval(self.arg.loss_args) if isinstance(self.arg.loss_args, str) else self.arg.loss_args
            return loss_class(**loss_args)
        except Exception as e:
            raise ValueError(f"Failed to load loss function '{self.arg.loss}' with args {self.arg.loss_args}: {e}")    
        
    def compute_metrics(self, probs, targets):
        """
        Compute comprehensive metrics for fall detection
        Args:
            probs: tensor of shape [B] - probabilities between 0 and 1
            targets: tensor of shape [B] - binary labels (0 or 1)
        """
        with torch.no_grad():
            # Ensure inputs have correct shape
            probs = probs.view(-1)
            targets = targets.view(-1)
            
            # Get predictions from probabilities
            predictions = (probs > 0.5).float()
            
            # Move to CPU for metric calculation
            predictions = predictions.cpu()
            targets = targets.cpu()
            
            # Calculate basic metrics
            correct = (predictions == targets).float()
            accuracy = correct.mean().item()
            
            # Handle edge cases where one class might be missing
            if not torch.any(targets == 1) or not torch.any(targets == 0):
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'accuracy': accuracy,
                    'false_alarm_rate': 0.0,
                    'miss_rate': 0.0
                }
            
            # Calculate true/false positives/negatives
            tp = torch.sum((predictions == 1) & (targets == 1)).float()
            tn = torch.sum((predictions == 0) & (targets == 0)).float()
            fp = torch.sum((predictions == 1) & (targets == 0)).float()
            fn = torch.sum((predictions == 0) & (targets == 1)).float()
            
            # Calculate metrics with epsilon to avoid division by zero
            epsilon = 1e-7
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            false_alarm_rate = fp / (fp + tn + epsilon)
            miss_rate = fn / (fn + tp + epsilon)
            
            return {
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'accuracy': accuracy,
                'false_alarm_rate': false_alarm_rate.item(),
                'miss_rate': miss_rate.item()
            }
        
    def train_epoch(self, epoch):
        """Training loop for a single epoch"""
        self.model.train()
        train_metrics = {'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'false_alarm_rate': 0, 'miss_rate': 0}
        total_samples = 0
        
        for inputs, targets, idx in tqdm(self.data_loader['train'], desc=f'Training epoch {epoch}'):
            # Move data to device
            acc_data = inputs['accelerometer'].to(self.device)
            targets = targets.to(self.device).float()
            
            # Forward pass with or without skeleton data
            if 'skeleton' in inputs:
                skl_data = inputs['skeleton'].to(self.device)
                probs, _ = self.model(acc_data, skl_data)
            else:
                probs, _ = self.model(acc_data, None)
            
            # Compute loss using binary cross entropy
            loss = F.binary_cross_entropy(probs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_metrics = self.compute_metrics(probs, targets)
            batch_size = targets.size(0)
            total_samples += batch_size
            
            for key in train_metrics:
                if key == 'loss':
                    train_metrics[key] += loss.item() * batch_size
                else:
                    train_metrics[key] += batch_metrics[key] * batch_size
        
        # Average metrics
        for key in train_metrics:
            train_metrics[key] /= total_samples
            
        return train_metrics

    def validate(self, epoch):
        """Validation loop"""
        self.model.eval()
        val_metrics = {'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'false_alarm_rate': 0, 'miss_rate': 0}
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets, idx in tqdm(self.data_loader['val'], desc=f'Validating epoch {epoch}'):
                # Move data to device
                acc_data = inputs['accelerometer'].to(self.device)
                targets = targets.to(self.device).float()
                
                # Forward pass with or without skeleton data
                if 'skeleton' in inputs:
                    skl_data = inputs['skeleton'].to(self.device)
                    probs, _ = self.model(acc_data, skl_data)
                else:
                    probs, _ = self.model(acc_data, None)
                
                # Compute loss
                loss = F.binary_cross_entropy(probs, targets)
                
                # Update metrics
                batch_metrics = self.compute_metrics(probs, targets)
                batch_size = targets.size(0)
                total_samples += batch_size
                
                for key in val_metrics:
                    if key == 'loss':
                        val_metrics[key] += loss.item() * batch_size
                    else:
                        val_metrics[key] += batch_metrics[key] * batch_size
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= total_samples
            
        return val_metrics
            
    def save_model(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        model_name = self.arg.model.split('.')[-1]
        current_fold = getattr(self, 'current_fold', None)
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Determine save directory (fold-specific if in cross-validation)
        save_dir = os.path.join(self.arg.work_dir, f'fold_{current_fold}') if current_fold is not None else self.arg.work_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Save checkpoint with metrics
        metrics_str = f"_f1_{metrics['f1']:.4f}_loss_{metrics['loss']:.4f}"
        checkpoint_name = f"{model_name}_epoch_{epoch}{metrics_str}.pt"
        torch.save(state_dict, os.path.join(save_dir, checkpoint_name))
        
        # Save best model separately
        if is_best:
            # Save complete model
            best_name = f"{model_name}_best_f1_{metrics['f1']:.4f}_loss_{metrics['loss']:.4f}.pt"
            best_path = os.path.join(save_dir, best_name)
            torch.save(state_dict, best_path)
            
            # Save weights-only version
            weights_name = f"{model_name}_best_weights_f1_{metrics['f1']:.4f}_loss_{metrics['loss']:.4f}.pt"
            weights_path = os.path.join(save_dir, weights_name)
            torch.save(self.model.state_dict(), weights_path)
            
            self.print_log(f"Saved best model for fold {current_fold if current_fold is not None else ''} at epoch {epoch}:")
            self.print_log(f"Complete model: {best_path}")
            self.print_log(f"Weights only: {weights_path}")
            self.print_log(f"Val Loss: {metrics['loss']:.4f}")
            self.print_log(f"Val F1: {metrics['f1']:.4f}")

    def save_results(self):
        """Save training results and metrics"""
        try:
            # Initialize empty lists if they don't exist
            if not hasattr(self, 'train_losses'):
                self.train_losses = []
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            if not hasattr(self, 'train_metrics'):
                self.train_metrics = []
            if not hasattr(self, 'val_metrics'):
                self.val_metrics = []

            # Get the maximum length among all lists
            max_len = max(
                len(self.train_losses),
                len(self.val_losses),
                len(self.train_metrics),
                len(self.val_metrics)
            )

            # Extend all lists to match the maximum length
            self.train_losses.extend([None] * (max_len - len(self.train_losses)))
            self.val_losses.extend([None] * (max_len - len(self.val_losses)))
            self.train_metrics.extend([None] * (max_len - len(self.train_metrics)))
            self.val_metrics.extend([None] * (max_len - len(self.val_metrics)))
            
            # Create DataFrame with all available metrics
            metrics_df = pd.DataFrame({
                'epoch': range(max_len),
                'train_loss': self.train_losses,
                'train_accuracy': [m.get('accuracy', None) if m else None for m in self.train_metrics],
                'train_precision': [m.get('precision', None) if m else None for m in self.train_metrics],
                'train_recall': [m.get('recall', None) if m else None for m in self.train_metrics],
                'train_f1': [m.get('f1', None) if m else None for m in self.train_metrics],
                'train_false_alarm_rate': [m.get('false_alarm_rate', None) if m else None for m in self.train_metrics],
                'train_miss_rate': [m.get('miss_rate', None) if m else None for m in self.train_metrics],
                'val_loss': self.val_losses,
                'val_accuracy': [m.get('accuracy', None) if m else None for m in self.val_metrics],
                'val_precision': [m.get('precision', None) if m else None for m in self.val_metrics],
                'val_recall': [m.get('recall', None) if m else None for m in self.val_metrics],
                'val_f1': [m.get('f1', None) if m else None for m in self.val_metrics],
                'val_false_alarm_rate': [m.get('false_alarm_rate', None) if m else None for m in self.val_metrics],
                'val_miss_rate': [m.get('miss_rate', None) if m else None for m in self.val_metrics]
            })
            
            # Save metrics to CSV
            metrics_path = os.path.join(self.arg.work_dir, 'metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            self.print_log(f"Saved metrics to {metrics_path}")
            
            # Plot training curves if we have data
            if len(self.train_metrics) > 0:
                self.plot_training_curves()
                
        except Exception as e:
            self.print_log(f"Error saving results: {str(e)}")
            traceback.print_exc()  # Print the full traceback for debugging
    
    def plot_training_curves(self):
        """Generate and save training visualization plots"""
        try:
            metrics_to_plot = [
                ('loss', 'Loss'),
                ('accuracy', 'Accuracy'),
                ('precision', 'Precision'),
                ('recall', 'Recall'),
                ('f1', 'F1 Score'),
                ('false_alarm_rate', 'False Alarm Rate'),
                ('miss_rate', 'Miss Rate')
            ]
            
            for metric_key, metric_name in metrics_to_plot:
                plt.figure(figsize=(10, 6))
                epochs = range(len(self.train_metrics))
                
                # Get metric values
                train_values = [m.get(metric_key, None) if m else None for m in self.train_metrics]
                val_values = [m.get(metric_key, None) if m else None for m in self.val_metrics]
                
                # Plot metric
                if train_values and any(v is not None for v in train_values):
                    plt.plot(epochs, train_values, 'b-', label=f'Training {metric_name}')
                if val_values and any(v is not None for v in val_values):
                    plt.plot(epochs, val_values, 'r-', label=f'Validation {metric_name}')
                
                plt.xlabel('Epoch')
                plt.ylabel(metric_name)
                plt.title(f'Training and Validation {metric_name}')
                plt.legend()
                plt.grid(True)
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(self.arg.work_dir, f'{metric_key}_curve.png'))
                plt.close()
                self.print_log(f"Saved {metric_name} curve plot to {os.path.join(self.arg.work_dir, f'{metric_key}_curve.png')}")
            
        except Exception as e:
            self.print_log(f"Error plotting training curves: {str(e)}")
    
    def create_folds(self, subjects, val_size=3):
        """Create folds for cross validation with fixed validation assignments"""
        # Define fixed fold assignments with validation subjects
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
            # Training subjects are all subjects not in validation
            train_subjects = [s for s in subjects if s not in val_subjects]
            folds.append((train_subjects, val_subjects))
            
            # Log fold information
            fold_num = len(folds)
            self.print_log(f"\nCreated {fold_desc}")
            self.print_log(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
            self.print_log(f"Training subjects ({len(train_subjects)}): {train_subjects}")
    
        return folds


    def start(self):
        """Start training process"""
        if self.arg.phase == 'train':
            try:
                # Phase 1: Cross-validation
                print("\nPhase 1: Leave-Three-Out Cross-validation")
                
                # Get all subjects
                all_subjects = list(range(29, 47))  # Subjects from 29 to 46
                
                # Create folds
                folds = self.create_folds(all_subjects, val_size=3)
                epoch_metrics = []
                best_fold_metrics = []
                
                # Train on each fold
                for fold, (train_subjects, val_subjects) in enumerate(folds, 1):
                    print(f"\nFold {fold}/5")
                    print(f"Training subjects ({len(train_subjects)}): {train_subjects}")
                    print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
                    
                    # Set current fold for model saving
                    self.current_fold = fold
                    
                    # Reset model and metrics for this fold
                    self.setup_components()
                    self.setup_metrics()
                    
                    # Load data for this fold
                    self.load_data(train_subjects, val_subjects)
                    
                    fold_train_losses = []
                    fold_val_losses = []
                    fold_metrics = []
                    best_fold_f1 = 0
                    best_fold_state = None
                    
                    for epoch in range(self.arg.num_epoch):
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                            
                            train_metrics = self.train_epoch(epoch)
                            val_metrics = self.validate(epoch)
                            
                            # Store losses and metrics
                            fold_train_losses.append(train_metrics['loss'])
                            fold_val_losses.append(val_metrics['loss'])
                            fold_metrics.append({**val_metrics, 'epoch': epoch})
                            
                            # Track best model for this fold
                            if val_metrics['f1'] > best_fold_f1:
                                best_fold_f1 = val_metrics['f1']
                                best_fold_state = {
                                    'epoch': epoch,
                                    'state_dict': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'scheduler': self.scheduler.state_dict(),
                                    'metrics': val_metrics
                                }
                            
                            # Store metrics for optimal epoch calculation
                            epoch_metrics.append({
                                'fold': fold,
                                'epoch': epoch,
                                'train_loss': train_metrics['loss'],
                                'val_loss': val_metrics['loss'],
                                'val_f1': val_metrics['f1']
                            })
                            
                            # Print epoch metrics
                            self.print_epoch_metrics(epoch, train_metrics, val_metrics)
                    
                    # Save fold results
                    self.save_fold_results(fold, fold_train_losses, fold_val_losses, fold_metrics)
                    
                    # Save best model for this fold
                    if best_fold_state:
                        best_fold_metrics.append({
                            'fold': fold,
                            'f1': best_fold_f1,
                            'epoch': best_fold_state['epoch'],
                            'state_dict': best_fold_state['state_dict']
                        })
                        
                        # Save best fold model
                        fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
                        model_name = self.arg.model.split('.')[-1]
                        best_name = f"{model_name}_fold{fold}_best_f1_{best_fold_f1:.4f}.pt"
                        torch.save(best_fold_state, os.path.join(fold_dir, best_name))
                        self.print_log(f"Saved best model for fold {fold} (F1: {best_fold_f1:.4f})")
            
                # Find best performing fold
                best_fold = max(best_fold_metrics, key=lambda x: x['f1'])
                self.print_log(f"\nBest performing fold: {best_fold['fold']} (F1: {best_fold['f1']:.4f})")
                
                # Phase 2: Train final model using best fold's configuration
                print("\nPhase 2: Training final model on all data")
                self.setup_components()  # Reset model and optimizer
                
                # Load the best fold's weights
                self.model.load_state_dict(best_fold['state_dict'])
                self.print_log(f"Loaded weights from best fold (Fold {best_fold['fold']})")
                
                # Train on full dataset for the same number of epochs as best fold
                self.load_data(all_subjects, all_subjects)  # Use all subjects
                best_epoch = best_fold['epoch']
                
                for epoch in range(best_epoch + 1):
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                        train_metrics = self.train_epoch(epoch)
                        self.print_epoch_metrics(epoch, train_metrics, None)
            
                # Save final model
                final_state = {
                    'epoch': best_epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'metrics': train_metrics,
                    'best_fold': best_fold['fold']
                }
                model_name = self.arg.model.split('.')[-1]
                final_name = f"{model_name}_final_from_fold{best_fold['fold']}.pt"
                torch.save(final_state, os.path.join(self.arg.work_dir, final_name))
                self.print_log(f"Saved final model: {final_name}")
                
                # Save overall results
                self.save_results()
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                traceback.print_exc()
                self.save_results()  # Save results even if there's an error
        else:
            self.test()
    
    def log_progress(self, epoch, train_metrics, val_metrics=None):
        """Log training progress"""
        log_str = f'\nEpoch {epoch}:\n'
        log_str += f"Training - Loss: {train_metrics['loss']:.4f}, "
        log_str += f"Accuracy: {train_metrics['accuracy']:.4f}, "
        log_str += f"Precision: {train_metrics['precision']:.4f}, "
        log_str += f"Recall: {train_metrics['recall']:.4f}, "
        log_str += f"F1: {train_metrics['f1']:.4f}\n"
        
        if val_metrics:
            log_str += f"Validation - Loss: {val_metrics['loss']:.4f}, "
            log_str += f"Accuracy: {val_metrics['accuracy']:.4f}, "
            log_str += f"Precision: {val_metrics['precision']:.4f}, "
            log_str += f"Recall: {val_metrics['recall']:.4f}, "
            log_str += f"F1: {val_metrics['f1']:.4f}\n"
            
        self.print_log(log_str)

    def calculate_optimal_epoch(self, epoch_metrics):
        """Calculate optimal epoch based on validation metrics"""
        # Group metrics by epoch and calculate average F1 score
        epoch_avg_metrics = {}
        for metric in epoch_metrics:
            epoch = metric['epoch']
            if epoch not in epoch_avg_metrics:
                epoch_avg_metrics[epoch] = {'val_f1': [], 'val_loss': []}
            epoch_avg_metrics[epoch]['val_f1'].append(metric['val_f1'])
            epoch_avg_metrics[epoch]['val_loss'].append(metric['val_loss'])
        
        # Calculate average metrics for each epoch
        avg_metrics = {
            epoch: {
                'val_f1': np.mean(metrics['val_f1']),
                'val_loss': np.mean(metrics['val_loss'])
            }
            for epoch, metrics in epoch_avg_metrics.items()
        }
        
        # Find epoch with best average F1 score
        best_epoch = max(avg_metrics.keys(), key=lambda e: avg_metrics[e]['val_f1'])
        
        return best_epoch

    def print_epoch_metrics(self, epoch, train_metrics, val_metrics):
        """Print metrics for current epoch"""
        # Format training metrics
        train_msg = (f"Epoch {epoch} - Train: "
                    f"Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.4f}, "
                    f"Prec: {train_metrics['precision']:.4f}, "
                    f"Rec: {train_metrics['recall']:.4f}, "
                    f"F1: {train_metrics['f1']:.4f}, "
                    f"FAR: {train_metrics['false_alarm_rate']:.4f}, "
                    f"MR: {train_metrics['miss_rate']:.4f}")
        
        # Format validation metrics if available
        if val_metrics and any(val_metrics.values()):
            val_msg = (f"Val: "
                      f"Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"Prec: {val_metrics['precision']:.4f}, "
                      f"Rec: {val_metrics['recall']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}, "
                      f"FAR: {val_metrics['false_alarm_rate']:.4f}, "
                      f"MR: {val_metrics['miss_rate']:.4f}")
            self.print_log(f"{train_msg} | {val_msg}")
        else:
            self.print_log(train_msg)

    @staticmethod
    def import_class(name):
        """Dynamically import a class"""
        mod_str, _sep, class_str = name.rpartition('.')
        __import__(mod_str)
        try:
            return getattr(sys.modules[mod_str], class_str)
        except AttributeError:
            raise ImportError(f'Class {class_str} cannot be found')

    def save_fold_results(self, fold, train_losses, val_losses, metrics):
        """Save results for each fold"""
        # Create fold directory
        fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save metrics
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': metrics
        }
        
        with open(os.path.join(fold_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log best metrics for this fold
        best_metrics = max(metrics, key=lambda x: x['f1'])
        self.print_log(f"\nBest metrics for fold {fold}:")
        self.print_log(f"Epoch: {best_metrics.get('epoch', 'N/A')}")
        self.print_log(f"Val Loss: {best_metrics['loss']:.4f}")
        self.print_log(f"F1: {best_metrics['f1']:.4f}")
        self.print_log(f"Precision: {best_metrics['precision']:.4f}")
        self.print_log(f"Recall: {best_metrics['recall']:.4f}")
        self.print_log(f"Results saved to: {fold_dir}")
        
        return best_metrics

    def train_fold(self, fold):
        """Train a single fold."""
        self.current_fold = fold  # Track current fold
        
        # Initialize fold metrics
        fold_train_losses = []
        fold_val_losses = []
        fold_metrics = []
        best_fold_f1 = float('-inf')
        epochs_without_improvement = 0
        early_stop = False
        best_fold_state = None
        
        for epoch in range(self.arg.num_epoch):
            if early_stop:
                self.print_log(f'Early stopping triggered for fold {fold}')
                break
                
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            # Add epoch to metrics
            val_metrics['epoch'] = epoch
            
            # Save metrics
            fold_train_losses.append(train_metrics['loss'])
            fold_val_losses.append(val_metrics['loss'])
            fold_metrics.append(val_metrics)
            
            # Early stopping check for this fold
            current_f1 = val_metrics.get('f1', 0)
            if current_f1 > best_fold_f1:
                best_fold_f1 = current_f1
                epochs_without_improvement = 0
                best_fold_state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'metrics': val_metrics
                }
                self.save_model(epoch, val_metrics, is_best=True)
                self.print_log(f'Fold {fold} - New best F1: {current_f1:.4f}')
            else:
                epochs_without_improvement += 1
                self.print_log(f'Fold {fold} - Epochs without improvement: {epochs_without_improvement}/{self.early_stopping_patience}')
                if epochs_without_improvement >= self.early_stopping_patience:
                    early_stop = True
                    self.print_log(f'Early stopping triggered for fold {fold} after {epoch + 1} epochs')
                    break
            
            # Print epoch metrics
            self.print_epoch_metrics(epoch, train_metrics, val_metrics)
        
        # Load best model state for this fold
        if best_fold_state is not None:
            self.model.load_state_dict(best_fold_state['state_dict'])
            self.optimizer.load_state_dict(best_fold_state['optimizer'])
            self.scheduler.load_state_dict(best_fold_state['scheduler'])
            self.print_log(f'Loaded best model state for fold {fold} (F1: {best_fold_f1:.4f})')
        
        return fold_train_losses, fold_val_losses, fold_metrics

    def cross_validate(self):
        """Perform k-fold cross validation."""
        all_fold_metrics = []
        
        for fold in range(self.arg.num_fold):
            self.print_log(f'\nTraining Fold {fold + 1}/{self.arg.num_fold}')
            self.print_log('-' * 50)
            
            # Reset model and optimizer for each fold
            self.setup_components()
            
            # Setup data loaders for this fold
            self.setup_fold_data(fold)
            
            # Train fold
            train_losses, val_losses, metrics = self.train_fold(fold)
            
            # Save fold results and get best metrics
            best_fold_metrics = self.save_fold_results(fold, train_losses, val_losses, metrics)
            all_fold_metrics.append(best_fold_metrics)
        
        # Calculate and log average metrics across all folds
        self.print_log('\nAverage Metrics Across All Folds:')
        self.print_log('-' * 50)
        avg_metrics = {
            'loss': np.mean([m['loss'] for m in all_fold_metrics]),
            'f1': np.mean([m['f1'] for m in all_fold_metrics]),
            'precision': np.mean([m['precision'] for m in all_fold_metrics]),
            'recall': np.mean([m['recall'] for m in all_fold_metrics])
        }
        
        self.print_log(f"Average Val Loss: {avg_metrics['loss']:.4f}")
        self.print_log(f"Average F1: {avg_metrics['f1']:.4f}")
        self.print_log(f"Average Precision: {avg_metrics['precision']:.4f}")
        self.print_log(f"Average Recall: {avg_metrics['recall']:.4f}")
        
        # Save average metrics
        with open(os.path.join(self.arg.work_dir, 'average_metrics.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=4)
    
    def test(self):
        """Evaluate model on test set"""
        self.model.eval()
        metrics = []
        confusion = np.zeros((2, 2))  # Binary classification
        
        with torch.no_grad():
            for inputs, targets, idx in tqdm(self.data_loader['test'], desc='Testing'):
                # Move data to device
                acc_data = inputs['accelerometer'].to(self.device)
                targets = targets.to(self.device).float()
                
                # Forward pass with or without skeleton data
                if 'skeleton' in inputs:
                    skl_data = inputs['skeleton'].to(self.device)
                    probs, _ = self.model(acc_data, skl_data)
                else:
                    probs, _ = self.model(acc_data, None)
                    
                predictions = (probs > 0.5).float()
                
                # Update confusion matrix
                for pred, target in zip(predictions.cpu(), targets.cpu()):
                    confusion[int(target), int(pred)] += 1
                
                # Compute metrics
                batch_metrics = self.compute_metrics(probs, targets)
                metrics.append(batch_metrics)
        
        # Aggregate metrics
        final_metrics = {
            key: np.mean([m[key] for m in metrics]) 
            for key in metrics[0].keys()
        }
        
        # Calculate confusion matrix metrics
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Add additional metrics
        final_metrics.update({
            'specificity': specificity,
            'npv': npv,
            'confusion_matrix': confusion.tolist()
        })
        
        # Save results
        results = {
            'metrics': final_metrics,
            'confusion_matrix': confusion.tolist()
        }
        
        save_path = os.path.join(self.arg.work_dir, 'test_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log results
        self.print_log('\nTest Results:')
        self.print_log(f'Precision: {final_metrics["precision"]:.4f}')
        self.print_log(f'Recall: {final_metrics["recall"]:.4f}')
        self.print_log(f'F1-Score: {final_metrics["f1"]:.4f}')
        self.print_log(f'False Alarm Rate: {final_metrics["false_alarm_rate"]:.4f}')
        self.print_log(f'Specificity: {final_metrics["specificity"]:.4f}')
        self.print_log(f'NPV: {final_metrics["npv"]:.4f}')
        
        # Plot confusion matrix
        self.plot_confusion_matrix(confusion)
    
    def plot_confusion_matrix(self, confusion):
        """Plot and save confusion matrix visualization"""
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        
        # Add labels
        classes = ['Non-Fall', 'Fall']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = confusion.max() / 2.
        for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
            plt.text(j, i, format(confusion[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.arg.work_dir, 'confusion_matrix.png'))
        plt.close()
    
    def print_log(self, msg, print_time=True):
        """Print and save log messages"""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            msg = f"[ {localtime} ] {msg}"
        print(msg)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(msg, file=f)
    
    def save_config(self):
        """Save configuration file"""
        shutil.copy2(
            self.arg.config,
            os.path.join(self.arg.work_dir, os.path.basename(self.arg.config))
        )

if __name__ == "__main__":
    parser = get_args()
    # Load config
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)

        # Validate arguments
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    # Initialize training
    init_seed(arg.seed)
    trainer = FallDetectionTrainer(arg)
    trainer.start()
class FallDetectionTrainer(FallDetectionTrainer):
    """Additional methods for FallDetectionTrainer"""
    
    def load_data(self, train_subjects, test_subjects):
        """Load and prepare datasets"""
        Feeder = self.import_class(self.arg.feeder)
        
        # Prepare datasets
        builder = prepare_smartfallmm(self.arg)
        
        # Training data
        train_data = filter_subjects(builder, train_subjects)
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(
                **self.arg.train_feeder_args,
                dataset=train_data
            ),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            pin_memory=True
        )
        
        # Validation data
        if self.arg.include_val:
            val_data = filter_subjects(builder, test_subjects)
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(
                    **self.arg.val_feeder_args,
                    dataset=val_data
                ),
                batch_size=self.arg.val_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                pin_memory=True
            )
    
    def plot_training_curves(self):
        """Generate and save training visualization plots"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot losses
        epochs = range(len(self.train_metrics))
        train_loss = [m['loss'] for m in self.train_metrics]
        val_loss = [m['loss'] for m in self.val_metrics]
        
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot metrics
        train_precision = [m['precision'] for m in self.train_metrics]
        train_recall = [m['recall'] for m in self.train_metrics]
        val_precision = [m['precision'] for m in self.val_metrics]
        val_recall = [m['recall'] for m in self.val_metrics]
        
        ax2.plot(epochs, train_precision, 'b-', label='Train Precision')
        ax2.plot(epochs, train_recall, 'b--', label='Train Recall')
        ax2.plot(epochs, val_precision, 'r-', label='Val Precision')
        ax2.plot(epochs, val_recall, 'r--', label='Val Recall')
        ax2.set_title('Training and Validation Metrics')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.arg.work_dir, 'training_curves.png'))
        plt.close()
        self.print_log(f"Saved training curves plot to {os.path.join(self.arg.work_dir, 'training_curves.png')}")
    
    def test(self):
        """Test loop"""
        self.model.eval()
        metrics = []
        confusion = np.zeros((2, 2))  # Binary classification
        
        with torch.no_grad():
            for inputs, targets, idx in tqdm(self.data_loader['test'], desc='Testing'):
                # Move data to device
                acc_data = inputs['accelerometer'].to(self.device)
                targets = targets.to(self.device).float()
                
                # Forward pass with or without skeleton data
                if 'skeleton' in inputs:
                    skl_data = inputs['skeleton'].to(self.device)
                    probs, _ = self.model(acc_data, skl_data)
                else:
                    probs, _ = self.model(acc_data, None)
                    
                predictions = (probs > 0.5).float()
                
                # Update confusion matrix
                for pred, target in zip(predictions.cpu(), targets.cpu()):
                    confusion[int(target), int(pred)] += 1
                
                # Compute metrics
                batch_metrics = self.compute_metrics(probs, targets)
                metrics.append(batch_metrics)
        
        # Aggregate metrics
        final_metrics = {
            key: np.mean([m[key] for m in metrics]) 
            for key in metrics[0].keys()
        }
        
        # Calculate confusion matrix metrics
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Add additional metrics
        final_metrics.update({
            'specificity': specificity,
            'npv': npv,
            'confusion_matrix': confusion.tolist()
        })
        
        # Save results
        results = {
            'metrics': final_metrics,
            'confusion_matrix': confusion.tolist()
        }
        
        save_path = os.path.join(self.arg.work_dir, 'test_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log results
        self.print_log('\nTest Results:')
        self.print_log(f'Precision: {final_metrics["precision"]:.4f}')
        self.print_log(f'Recall: {final_metrics["recall"]:.4f}')
        self.print_log(f'F1-Score: {final_metrics["f1"]:.4f}')
        self.print_log(f'False Alarm Rate: {final_metrics["false_alarm_rate"]:.4f}')
        self.print_log(f'Specificity: {final_metrics["specificity"]:.4f}')
        self.print_log(f'NPV: {final_metrics["npv"]:.4f}')
        
        # Plot confusion matrix
        self.plot_confusion_matrix(confusion)
    
    def plot_confusion_matrix(self, confusion):
        """Plot and save confusion matrix visualization"""
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        
        # Add labels
        classes = ['Non-Fall', 'Fall']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = confusion.max() / 2.
        for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
            plt.text(j, i, format(confusion[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.arg.work_dir, 'confusion_matrix.png'))
        plt.close()
    
    def print_log(self, msg, print_time=True):
        """Print and save log messages"""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            msg = f"[ {localtime} ] {msg}"
        print(msg)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(msg, file=f)
    
    def save_config(self):
        """Save configuration file"""
        shutil.copy2(
            self.arg.config,
            os.path.join(self.arg.work_dir, os.path.basename(self.arg.config))
        )