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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, classification_report
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
    """Initialize random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # This may slow down training but ensures reproducibility

class FallDetectionTrainer:
    def __init__(self, arg):
        self.arg = arg
        self.setup_environment()
        self.setup_components()
        self.setup_metrics()
        
    def setup_environment(self):
        """Initialize training environment and directories"""
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            self.save_config()
            
        self.device = (f'cuda:{self.arg.device[0]}' if isinstance(self.arg.device, list) 
                      else f'cuda:{self.arg.device}' if torch.cuda.is_available() else 'cpu')
        
        self.global_step = 0
        self.best_metrics = {'precision': 0, 'f1': 0, 'recall': 0, 'accuracy': 0}
        self.best_val_loss = float('inf')
        
        self.patience = 16
        self.patience_counter = 0
        self.min_delta = 1e-4
        
    def setup_components(self):
        """Initialize model, optimizer, and related components"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = {}
        self.model = self.load_model()
        self.model = self.model.to(self.device)
        
        if self.arg.phase == 'train':
            self.criterion = self.load_loss()
            self.optimizer = self.load_optimizer()
            self.scheduler = self.load_scheduler()
            
        self.train_metrics = []
        self.val_metrics = []
        
    def load_data(self, train_subjects, test_subjects):
        """Prepare data loaders for training, validation, and testing."""
        try:
            Feeder = self.import_class(self.arg.feeder)
            builder = prepare_smartfallmm(self.arg)
            train_data = filter_subjects(builder, train_subjects)
            test_data = filter_subjects(builder, test_subjects)
            
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
            if self.arg.loss.startswith('torch.nn.'):
                loss_class = getattr(torch.nn, self.arg.loss.split('.')[-1])
            else:
                module_name, _sep, class_str = self.arg.loss.rsplit('.', 1)
                loss_module = __import__(module_name, fromlist=[class_str])
                loss_class = getattr(loss_module, class_str)
            
            loss_args = eval(self.arg.loss_args) if isinstance(self.arg.loss_args, str) else self.arg.loss_args
            return loss_class(**loss_args)
        except Exception as e:
            raise ValueError(f"Failed to load loss function '{self.arg.loss}' with args {self.arg.loss_args}: {e}")    
        
    def compute_metrics(self, probs, targets):
        """Compute window-level precision, recall, F1 score."""
        predictions = (probs > 0.5).float()
        tp = torch.sum((predictions == 1) & (targets == 1)).float()
        fp = torch.sum((predictions == 1) & (targets == 0)).float()
        fn = torch.sum((predictions == 0) & (targets == 1)).float()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        accuracy = torch.sum(predictions == targets).float() / targets.size(0)
        
        far = fp / (fp + tp + 1e-10)  
        mr = fn / (fn + tp + 1e-10)   
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'accuracy': accuracy.item(),
            'false_alarm_rate': far.item(),
            'miss_rate': mr.item()
        }

    def aggregate_trial_predictions(self, aggregator):
        """
        Aggregates window-level predictions into trial-level predictions (average pooling).
        aggregator: Dict[int, Dict[str, Any]] -> {trial_id: {'probs': [list_of_probs], 'label': 0_or_1}}
        
        Returns:
            trial_probs: list of aggregated probabilities (average across windows)
            trial_labels: list of single label per trial
        """
        trial_probs = []
        trial_labels = []
        for trial_id, info in aggregator.items():
            # Average aggregator
            avg_prob = np.mean(info['probs'])
            trial_probs.append(avg_prob)
            trial_labels.append(info['label'])
        return np.array(trial_probs), np.array(trial_labels)

    def compute_trial_metrics(self, trial_probs, trial_labels):
        """
        Compute trial-level metrics from aggregated probabilities and single trial label.
        trial_probs: np.array of shape [Ntrials]
        trial_labels: np.array of shape [Ntrials]
        """
        predictions = (trial_probs > 0.5).astype(np.float32)
        tp = np.sum((predictions == 1) & (trial_labels == 1)).astype(float)
        fp = np.sum((predictions == 1) & (trial_labels == 0)).astype(float)
        fn = np.sum((predictions == 0) & (trial_labels == 1)).astype(float)
        tn = np.sum((predictions == 0) & (trial_labels == 0)).astype(float)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        
        far = fp / (fp + tp + 1e-10)
        mr = fn / (fn + tp + 1e-10)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'false_alarm_rate': far,
            'miss_rate': mr
        }

    def train_epoch(self, epoch):
        self.model.train()
        train_metrics = {'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'false_alarm_rate': 0, 'miss_rate': 0}
        total_samples = 0
        
        for inputs, targets, idx in tqdm(self.data_loader['train'], desc=f'Training epoch {epoch}'):
            acc_data = inputs['accelerometer'].to(self.device)
            targets = targets.to(self.device).float()
            
            # Model returns probabilities of shape [batch_size]
            probs = self.model(acc_data.float())
            loss = F.binary_cross_entropy(probs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            batch_metrics = self.compute_metrics(probs, targets)
            batch_size = targets.size(0)
            total_samples += batch_size
            
            for key in train_metrics:
                if key == 'loss':
                    train_metrics[key] += loss.item() * batch_size
                else:
                    train_metrics[key] += batch_metrics[key] * batch_size
        
        for key in train_metrics:
            train_metrics[key] /= total_samples
        
        self.train_losses.append(train_metrics['loss'])
        self.train_metrics.append(train_metrics)
        
        return train_metrics

    def validate(self, epoch):
        """Validation loop with window-level AND trial-level aggregation."""
        self.model.eval()
        val_metrics = {'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'false_alarm_rate': 0, 'miss_rate': 0}
        total_samples = 0
        trial_aggregator = defaultdict(lambda: {'probs': [], 'label': None})
        
        with torch.no_grad():
            for inputs, targets, idx in tqdm(self.data_loader['val'], desc=f'Validating epoch {epoch}'):
                acc_data = inputs['accelerometer'].to(self.device)
                targets = targets.to(self.device).float()
                
                # Model returns probabilities of shape [batch_size]
                probs = self.model(acc_data.float())
                loss = F.binary_cross_entropy(probs, targets)
                
                batch_metrics = self.compute_metrics(probs, targets)
                batch_size = targets.size(0)
                total_samples += batch_size
                
                for key in val_metrics:
                    if key == 'loss':
                        val_metrics[key] += loss.item() * batch_size
                    else:
                        val_metrics[key] += batch_metrics[key] * batch_size
                
                # Store window-level predictions for later trial-level aggregation
                for i, trial_id in enumerate(idx):
                    trial_aggregator[trial_id.item()]['probs'].append(probs[i].item())
                    trial_aggregator[trial_id.item()]['label'] = targets[i].item()
        
        # Normalize metrics by total samples
        for key in val_metrics:
            val_metrics[key] /= total_samples
        
        # Compute trial-level metrics
        trial_probs, trial_labels = self.aggregate_trial_predictions(trial_aggregator)
        trial_metrics = self.compute_trial_metrics(trial_probs, trial_labels)
        
        # Add trial-level metrics to val_metrics
        val_metrics.update({f'trial_{k}': v for k, v in trial_metrics.items()})
        
        self.val_losses.append(val_metrics['loss'])
        self.val_metrics.append(val_metrics)
        
        return val_metrics

    def save_model(self, epoch, metrics, is_best=False):
        model_name = self.arg.model.split('.')[-1]
        current_fold = getattr(self, 'current_fold', None)
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'metrics': metrics
        }
        
        save_dir = os.path.join(self.arg.work_dir, f'fold_{current_fold}') if current_fold is not None else self.arg.work_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if is_best:
            best_name = f"{model_name}_best_f1_{metrics['f1']:.4f}_loss_{metrics['loss']:.4f}.pt"
            best_path = os.path.join(save_dir, best_name)
            torch.save(state_dict, best_path)
            
            weights_name = f"{model_name}_best_weights_f1_{metrics['f1']:.4f}_loss_{metrics['loss']:.4f}.pt"
            weights_path = os.path.join(save_dir, weights_name)
            torch.save(self.model.state_dict(), weights_path)
            
            self.print_log(f"Saved best model for fold {current_fold if current_fold is not None else ''} at epoch {epoch}:")
            self.print_log(f"Complete model: {best_path}")
            self.print_log(f"Weights only: {weights_path}")
            self.print_log(f"Val Loss: {metrics['loss']:.4f}")
            self.print_log(f"Val F1: {metrics['f1']:.4f}")
            
    def save_results(self):
        try:
            if not hasattr(self, 'train_losses'):
                self.train_losses = []
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            if not hasattr(self, 'train_metrics'):
                self.train_metrics = []
            if not hasattr(self, 'val_metrics'):
                self.val_metrics = []
            
            max_len = max(
                len(self.train_losses),
                len(self.val_losses),
                len(self.train_metrics),
                len(self.val_metrics)
            )
            
            self.train_losses.extend([None] * (max_len - len(self.train_losses)))
            self.val_losses.extend([None] * (max_len - len(self.val_losses)))
            self.train_metrics.extend([None] * (max_len - len(self.train_metrics)))
            self.val_metrics.extend([None] * (max_len - len(self.val_metrics)))
            
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
                'val_miss_rate': [m.get('miss_rate', None) if m else None for m in self.val_metrics],
                
                # Additional columns if we have trial-level aggregator metrics in val_metrics
                'val_trial_f1': [m.get('trial_f1', None) if m else None for m in self.val_metrics],
                'val_trial_precision': [m.get('trial_precision', None) if m else None for m in self.val_metrics],
                'val_trial_recall': [m.get('trial_recall', None) if m else None for m in self.val_metrics],
                'val_trial_accuracy': [m.get('trial_accuracy', None) if m else None for m in self.val_metrics],
                'val_trial_false_alarm_rate': [m.get('trial_false_alarm_rate', None) if m else None for m in self.val_metrics],
                'val_trial_miss_rate': [m.get('trial_miss_rate', None) if m else None for m in self.val_metrics],
            })
            
            metrics_path = os.path.join(self.arg.work_dir, 'metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            self.print_log(f"Saved metrics to {metrics_path}")
            
            if len(self.train_metrics) > 0:
                self.plot_training_curves()
                
        except Exception as e:
            self.print_log(f"Error saving results: {str(e)}")
            traceback.print_exc()

    def plot_training_curves(self):
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
                
                train_values = [m.get(metric_key, None) if m else None for m in self.train_metrics]
                val_values = [m.get(metric_key, None) if m else None for m in self.val_metrics]
                
                if train_values and any(v is not None for v in train_values):
                    plt.plot(epochs, train_values, 'b-', label=f'Training {metric_name}')
                if val_values and any(v is not None for v in val_values):
                    plt.plot(epochs, val_values, 'r-', label=f'Validation {metric_name}')
                
                plt.xlabel('Epoch')
                plt.ylabel(metric_name)
                plt.title(f'Training and Validation {metric_name}')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.arg.work_dir, f'{metric_key}_curve.png'))
                plt.close()
                self.print_log(f"Saved {metric_name} curve plot to {os.path.join(self.arg.work_dir, f'{metric_key}_curve.png')}")
            
        except Exception as e:
            self.print_log(f"Error plotting training curves: {str(e)}")

    def create_folds(self, subjects, val_size=3):
        fold_assignments = [
            ([43, 35, 36], "Fold 1: 38.3% falls"),
            ([44, 34, 32], "Fold 2: 39.7% falls"),
            ([45, 37, 38], "Fold 3: 44.8% falls"),
            ([46, 29, 31], "Fold 4: 41.4% falls"),
            ([30, 39], "Fold 5: 43.3% falls")
        ]
        
        folds = []
        for val_subjects, fold_desc in fold_assignments:
            train_subjects = [s for s in subjects if s not in val_subjects]
            folds.append((train_subjects, val_subjects))
            
            fold_num = len(folds)
            self.print_log(f"\nCreated {fold_desc}")
            self.print_log(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
            self.print_log(f"Training subjects ({len(train_subjects)}): {train_subjects}")
    
        return folds

    def start(self):
        if self.arg.phase == 'train':
            try:
                init_seed(42)
                
                print("\nPhase 1: Leave-Three-Out Cross-validation")
                
                all_subjects = list(range(29, 47))
                folds = self.create_folds(all_subjects, val_size=3)
                epoch_metrics = []
                best_fold_metrics = []
                all_fold_results = []
                
                for fold, (train_subjects, val_subjects) in enumerate(folds, 1):
                    init_seed(42 + fold)
                    
                    print(f"\nFold {fold}/5")
                    print(f"Training subjects ({len(train_subjects)}): {train_subjects}")
                    print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
                    
                    self.current_fold = fold
                    self.setup_components()
                    self.setup_metrics()
                    self.load_data(train_subjects, val_subjects)
                    
                    fold_train_losses = []
                    fold_val_losses = []
                    fold_metrics = []
                    best_fold_f1 = 0
                    best_fold_metrics_dict = None
                    best_fold_state = None
                    patience = 10  # Number of epochs to wait for improvement
                    epochs_without_improvement = 0
                    
                    for epoch in range(self.arg.num_epoch):
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                            
                            train_metrics = self.train_epoch(epoch)
                            val_metrics = self.validate(epoch)
                            
                            fold_train_losses.append(train_metrics['loss'])
                            fold_val_losses.append(val_metrics['loss'])
                            fold_metrics.append(val_metrics)
                            
                            if val_metrics['f1'] > best_fold_f1:
                                best_fold_f1 = val_metrics['f1']
                                best_fold_metrics_dict = val_metrics.copy()
                                best_fold_state = {
                                    'epoch': epoch,
                                    'state_dict': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'scheduler': self.scheduler.state_dict(),
                                    'metrics': val_metrics,
                                    'fold': fold
                                }
                                epochs_without_improvement = 0  # Reset counter
                            else:
                                epochs_without_improvement += 1
                                if epochs_without_improvement >= patience:
                                    print(f"\nEarly stopping triggered! No improvement in validation F1 for {patience} epochs.")
                                    print(f"Best validation F1: {best_fold_f1:.4f} at epoch {best_fold_state['epoch']}")
                                    break
                        
                            epoch_metrics.append({
                                'fold': fold,
                                'epoch': epoch,
                                'train_loss': train_metrics['loss'],
                                'val_loss': val_metrics['loss'],
                                'val_f1': val_metrics['f1']
                            })
                            
                            self.log_epoch_metrics(epoch, train_metrics, val_metrics)
                    
                    all_fold_results.append({
                        'fold': fold,
                        'train_subjects': train_subjects,
                        'val_subjects': val_subjects,
                        'best_epoch': best_fold_state['epoch'],
                        'metrics': best_fold_metrics_dict
                    })
                    
                    self.save_fold_results(fold, fold_train_losses, fold_val_losses, fold_metrics)
                    
                    if best_fold_state:
                        best_fold_metrics.append({
                            'fold': fold,
                            'f1': best_fold_f1,
                            'epoch': best_fold_state['epoch'],
                            'state_dict': best_fold_state['state_dict']
                        })
                
                print("\n" + "="*70)
                print("Cross-validation Results Summary")
                print("="*70)
                
                all_metrics = {
                    'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'best_epoch': []
                }
                
                print("\nResults for Each Fold:")
                print("-"*70)
                print(f"{'Fold':^6} {'Best Epoch':^12} {'Accuracy':^10} {'Precision':^10} {'Recall':^10} {'F1':^10}")
                print("-"*70)
                
                for result in all_fold_results:
                    fold = result['fold']
                    metrics = result['metrics']
                    
                    print(f"{fold:^6d} {result['best_epoch']:^12d} "
                          f"{metrics['accuracy']:^10.4f} {metrics['precision']:^10.4f} "
                          f"{metrics['recall']:^10.4f} {metrics['f1']:^10.4f}")
                    
                    for metric in all_metrics:
                        if metric == 'best_epoch':
                            all_metrics[metric].append(result['best_epoch'])
                        else:
                            all_metrics[metric].append(metrics[metric])
                
                print("-"*70)
                
                print("\nAverage Metrics Across All Folds:")
                print("-"*70)
                print(f"{'Metric':^15} {'Mean':^12} {'Std Dev':^12}")
                print("-"*70)
                
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'best_epoch']:
                    values = np.array(all_metrics[metric])
                    mean = np.mean(values)
                    std = np.std(values)
                    metric_name = metric.replace('_', ' ').title()
                    print(f"{metric_name:^15} {mean:^12.4f} {std:^12.4f}")
                
                print("="*70)
                
                best_fold = max(best_fold_metrics, key=lambda x: x['f1'])
                print(f"\nBest Fold: {best_fold['fold']} (F1: {best_fold['f1']:.4f})")
                
                print("\nPhase 2: Training final model on all data")
                self.setup_components()
                
                self.model.load_state_dict(best_fold['state_dict'])
                self.print_log(f"Loaded weights from best fold (Fold {best_fold['fold']})")
                
                self.load_data(all_subjects, all_subjects)
                best_epoch = best_fold['epoch']
                
                for epoch in range(best_epoch + 1):
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                        train_metrics = self.train_epoch(epoch)
                        self.log_epoch_metrics(epoch, train_metrics, None)
            
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
                
                self.save_results()
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                traceback.print_exc()
                self.save_results()
        else:
            self.test()
    
    def log_epoch_metrics(self, epoch, train_metrics, val_metrics):
        msg = f"Epoch {epoch} - Train: Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, FAR: {train_metrics['false_alarm_rate']:.4f}, MR: {train_metrics['miss_rate']:.4f}"
        if val_metrics:
            msg += (f" | Val: Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['accuracy']:.4f}, "
                    f"Prec: {val_metrics['precision']:.4f}, "
                    f"Rec: {val_metrics['recall']:.4f}, "
                    f"F1: {val_metrics['f1']:.4f}, "
                    f"FAR: {val_metrics['false_alarm_rate']:.4f}, "
                    f"MR: {val_metrics['miss_rate']:.4f}, "
                    f"Trial-F1: {val_metrics.get('trial_f1', None):.4f}")
        self.print_log(msg)

    def print_log(self, msg, print_time=True):
        if print_time:
            time_str = time.strftime('[ %Y-%m-%d %H:%M:%S ]', time.localtime())
            msg = f'{time_str} {msg}'
        print(msg)

    def calculate_optimal_epoch(self, epoch_metrics):
        epoch_avg_metrics = {}
        for metric in epoch_metrics:
            epoch_ = metric['epoch']
            if epoch_ not in epoch_avg_metrics:
                epoch_avg_metrics[epoch_] = {'val_f1': [], 'val_loss': []}
            epoch_avg_metrics[epoch_]['val_f1'].append(metric['val_f1'])
            epoch_avg_metrics[epoch_]['val_loss'].append(metric['val_loss'])
        
        avg_metrics = {
            epoch_: {
                'val_f1': np.mean(m['val_f1']),
                'val_loss': np.mean(m['val_loss'])
            }
            for epoch_, m in epoch_avg_metrics.items()
        }
        
        best_epoch = max(avg_metrics.keys(), key=lambda e: avg_metrics[e]['val_f1'])
        return best_epoch

    def save_fold_results(self, fold, train_losses, val_losses, metrics):
        fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': metrics
        }
        with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - Fold {fold}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fold_dir, 'loss_curve.png'))
        plt.close()
        
        self.print_log(f"\nFold {fold} Results:")
        self.print_log(f"Final Train Loss: {train_losses[-1]:.4f}")
        self.print_log(f"Final Val Loss: {val_losses[-1]:.4f}")
        self.print_log(f"Best Val F1: {max(m['f1'] for m in metrics):.4f}")
        self.print_log(f"Results saved to: {fold_dir}")

    def test(self):
        """Test loop with trial-level aggregation as well."""
        self.model.eval()
        test_metrics = {'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'false_alarm_rate': 0, 'miss_rate': 0}
        total_samples = 0
        trial_aggregator = defaultdict(lambda: {'probs': [], 'label': None})
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets, idx in tqdm(self.data_loader['test'], desc='Testing'):
                acc_data = inputs['accelerometer'].to(self.device)
                targets = targets.to(self.device).float()
                
                # Model returns probabilities of shape [batch_size]
                probs = self.model(acc_data.float())
                loss = F.binary_cross_entropy(probs, targets)
                
                batch_metrics = self.compute_metrics(probs, targets)
                batch_size = targets.size(0)
                total_samples += batch_size
                
                for key in test_metrics:
                    if key == 'loss':
                        test_metrics[key] += loss.item() * batch_size
                    else:
                        test_metrics[key] += batch_metrics[key] * batch_size
                
                # Store predictions for confusion matrix
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                
                # Store window-level predictions for later trial-level aggregation
                for i, trial_id in enumerate(idx):
                    trial_aggregator[trial_id.item()]['probs'].append(probs[i].item())
                    trial_aggregator[trial_id.item()]['label'] = targets[i].item()
        
        # Normalize metrics by total samples
        for key in test_metrics:
            test_metrics[key] /= total_samples
        
        # Compute trial-level metrics
        trial_probs, trial_labels = self.aggregate_trial_predictions(trial_aggregator)
        trial_metrics = self.compute_trial_metrics(trial_probs, trial_labels)
        
        # Add trial-level metrics to test_metrics
        test_metrics.update({f'trial_{k}': v for k, v in trial_metrics.items()})
        
        # Plot confusion matrix
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        predictions = (all_probs > 0.5).astype(int)
        confusion = confusion_matrix(all_labels, predictions)
        self.plot_confusion_matrix(confusion)
        
        # Perform threshold sweep evaluation
        self.print_log("\nWindow-level Threshold Sweep Analysis:")
        window_sweep = self.evaluate_threshold_sweep(all_probs, all_labels)
        
        self.print_log("\nTrial-level Threshold Sweep Analysis:")
        trial_sweep = self.evaluate_threshold_sweep(trial_probs, trial_labels)
        
        # Save all results
        results = {
            'window_metrics': test_metrics,
            'trial_metrics': trial_metrics,
            'window_sweep': window_sweep,
            'trial_sweep': trial_sweep,
            'confusion_matrix': confusion.tolist()
        }
        
        save_path = os.path.join(self.arg.work_dir, 'test_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return test_metrics

    def evaluate_threshold_sweep(self, probabilities, targets):
        """
        Evaluate model performance across different thresholds with enhanced metrics and visualizations.
        Args:
            probabilities: numpy array of shape (num_samples,) containing predicted probabilities
            targets: numpy array of shape (num_samples,) containing ground truth labels (0/1)
        Returns:
            dict: Dictionary containing best metrics and their corresponding thresholds
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
        import seaborn as sns
        
        # Calculate ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
        roc_auc = roc_auc_score(targets, probabilities)
        
        # Calculate Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(targets, probabilities)
        
        # Initialize best metrics tracking
        best_metrics = {
            'f1': {'score': 0, 'threshold': 0},
            'precision': {'score': 0, 'threshold': 0},
            'recall': {'score': 0, 'threshold': 0},
            'accuracy': {'score': 0, 'threshold': 0},
            'specificity': {'score': 0, 'threshold': 0},
            'balanced_accuracy': {'score': 0, 'threshold': 0}
        }
        
        # Create figure for ROC and PR curves
        plt.figure(figsize=(15, 5))
        
        # Plot ROC curve
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Plot Precision-Recall curve
        plt.subplot(1, 3, 2)
        plt.plot(recall_curve, precision_curve, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        # Print header for threshold sweep results
        self.print_log("\nThreshold Sweep Results:")
        self.print_log("-" * 80)
        header = f"{'Threshold':^10} | {'F1':^8} | {'Precision':^10} | {'Recall':^8} | {'Accuracy':^10} | {'Specificity':^12} | {'Bal Acc':^8}"
        self.print_log(header)
        self.print_log("-" * 80)
        
        # Perform threshold sweep
        for threshold in np.arange(0.05, 1.0, 0.05):
            preds_binary = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(targets, preds_binary).ravel()
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            balanced_acc = (recall + specificity) / 2
            
            # Update best metrics
            metrics_dict = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'specificity': specificity,
                'balanced_accuracy': balanced_acc
            }
            
            for metric, value in metrics_dict.items():
                if value > best_metrics[metric]['score']:
                    best_metrics[metric] = {'score': value, 'threshold': threshold}
            
            # Print results in table format
            result_line = f"{threshold:^10.2f} | {f1:^8.4f} | {precision:^10.4f} | {recall:^8.4f} | "
            result_line += f"{accuracy:^10.4f} | {specificity:^12.4f} | {balanced_acc:^8.4f}"
            self.print_log(result_line)
        
        # Plot confusion matrix for best F1 threshold
        plt.subplot(1, 3, 3)
        best_threshold = best_metrics['f1']['threshold']
        best_preds = (probabilities >= best_threshold).astype(int)
        cm = confusion_matrix(targets, best_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix\n(threshold={best_threshold:.2f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save the plots
        plt.tight_layout()
        plt.savefig(os.path.join(self.arg.work_dir, 'evaluation_metrics.png'))
        plt.close()
        
        # Print best metrics summary
        self.print_log("\nBest Metrics Summary:")
        self.print_log("-" * 80)
        for metric, values in best_metrics.items():
            self.print_log(f"Best {metric:15s}: {values['score']:.4f} at threshold {values['threshold']:.2f}")
        
        return best_metrics

    def plot_confusion_matrix(self, confusion):
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        
        classes = ['Non-Fall', 'Fall']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        thresh = confusion.max() / 2.
        for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
            plt.text(j, i, format(confusion[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.arg.work_dir, 'confusion_matrix.png'))
        plt.close()

    def save_config(self):
        if not hasattr(self.arg, 'config') or not os.path.isfile(self.arg.config):
            raise ValueError("Config file not found or not specified.")
        
        dest_path = os.path.join(self.arg.work_dir, os.path.basename(self.arg.config))
        shutil.copy2(self.arg.config, dest_path)
        self.print_log(f"Configuration saved to {dest_path}")
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def import_class(self, name):
        try:
            components = name.split('.')
            module = ".".join(components[:-1])
            class_name = components[-1]
            
            if module:
                m = __import__(module)
                for comp in components[1:-1]:
                    m = getattr(m, comp)
                return getattr(m, class_name)
            else:
                return globals()[class_name]
        except Exception as e:
            print(f"Error importing class {name}: {str(e)}")
            raise


class FallDetectionTrainer(FallDetectionTrainer):
    """Additional methods for FallDetectionTrainer"""
    
    def load_data(self, train_subjects, test_subjects):
        """Load and prepare datasets with existing approach."""
        Feeder = self.import_class(self.arg.feeder)
        builder = prepare_smartfallmm(self.arg)
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
        """Plot and save training visualization plots with existing approach."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        epochs = range(len(self.train_metrics))
        train_loss = [m['loss'] for m in self.train_metrics]
        val_loss = [m['loss'] for m in self.val_metrics]
        
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.arg.work_dir, 'training_curves.png'))
        plt.close()
        self.print_log(f"Saved training curves plot to {os.path.join(self.arg.work_dir, 'training_curves.png')}")

    def test(self):
        """Test loop with window-level and trial-level aggregator."""
        self.model.eval()
        test_metrics = {'loss': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'false_alarm_rate': 0, 'miss_rate': 0}
        total_samples = 0
        trial_aggregator = defaultdict(lambda: {'probs': [], 'label': None})
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets, idx in tqdm(self.data_loader['test'], desc='Testing'):
                acc_data = inputs['accelerometer'].to(self.device)
                targets = targets.to(self.device).float()
                
                # Model returns only probabilities, no need to unpack
                probs = self.model(acc_data.float())
                if isinstance(probs, tuple):
                    probs = probs[0]  # Handle case where model returns tuple
                
                loss = F.binary_cross_entropy(probs, targets)
                
                batch_metrics = self.compute_metrics(probs, targets)
                batch_size = targets.size(0)
                total_samples += batch_size
                
                for key in test_metrics:
                    if key == 'loss':
                        test_metrics[key] += loss.item() * batch_size
                    else:
                        test_metrics[key] += batch_metrics[key] * batch_size
                
                # Store predictions for confusion matrix
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                
                # Store window-level predictions for later trial-level aggregation
                for i, trial_id in enumerate(idx):
                    trial_aggregator[trial_id.item()]['probs'].append(probs[i].item())
                    trial_aggregator[trial_id.item()]['label'] = targets[i].item()
        
        # Normalize metrics by total samples
        for key in test_metrics:
            test_metrics[key] /= total_samples
        
        # Compute trial-level metrics
        trial_probs, trial_labels = self.aggregate_trial_predictions(trial_aggregator)
        trial_metrics = self.compute_trial_metrics(trial_probs, trial_labels)
        
        # Add trial-level metrics to test_metrics
        test_metrics.update({f'trial_{k}': v for k, v in trial_metrics.items()})
        
        # Plot confusion matrix
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        predictions = (all_probs > 0.5).astype(int)
        confusion = confusion_matrix(all_labels, predictions)
        self.plot_confusion_matrix(confusion)
        
        # Perform threshold sweep evaluation
        self.print_log("\nWindow-level Threshold Sweep Analysis:")
        window_sweep = self.evaluate_threshold_sweep(all_probs, all_labels)
        
        self.print_log("\nTrial-level Threshold Sweep Analysis:")
        trial_sweep = self.evaluate_threshold_sweep(trial_probs, trial_labels)
        
        # Save all results
        results = {
            'window_metrics': test_metrics,
            'trial_metrics': trial_metrics,
            'window_sweep': window_sweep,
            'trial_sweep': trial_sweep,
            'confusion_matrix': confusion.tolist()
        }
        
        save_path = os.path.join(self.arg.work_dir, 'test_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return test_metrics

    def compute_trial_metrics(self, trial_probs, trial_labels):
        """
        Compute trial-level metrics from aggregated predictions.
        """
        predictions = (trial_probs > 0.5).astype(np.float32)
        tp = ((predictions == 1) & (trial_labels == 1)).sum().astype(float)
        fp = ((predictions == 1) & (trial_labels == 0)).sum().astype(float)
        fn = ((predictions == 0) & (trial_labels == 1)).sum().astype(float)
        tn = ((predictions == 0) & (trial_labels == 0)).sum().astype(float)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        far = fp / (fp + tp + 1e-10)
        mr = fn / (fn + tp + 1e-10)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'false_alarm_rate': far,
            'miss_rate': mr
        }

if __name__ == "__main__":
    parser = get_args()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    trainer = FallDetectionTrainer(arg)
    trainer.start()
