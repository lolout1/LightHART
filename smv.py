import traceback
from typing import Dict, List, Tuple
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
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, precision_score, recall_score
from utils.dataset import prepare_smartfallmm, filter_subjects
from Models.mobile import EnhancedDualPathFallDetector
from main import import_class, str2bool, get_args
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn explicitly
from collections import defaultdict
import datetime
import copy

# Removed matplotlib imports to disable plotting globally

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class SMVAugmentation:
    """Data augmentation specific to SMV signals"""
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        """No augmentation, return data as is"""
        return data

class SMVOptimizedTrainer:
    def __init__(self, arg):
        """Initialize trainer with arguments"""
        self.arg = arg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_step = 0
        self.best_accuracy = 0
        self.best_f1 = 0
        self.early_stopping_patience = 10
        self.early_stopping_counter = 0
        
        # Initialize metrics history
        self.metrics_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_precision': [], 
            'train_sensitivity': [], 'train_specificity': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 
            'val_sensitivity': [], 'val_specificity': [], 'val_auc': [],
            'test_loss': [], 'test_acc': [], 'test_f1': [], 'test_precision': [], 
            'test_sensitivity': [], 'test_specificity': [], 'test_auc': []
        }
        
        self.setup_logging()
        self.setup_basic_attributes()
        self.setup_environment()
        self.initialize_components()
        
        # Store metrics for both runs
        self.run_metrics = []
        
    def setup_logging(self):
        """Setup logging directory and file"""
        # Create work directory if it doesn't exist
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        self.log_file = os.path.join(self.arg.work_dir, 'log.txt')

    def print_log(self, msg, print_time=True):
        """Print logs to console and file"""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            msg = f"[ {localtime} ] {msg}"
        print(msg)
        if hasattr(self, 'log_file'):
            with open(self.log_file, 'a') as f:
                print(msg, file=f)

    def setup_basic_attributes(self):
        """Initialize basic class attributes"""
        self.print_log("Initializing basic attributes...")
        self.results = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 
                                           'train_acc', 'val_acc', 'f1', 'auc', 'sensitivity', 'specificity'])
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_f1 = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0
        self.train_subjects = []
        self.val_subjects = []
        self.test_subjects = []
        self.early_stopping_counter = 0
        self.early_stopping_patience = 25
        self.best_model_path = None
        self.data_loader = {}
        self.train_class_counts = None

        # Initialize sensors list
        self.inertial_sensors = []
        for modality in self.arg.dataset_args['modalities']:
            if modality != 'skeleton':
                self.inertial_sensors.extend(
                    [f"{modality}_{sensor}" for sensor in self.arg.dataset_args['sensors'][modality]]
                )

    def setup_environment(self):
        """Setup working directory and CUDA devices"""
        self.print_log("Setting up environment...")
        # Save config
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            self.save_config(self.arg.config, self.arg.work_dir)

        # Setup CUDA
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
            self.device = f'cuda:{self.output_device}'
            torch.cuda.set_device(self.output_device)
            self.print_log(f'Using GPU device {self.output_device}')
        else:
            self.device = 'cpu'
            self.output_device = 'cpu'
            self.print_log('Using CPU')

    def save_config(self, src_path: str, desc_path: str) -> None:
        """Save configuration file"""
        config_name = src_path.rpartition("/")[-1]
        dest_path = f'{desc_path}/{config_name}'
        self.print_log(f'Saving config to {dest_path}')
        shutil.copy(src_path, dest_path)

    def initialize_components(self):
        """Initialize model, optimizer, and training components"""
        self.print_log("Initializing components...")
        # Initialize model
        self.model = self.load_model(self.arg.model, self.arg.model_args)
        
        # Initialize training components
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self.load_optimizer()
        self.scaler = GradScaler()  # For mixed precision training
        self.augmentation = SMVAugmentation()  # Data augmentation

    def load_model(self, model, model_args):
        """Load and initialize the model"""
        Model = import_class(model)
        model = Model(**model_args).to(self.device)
        self.print_log(f"Model {model} loaded successfully")
        return model

    def load_optimizer(self):
        """Initialize optimizer"""
        if self.arg.optimizer.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=self.arg.base_lr)
        elif self.arg.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")

    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = len(self.data_loader['train']) * self.arg.num_epoch
        warmup_steps = len(self.data_loader['train']) * getattr(self.arg, 'warmup_epochs', 10)
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.arg.base_lr,
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0
        )

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, features: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute combined loss with SMV components"""
        # Classification loss
        cls_loss = self.criterion(logits, targets)
        
        # Prepare binary targets (batch_size,)
        binary_targets = (targets > 0).float()
        
        # Get SMV features (batch_size,)
        phone_smv = features['phone_smv']
        watch_smv = features['watch_smv']
        
        # Ensure dimensions match for BCE loss
        if len(phone_smv.shape) == 1:
            phone_smv = phone_smv.view(-1)
        if len(watch_smv.shape) == 1:
            watch_smv = watch_smv.view(-1)
        if len(binary_targets.shape) == 1:
            binary_targets = binary_targets.view(-1)
        
        # SMV loss
        smv_loss = F.binary_cross_entropy_with_logits(
            phone_smv, binary_targets
        ) + F.binary_cross_entropy_with_logits(
            watch_smv, binary_targets
        )
        
        # Consistency loss
        consistency_loss = F.mse_loss(phone_smv, watch_smv)
        
        # Combine losses with weights
        total_loss = (
            cls_loss + 
            self.arg.loss_args.get('smv_weight', 0.2) * smv_loss +
            self.arg.loss_args.get('consistency_weight', 0.1) * consistency_loss
        )
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'smv_loss': smv_loss.item(),
            'consistency_loss': consistency_loss.item()
        }

    def load_data(self):
        """Load and prepare datasets"""
        try:
            # Get dataset builder
            builder = prepare_smartfallmm(self.arg)
            
            # Get all available subjects
            all_subjects = sorted(list({trial.subject_id for trial in builder.dataset.matched_trials}))
            available_subjects = sorted(list(set(all_subjects) & set(self.arg.subjects)))
            
            if not available_subjects:
                self.print_log("No subjects available for training!")
                return False
            
            # Set specific test and validation subjects
            self.train_subjects = [s for s in available_subjects if s not in self.test_subjects + self.val_subjects]
            
            self.print_log(f"\nSubject Split:")
            self.print_log(f"Training subjects: {self.train_subjects}")
            self.print_log(f"Validation subjects: {self.val_subjects}")
            self.print_log(f"Test subjects: {self.test_subjects}")
            
            # Prepare datasets
            train_data = filter_subjects(builder, self.train_subjects)
            val_data = filter_subjects(builder, self.val_subjects)
            test_data = filter_subjects(builder, self.test_subjects)
            
            if not all([train_data, val_data, test_data]):
                self.print_log("Error: One or more datasets are empty!")
                return False
            
            # Create training dataset with balanced sampling
            train_dataset = import_class(self.arg.feeder)(dataset=train_data, batch_size=self.arg.batch_size)
            labels = train_dataset.labels
            class_counts = np.bincount(labels)
            total_samples = len(labels)
            class_weights = total_samples / (len(class_counts) * class_counts)
            sample_weights = class_weights[labels]
            
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(train_dataset),
                replacement=True
            )
            
            # Create dataloaders
            self.data_loader = {}
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                sampler=sampler,
                num_workers=self.arg.num_worker,
                pin_memory=True if self.use_cuda else False,
                drop_last=True
            )
            
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=import_class(self.arg.feeder)(dataset=val_data, batch_size=self.arg.batch_size),
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                pin_memory=True if self.use_cuda else False
            )
            
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=import_class(self.arg.feeder)(dataset=test_data, batch_size=self.arg.batch_size),
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                pin_memory=True if self.use_cuda else False
            )
            
            self.print_log(f"Created training dataloader with {len(train_dataset)} samples")
            self.print_log(f"Created validation dataloader with {len(self.data_loader['val'].dataset)} samples")
            self.print_log(f"Created test dataloader with {len(self.data_loader['test'].dataset)} samples")
            
            return True
            
        except Exception as e:
            self.print_log(f"Error in data loading: {str(e)}")
            traceback.print_exc()
            return False

    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch"""
        self.model.train()
        
        loader = self.data_loader['train']
        process = tqdm(loader, dynamic_ncols=True)
        
        loss_value = []
        acc_value = []
        f1_value = []
        auc_value = []
        
        for batch_idx, (sensor_data, label, _) in enumerate(process):  
            self.global_step += 1
            # Get data
            with torch.no_grad():
                sensor_data = {k: v.to(self.device) for k, v in sensor_data.items()}
                label = label.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(sensor_data)  
            
            # Calculate loss
            loss = self.criterion(logits, label)  
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            y_pred = torch.argmax(logits.data, dim=1)
            acc = torch.mean((y_pred == label).float())
            
            f1 = f1_score(label.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            try:
                auc = roc_auc_score(label.cpu().numpy(), torch.softmax(logits.data, dim=1)[:, 1].cpu().numpy())
            except ValueError:
                auc = 0.0
            
            loss_value.append(loss.item())
            acc_value.append(acc.item())
            f1_value.append(f1)
            auc_value.append(auc)
            
            # Print
            lr = self.optimizer.param_groups[0]['lr']
            process.set_description(f'Train Epoch {epoch}: {batch_idx+1}/{len(loader)} Loss: {np.mean(loss_value):.4f} Acc: {np.mean(acc_value):.4f} F1: {np.mean(f1_value):.4f} AUC: {np.mean(auc_value):.4f} LR: {lr:.6f}')
        
        # Statistics
        train_metrics = {
            'loss': np.mean(loss_value),
            'accuracy': np.mean(acc_value) * 100,
            'f1': np.mean(f1_value),
            'auc': np.mean(auc_value)
        }
        
        return train_metrics

    def get_detailed_metrics(self, targets, predictions, logits=None):
        """Calculate detailed metrics including F1, precision, sensitivity, specificity, and AUC"""
        targets = np.array(targets)
        predictions = np.array(predictions)
        
        # Calculate confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics = {
            'f1': f1,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
        }
        
        # Calculate AUC if logits are provided
        if logits is not None:
            try:
                probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                auc = roc_auc_score(targets, probabilities)
                metrics['auc'] = auc
            except:
                metrics['auc'] = 0.5  # Default AUC if calculation fails
        else:
            metrics['auc'] = 0.5
            
        return metrics

    def validate(self, loader='val'):
        """Validate the model"""
        self.model.eval()
        
        metrics = {
            'loss': 0,
            'accuracy': 0,
            'predictions': [],
            'targets': [],
            'logits': []
        }
        
        loader_name = 'validation' if loader == 'val' else 'test'
        process = tqdm(self.data_loader[loader], desc=f'{loader_name.capitalize()}')
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(process):
                sensor_data = {k: v.to(self.device) for k, v in inputs.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(sensor_data)
                loss = self.criterion(logits, targets)
                
                # Update metrics
                predictions = torch.argmax(logits, 1)
                metrics['predictions'].extend(predictions.cpu().numpy())
                metrics['targets'].extend(targets.cpu().numpy())
                metrics['logits'].append(logits.detach())
                metrics['loss'] += loss.item()
                metrics['accuracy'] += (predictions == targets).sum().item()
        
        # Calculate final metrics
        n_samples = len(self.data_loader[loader].dataset)
        metrics['loss'] /= len(self.data_loader[loader])
        metrics['accuracy'] = 100 * metrics['accuracy'] / n_samples
        metrics['logits'] = torch.cat(metrics['logits'], dim=0)
        
        # Calculate additional metrics
        detailed_metrics = self.get_detailed_metrics(
            metrics['targets'], 
            metrics['predictions'],
            metrics['logits']
        )
        metrics.update(detailed_metrics)
        
        # Store metrics in history
        prefix = f'{loader}_'
        self.metrics_history[f'{prefix}loss'].append(metrics['loss'])
        self.metrics_history[f'{prefix}acc'].append(metrics['accuracy'])
        self.metrics_history[f'{prefix}f1'].append(metrics['f1'])
        self.metrics_history[f'{prefix}precision'].append(metrics['precision'])
        self.metrics_history[f'{prefix}sensitivity'].append(metrics['sensitivity'])
        self.metrics_history[f'{prefix}specificity'].append(metrics['specificity'])
        self.metrics_history[f'{prefix}auc'].append(metrics['auc'])
        
        return metrics

    def print_model_info(self):
        """Print model information and configuration"""
        try:
            # Print model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.print_log("\nModel Information:")
            self.print_log(f"Architecture: {self.arg.model}")
            self.print_log(f"Total Parameters: {total_params:,}")
            self.print_log(f"Trainable Parameters: {trainable_params:,}")
            self.print_log(f"Model Size: {total_params * 4 / (1024*1024):.2f} MB")
            self.print_log(f"Training Device: {self.device}")
            self.print_log(f"Optimizer: {self.arg.optimizer}")
            self.print_log(f"Learning Rate: {self.arg.base_lr}")
            self.print_log(f"Weight Decay: {self.arg.weight_decay}")
            self.print_log(f"Batch Size: {self.arg.batch_size}")
            self.print_log(f"Number of Epochs: {self.arg.num_epoch}")
            
            # Print dataset info
            self.print_log("\nDataset Information:")
            if 'train' in self.data_loader:
                self.print_log(f"Training Samples: {len(self.data_loader['train'].dataset)}")
            if 'val' in self.data_loader:
                self.print_log(f"Validation Samples: {len(self.data_loader['val'].dataset)}")
            
            # Print training configuration
            self.print_log("\nTraining Configuration:")
            self.print_log(f"Loss Settings:")
            self.print_log(f"  - SMV Weight: {self.arg.loss_args.get('smv_weight', 0.2)}")
            self.print_log(f"  - Consistency Weight: {self.arg.loss_args.get('consistency_weight', 0.1)}")
            
            # Save complete configuration
            config_path = os.path.join(self.arg.work_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(vars(self.arg), f, default_flow_style=False)
            self.print_log(f"\nFull configuration saved to: {config_path}")
            
            # Print hardware info
            if self.use_cuda:
                gpu_name = torch.cuda.get_device_name(self.output_device)
                memory_allocated = torch.cuda.memory_allocated(self.output_device) / (1024**3)
                memory_cached = torch.cuda.memory_reserved(self.output_device) / (1024**3)
                self.print_log("\nHardware Information:")
                self.print_log(f"GPU: {gpu_name}")
                self.print_log(f"Memory Allocated: {memory_allocated:.2f} GB")
                self.print_log(f"Memory Cached: {memory_cached:.2f} GB")
            
            # Model architecture
            self.print_log("\nModel Architecture:")
            self.print_log(str(self.model))
            
        except Exception as e:
            self.print_log(f"Error in print_model_info: {str(e)}")
            traceback.print_exc()

    def save_model(self, epoch, val_metrics, test_metrics, split_info=None):
        """Save model with detailed metrics"""
        save_path = os.path.join(
            self.arg.work_dir,
            f'model_valf1_{val_metrics["f1"]:.4f}_testf1_{test_metrics["f1"]:.4f}.pt'
        )
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'test_subjects': self.test_subjects,
            'val_subjects': self.val_subjects,
            'config': self.arg.__dict__
        }
        
        torch.save(save_dict, save_path)
        self.print_log(f'Model saved to {save_path}')

    def log_split_summary(self, split_info, best_metrics):
        """Log detailed summary of split results"""
        self.print_log("\n" + "="*50)
        self.print_log(f"SPLIT {split_info['name']} SUMMARY")
        self.print_log("="*50)
        self.print_log(f"\nSplit Configuration:")
        self.print_log(f"Description: {split_info['desc']}")
        self.print_log(f"Test Subjects: {split_info['test']}")
        self.print_log(f"Validation Subjects: {split_info['val']}")
        
        # Log best validation metrics
        self.print_log("\nBest Validation Metrics:")
        self.print_log(f"Loss: {best_metrics['val']['loss']:.4f}")
        self.print_log(f"Accuracy: {best_metrics['val']['accuracy']:.2f}%")
        self.print_log(f"F1 Score: {best_metrics['val']['f1']:.4f}")
        self.print_log(f"Precision: {best_metrics['val']['precision']:.4f}")
        self.print_log(f"Sensitivity (Recall): {best_metrics['val']['sensitivity']:.4f}")
        self.print_log(f"Specificity: {best_metrics['val']['specificity']:.4f}")
        self.print_log(f"AUC: {best_metrics['val']['auc']:.4f}")
        
        # Log corresponding test metrics
        self.print_log("\nCorresponding Test Metrics:")
        self.print_log(f"Loss: {best_metrics['test']['loss']:.4f}")
        self.print_log(f"Accuracy: {best_metrics['test']['accuracy']:.2f}%")
        self.print_log(f"F1 Score: {best_metrics['test']['f1']:.4f}")
        self.print_log(f"Precision: {best_metrics['test']['precision']:.4f}")
        self.print_log(f"Sensitivity (Recall): {best_metrics['test']['sensitivity']:.4f}")
        self.print_log(f"Specificity: {best_metrics['test']['specificity']:.4f}")
        self.print_log(f"AUC: {best_metrics['test']['auc']:.4f}")
        
        self.print_log("\n" + "="*50 + "\n")

    def save_plots(self, save_dir, split_info, epoch_metrics, best_metrics, best_epoch):
        """Save comprehensive training plots with detailed metrics at split end"""
        plt.style.use('seaborn')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Training Metrics for {split_info["name"]}\n{split_info["desc"]}', fontsize=16, y=0.95)
        
        # Plot Loss
        ax1.plot(epoch_metrics['train_loss'], label='Train', marker='o')
        ax1.plot(epoch_metrics['val_loss'], label='Validation', marker='s')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Accuracy
        ax2.plot(epoch_metrics['train_acc'], label='Train', marker='o')
        ax2.plot(epoch_metrics['val_acc'], label='Validation', marker='s')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot F1 Score
        ax3.plot(epoch_metrics['train_f1'], label='Train', marker='o')
        ax3.plot(epoch_metrics['val_f1'], label='Validation', marker='s')
        ax3.set_title('F1 Score Curves')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True)
        
        # Plot ROC AUC
        ax4.plot(epoch_metrics['train_auc'], label='Train', marker='o')
        ax4.plot(epoch_metrics['val_auc'], label='Validation', marker='s')
        ax4.set_title('ROC AUC Curves')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('AUC')
        ax4.legend()
        ax4.grid(True)

        # Add text box with best metrics
        metrics_text = (
            f'Best Validation Metrics (Epoch {best_epoch}):\n'
            f'Loss: {best_metrics["val"]["loss"]:.4f}\n'
            f'Accuracy: {best_metrics["val"]["accuracy"]:.2f}%\n'
            f'F1 Score: {best_metrics["val"]["f1"]:.4f}\n'
            f'AUC: {best_metrics["val"]["auc"]:.4f}\n\n'
            f'Corresponding Test Metrics:\n'
            f'Loss: {best_metrics["test"]["loss"]:.4f}\n'
            f'Accuracy: {best_metrics["test"]["accuracy"]:.2f}%\n'
            f'F1 Score: {best_metrics["test"]["f1"]:.4f}\n'
            f'AUC: {best_metrics["test"]["auc"]:.4f}'
        )
        
        # Add configuration details
        config_text = (
            f'Configuration:\n'
            f'Model: {self.arg.model}\n'
            f'Learning Rate: {self.arg.base_lr}\n'
            f'Batch Size: {self.arg.batch_size}\n'
            f'Split Info:\n'
            f'  Test: {split_info["test"]}\n'
            f'  Val: {split_info["val"]}'
        )
        
        fig.text(0.02, 0.02, config_text, fontsize=10, va='bottom', ha='left')
        fig.text(0.98, 0.02, metrics_text, fontsize=10, va='bottom', ha='right')
        
        # Save plot
        plot_path = os.path.join(save_dir, f'split_{split_info["name"]}_final_metrics.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.print_log(f'Final training curves saved to: {plot_path}')

    def save_split_results(self, split_dir, results):
        """Save detailed results for a split"""
        results_file = os.path.join(split_dir, 'detailed_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Detailed Results for {results['split_name']}\n")
            f.write("="*50 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"Description: {results['description']}\n")
            f.write(f"Test Subjects: {results['test_subjects']}\n")
            f.write(f"Validation Subjects: {results['val_subjects']}\n\n")
            
            f.write("Best Validation Metrics:\n")
            f.write("-" * 25 + "\n")
            for metric, value in results['val_metrics'].items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nCorresponding Test Metrics:\n")
            f.write("-" * 25 + "\n")
            for metric, value in results['test_metrics'].items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric}: {value:.4f}\n")

    def save_overall_summary(self, all_results):
        """Save comprehensive summary of all splits"""
        summary_file = os.path.join(self.arg.work_dir, 'all_splits_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE SUMMARY OF ALL SPLITS\n")
            f.write("="*50 + "\n\n")
            
            # Save training configuration
            f.write("Training Configuration\n")
            f.write("-"*25 + "\n")
            f.write(f"Model: {self.arg.model}\n")
            f.write(f"Learning Rate: {self.arg.base_lr}\n")
            f.write(f"Batch Size: {self.arg.batch_size}\n")
            f.write(f"Max Epochs: {self.arg.num_epoch}\n")
            f.write(f"Early Stopping Patience: {self.early_stopping_patience}\n\n")
            
            # Individual split results
            f.write("Individual Split Results\n")
            f.write("-"*25 + "\n\n")
            for result in all_results:
                f.write(f"Split: {result['split_name']}\n")
                f.write(f"Description: {result['description']}\n")
                f.write(f"Test Subjects: {result['test_subjects']}\n")
                f.write(f"Val Subjects: {result['val_subjects']}\n\n")
                
                f.write("Validation Metrics:\n")
                for metric, value in result['val_metrics'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric}: {value:.4f}\n")
                
                f.write("\nTest Metrics:\n")
                for metric, value in result['test_metrics'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n" + "="*50 + "\n\n")
            
            # Calculate and write statistics across all splits
            f.write("\nAGGREGATE STATISTICS ACROSS ALL SPLITS\n")
            f.write("-"*50 + "\n\n")
            
            metrics_to_track = ['loss', 'accuracy', 'f1', 'precision', 'sensitivity', 'specificity', 'auc']
            
            for phase in ['val', 'test']:
                f.write(f"{phase.upper()} METRICS:\n")
                f.write("-"*20 + "\n")
                
                for metric in metrics_to_track:
                    values = [r[f'{phase}_metrics'][metric] for r in all_results]
                    mean = np.mean(values)
                    std = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    
                    if metric == 'accuracy':
                        f.write(f"{metric}:\n")
                        f.write(f"  Mean ± Std: {mean:.2f}% ± {std:.2f}%\n")
                        f.write(f"  Range: [{min_val:.2f}% - {max_val:.2f}%]\n")
                    else:
                        f.write(f"{metric}:\n")
                        f.write(f"  Mean ± Std: {mean:.4f} ± {std:.4f}\n")
                        f.write(f"  Range: [{min_val:.4f} - {max_val:.4f}]\n")
                f.write("\n")
            
            # Add timestamp
            f.write(f"\nSummary generated at: {datetime.datetime.now()}\n")
        
        self.print_log(f"\nComprehensive summary saved to: {summary_file}")

    def plot_average_metrics(self, all_results):
        """Plot average validation and training metrics across all splits"""
        plt.style.use('seaborn-v0_8')  # Fix for matplotlib deprecation warning
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Average Training Metrics Across All {len(all_results)} Splits', fontsize=16, y=0.95)
        
        # Calculate average metrics
        metrics_to_track = ['loss', 'accuracy', 'f1', 'auc']
        phases = ['train', 'val']
        avg_metrics = {
            phase: {metric: [] for metric in metrics_to_track}
            for phase in phases
        }
        
        # Initialize metrics history if not present
        for result in all_results:
            if 'metrics_history' not in result:
                result['metrics_history'] = {}
                # Use the final metrics for each phase
                for phase in phases:
                    phase_metrics = result.get(f'{phase}_metrics', {})
                    for metric in metrics_to_track:
                        if metric in phase_metrics:
                            result['metrics_history'][f'{phase}_{metric}'] = [phase_metrics[metric]]
        
        # Collect metrics from all splits
        for result in all_results:
            for phase in phases:
                for metric in metrics_to_track:
                    metric_key = f'{phase}_{metric}'
                    if metric_key in result['metrics_history']:
                        avg_metrics[phase][metric].append(result['metrics_history'][metric_key])
                    elif metric in result.get(f'{phase}_metrics', {}):
                        # If no history, use the final metric as a single point
                        avg_metrics[phase][metric].append([result[f'{phase}_metrics'][metric]])
        
        # Ensure that there is data to process
        if avg_metrics['train']['loss']:
            max_epochs = max(len(metrics[0]) for metrics in avg_metrics['train']['loss'])
            # Proceed with further calculations
            mean_metrics = {
                phase: {metric: np.zeros(max_epochs) for metric in metrics_to_track}
                for phase in phases
            }
            std_metrics = {
                phase: {metric: np.zeros(max_epochs) for metric in metrics_to_track}
                for phase in phases
            }
            
            # Calculate statistics
            for phase in phases:
                for metric in metrics_to_track:
                    metric_data = avg_metrics[phase][metric]
                    # Pad shorter sequences with last value
                    padded_data = [
                        np.pad(m, (0, max_epochs - len(m)), 'edge')
                        for m in metric_data
                    ]
                    metric_array = np.array(padded_data)
                    mean_metrics[phase][metric] = np.mean(metric_array, axis=0)
                    std_metrics[phase][metric] = np.std(metric_array, axis=0)
            
            epochs = np.arange(max_epochs)
            
            # Plot Loss
            ax1.plot(epochs, mean_metrics['train']['loss'], label='Train', color='blue')
            ax1.fill_between(epochs, 
                            mean_metrics['train']['loss'] - std_metrics['train']['loss'],
                            mean_metrics['train']['loss'] + std_metrics['train']['loss'],
                            alpha=0.2, color='blue')
            ax1.plot(epochs, mean_metrics['val']['loss'], label='Validation', color='red')
            ax1.fill_between(epochs,
                            mean_metrics['val']['loss'] - std_metrics['val']['loss'],
                            mean_metrics['val']['loss'] + std_metrics['val']['loss'],
                            alpha=0.2, color='red')
            ax1.set_title('Average Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Plot Accuracy
            ax2.plot(epochs, mean_metrics['train']['accuracy'], label='Train', color='blue')
            ax2.fill_between(epochs,
                            mean_metrics['train']['accuracy'] - std_metrics['train']['accuracy'],
                            mean_metrics['train']['accuracy'] + std_metrics['train']['accuracy'],
                            alpha=0.2, color='blue')
            ax2.plot(epochs, mean_metrics['val']['accuracy'], label='Validation', color='red')
            ax2.fill_between(epochs,
                            mean_metrics['val']['accuracy'] - std_metrics['val']['accuracy'],
                            mean_metrics['val']['accuracy'] + std_metrics['val']['accuracy'],
                            alpha=0.2, color='red')
            ax2.set_title('Average Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            
            # Plot F1 Score
            ax3.plot(epochs, mean_metrics['train']['f1'], label='Train', color='blue')
            ax3.fill_between(epochs,
                            mean_metrics['train']['f1'] - std_metrics['train']['f1'],
                            mean_metrics['train']['f1'] + std_metrics['train']['f1'],
                            alpha=0.2, color='blue')
            ax3.plot(epochs, mean_metrics['val']['f1'], label='Validation', color='red')
            ax3.fill_between(epochs,
                            mean_metrics['val']['f1'] - std_metrics['val']['f1'],
                            mean_metrics['val']['f1'] + std_metrics['val']['f1'],
                            alpha=0.2, color='red')
            ax3.set_title('Average F1 Score')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('F1 Score')
            ax3.legend()
            
            # Plot AUC
            ax4.plot(epochs, mean_metrics['train']['auc'], label='Train', color='blue')
            ax4.fill_between(epochs,
                            mean_metrics['train']['auc'] - std_metrics['train']['auc'],
                            mean_metrics['train']['auc'] + std_metrics['train']['auc'],
                            alpha=0.2, color='blue')
            ax4.plot(epochs, mean_metrics['val']['auc'], label='Validation', color='red')
            ax4.fill_between(epochs,
                            mean_metrics['val']['auc'] - std_metrics['val']['auc'],
                            mean_metrics['val']['auc'] + std_metrics['val']['auc'],
                            alpha=0.2, color='red')
            ax4.set_title('Average AUC')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('AUC')
            ax4.legend()
            
            # Add text box with average best metrics
            best_metrics_text = f"Average Best Metrics (N={len(all_results)} splits):\n\n"
            for phase in phases:
                best_metrics_text += f"{phase.upper()}:\n"
                for metric in metrics_to_track:
                    mean_val = np.mean([max(metrics) for metrics in avg_metrics[phase][metric]])
                    std_val = np.std([max(metrics) for metrics in avg_metrics[phase][metric]])
                    best_metrics_text += f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n"
                best_metrics_text += "\n"
            
            fig.text(1.02, 0.5, best_metrics_text, fontsize=10, va='center')
            
            # Save plot
            plot_path = os.path.join(self.arg.work_dir, 'average_metrics_across_splits.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Log average metrics
            self.print_log(f"\nAVERAGE BEST METRICS ACROSS ALL {len(all_results)} SPLITS")
            self.print_log("=" * 50)
            for phase in phases:
                self.print_log(f"\n{phase.upper()} Metrics:")
                for metric in metrics_to_track:
                    values = [max(metrics) for metrics in avg_metrics[phase][metric]]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    self.print_log(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
            
            self.print_log(f"\nAverage metrics plot saved to: {plot_path}")
            
            return {
                'mean_metrics': mean_metrics,
                'std_metrics': std_metrics,
                'best_metrics': {
                    phase: {
                        metric: (np.mean([max(metrics) for metrics in avg_metrics[phase][metric]]),
                               np.std([max(metrics) for metrics in avg_metrics[phase][metric]]))
                        for metric in metrics_to_track
                    }
                    for phase in phases
                }
            }
        else:
            print("No training loss data available to calculate max_epochs.")

    def organize_results(self, all_results, avg_metrics):
        """Organize all results into a new directory named with average test F1 score"""
        # Calculate average test F1 score
        test_f1_scores = [r['test_metrics']['f1'] for r in all_results]
        avg_test_f1 = np.mean(test_f1_scores)
        
        # Create directory name with timestamp and metrics
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self.arg.model}_avgF1_{avg_test_f1:.4f}_{timestamp}"
        results_dir = os.path.join(os.path.dirname(self.arg.work_dir), "experiments", dir_name)
        os.makedirs(results_dir, exist_ok=True)
        
        self.print_log(f"\nOrganizing results in directory: {results_dir}")
        
        # Copy config file
        config_dir = os.path.join(results_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)
        config_src = self.arg.config
        config_dst = os.path.join(config_dir, os.path.basename(self.arg.config))
        shutil.copy2(config_src, config_dst)
        
        # Create weights directory
        weights_dir = os.path.join(results_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        # Copy all split results
        splits_dir = os.path.join(results_dir, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        for split_idx, result in enumerate(all_results):
            split_name = f"split_{split_idx+1}"
            split_dir = os.path.join(splits_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            # Copy split's model file
            if 'model_path' in result:
                model_src = result['model_path']
                model_dst = os.path.join(split_dir, os.path.basename(model_src))
                shutil.copy2(model_src, model_dst)
                
                # Extract and save weights separately
                model_state = torch.load(model_src)
                weights_path = os.path.join(weights_dir, f"{split_name}_weights.pt")
                torch.save(model_state['model_state_dict'], weights_path)
            
            # Save split metrics
            metrics_path = os.path.join(split_dir, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Split {split_idx+1} Metrics\n")
                f.write("="*50 + "\n\n")
                f.write("Validation Metrics:\n")
                for k, v in result['val_metrics'].items():
                    if isinstance(v, (int, float)):
                        f.write(f"{k}: {v:.4f}\n")
                f.write("\nTest Metrics:\n")
                for k, v in result['test_metrics'].items():
                    if isinstance(v, (int, float)):
                        f.write(f"{k}: {v:.4f}\n")
        
        # Copy average metrics plot
        if hasattr(self, 'avg_metrics_plot_path'):
            plot_dst = os.path.join(results_dir, 'average_metrics.png')
            shutil.copy2(self.avg_metrics_plot_path, plot_dst)
        
        # Save overall summary
        summary_path = os.path.join(results_dir, 'experiment_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Experiment Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: {self.arg.model}\n")
            f.write(f"Average Test F1: {avg_test_f1:.4f}\n")
            f.write(f"Number of Splits: {len(all_results)}\n\n")
            
            f.write("Average Metrics Across Splits:\n")
            f.write("-"*30 + "\n")
            for phase in ['train', 'val', 'test']:
                f.write(f"\n{phase.upper()} Metrics:\n")
                metrics = avg_metrics['best_metrics'].get(phase, {})
                for metric, (mean, std) in metrics.items():
                    f.write(f"{metric}: {mean:.4f} ± {std:.4f}\n")
            
            f.write("\nConfiguration:\n")
            f.write("-"*30 + "\n")
            for k, v in vars(self.arg).items():
                f.write(f"{k}: {v}\n")
        
        self.print_log(f"Results organized in: {results_dir}")
        return results_dir

    def run_all_splits(self):
        """Run training and evaluation on all specified splits"""
        splits = [
            {
                'name': 'Split 1',
                'test': [44, 46],
                'val': [40, 42],
                'desc': 'Initial split with test=[44,46], val=[40,42]'
            },
            {
                'name': 'Split 2',
                'test': [40, 42],
                'val': [44, 46],
                'desc': 'Reversed split with test=[40,42], val=[44,46]'
            },
            {
                'name': 'Split 3',
                'test': [45, 46],
                'val': [44, 43],
                'desc': 'New split with test=[45,46], val=[44,43]'
            },
            {
                'name': 'Split 4',
                'test': [44, 43],
                'val': [45, 46],
                'desc': 'Reversed split with test=[44,43], val=[45,46]'
            },
            {
                'name': 'Split 5',
                'test': [44, 39],
                'val': [38, 42],
                'desc': 'New split with test=[44,39], val=[38,42]'
            },
            {
                'name': 'Split 6',
                'test': [38, 42],
                'val': [39, 44],
                'desc': 'New split with test=[38,42], val=[39,44]'
            },
            {
                'name': 'Split 7',
                'test': [45, 46],
                'val': [40, 42],
                'desc': 'New split with test=[45,46], val=[40,42]'
            },
            {
                'name': 'Split 8',
                'test': [45, 46],
                'val': [42, 44],
                'desc': 'Final split with test=[45,46], val=[42,44]'
            }
        ]
        
        all_results = []
        
        for split in splits:
            split_dir = os.path.join(self.arg.work_dir, f"split_{split['name'].lower()}")
            os.makedirs(split_dir, exist_ok=True)
            
            self.print_log(f"\n{'='*50}")
            self.print_log(f"Starting {split['name']}: {split['desc']}")
            self.print_log(f"Test subjects: {split['test']}")
            self.print_log(f"Validation subjects: {split['val']}")
            self.print_log('='*50)
            
            # Set up data for this split
            self.test_subjects = split['test']
            self.val_subjects = split['val']
            self.load_data()
            
            # Initialize new model for each split
            self.initialize_components()
            
            # Train model
            best_metrics = self.train(split_info=split)
            
            # Log detailed split summary
            self.log_split_summary(split, best_metrics)
            
            # Save results
            results = {
                'split_name': split['name'],
                'description': split['desc'],
                'test_subjects': split['test'],
                'val_subjects': split['val'],
                'val_metrics': best_metrics['val'],
                'test_metrics': best_metrics['test']
            }
            all_results.append(results)
            
            # Save detailed results to file
            self.save_split_results(split_dir, results)
            
            # Clear GPU memory
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'optimizer'):
                del self.optimizer
            torch.cuda.empty_cache()
        
        # After all splits are done, plot average metrics
        avg_metrics = self.plot_average_metrics(all_results)
        
        # Save comprehensive summary
        self.save_overall_summary(all_results)
        
        # Organize results in a new directory
        self.organize_results(all_results, avg_metrics)
        
    def train(self, split_info=None):
        """Training process with enhanced logging"""
        self.print_log('\n' + '='*50)
        self.print_log('STARTING TRAINING PHASE')
        self.print_log('='*50)
        self.print_log(f'Training with validation subjects: {split_info["val"]}')
        self.print_log(f'Test subjects: {split_info["test"]} (will be evaluated only after training)')
        
        best_metrics = {
            'val': {'f1': 0},
            'test': None  # Will be evaluated only once after training is complete
        }
        
        epoch_metrics = defaultdict(list)
        best_epoch = 0
        best_model_state = None
        
        # Training Loop - Only use train and validation sets
        for epoch in range(self.arg.num_epoch):
            self.print_log(f'\nEpoch {epoch}/{self.arg.num_epoch-1}')
            self.print_log('-' * 20)
            
            # Training Phase
            self.model.train()
            train_metrics = self.train_epoch(epoch)
            self.print_log(
                f'Training - Loss: {train_metrics["loss"]:.4f}, '
                f'Acc: {train_metrics["accuracy"]:.2f}%, '
                f'F1: {train_metrics["f1"]:.4f}'
            )
            
            # Validation Phase
            self.model.eval()
            val_metrics = self.validate(loader='val')
            self.print_log(
                f'Validation - Loss: {val_metrics["loss"]:.4f}, '
                f'Acc: {val_metrics["accuracy"]:.2f}%, '
                f'F1: {val_metrics["f1"]:.4f}'
            )
            
            # Store metrics history
            epoch_metrics['train_loss'].append(train_metrics['loss'])
            epoch_metrics['train_acc'].append(train_metrics['accuracy'])
            epoch_metrics['train_f1'].append(train_metrics['f1'])
            epoch_metrics['train_auc'].append(train_metrics['auc'])
            epoch_metrics['val_loss'].append(val_metrics['loss'])
            epoch_metrics['val_acc'].append(val_metrics['accuracy'])
            epoch_metrics['val_f1'].append(val_metrics['f1'])
            epoch_metrics['val_auc'].append(val_metrics['auc'])
            
            # Update best model if validation F1 improves
            if val_metrics['f1'] > best_metrics['val']['f1']:
                best_metrics['val'] = val_metrics
                best_epoch = epoch
                best_model_state = copy.deepcopy(self.model.state_dict())
                self.print_log(f'New best model found at epoch {epoch}')
            
            # Early stopping check
            if epoch - best_epoch >= self.early_stopping_patience:
                self.print_log(f'Early stopping triggered after {epoch} epochs')
                break
        
        self.print_log('\n' + '='*50)
        self.print_log('TRAINING COMPLETED - STARTING TEST EVALUATION')
        self.print_log('='*50)
        
        # Load best model for test evaluation
        self.print_log(f"\nLoading best model from epoch {best_epoch} for test evaluation...")
        self.model.load_state_dict(best_model_state)
        self.model.eval()
        
        # Evaluate on test set exactly once
        self.print_log(f"\nEvaluating on test subjects: {split_info['test']}")
        test_metrics = self.validate(loader='test')
        best_metrics['test'] = test_metrics
        
        # Log final metrics
        self.print_log("\nFINAL METRICS SUMMARY")
        self.print_log("=" * 50)
        self.print_log(f"\nBest Validation Metrics (Epoch {best_epoch}):")
        for k, v in best_metrics['val'].items():
            if isinstance(v, (int, float)):
                self.print_log(f"{k}: {v:.4f}")
        
        self.print_log(f"\nTest Metrics (Evaluated once after training):")
        for k, v in best_metrics['test'].items():
            if isinstance(v, (int, float)):
                self.print_log(f"{k}: {v:.4f}")
        
        # Save final model with all metrics
        model_name = (f'model_split{split_info["name"]}_epoch{best_epoch}_'
                     f'valf1_{best_metrics["val"]["f1"]:.4f}_'
                     f'testf1_{best_metrics["test"]["f1"]:.4f}.pt')
        model_path = os.path.join(self.arg.work_dir, model_name)
        
        save_dict = {
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': best_metrics['val'],
            'test_metrics': best_metrics['test'],
            'split_info': split_info,
            'training_config': {
                'val_subjects': split_info['val'],
                'test_subjects': split_info['test'],
                'early_stopping_patience': self.early_stopping_patience,
                'total_epochs': epoch + 1
            }
        }
        
        torch.save(save_dict, model_path)
        self.print_log(f'\nBest model and metrics saved to: {model_path}')
        
        # Save final plots
        self.save_plots(self.arg.work_dir, split_info, epoch_metrics, best_metrics, best_epoch)
        
        return best_metrics

    def start(self):
        """Initialize and start training"""
        if not self.load_data():
            return
        
        self.setup_scheduler()
        self.run_all_splits()

def get_args():
    '''
    Function to build Argument Parser
    '''
    parser = argparse.ArgumentParser(description='SMV-Enhanced Fall Detection Training')
    
    # Basic parameters
    parser.add_argument('--config', default='./config/smartfallmm/mobile.yaml')
    parser.add_argument('--dataset', type=str, default='utd')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                       help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=8, 
                       help='input batch size for testing (default: 8)')
    parser.add_argument('--val-batch-size', type=int, default=8,
                       help='input batch size for validation (default: 8)')
    parser.add_argument('--num-epoch', type=int, default=300, metavar='N',
                       help='number of epochs to train (default: 300)')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--warmup-epochs', type=int, default=10,
                       help='number of warmup epochs (default: 10)')

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR',
                       help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0004)

    # Model parameters
    parser.add_argument('--model', default=None, help='Name of Model to load')
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--model-args', default=dict(), help='A dictionary for model args')
    parser.add_argument('--weights', type=str, help='Location of weight file')
    parser.add_argument('--model-saved-name', type=str, default='mobile_falldet',
                       help='Weight name')

    # Loss parameters
    parser.add_argument('--loss', default='loss.BCE', help='Name of loss function to use')
    parser.add_argument('--loss-args', default=dict(), help='A dictionary for loss')
    parser.add_argument('--smv-weight', type=float, default=0.2,
                       help='Weight for SMV loss')
    parser.add_argument('--consistency-weight', type=float, default=0.1,
                       help='Weight for consistency loss')

    # Dataset parameters
    parser.add_argument('--dataset-args', default=dict(), help='Arguments for dataset')
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--feeder', default=None, help='Dataloader location')
    parser.add_argument('--train-feeder-args', default=dict(),
                       help='A dict for dataloader args')
    parser.add_argument('--val-feeder-args', default=dict(),
                       help='A dict for validation data loader')
    parser.add_argument('--test-feeder-args', default=dict(),
                       help='A dict for test data loader')
    parser.add_argument('--include-val', type=str2bool, default=True,
                       help='If we will have the validation set or not')

    # Other parameters
    parser.add_argument('--seed', type=int, default=2,
                       help='random seed (default: 2)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                       help='how many batches to wait before logging training status')
    parser.add_argument('--work-dir', type=str, default='work_dir/mobile_falldet',
                       help='Working Directory')
    parser.add_argument('--print-log', type=str2bool, default=True,
                       help='print logging or not')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=4,
                       help='number of workers for data loading')
    parser.add_argument('--result-file', type=str, help='Name of result file')
    parser.add_argument('--cross-validation', type=str2bool, default=False,
                       help='Perform k-fold cross-validation')
    parser.add_argument('--n-folds', type=int, default=1,
                       help='Number of folds for cross-validation')
    parser.add_argument('--aug-args', default=dict(), help='A dictionary for augmentation args')

    return parser

def main():
    parser = get_args()
    
    # Load arg from config file
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
    trainer = SMVOptimizedTrainer(arg)
    trainer.start()

if __name__ == '__main__':
    main()
