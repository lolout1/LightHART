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
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
from utils.dataset import prepare_smartfallmm, filter_subjects
from Models.mobile import EnhancedDualPathFallDetector
from main import import_class, str2bool, get_args

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
    def __init__(self, noise_std=0.01, mask_ratio=0.1):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        augmented_data = {}
        for key, tensor in data.items():
            aug_tensor = tensor.clone()
            xyz_data = tensor[:, :, :3]  # Original XYZ data
            smv_data = tensor[:, :, 3:]  # SMV channel
            
            # Apply augmentations only to SMV
            smv_data = smv_data * (torch.randn_like(smv_data) * self.noise_std + 1)  # Noise
            smv_data = smv_data * (torch.rand_like(smv_data) > self.mask_ratio)  # Masking
            smv_data = smv_data * (torch.randn(1).exp() * 0.1 + 1)  # Scaling
            
            augmented_data[key] = torch.cat([xyz_data, smv_data], dim=-1)
        return augmented_data

class SMVOptimizedTrainer:
    def __init__(self, arg):
        self.arg = arg
        # First setup logging
        self.setup_logging()
        # Then setup other components in order
        self.setup_basic_attributes()
        self.setup_environment()
        self.initialize_components()
        self.load_data()
        self.print_model_info()

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
                                           'train_acc', 'val_acc', 'f1', 'auc'])
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_f1 = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0
        self.train_subjects = []
        self.test_subject = []
        self.early_stopping_counter = 0
        self.early_stopping_patience = 75
        self.best_model_path = None
        self.data_loader = {}

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
        """Loads training and validation datasets"""
        try:
            self.print_log("Loading dataset...")
            Feeder = import_class(self.arg.feeder)
        
            # First get all matched trials
            builder = prepare_smartfallmm(self.arg)

            # Get available subjects from matched trials
            all_trial_subjects = set(trial.subject_id for trial in builder.dataset.matched_trials)
            self.print_log(f"All available subjects in matched trials: {sorted(list(all_trial_subjects))}")
            
            # Filter the subjects that are in our arg.subjects list and in the matched trials
            available_subjects = sorted(list(all_trial_subjects & set(self.arg.subjects)))
            self.print_log(f"Subjects available for training/validation: {available_subjects}")
            
            if not available_subjects:
                self.print_log("No subjects available that match both matched trials and requested subjects!")
                return False
                
            # Split for training and validation
            val_subjects = available_subjects[-3:]  # Take last 3 subjects for validation
            train_subjects = available_subjects[:-3]  # Rest for training
            
            self.print_log(f"\nSplit:")
            self.print_log(f"Training subjects: {train_subjects}")
            self.print_log(f"Validation subjects: {val_subjects}")

            self.train_subjects = train_subjects
            self.test_subject = val_subjects

            # Prepare training data
            self.print_log("\nPreparing training data...")
            norm_train = filter_subjects(builder, train_subjects)
            if not norm_train:
                self.print_log("No training data was loaded. Exiting dataset preparation.")
                return False

            self.print_log("\nCreating training dataloader...")
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(dataset=norm_train, batch_size=self.arg.batch_size),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                pin_memory=True if self.use_cuda else False
            )
            
            self.print_log(f"Created training dataloader with {len(self.data_loader['train'].dataset)} samples")

            # Prepare validation data
            if self.arg.include_val:
                self.print_log("\nPreparing validation data...")
                norm_val = filter_subjects(builder, val_subjects)
                if not norm_val:
                    self.print_log("No validation data was loaded. Continuing without validation.")
                else:
                    self.data_loader['val'] = torch.utils.data.DataLoader(
                        dataset=Feeder(dataset=norm_val, batch_size=self.arg.batch_size),
                        batch_size=self.arg.batch_size,
                        shuffle=False,
                        num_workers=self.arg.num_worker,
                        pin_memory=True if self.use_cuda else False
                    )
                    self.print_log(f"Created validation dataloader with {len(self.data_loader['val'].dataset)} samples")
        
            return True
        except Exception as e:
            self.print_log(f"Error in load_data: {str(e)}")
            traceback.print_exc()
            return False

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        metrics = {
            'loss': 0,
            'cls_loss': 0,
            'smv_loss': 0,
            'consistency_loss': 0,
            'accuracy': 0,
            'predictions': [],
            'targets': []
        }
        
        process = tqdm(self.data_loader['train'], desc=f'Train Epoch {epoch}')
        
        for batch_idx, (inputs, targets, _) in enumerate(process):
            # Move data to device and apply augmentation
            inputs = self.augmentation(inputs)
            sensor_data = {k: v.to(self.device) for k, v in inputs.items()}
            targets = targets.to(self.device)
            
            # Mixed precision training with updated autocast
            with autocast('cuda'):  # Specify device type
                logits, features = self.model(sensor_data)
                loss, loss_dict = self.compute_loss(logits, targets, features)
            
            # Optimizer step
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Update metrics
            predictions = torch.argmax(logits, 1)
            metrics['predictions'].extend(predictions.cpu().numpy())
            metrics['targets'].extend(targets.cpu().numpy())
            metrics['loss'] += loss.item()
            metrics['cls_loss'] += loss_dict['cls_loss']
            metrics['smv_loss'] += loss_dict['smv_loss']
            metrics['consistency_loss'] += loss_dict['consistency_loss']
            metrics['accuracy'] += (predictions == targets).sum().item()
            
            # Update progress bar
            process.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * metrics['accuracy'] / ((batch_idx + 1) * targets.size(0)):.2f}%"
            })
        
        # Compute final metrics
        n_samples = len(self.data_loader['train'].dataset)
        for key in ['loss', 'cls_loss', 'smv_loss', 'consistency_loss']:
            metrics[key] /= len(self.data_loader['train'])
        metrics['accuracy'] = 100 * metrics['accuracy'] / n_samples
        metrics['f1'] = f1_score(metrics['targets'], metrics['predictions'], average='binary')  # Use 'binary' for binary classification
        
        return metrics

    def update_metrics(self, metrics, predictions, targets, loss, loss_dict):
        """Update training metrics"""
        metrics['predictions'].extend(predictions.cpu().numpy())
        metrics['targets'].extend(targets.cpu().numpy())
        metrics['loss'] += loss.item()
        metrics['cls_loss'] += loss_dict['cls_loss']
        metrics['smv_loss'] += loss_dict['smv_loss']
        metrics['consistency_loss'] += loss_dict['consistency_loss']
        metrics['accuracy'] += (predictions == targets).sum().item()
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

    def train(self):
        """Main training loop with early stopping"""
        try:
            self.results = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 
                                            'train_acc', 'val_acc', 'f1', 'auc'])
            
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.print_log(f"\nEpoch {epoch}/{self.arg.num_epoch}")
                
                # Training phase
                train_metrics = self.train_epoch(epoch)
                self.print_log(
                    f"Training - Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.2f}%, "
                    f"F1: {train_metrics['f1']:.4f}"
                )
                
                # Validation phase
                if 'val' in self.data_loader:
                    val_metrics = self.validate()
                    self.print_log(
                        f"Validation - Loss: {val_metrics['loss']:.4f}, "
                        f"Acc: {val_metrics['accuracy']:.2f}%, "
                        f"F1: {val_metrics['f1']:.4f}, "
                        f"AUC: {val_metrics['auc']:.4f}"
                    )
                    
                    # Save training history
                    self.results.loc[epoch] = [
                        epoch,
                        train_metrics['loss'], val_metrics['loss'],
                        train_metrics['accuracy'], val_metrics['accuracy'],
                        val_metrics['f1'], val_metrics['auc']
                    ]
                    
                    # Check for improvement
                    if val_metrics['f1'] > self.best_f1:
                        improvement = val_metrics['f1'] - self.best_f1
                        self.print_log(f"F1 score improved by {improvement:.4f}!")
                        self.best_f1 = val_metrics['f1']
                        self.best_accuracy = val_metrics['accuracy']
                        self.best_loss = val_metrics['loss']
                        
                        # Save best model
                        if self.best_model_path and os.path.exists(self.best_model_path):
                            os.remove(self.best_model_path)
                        
                        self.best_model_path = os.path.join(
                            self.arg.work_dir,
                            f'model_epoch_{epoch}_f1_{val_metrics["f1"]:.4f}.pth'
                        )
                        
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                            'best_f1': self.best_f1,
                            'metrics': val_metrics
                        }, self.best_model_path)
                        
                        self.print_log(f"Model saved to {self.best_model_path}")
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                        self.print_log(
                            f"No improvement in F1 score for {self.early_stopping_counter} epochs. "
                            f"Best F1: {self.best_f1:.4f}"
                        )
                    
                    # Early stopping check
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        self.print_log(
                            f"\nEarly stopping triggered after {epoch} epochs. "
                            f"Best F1: {self.best_f1:.4f}"
                        )
                        break
                
                # Disabled plotting
                # self.save_training_curves(epoch)
                
            # Final results
            self.print_log("\nTraining completed!")
            self.print_log(f"Best Validation Metrics:")
            self.print_log(f"F1 Score: {self.best_f1:.4f}")
            self.print_log(f"Accuracy: {self.best_accuracy:.2f}%")
            self.print_log(f"Loss: {self.best_loss:.4f}")
            
            # Save final results
            self.results.to_csv(os.path.join(self.arg.work_dir, 'training_history.csv'))
            
        except Exception as e:
            self.print_log(f"Error in training loop: {str(e)}")
            traceback.print_exc()

    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        metrics = {
            'loss': 0,
            'accuracy': 0,
            'predictions': [],
            'targets': [],
            'probabilities': []
        }
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(self.data_loader['val'], desc='Validation'):
                sensor_data = {k: v.to(self.device) for k, v in inputs.items()}
                targets = targets.to(self.device)
                
                logits, features = self.model(sensor_data)
                loss, _ = self.compute_loss(logits, targets, features)
                
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, 1)
                
                metrics['loss'] += loss.item()
                metrics['accuracy'] += (predictions == targets).sum().item()
                metrics['predictions'].extend(predictions.cpu().numpy())
                metrics['targets'].extend(targets.cpu().numpy())
                metrics['probabilities'].extend(probabilities.cpu().numpy())
        
        # Compute final metrics
        n_samples = len(self.data_loader['val'].dataset)
        metrics['loss'] /= len(self.data_loader['val'])
        metrics['accuracy'] = 100 * metrics['accuracy'] / n_samples
        metrics['f1'] = f1_score(metrics['targets'], metrics['predictions'], average='binary')  # Use 'binary' for binary classification
        
        # Convert probabilities to numpy array and extract positive class probabilities
        probabilities_np = np.array(metrics['probabilities'])  # Shape: [n_samples, 2]
        positive_class_probs = probabilities_np[:, 1]           # Shape: [n_samples]
        
        # Compute ROC AUC Score for binary classification
        metrics['auc'] = roc_auc_score(metrics['targets'], positive_class_probs)
        
        return metrics   

    def save_training_curves(self, current_epoch):
        """Save training curves as plots - Disabled"""
        pass  # Plotting is disabled

    def finalize_metrics(self, metrics):
        """Compute final metrics for the epoch"""
        n_samples = len(self.data_loader['train'].dataset)
        for key in ['loss', 'cls_loss', 'smv_loss', 'consistency_loss']:
            metrics[key] /= len(self.data_loader['train'])
        metrics['accuracy'] = 100 * metrics['accuracy'] / n_samples
        metrics['f1'] = f1_score(metrics['targets'], metrics['predictions'], average='macro')
        return metrics

    def start(self):
        """Initialize and start training"""
        if not self.load_data():
            return
        
        self.setup_scheduler()
        self.print_log('Starting training...')
        self.train()

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

    return parser

def main():
    """Main function to start training"""
    parser = get_args()
    p = parser.parse_args()
    
    # Load config
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
