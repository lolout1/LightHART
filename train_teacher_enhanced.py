#!/usr/bin/env python
"""
train_teacher_enhanced.py

Train a teacher model with enhanced IMU fusion for fall detection.
This script supports:
1. Multiple IMU fusion methods (Standard KF, EKF, UKF)
2. Skeleton + IMU alignment and fusion
3. Comprehensive logging and visualization
4. Model checkpointing and early stopping

Usage:
    python train_teacher_enhanced.py --config config/smartfallmm/teacher_enhanced.yaml --device 0
"""

import argparse
import yaml
import os
import torch
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

from utils.dataset import prepare_smartfallmm, split_by_subjects
from utils.enhanced_imu_fusion import fuse_inertial_modalities
from Feeder.multimodal_quat_feeder import MultimodalQuatFeeder, pad_collate_fn
from Models.transformer_quat_enhanced import QuatTeacherEnhanced

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_seed(seed):
    """Initialize random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train enhanced teacher model")
    parser.add_argument('--config', default='config/smartfallmm/teacher_enhanced.yaml',
                        help='Path to config file')
    parser.add_argument('--device', default='0', help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug mode')
    parser.add_argument('--train_subjects', type=str, default=None,
                        help='Comma-separated list of subject IDs for training')
    parser.add_argument('--val_subjects', type=str, default=None,
                        help='Comma-separated list of subject IDs for validation')
    return parser.parse_args()

class TeacherTrainer:
    """
    Trainer for enhanced teacher model utilizing IMU fusion.
    """
    def __init__(self, args):
        # Load config
        with open(args.config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize seed
        self.seed = args.seed
        init_seed(self.seed)
        
        # Set debug mode
        self.debug = args.debug
        
        # Create work directory
        self.work_dir = self.cfg.get('work_dir', 'exps/teacher_enhanced')
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.work_dir, 'train.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TeacherTrainer')
        
        # Add console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(console)
        
        # Parse subject lists
        if args.train_subjects:
            self.train_subjects = [int(s) for s in args.train_subjects.split(',')]
        else:
            # Use subjects from config
            self.train_subjects = self.cfg.get('subjects', [])
            
        if args.val_subjects:
            self.val_subjects = [int(s) for s in args.val_subjects.split(',')]
        else:
            # Default: use 20% of train_subjects for validation
            num_val = max(1, len(self.train_subjects) // 5)
            self.val_subjects = self.train_subjects[-num_val:]
            self.train_subjects = self.train_subjects[:-num_val]
        
        # Initialize training metrics history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        
        self.logger.info(f"Initialized TeacherTrainer with config: {args.config}")
        self.logger.info(f"Train subjects: {self.train_subjects}")
        self.logger.info(f"Val subjects: {self.val_subjects}")
        
    def build_dataset(self):
        """Build datasets and dataloaders."""
        self.logger.info("Building dataset...")
        
        # Get dataset builder
        builder = prepare_smartfallmm(self.cfg)
        
        # Build train dataset
        train_data = split_by_subjects(builder, self.train_subjects, fuse=True)
        self.train_dataset = MultimodalQuatFeeder(train_data)
        
        # Build val dataset
        val_data = split_by_subjects(builder, self.val_subjects, fuse=True)
        self.val_dataset = MultimodalQuatFeeder(val_data)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.get('batch_size', 32),
            shuffle=True,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=True,
            collate_fn=pad_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.get('val_batch_size', 32),
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=True,
            collate_fn=pad_collate_fn
        )
        
        self.logger.info(f"Built dataset: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")
    
    def build_model(self):
        """Build enhanced teacher model."""
        self.logger.info("Building model...")
        
        model_args = self.cfg.get('model_args', {})
        
        # Create model
        self.model = QuatTeacherEnhanced(**model_args)
        self.model = self.model.to(self.device)
        
        # Calculate model size
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model built with {num_params:,} trainable parameters")
    
    def build_optimizer(self):
        """Build optimizer and scheduler."""
        self.logger.info("Building optimizer...")
        
        # Get optimizer params
        optimizer_type = self.cfg.get('optimizer', 'adam').lower()
        lr = self.cfg.get('base_lr', 0.001)
        weight_decay = self.cfg.get('weight_decay', 0.0001)
        
        # Create optimizer
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else: # sgd
            momentum = self.cfg.get('momentum', 0.9)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        
        # Create scheduler
        scheduler_type = self.cfg.get('scheduler', None)
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.get('num_epoch', 100),
                eta_min=self.cfg.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            step_size = self.cfg.get('step_size', 20)
            gamma = self.cfg.get('gamma', 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'plateau':
            patience = self.cfg.get('lr_patience', 10)
            factor = self.cfg.get('lr_factor', 0.1)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                verbose=True
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Built optimizer: {optimizer_type}, lr={lr}, weight_decay={weight_decay}")
        if self.scheduler is not None:
            self.logger.info(f"Built scheduler: {scheduler_type}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        # Create progress bar if tqdm is available
        try:
            from tqdm import tqdm
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        except ImportError:
            pbar = self.train_loader
        
        for i, (imu_tensor, imu_mask, skel_tensor, skel_mask, labels) in enumerate(pbar):
            # Move to device
            imu_tensor = imu_tensor.to(self.device)
            imu_mask = imu_mask.to(self.device)
            skel_tensor = skel_tensor.to(self.device)
            skel_mask = skel_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(skel_tensor, imu_tensor, skel_mask, imu_mask)
            logits = outputs["logits"]
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # Backward and optimize
            loss.backward()
            
            # Gradient clipping
            if self.cfg.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.get('grad_clip')
                )
            
            self.optimizer.step()
            
            # Update metrics
            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        # Calculate epoch metrics
        train_loss = epoch_loss / total
        train_acc = 100.0 * correct / total
        
        # Calculate F1 score
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['train_f1'].append(train_f1)
        
        return train_loss, train_acc, train_f1
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imu_tensor, imu_mask, skel_tensor, skel_mask, labels in self.val_loader:
                # Move to device
                imu_tensor = imu_tensor.to(self.device)
                imu_mask = imu_mask.to(self.device)
                skel_tensor = skel_tensor.to(self.device)
                skel_mask = skel_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(skel_tensor, imu_tensor, skel_mask, imu_mask)
                logits = outputs["logits"]
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(logits, labels)
                
                # Update metrics
                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                
                _, preds = torch.max(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = val_loss / total
        val_acc = 100.0 * correct / total
        
        # Calculate additional metrics
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')
        
        # Update history
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)
        self.history['val_precision'].append(val_precision)
        self.history['val_recall'].append(val_recall)
        
        # Log metrics
        self.logger.info(
            f"Epoch {epoch+1} Val: "
            f"Loss={val_loss:.4f}, Acc={val_acc:.2f}%, "
            f"F1={val_f1:.4f}, Prec={val_precision:.4f}, Recall={val_recall:.4f}"
        )
        
        # If in debug mode, save confusion matrix
        if self.debug:
            self.plot_confusion_matrix(all_labels, all_preds, epoch+1)
        
        return val_loss, val_acc, val_f1, val_precision, val_recall
    
    def train(self):
        """Train the model."""
        # Get training parameters
        num_epochs = self.cfg.get('num_epoch', 100)
        early_stop_patience = self.cfg.get('early_stop_patience', 20)
        
        # Setup for early stopping
        best_val_f1 = 0.0
        best_epoch = -1
        patience_counter = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train one epoch
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall = self.validate(epoch)
            
            # Log epoch metrics
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Train F1={train_f1:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val F1={val_f1:.4f}"
            )
            
            # Update learning rate if using ReduceLROnPlateau
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Check for improvement
            is_best = val_f1 > best_val_f1
            
            if is_best:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                self.save_model(os.path.join(self.work_dir, 'teacher_best.pth'))
                self.logger.info(f"Saved new best model with val F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Save latest model
            self.save_model(os.path.join(self.work_dir, 'teacher_latest.pth'))
            
            # Plot training curves for visualization
            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                self.plot_training_curves()
        
        # Training finished
        total_time = time.time() - start_time
        self.logger.info(f"Training finished in {total_time:.2f} seconds")
        self.logger.info(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")
        
        # Save final model
        self.save_model(os.path.join(self.work_dir, 'teacher_final.pth'))
        
        # Plot final training curves
        self.plot_training_curves()
        
        return best_val_f1, best_epoch
    
    def save_model(self, path):
        """Save model weights."""
        torch.save({
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history
        }, path)
    
    def load_model(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        self.logger.info(f"Loaded model from {path}")
    
    def plot_training_curves(self):
        """Plot training curves for loss and metrics."""
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Plot F1 score
        plt.subplot(2, 2, 3)
        plt.plot(self.history['train_f1'], label='Train F1')
        plt.plot(self.history['val_f1'], label='Val F1')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # Plot precision and recall
        plt.subplot(2, 2, 4)
        plt.plot(self.history['val_precision'], label='Val Precision')
        plt.plot(self.history['val_recall'], label='Val Recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.work_dir, 'training_curves.png'))
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """Plot confusion matrix."""
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.work_dir, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create trainer
    trainer = TeacherTrainer(args)
    
    # Build dataset
    trainer.build_dataset()
    
    # Build model
    trainer.build_model()
    
    # Build optimizer
    trainer.build_optimizer()
    
    # Train model
    best_f1, best_epoch = trainer.train()
    
    print(f"Training completed. Best val F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Results saved to {trainer.work_dir}")

if __name__ == '__main__':
    main()
