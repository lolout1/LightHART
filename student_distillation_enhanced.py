#!/usr/bin/env python
"""
distill_student_enhanced.py

Distill knowledge from teacher model to student model for fall detection.
The student model only uses watch sensor data for inference.

Usage:
    python distill_student_enhanced.py --config config/smartfallmm/student_enhanced.yaml 
                                       --teacher_path exps/teacher_enhanced/teacher_best.pth
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
from Models.transformer_quat_enhanced import QuatTeacherEnhanced, QuatStudentEnhanced
from utils.cross_modal_loss import EnhancedDistillationLoss

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
    parser = argparse.ArgumentParser(description="Distill knowledge to student model")
    parser.add_argument('--config', default='config/smartfallmm/student_enhanced.yaml',
                        help='Path to config file')
    parser.add_argument('--teacher_path', required=True,
                        help='Path to pretrained teacher model')
    parser.add_argument('--device', default='0', help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug mode')
    parser.add_argument('--train_subjects', type=str, default=None,
                        help='Comma-separated list of subject IDs for training')
    parser.add_argument('--val_subjects', type=str, default=None,
                        help='Comma-separated list of subject IDs for validation')
    return parser.parse_args()

class EnhancedDistillTrainer:
    """
    Trainer for distilling knowledge from teacher to student model.
    """
    def __init__(self, args):
        # Load config
        with open(args.config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Save teacher path
        self.teacher_path = args.teacher_path
        
        # Initialize seed
        self.seed = args.seed
        init_seed(self.seed)
        
        # Set debug mode
        self.debug = args.debug
        
        # Create work directory
        self.work_dir = self.cfg.get('work_dir', 'exps/student_enhanced')
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.work_dir, 'distill.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('EnhancedDistillTrainer')
        
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
        
        # Initialize metrics history
        self.metrics_history = {
            'train_loss_total': [],
            'train_loss_kl': [],
            'train_loss_ce': [],
            'train_loss_feat': [],
            'train_loss_attn': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        
        self.logger.info(f"Initialized distillation trainer with config: {args.config}")
        self.logger.info(f"Teacher model: {self.teacher_path}")
        self.logger.info(f"Train subjects: {self.train_subjects}")
        self.logger.info(f"Val subjects: {self.val_subjects}")
        
    def build_data(self):
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
    
    def build_models(self):
        """Build teacher and student models."""
        self.logger.info("Building models...")
        
        # Get model arguments
        teacher_args = self.cfg.get('teacher_args', {})
        student_args = self.cfg.get('student_args', {})
        
        # Create teacher model
        self.teacher = QuatTeacherEnhanced(**teacher_args)
        self.teacher = self.teacher.to(self.device)
        
        # Create student model
        self.student = QuatStudentEnhanced(**student_args)
        self.student = self.student.to(self.device)
        
        # Calculate model sizes
        teacher_params = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        student_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        
        self.logger.info(f"Teacher model: {teacher_params:,} parameters")
        self.logger.info(f"Student model: {student_params:,} parameters")
        self.logger.info(f"Size reduction: {100 * (1 - student_params / teacher_params):.2f}%")
    
    def load_teacher_weights(self):
        """Load pretrained teacher weights."""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.teacher_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.teacher.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.teacher.load_state_dict(checkpoint)
            
            self.logger.info(f"Loaded teacher weights from {self.teacher_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load teacher weights: {e}")
            return False
    
    def build_optimizer_loss(self):
        """Build optimizer, scheduler, and loss function."""
        self.logger.info("Building optimizer and loss...")
        
        # Get optimizer params
        optimizer_type = self.cfg.get('optimizer', 'adam').lower()
        lr = self.cfg.get('base_lr', 0.001)
        weight_decay = self.cfg.get('weight_decay', 0.0001)
        
        # Create optimizer
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.student.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else: # sgd
            momentum = self.cfg.get('momentum', 0.9)
            self.optimizer = torch.optim.SGD(
                self.student.parameters(),
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
        
        # Create loss function
        loss_args = self.cfg.get('distill_args', {})
        self.distill_loss = EnhancedDistillationLoss(
            temperature=loss_args.get('temperature', 3.0),
            alpha=loss_args.get('alpha', 0.5),
            beta=loss_args.get('beta', 1.0),
            gamma=loss_args.get('gamma', 0.2),
            teacher_feat_dim=loss_args.get('teacher_feat_dim', 128),
            student_feat_dim=loss_args.get('student_feat_dim', 64),
            quat_indices=loss_args.get('quat_indices', None)
        )
        
        self.logger.info(f"Built optimizer: {optimizer_type}, lr={lr}, weight_decay={weight_decay}")
        if self.scheduler is not None:
            self.logger.info(f"Built scheduler: {scheduler_type}")
        self.logger.info(f"Built distillation loss with parameters: {loss_args}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.student.train()
        self.teacher.eval()  # Teacher always in eval mode
        
        epoch_losses = {
            'total': 0.0,
            'kl': 0.0,
            'ce': 0.0,
            'feat': 0.0,
            'attn': 0.0
        }
        correct = 0
        total = 0
        
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
            
            # Forward pass - student
            self.optimizer.zero_grad()
            student_outputs = self.student(imu_tensor, imu_mask)
            
            # Forward pass - teacher (no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher(skel_tensor, imu_tensor, skel_mask, imu_mask)
            
            # Calculate distillation loss
            loss = self.distill_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                labels=labels
            )
            
            # Extract component losses for logging
            batch_losses = {
                'total': loss.item(),
                'kl': getattr(self.distill_loss, 'last_kl_loss', 0.0),
                'ce': getattr(self.distill_loss, 'last_ce_loss', 0.0),
                'feat': getattr(self.distill_loss, 'last_feat_loss', 0.0),
                'attn': getattr(self.distill_loss, 'last_attn_loss', 0.0)
            }
            
            # Backward and optimize
            loss.backward()
            
            # Gradient clipping
            if self.cfg.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    self.cfg.get('grad_clip')
                )
            
            self.optimizer.step()
            
            # Update metrics
            batch_size = labels.size(0)
            for k in epoch_losses:
                epoch_losses[k] += batch_losses[k] * batch_size
            
            _, preds = torch.max(student_outputs["logits"], dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            
            # Update progress bar
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{batch_losses["total"]:.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        # Calculate epoch metrics
        for k in epoch_losses:
            epoch_losses[k] /= total
        
        train_acc = 100.0 * correct / total
        
        # Update history
        self.metrics_history['train_loss_total'].append(epoch_losses['total'])
        self.metrics_history['train_loss_kl'].append(epoch_losses['kl'])
        self.metrics_history['train_loss_ce'].append(epoch_losses['ce'])
        self.metrics_history['train_loss_feat'].append(epoch_losses['feat'])
        self.metrics_history['train_loss_attn'].append(epoch_losses['attn'])
        self.metrics_history['train_acc'].append(train_acc)
        
        # Log metrics
        self.logger.info(
            f"Epoch {epoch+1} Train: "
            f"Loss={epoch_losses['total']:.4f}, "
            f"KL={epoch_losses['kl']:.4f}, "
            f"CE={epoch_losses['ce']:.4f}, "
            f"Feat={epoch_losses['feat']:.4f}, "
            f"Attn={epoch_losses['attn']:.4f}, "
            f"Acc={train_acc:.2f}%"
        )
        
        return epoch_losses, train_acc
    
    def eval_epoch(self, epoch):
        """Evaluate student model on validation set."""
        self.student.eval()
        
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for imu_tensor, imu_mask, skel_tensor, skel_mask, labels in self.val_loader:
                # Move data to device
                imu_tensor = imu_tensor.to(self.device)
                imu_mask = imu_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass for student
                student_outputs = self.student(imu_tensor, imu_mask)
                logits = student_outputs["logits"]
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                
                # Get predictions
                _, predicted = torch.max(logits, dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(all_labels) if all_labels else 0
        val_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate F1, precision, recall
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        if len(np.unique(all_labels)) <= 2:
            # Binary classification
            val_f1 = f1_score(all_labels, all_preds, average="binary")
            val_precision = precision_score(all_labels, all_preds, average="binary")
            val_recall = recall_score(all_labels, all_preds, average="binary")
        else:
            # Multi-class classification
            val_f1 = f1_score(all_labels, all_preds, average="macro")
            val_precision = precision_score(all_labels, all_preds, average="macro")
            val_recall = recall_score(all_labels, all_preds, average="macro")
        
        # Update metrics history
        self.metrics_history["val_loss"].append(val_loss)
        self.metrics_history["val_acc"].append(val_acc)
        self.metrics_history["val_f1"].append(val_f1)
        self.metrics_history["val_precision"].append(val_precision)
        self.metrics_history["val_recall"].append(val_recall)
        
        # Log metrics
        self.logger.info(
            f"Epoch {epoch+1} Val: "
            f"Loss={val_loss:.4f}, Acc={val_acc:.2f}%, "
            f"F1={val_f1:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}"
        )
        
        return val_loss, val_acc, val_f1, val_precision, val_recall
    
    def distill(self):
        """Start the enhanced distillation process."""
        # Get training parameters
        num_epochs = self.cfg.get("num_epoch", 50)
        
        # Initialize best validation metrics
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        
        # Initialize early stopping
        patience = self.cfg.get("early_stop_patience", 15)
        early_stop_counter = 0
        
        self.logger.info(f"Starting enhanced distillation for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_losses, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall = self.eval_epoch(epoch)
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Check if this is the best model
            is_best = False
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                is_best = True
                
                # Save best model
                model_path = os.path.join(self.work_dir, f"{self.cfg.get('model_saved_name', 'student_enhanced_best')}.pth")
                torch.save(self.student.state_dict(), model_path)
                self.logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
                
                # Reset early stopping counter
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Check for early stopping
            if early_stop_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} Summary: "
                f"Train Loss={train_losses['total']:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                f"F1={val_f1:.4f} "
                f"{'(Best)' if is_best else ''}"
            )
            
        # Save final model
        final_model_path = os.path.join(self.work_dir, f"{self.cfg.get('model_saved_name', 'student_enhanced')}_final.pth")
        torch.save(self.student.state_dict(), final_model_path)
        
        # Save training history
        history_path = os.path.join(self.work_dir, "distillation_history.json")
        import json
        with open(history_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_history = {}
            for k, v in self.metrics_history.items():
                serializable_history[k] = [float(x) for x in v]
            
            json.dump(serializable_history, f, indent=2)
        
        # Plot training curves for visualization
        self.plot_training_curves()
        
        self.logger.info(
            f"Distillation completed. Best F1: {best_val_f1:.4f} at epoch {best_epoch}. "
            f"Final metrics - Accuracy: {best_val_acc:.2f}%, F1: {best_val_f1:.4f}"
        )
        
        return best_val_acc, best_val_f1
    
    def plot_training_curves(self):
        """Plot training curves for loss and metrics."""
        import matplotlib.pyplot as plt
        
        # Create figure for loss components
        plt.figure(figsize=(12, 8))
        
        # Plot loss components
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history["train_loss_total"], label="Total Loss")
        plt.plot(self.metrics_history["train_loss_kl"], label="KL Loss")
        plt.plot(self.metrics_history["train_loss_ce"], label="CE Loss")
        plt.title("Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot feature and attention losses
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history["train_loss_feat"], label="Feature Loss")
        plt.plot(self.metrics_history["train_loss_attn"], label="Attention Loss")
        plt.title("Feature and Attention Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics_history["train_acc"], label="Train Accuracy")
        plt.plot(self.metrics_history["val_acc"], label="Validation Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        
        # Plot F1, precision, recall
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics_history["val_f1"], label="F1 Score")
        plt.plot(self.metrics_history["val_precision"], label="Precision")
        plt.plot(self.metrics_history["val_recall"], label="Recall")
        plt.title("F1, Precision, Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.work_dir, "distillation_curves.png"), dpi=150)
        plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Initialize the trainer
    trainer = EnhancedDistillTrainer(args)
    
    # Build data
    trainer.build_data()
    
    # Build models
    trainer.build_models()
    
    # Load teacher weights
    if not trainer.load_teacher_weights():
        print("Failed to load teacher weights. Aborting.")
        return
    
    # Freeze teacher model
    for param in trainer.teacher.parameters():
        param.requires_grad = False
    
    trainer.teacher.eval()
    
    # Build optimizer and loss
    trainer.build_optimizer_loss()
    
    # Start distillation
    trainer.distill()

if __name__ == "__main__":
    main()
