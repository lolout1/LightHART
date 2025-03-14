#!/usr/bin/env python
"""
Train quaternion-enhanced teacher model for fall detection.

This script trains the teacher model using both skeleton and IMU data.
The teacher model uses quaternion-based orientation features for better fall detection.
"""

import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils.loader_quat import DatasetBuilderQuat, MultimodalQuatFeeder, pad_collate_fn
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train quaternion-enhanced teacher model for fall detection."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/smartfallmm/teacher_quat.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="0",
        help="GPU device ID"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--cross-val",
        action="store_true",
        help="Use 5-fold cross-validation"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize training results"
    )
    return parser.parse_args()

class TeacherTrainer:
    """Trainer for quaternion-enhanced teacher model."""
    
    def __init__(self, args):
        """
        Initialize trainer.
        
        Args:
            args: Command line arguments
        """
        # Load configuration
        with open(args.config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Set device
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set seed
        init_seed(args.seed)
        
        # Set cross-validation mode
        self.cross_val = args.cross_val
        
        # Set visualization flag
        self.visualize = args.visualize
        
        # Create experiment directory
        self.work_dir = self.cfg.get("work_dir", "./exps/teacher_quat")
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Create visualization directory if needed
        if self.visualize:
            self.viz_dir = os.path.join(self.work_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.work_dir, "teacher_training.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TeacherTrainer")
        
        # Add console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(console)
        
        # Initialize metrics history
        self.metrics_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": []
        }
    
    def _get_dataset(self):
        """
        Get dataset object.
        This is a placeholder method that should be overridden.
        """
        # This is a minimal placeholder - replace with your actual dataset
        class DummyDataset:
            def __init__(self):
                self.matched_trials = []
        
        return DummyDataset()
    
    def build_data(self, subjects=None):
        """
        Build dataset and dataloaders.
        
        Args:
            subjects: Optional list of subject IDs to include
        """
        # Get dataset arguments
        db_args = self.cfg.get('dataset_args', {})
        
        # Get dataset builder
        dataset = self._get_dataset()
        self.builder = DatasetBuilderQuat(dataset=dataset, **db_args)
        
        # Use provided subjects or default from config
        if subjects is None:
            subjects = self.cfg.get("subjects", [])
        
        # Build data
        data_dict = self.builder.make_dataset(subjects)
        
        # Create dataset
        ds_all = MultimodalQuatFeeder(data_dict)
        
        # Check if we're building for cross-validation
        if self.cross_val:
            self.dataset = ds_all
            return
        
        # Split into train/val
        n = len(ds_all)
        val_size = int(n * 0.2)
        train_size = n - val_size
        
        # Use random split with seed
        generator = torch.Generator().manual_seed(42)
        ds_train, ds_val = random_split(ds_all, [train_size, val_size], generator=generator)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            ds_train, 
            batch_size=self.cfg.get("batch_size", 16),
            shuffle=True, 
            collate_fn=lambda b: pad_collate_fn(b, fixed_imu_len=128),
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            ds_val, 
            batch_size=self.cfg.get("val_batch_size", 16),
            shuffle=False, 
            collate_fn=lambda b: pad_collate_fn(b, fixed_imu_len=128),
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True
        )
        
        self.logger.info(f"Dataset built: {train_size} train, {val_size} val samples")
    
    def build_model(self):
        """Build quaternion-enhanced teacher model."""
        # Get model arguments
        model_args = self.cfg.get("model_args", {})
        
        # Build model
        self.model = QuatTeacherEnhanced(**model_args).to(self.device)
        
        # Get model summary
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Teacher model built: {num_params:,} parameters")
        
        # Print model architecture
        arch_summary = str(self.model)
        self.logger.info(f"Model architecture:\n{arch_summary}")
    
    def build_optimizer(self):
        """Build optimizer and learning rate scheduler."""
        # Get optimizer type
        optimizer_name = self.cfg.get("optimizer", "adam").lower()
        
        # Build optimizer
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg.get("base_lr", 0.001),
                weight_decay=self.cfg.get("weight_decay", 0.0001)
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.get("base_lr", 0.001),
                weight_decay=self.cfg.get("weight_decay", 0.01)
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.cfg.get("base_lr", 0.01),
                momentum=self.cfg.get("momentum", 0.9),
                weight_decay=self.cfg.get("weight_decay", 0.0001)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Build scheduler
        scheduler_name = self.cfg.get("scheduler", None)
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.get("num_epoch", 100),
                eta_min=self.cfg.get("min_lr", 1e-6)
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.get("step_size", 30),
                gamma=self.cfg.get("gamma", 0.1)
            )
        elif scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.cfg.get("gamma", 0.1),
                patience=self.cfg.get("patience", 10),
                verbose=True
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        if self.scheduler:
            self.logger.info(f"Scheduler: {self.scheduler.__class__.__name__}")
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Create loss function
        criterion = nn.CrossEntropyLoss()
        
        # Use tqdm for progress bar
        progress_bar = tqdm(
            enumerate(self.train_loader), 
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}/{self.cfg.get('num_epoch', 100)}"
        )
        
        for batch_idx, (imu_tensor, imu_mask, skel_tensor, skel_mask, labels) in progress_bar:
            # Move data to device
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
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.cfg.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.cfg.get("grad_clip")
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / total:.2f}%"
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / total if total > 0 else 0
        epoch_acc = 100.0 * correct / total if total > 0 else 0
        
        # Update metrics history
        self.metrics_history["train_loss"].append(epoch_loss)
        self.metrics_history["train_acc"].append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def eval_epoch(self, epoch):
        """
        Evaluate on validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (loss, accuracy, f1, precision, recall)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        # Create loss function
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for imu_tensor, imu_mask, skel_tensor, skel_mask, labels in self.val_loader:
                # Move data to device
                imu_tensor = imu_tensor.to(self.device)
                imu_mask = imu_mask.to(self.device)
                skel_tensor = skel_tensor.to(self.device)
                skel_mask = skel_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(skel_tensor, imu_tensor, skel_mask, imu_mask)
                logits = outputs["logits"]
                
                # Calculate loss
                loss = criterion(logits, labels)
                
                # Update metrics
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(logits, 1)
                
                # Collect predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(all_labels) if all_labels else 0
        val_acc = 100.0 * accuracy_score(all_labels, all_preds) if all_labels else 0
        
        # Calculate F1 score
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
        
        return val_loss, val_acc, val_f1, val_precision, val_recall
    
    def train(self):
        """
        Train the model for specified number of epochs.
        
        Returns:
            Best validation accuracy
        """
        # Get training parameters
        num_epochs = self.cfg.get("num_epoch", 100)
        
        # Initialize best validation metrics
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        
        # Initialize early stopping
        patience = self.cfg.get("early_stop_patience", 20)
        early_stop_counter = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall = self.eval_epoch(epoch)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
            )
            
            # Check if this is the best model
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                
                # Save best model
                model_path = os.path.join(self.work_dir, "teacher_quat_best.pth")
                torch.save(self.model.state_dict(), model_path)
                
                # Reset early stopping counter
                early_stop_counter = 0
                
                self.logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
            else:
                early_stop_counter += 1
            
            # Check for early stopping
            if early_stop_counter >= patience:
                self.logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save final model
        final_model_path = os.path.join(self.work_dir, "teacher_quat_final.pth")
        torch.save(self.model.state_dict(), final_model_path)
        
        # Save training history
        history_path = os.path.join(self.work_dir, "teacher_training_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_history = {}
            for k, v in self.metrics_history.items():
                serializable_history[k] = [float(x) for x in v]
            
            json.dump(serializable_history, f, indent=2)
        
        # Log final results
        self.logger.info(
            f"Training completed - Best F1: {best_val_f1:.4f} at epoch {best_epoch}, "
            f"Final F1: {val_f1:.4f}"
        )
        
        # Plot training curves
        if self.visualize:
            self.plot_training_curves()
        
        return best_val_acc, best_val_f1
    
    def start_cross_validation(self):
        """
        Run 5-fold cross-validation.
        
        Returns:
            Average F1 score across folds
        """
        folds = [
            ([43, 35, 36], "Fold1"),
            ([44, 34, 32], "Fold2"),
            ([45, 37, 38], "Fold3"),
            ([46, 29, 31], "Fold4"),
            ([30, 33, 39], "Fold5")
        ]
        
        self.logger.info("Starting 5-fold cross-validation")
        
        all_metrics = []
        
        for i, (val_subjects, fold_name) in enumerate(folds, start=1):
            self.logger.info(f"=== Fold {i}: {fold_name} (Validation subjects: {val_subjects}) ===")
            
            # Reset metrics history for each fold
            self.metrics_history = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
                "val_f1": [],
                "val_precision": [],
                "val_recall": []
            }
            
            # Determine train subjects (all except validation)
            all_subjects = self.cfg.get("subjects", [])
            train_subjects = [s for s in all_subjects if s not in val_subjects]
            
            # Build data for this fold
            self.build_data(train_subjects)
            train_ds = MultimodalQuatFeeder(self.builder.processed_data)
            
            self.train_loader = DataLoader(
                train_ds, 
                batch_size=self.cfg.get("batch_size", 16),
                shuffle=True, 
                collate_fn=lambda b: pad_collate_fn(b, fixed_imu_len=128),
                num_workers=self.cfg.get("num_workers", 4),
                pin_memory=True
            )
            
            self.build_data(val_subjects)
            val_ds = MultimodalQuatFeeder(self.builder.processed_data)
            
            self.val_loader = DataLoader(
                val_ds, 
                batch_size=self.cfg.get("val_batch_size", 16),
                shuffle=False, 
                collate_fn=lambda b: pad_collate_fn(b, fixed_imu_len=128),
                num_workers=self.cfg.get("num_workers", 4),
                pin_memory=True
            )
            
            # Build model and optimizer for each fold
            self.build_model()
            self.build_optimizer()
            
            # Train model
            best_val_acc, best_val_f1 = self.train()
            
            # Save fold-specific model
            fold_model_path = os.path.join(self.work_dir, f"{fold_name}_teacher_quat_best.pth")
            best_model_path = os.path.join(self.work_dir, "teacher_quat_best.pth")
            
            # Copy best model to fold-specific name
            import shutil
            shutil.copy(best_model_path, fold_model_path)
            
            # Save fold training history
            fold_history_path = os.path.join(self.work_dir, f"{fold_name}_teacher_training_history.json")
            with open(fold_history_path, 'w') as f:
                serializable_history = {}
                for k, v in self.metrics_history.items():
                    serializable_history[k] = [float(x) for x in v]
                
                json.dump(serializable_history, f, indent=2)
            
            # Plot fold training curves
            if self.visualize:
                self.plot_training_curves(fold_name)
            
            # Store best metrics for this fold
            all_metrics.append({
                "fold": fold_name,
                "val_subjects": val_subjects,
                "best_epoch": best_epoch,
                "accuracy": best_val_acc,
                "f1": best_val_f1
            })
        
        # Calculate average metrics across folds
        avg_acc = sum(m["accuracy"] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)
        
        # Log cross-validation results
        self.logger.info("=== Cross-Validation Results ===")
        for m in all_metrics:
            self.logger.info(f"{m['fold']} (Val subjects: {m['val_subjects']}): "
                            f"Acc={m['accuracy']:.2f}%, F1={m['f1']:.4f}")
        
        self.logger.info(f"Average Accuracy: {avg_acc:.2f}%")
        self.logger.info(f"Average F1: {avg_f1:.4f}")
        
        # Save cross-validation results
        cv_results = {
            "folds": all_metrics,
            "average": {
                "accuracy": float(avg_acc),
                "f1": float(avg_f1)
            }
        }
        
        cv_path = os.path.join(self.work_dir, "cross_validation_results.json")
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        return avg_f1
    
    def start(self):
        """
        Start training process.
        
        Returns:
            Best validation metrics
        """
        self.logger.info("=== Starting Teacher Training ===")
        
        # Build data, model, and optimizer
        self.build_data()
        self.build_model()
        self.build_optimizer()
        
        # Start training or cross-validation
        if self.cross_val:
            return self.start_cross_validation()
        else:
            return self.train()
    
    def plot_training_curves(self, fold_name=None):
        """
        Plot training curves.
        
        Args:
            fold_name: Optional fold name for cross-validation
        """
        # Create figure for accuracy and loss
        plt.figure(figsize=(12, 10))
        
        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history["train_loss"], label="Train Loss")
        plt.plot(self.metrics_history["val_loss"], label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot training and validation accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history["train_acc"], label="Train Accuracy")
        plt.plot(self.metrics_history["val_acc"], label="Validation Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        
        # Plot F1 score
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics_history["val_f1"], label="F1 Score")
        plt.title("F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        
        # Plot precision and recall
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics_history["val_precision"], label="Precision")
        plt.plot(self.metrics_history["val_recall"], label="Recall")
        plt.title("Precision and Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        if fold_name:
            plt.savefig(os.path.join(self.viz_dir, f"{fold_name}_training_curves.png"), dpi=150)
        else:
            plt.savefig(os.path.join(self.viz_dir, "training_curves.png"), dpi=150)
        
        plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create trainer
    trainer = TeacherTrainer(args)
    
    # Start training
    trainer.start()

if __name__ == "__main__":
    main()
