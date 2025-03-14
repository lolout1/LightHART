# train_teacher_enhanced.py

#!/usr/bin/env python
"""
Train enhanced teacher model for fall detection.
"""

import argparse
import yaml
import os
import torch
import logging
import time
from torch.utils.data import DataLoader, random_split
import numpy as np

from utils.enhanced_dataset_builder import EnhancedDatasetBuilder
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train enhanced teacher model for fall detection."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/smartfallmm/teacher_enhanced.yaml",
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
    return parser.parse_args()

class EnhancedTeacherTrainer:
    """
    Trainer for enhanced teacher model.
    """
    
    def __init__(self, args):
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
        
        # Create experiment directory
        self.work_dir = self.cfg.get("work_dir", "./exps/teacher_enhanced")
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.work_dir, "teacher_training.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("EnhancedTeacherTrainer")
        
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
        """Get dataset object."""
        # This should be replaced with your actual dataset
        # Implement based on your SmartFallMM dataset class
        pass
    
    def build_data(self, subjects=None):
        """Build dataset with enhanced alignment and fusion."""
        # Get dataset arguments
        db_args = self.cfg.get('dataset_args', {})
        
        # Make sure wrist_idx is set
        wrist_idx = db_args.get('wrist_idx', 9)
        db_args['wrist_idx'] = wrist_idx
        
        # Get dataset builder
        dataset = self._get_dataset()
        self.builder = EnhancedDatasetBuilder(dataset=dataset, **db_args)
        
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
        """Build enhanced teacher model optimized for fall detection."""
        # Get model arguments
        model_args = self.cfg.get("model_args", {})
        
        # Build model
        self.model = QuatTeacherEnhanced(**model_args).to(self.device)
        
        # Get model summary
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Enhanced teacher model built: {num_params:,} parameters")
    
    def build_optimizer(self):
        """Build optimizer and learning rate scheduler."""
        # Get optimizer type
        optimizer_name = self.cfg.get("optimizer", "adam").lower()
        
        # Build optimizer
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.get("base_lr", 0.001),
                weight_decay=self.cfg.get("weight_decay", 0.0001)
            )
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.get("base_lr", 0.001),
                weight_decay=self.cfg.get("weight_decay", 0.01)
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.get("base_lr", 0.01),
                momentum=self.cfg.get("momentum", 0.9),
                weight_decay=self.cfg.get("weight_decay", 0.0001)
            )
        
        # Build scheduler
        scheduler_name = self.cfg.get("scheduler", None)
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.get("num_epoch", 100),
                eta_min=self.cfg.get("min_lr", 1e-6)
            )
        elif scheduler_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.get("step_size", 30),
                gamma=self.cfg.get("gamma", 0.1)
            )
        elif scheduler_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Create loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        for batch_idx, (imu_tensor, imu_mask, skel_tensor, skel_mask, labels) in enumerate(self.train_loader):
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
            
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.2f}%"
                )
        
        # Calculate epoch metrics
        epoch_loss = total_loss / total if total > 0 else 0
        epoch_acc = 100.0 * correct / total if total > 0 else 0
        
        # Update metrics history
        self.metrics_history["train_loss"].append(epoch_loss)
        self.metrics_history["train_acc"].append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def eval_epoch(self, epoch):
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        # Create loss function
        criterion = torch.nn.CrossEntropyLoss()
        
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
        
        return val_loss, val_acc, val_f1, val_precision, val_recall
    
    def train(self):
        """Train the model."""
        # Get training parameters
        num_epochs = self.cfg.get("num_epoch", 100)
        
        # Initialize best validation metrics
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        
        # Initialize early stopping
        patience = self.cfg.get("early_stop_patience", 20)
        early_stop_counter = 0
        
        self.logger.info(f"Starting enhanced teacher training for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall = self.eval_epoch(epoch)
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
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
                model_path = os.path.join(self.work_dir, "teacher_enhanced_best.pth")
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
        final_model_path = os.path.join(self.work_dir, "teacher_enhanced_final.pth")
        torch.save(self.model.state_dict(), final_model_path)
        
        # Save training history
        history_path = os.path.join(self.work_dir, "teacher_training_history.json")
        import json
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
        
        return best_val_acc, best_val_f1
    
    def start(self):
        """Start the training process."""
        self.logger.info("=== Starting Enhanced Teacher Training ===")
        
        # Build data, model, and optimizer
        self.build_data()
        self.build_model()
        self.build_optimizer()
        
        # Start training
        if self.cross_val:
            return self.start_cross_validation()
        else:
            return self.train()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create trainer
    trainer = EnhancedTeacherTrainer(args)
    
    # Start training
    trainer.start()

if __name__ == "__main__":
    main()
