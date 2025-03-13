#!/usr/bin/env python
"""
Enhanced cross-modal knowledge distillation from teacher to student for fall detection.
"""

import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import time
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from importlib import import_module
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.loader_quat import DatasetBuilderQuat
from Feeder.multimodal_quat_feeder import MultimodalQuatFeeder, pad_collate_fn
from utils.cross_modal_loss import get_distillation_loss
from Models.transformer_quat_enhanced import QuatTeacherEnhanced, QuatStudentEnhanced

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
        description="Enhanced cross-modal distillation from teacher to student model."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/smartfallmm/distill_student_enhanced.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--teacher-weights", 
        type=str, 
        default="",
        help="Path to teacher model weights (empty for auto-detection)"
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
        "--loss-type",
        type=str,
        default="enhanced",
        choices=["enhanced", "adversarial"],
        help="Type of distillation loss to use"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize attention maps during training"
    )
    return parser.parse_args()

class EnhancedDistillTrainer:
    """Enhanced trainer for cross-modal knowledge distillation from teacher to student."""
    
    def __init__(self, arg):
        """
        Initialize distillation trainer.
        
        Args:
            arg: Command line arguments
        """
        # Load configuration
        with open(arg.config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Set device
        os.environ["CUDA_VISIBLE_DEVICES"] = arg.device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Set seed
        init_seed(arg.seed)
        
        # Set cross-validation mode
        self.cross_val = arg.cross_val
        
        # Set teacher weights path (if provided)
        self.teacher_weights = arg.teacher_weights
        
        # Set loss type
        self.loss_type = arg.loss_type
        
        # Set visualization flag
        self.visualize = arg.visualize
        
        # Create experiment directory
        self.work_dir = self.cfg.get("work_dir", "./exps/student_enhanced")
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Create visualization directory if needed
        if self.visualize:
            self.viz_dir = os.path.join(self.work_dir, "visualizations")
            os.makedirs(self.viz_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_history = {
            "train_loss_total": [],
            "train_loss_kl": [],
            "train_loss_ce": [],
            "train_loss_feat": [],
            "train_loss_attn": [],
            "train_loss_inter": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": []
        }
    
    def _get_dataset(self):
        """
        Get dataset object.
        
        This placeholder method should be replaced with your actual dataset.
        """
        # This is a minimal placeholder for the dataset
        # In your actual implementation, return your real dataset object
        class DummyDataset:
            def __init__(self):
                self.matched_trials = []
        
        return DummyDataset()
    
    def _build_data(self):
        """Build dataset and dataloaders."""
        # Get dataset arguments
        db_args = self.cfg.get('dataset_args', {})
        
        # Make sure wrist_idx is set
        wrist_idx = db_args.get('wrist_idx', 9)
        db_args['wrist_idx'] = wrist_idx
        
        # Get dataset builder
        dataset = self._get_dataset()
        self.builder = DatasetBuilderQuat(dataset=dataset, **db_args)
        
        # Check for cross-validation mode
        if self.cross_val:
            # Will build data for each fold during cross-validation
            return
        
        # Build data for normal training
        data_dict = self.builder.make_dataset(self.cfg["subjects"], max_workers=4)
        
        # Create dataset
        ds_all = MultimodalQuatFeeder(data_dict)
        
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
            batch_size=self.cfg["batch_size"],
            shuffle=True, 
            collate_fn=lambda b: pad_collate_fn(b, fixed_imu_len=128),
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            ds_val, 
            batch_size=self.cfg["val_batch_size"],
            shuffle=False, 
            collate_fn=lambda b: pad_collate_fn(b, fixed_imu_len=128),
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True
        )
    
    def _build_models(self):
        """Build enhanced teacher and student models."""
        # Create teacher model
        t_args = self.cfg["teacher_args"]
        self.teacher = QuatTeacherEnhanced(**t_args).to(self.device)
        
        # Create student model
        s_args = self.cfg["student_args"]
        self.student = QuatStudentEnhanced(**s_args).to(self.device)
        
        # Display model info
        t_params = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        s_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        
        print(f"Teacher: QuatTeacherEnhanced, Parameters: {t_params:,}")
        print(f"Student: QuatStudentEnhanced, Parameters: {s_params:,}")
        print(f"Size reduction: {(1 - s_params/t_params) * 100:.1f}%")
    
    def _load_teacher_weights(self, fold_name=None):
        """
        Load pre-trained teacher weights.
        
        Args:
            fold_name: Name of the fold for cross-validation
        
        Returns:
            True if loaded successfully, False otherwise
        """
        # Determine weights path
        if self.teacher_weights:
            # Use provided path
            weights_path = self.teacher_weights
        elif fold_name:
            # Try to find weights for this fold
            teacher_dir = self.cfg.get("teacher_weight_dir", "exps/teacher_quat")
            weights_path = os.path.join(teacher_dir, f"{fold_name}_teacher_quat_best.pth")
            
            # If not found, try alternative naming
            if not os.path.exists(weights_path):
                weights_path = os.path.join(teacher_dir, f"{fold_name}_teacher_best.pth")
        else:
            # Default path
            teacher_dir = self.cfg.get("teacher_weight_dir", "exps/teacher_quat")
            weights_path = os.path.join(teacher_dir, "teacher_quat_best.pth")
            
            # If not found, try alternative naming
            if not os.path.exists(weights_path):
                weights_path = os.path.join(teacher_dir, "teacher_best.pth")
        
        # Load weights
        if os.path.exists(weights_path):
            try:
                # Load weights with strict=False to handle architectural differences
                state_dict = torch.load(weights_path, map_location=self.device)
                
                # Handle potential key mismatches due to model architecture differences
                # This is needed when loading standard model weights into enhanced model
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('skel_enc.') or k.startswith('imu_enc.'):
                        # Adjust transformer keys
                        parts = k.split('.')
                        if len(parts) > 2 and parts[1].isdigit():
                            # Change 'skel_enc.0.xxx' to 'skel_enc.layers.0.xxx'
                            new_key = f"{parts[0]}.layers.{parts[1]}.{'.'.join(parts[2:])}"
                            new_state_dict[new_key] = v
                        else:
                            new_state_dict[k] = v
                    else:
                        new_state_dict[k] = v
                
                # Try loading with the modified state dict
                missing, unexpected = self.teacher.load_state_dict(new_state_dict, strict=False)
                
                if missing:
                    print(f"Warning: Missing keys in teacher: {missing}")
                if unexpected:
                    print(f"Warning: Unexpected keys in teacher: {unexpected}")
                
                print(f"Loaded teacher weights from {weights_path}")
                return True
            except Exception as e:
                print(f"Error loading teacher weights: {e}")
                return False
        else:
            print(f"Teacher weights not found at {weights_path}")
            return False
    
    def _build_optimizer_loss(self):
        """Build optimizer and enhanced distillation loss."""
        # Get optimizer type
        opt_name = self.cfg["optimizer"].lower()
        
        # Set up optimizer (for student only)
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.student.parameters(),
                lr=self.cfg["base_lr"],
                weight_decay=self.cfg.get("weight_decay", 0.0004)
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=self.cfg["base_lr"],
                weight_decay=self.cfg.get("weight_decay", 0.0004)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        
        # Set up learning rate scheduler
        scheduler_type = self.cfg.get("scheduler", None)
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg["num_epoch"],
                eta_min=self.cfg["base_lr"] / 100
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.get("lr_step", 20),
                gamma=self.cfg.get("lr_factor", 0.1)
            )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.cfg.get("lr_factor", 0.1),
                patience=self.cfg.get("lr_patience", 5),
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Set up distillation loss
        dist_args = self.cfg.get("distill_args", {})
        if not isinstance(dist_args, dict):
            dist_args = {}
        
        # Create distillation loss
        self.distill_loss = get_distillation_loss(
            loss_type=self.loss_type,
            **dist_args
        ).to(self.device)
        
        # If using adversarial loss, set up discriminator optimizer
        if self.loss_type == "adversarial":
            self.d_optimizer = torch.optim.Adam(
                self.distill_loss.discriminator.parameters(),
                lr=self.cfg["base_lr"] * 0.1,  # Lower LR for discriminator
                weight_decay=self.cfg.get("weight_decay", 0.0004)
            )
        
        # Training parameters
        self.num_epoch = self.cfg["num_epoch"]
        self.save_name = self.cfg.get("model_saved_name", "student_enhanced_best")
        self.print_log = self.cfg.get("print_log", True)
        
        # For gradient clipping
        self.grad_clip = self.cfg.get("grad_clip", None)
    
    def visualize_attention_maps(self, teacher_attn, student_attn, epoch, batch_idx=0, sample_idx=0):
        """
        Visualize attention maps from teacher and student.
        
        Args:
            teacher_attn: List of teacher attention maps
            student_attn: List of student attention maps
            epoch: Current epoch
            batch_idx: Batch index
            sample_idx: Sample index in batch
        """
        if not self.visualize:
            return
        
        # Create figure
        fig, axes = plt.subplots(
            2, max(len(teacher_attn), len(student_attn)), 
            figsize=(15, 6)
        )
        
        # Plot teacher attention maps
        for i, attn in enumerate(teacher_attn):
            if i >= axes.shape[1]:
                break
                
            # Select sample and average across heads
            attn_map = attn[sample_idx].mean(dim=0).cpu().numpy()
            
            # Plot heatmap
            im = axes[0, i].imshow(attn_map, cmap='viridis')
            axes[0, i].set_title(f"Teacher Layer {i+1}")
            axes[0, i].axis('off')
            
            # Add colorbar
            fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Plot student attention maps
        for i, attn in enumerate(student_attn):
            if i >= axes.shape[1]:
                break
                
            # Select sample and average across heads
            attn_map = attn[sample_idx].mean(dim=0).cpu().numpy()
            
            # Plot heatmap
            im = axes[1, i].imshow(attn_map, cmap='viridis')
            axes[1, i].set_title(f"Student Layer {i+1}")
            axes[1, i].axis('off')
            
            # Add colorbar
            fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # Set overall title
        plt.suptitle(f"Attention Maps - Epoch {epoch+1}, Batch {batch_idx}, Sample {sample_idx}")
        
        # Save figure
        viz_path = os.path.join(
            self.viz_dir, 
            f"attention_ep{epoch+1}_batch{batch_idx}_sample{sample_idx}.png"
        )
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def start(self):
        """Start the enhanced distillation process."""
        if self.cross_val:
            return self.start_cross_validation()
        
        print("\n" + "="*60)
        print(f"Starting enhanced distillation with {self.cfg['dataset_args']['imu_fusion']} fusion")
        print(f"Loss type: {self.loss_type}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        # Build data, models, and optimizer
        self._build_data()
        self._build_models()
        
        # Load teacher weights
        if not self._load_teacher_weights():
            print("Failed to load teacher weights, aborting.")
            return
        
        # Freeze teacher weights
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher.eval()
        
        # Build optimizer and loss
        self._build_optimizer_loss()
        
        # Distill knowledge
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        early_stop_count = 0
        early_stop_patience = self.cfg.get("early_stop_patience", 15)
        
        for epoch in range(self.num_epoch):
            # Train
            train_losses, train_acc = self._train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall = self._eval_epoch(epoch)
            
            # Update scheduler if using plateau scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                
                # Save model
                model_path = os.path.join(self.work_dir, f"{self.save_name}.pth")
                torch.save(self.student.state_dict(), model_path)
                print(f"Model saved to {model_path}")
                
                # Reset early stopping counter
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            # Check early stopping
            if early_stop_count >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Print current results
            if self.print_log:
                print(f"[Epoch {epoch+1}/{self.num_epoch}] "
                      f"Train Loss: {train_losses['total']:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                      f"F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                      f"Best F1: {best_val_f1:.4f} (Epoch {best_epoch})")
            
            # Update metrics history
            self.metrics_history["train_loss_total"].append(train_losses["total"])
            self.metrics_history["train_loss_kl"].append(train_losses.get("kl", 0))
            self.metrics_history["train_loss_ce"].append(train_losses.get("ce", 0))
            self.metrics_history["train_loss_feat"].append(train_losses.get("feat", 0))
            self.metrics_history["train_loss_attn"].append(train_losses.get("attn", 0))
            self.metrics_history["train_loss_inter"].append(train_losses.get("inter", 0))
            self.metrics_history["val_loss"].append(val_loss)
            self.metrics_history["train_acc"].append(train_acc)
            self.metrics_history["val_acc"].append(val_acc)
            self.metrics_history["val_f1"].append(val_f1)
            self.metrics_history["val_precision"].append(val_precision)
            self.metrics_history["val_recall"].append(val_recall)
        
        # Save training history
        history_path = os.path.join(self.work_dir, "enhanced_distillation_history.json")
        with open(history_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_history = {}
            for k, v in self.metrics_history.items():
                serializable_history[k] = [float(x) for x in v]
            
            json.dump(serializable_history, f, indent=2)
        
        print("\n" + "="*60)
        print(f"Enhanced distillation completed. Best F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        print(f"Final metrics - Accuracy: {best_val_acc:.2f}%, F1: {best_val_f1:.4f}")
        print("="*60 + "\n")
        
        # Plot training curves
        self.plot_training_curves()
        
        return best_val_acc, best_val_f1
    
    def start_cross_validation(self):
        """Run 5-fold cross-validation with enhanced distillation."""
        folds = [
            ([43, 35, 36], "Fold1"),
            ([44, 34, 32], "Fold2"),
            ([45, 37, 38], "Fold3"),
            ([46, 29, 31], "Fold4"),
            ([30, 33, 39], "Fold5")
        ]
        
        all_metrics = []
        
        print("\n" + "="*60)
        print(f"Starting 5-fold cross-validation enhanced distillation with {self.cfg['dataset_args']['imu_fusion']} fusion")
        print(f"Loss type: {self.loss_type}")
        print("="*60 + "\n")
        
        # Build models once to check structure
        self._build_models()
        
        # Get dataset
        dataset = self._get_dataset()
        db_args = self.cfg.get('dataset_args', {})
        self.builder = DatasetBuilderQuat(dataset=dataset, **db_args)
        
        for fold_idx, (val_subjects, fold_name) in enumerate(folds, start=1):
            print(f"\n--- {fold_name} (Validation Subjects: {val_subjects}) ---")
            
            # Determine train subjects (all except validation)
            all_subjects = self.cfg["subjects"]
            train_subjects = [s for s in all_subjects if s not in val_subjects]
            
            # Build data for this fold
            train_data = self.builder.make_dataset(train_subjects, max_workers=4)
            val_data = self.builder.make_dataset(val_subjects, max_workers=4)
            
            # Create datasets
            train_ds = MultimodalQuatFeeder(train_data)
            val_ds = MultimodalQuatFeeder(val_data)
            
            # Create dataloaders
            self.train_loader = DataLoader(
                train_ds, 
                batch_size=self.cfg["batch_size"],
                shuffle=True, 
                collate_fn=lambda b: pad_collate_fn(b, fixed_imu_len=128),
                num_workers=self.cfg.get("num_workers", 4),
                pin_memory=True
            )
            
            self.val_loader = DataLoader(
                val_ds, 
                batch_size=self.cfg["val_batch_size"],
                shuffle=False, 
                collate_fn=lambda b: pad_collate_fn(b, fixed_imu_len=128),
                num_workers=self.cfg.get("num_workers", 4),
                pin_memory=True
            )
            
            # Rebuild models for each fold
            self._build_models()
            
            # Load teacher weights for this fold
            if not self._load_teacher_weights(fold_name):
                print(f"Failed to load teacher weights for {fold_name}, skipping.")
                continue
            
            # Freeze teacher
            for param in self.teacher.parameters():
                param.requires_grad = False
            
            self.teacher.eval()
            
            # Build optimizer and loss
            self._build_optimizer_loss()
            
            # Initialize fold metrics
            best_val_acc = 0.0
            best_val_f1 = 0.0
            best_epoch = 0
            fold_metrics = {
                "train_loss_total": [],
                "train_loss_kl": [],
                "train_loss_ce": [],
                "train_loss_feat": [],
                "train_loss_attn": [],
                "train_loss_inter": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": [],
                "val_f1": [],
                "val_precision": [],
                "val_recall": []
            }
            
            for epoch in range(self.num_epoch):
                # Train
                train_losses, train_acc = self._train_epoch(epoch, fold_name)
                
                # Validate
                val_loss, val_acc, val_f1, val_precision, val_recall = self._eval_epoch(epoch)
                
                # Update scheduler
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler is not None:
                    self.scheduler.step()
                
                # Save best model
                if val_f1 > best_val_f1:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_epoch = epoch + 1
                    
                    # Save model
                    model_path = os.path.join(self.work_dir, f"{fold_name}_{self.save_name}.pth")
                    torch.save(self.student.state_dict(), model_path)
                
                # Update fold metrics
                fold_metrics["train_loss_total"].append(train_losses["total"])
                fold_metrics["train_loss_kl"].append(train_losses.get("kl", 0))
                fold_metrics["train_loss_ce"].append(train_losses.get("ce", 0))
                fold_metrics["train_loss_feat"].append(train_losses.get("feat", 0))
                fold_metrics["train_loss_attn"].append(train_losses.get("attn", 0))
                fold_metrics["train_loss_inter"].append(train_losses.get("inter", 0))
                fold_metrics["val_loss"].append(val_loss)
                fold_metrics["train_acc"].append(train_acc)
                fold_metrics["val_acc"].append(val_acc)
                fold_metrics["val_f1"].append(val_f1)
                fold_metrics["val_precision"].append(val_precision)
                fold_metrics["val_recall"].append(val_recall)
                
                # Print results
                if self.print_log:
                    print(f"[{fold_name} Epoch {epoch+1}/{self.num_epoch}] "
                          f"Train Loss: {train_losses['total']:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                          f"F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            
            # Save fold metrics
            fold_metrics_path = os.path.join(self.work_dir, f"{fold_name}_enhanced_metrics.json")
            with open(fold_metrics_path, 'w') as f:
                # Convert numpy values to Python native types for JSON serialization
                serializable_metrics = {}
                for k, v in fold_metrics.items():
                    serializable_metrics[k] = [float(x) for x in v]
                
                json.dump(serializable_metrics, f, indent=2)
            
            # Plot fold training curves
            self.plot_training_curves(fold_metrics, fold_name)
            
            # Store best metrics for this fold
            all_metrics.append({
                "fold": fold_name,
                "val_subjects": val_subjects,
                "best_epoch": best_epoch,
                "accuracy": best_val_acc,
                "f1": best_val_f1,
                "precision": val_precision,
                "recall": val_recall
            })
            
            print(f"[{fold_name}] Best F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        
        # Calculate and print average results
        avg_acc = sum(m["accuracy"] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)
        avg_precision = sum(m["precision"] for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)
        
        print("\n" + "="*60)
        print("Cross-Validation Results:")
        for m in all_metrics:
            print(f"{m['fold']} (Val: {m['val_subjects']}): Acc={m['accuracy']:.2f}%, F1={m['f1']:.4f}")
        print("-"*60)
        print(f"Average Accuracy: {avg_acc:.2f}%")
        print(f"Average F1 Score: {avg_f1:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print("="*60 + "\n")
        
        # Save cross-validation results
        cv_results = {
            "folds": all_metrics,
            "average": {
                "accuracy": float(avg_acc),
                "f1": float(avg_f1),
                "precision": float(avg_precision),
                "recall": float(avg_recall)
            }
        }
        
        cv_path = os.path.join(self.work_dir, "cross_validation_enhanced_results.json")
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=2)
        
        return avg_acc, avg_f1
    
    def _train_epoch(self, epoch, fold_name=None):
        """
        Train for one epoch with enhanced distillation.
        
        Args:
            epoch: Current epoch number
            fold_name: Name of the fold (for visualization)
            
        Returns:
            Tuple of (train_losses, train_accuracy)
        """
        self.student.train()
        self.teacher.eval()  # Teacher is always in eval mode
        
        total_loss = 0.0
        total_kl_loss = 0.0
        total_ce_loss = 0.0
        total_feat_loss = 0.0
        total_attn_loss = 0.0
        total_inter_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # For adversarial loss
        total_d_loss = 0.0
        total_g_loss = 0.0
        
        # Get the current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Learning Rate: {current_lr:.6f}")
        
        # Use tqdm for progress bar
        for batch_idx, (imu_tensor, imu_mask, skel_tensor, skel_mask, labels) in enumerate(tqdm(
            self.train_loader, desc=f"Epoch {epoch+1} Train"
        )):
            # Move data to device
            imu_tensor = imu_tensor.to(self.device)
            imu_mask = imu_mask.to(self.device)
            skel_tensor = skel_tensor.to(self.device)
            skel_mask = skel_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass for teacher (with no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher(skel_tensor, imu_tensor, skel_mask, imu_mask)
            
            # For adversarial loss: train discriminator first
            if self.loss_type == "adversarial":
                self.d_optimizer.zero_grad()
                
                # Forward pass for student (detached for discriminator)
                student_outputs = self.student(imu_tensor, imu_mask)
                
                # Train discriminator
                losses = self.distill_loss(student_outputs, teacher_outputs, labels)
                d_loss = losses["d_loss"]
                d_loss.backward()
                self.d_optimizer.step()
                
                total_d_loss += d_loss.item() * labels.size(0)
            
            # Forward pass for student
            self.optimizer.zero_grad()
            student_outputs = self.student(imu_tensor, imu_mask)
            
            # Enhanced distillation loss
            losses = self.distill_loss(student_outputs, teacher_outputs, labels)
            loss = losses["total"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if enabled
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses["total"].item() * labels.size(0)
            if "kl" in losses:
                total_kl_loss += losses["kl"].item() * labels.size(0)
            if "ce" in losses:
                total_ce_loss += losses["ce"].item() * labels.size(0)
            if "feat" in losses:
                total_feat_loss += losses["feat"].item() * labels.size(0)
            if "attn" in losses:
                total_attn_loss += losses["attn"].item() * labels.size(0)
            if "inter" in losses:
                total_inter_loss += losses["inter"].item() * labels.size(0)
            if "adv" in losses:
                total_g_loss += losses["adv"].item() * labels.size(0)
            
            _, predicted = torch.max(student_outputs["logits"], dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Visualize attention maps occasionally
            if self.visualize and batch_idx % 50 == 0:
                if "attentions" in student_outputs and "imu_attentions" in teacher_outputs:
                    self.visualize_attention_maps(
                        teacher_outputs["imu_attentions"],
                        student_outputs["attentions"],
                        epoch,
                        batch_idx,
                        0  # First sample in batch
                    )
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_kl_loss = total_kl_loss / total_samples if total_samples > 0 else 0
        avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else 0
        avg_feat_loss = total_feat_loss / total_samples if total_samples > 0 else 0
        avg_attn_loss = total_attn_loss / total_samples if total_samples > 0 else 0
        avg_inter_loss = total_inter_loss / total_samples if total_samples > 0 else 0
        
        if self.loss_type == "adversarial":
            avg_d_loss = total_d_loss / total_samples if total_samples > 0 else 0
            avg_g_loss = total_g_loss / total_samples if total_samples > 0 else 0
        
        epoch_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
        
        # Collect all losses
        losses_dict = {
            "total": avg_loss,
            "kl": avg_kl_loss,
            "ce": avg_ce_loss,
            "feat": avg_feat_loss,
            "attn": avg_attn_loss,
            "inter": avg_inter_loss
        }
        
        if self.loss_type == "adversarial":
            losses_dict.update({
                "d_loss": avg_d_loss,
                "g_loss": avg_g_loss
            })
        
        return losses_dict, epoch_accuracy
    
    def _eval_epoch(self, epoch):
        """
        Evaluate student model on validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (val_loss, val_accuracy, val_f1, val_precision, val_recall)
        """
        self.student.eval()
        self.teacher.eval()
        
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for imu_tensor, imu_mask, skel_tensor, skel_mask, labels in self.val_loader:
                # Move data to device
                imu_tensor = imu_tensor.to(self.device)
                imu_mask = imu_mask.to(self.device)
                skel_tensor = skel_tensor.to(self.device)
                skel_mask = skel_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass for teacher
                teacher_outputs = self.teacher(skel_tensor, imu_tensor, skel_mask, imu_mask)
                
                # Forward pass for student
                student_outputs = self.student(imu_tensor, imu_mask)
                
                # Compute distillation loss (use only total)
                losses = self.distill_loss(student_outputs, teacher_outputs, labels)
                loss = losses["total"]
                
                # Update metrics
                total_loss += loss.item() * labels.size(0)
                
                _, predicted = torch.max(student_outputs["logits"], dim=1)
                
                # Collect all predictions and labels for metrics
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # Calculate epoch metrics
        val_loss = total_loss / len(all_labels) if all_labels else 0
        val_accuracy = 100.0 * accuracy_score(all_labels, all_preds) if all_labels else 0
        
        # For binary classification
        if self.cfg.get("student_args", {}).get("num_classes", 2) == 2:
            val_f1 = f1_score(all_labels, all_preds, average='binary')
            val_precision = precision_score(all_labels, all_preds, average='binary')
            val_recall = recall_score(all_labels, all_preds, average='binary')
        else:
            # For multi-class
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            val_precision = precision_score(all_labels, all_preds, average='macro')
            val_recall = recall_score(all_labels, all_preds, average='macro')
        
        return val_loss, val_accuracy, val_f1, val_precision, val_recall
    
    def plot_training_curves(self, metrics=None, prefix=None):
        """
        Plot training curves.
        
        Args:
            metrics: Metrics dictionary (if None, use self.metrics_history)
            prefix: Prefix for filenames (e.g., fold name)
        """
        if metrics is None:
            metrics = self.metrics_history
        
        prefix = f"{prefix}_" if prefix else ""
        
        # Create figure for accuracy, f1, precision, recall
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(metrics["train_acc"], label="Train Accuracy")
        plt.plot(metrics["val_acc"], label="Validation Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        
        # Plot F1 score
        plt.subplot(2, 2, 2)
        plt.plot(metrics["val_f1"], label="Validation F1")
        plt.title("F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        
        # Plot precision
        plt.subplot(2, 2, 3)
        plt.plot(metrics["val_precision"], label="Validation Precision")
        plt.title("Precision")
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        
        # Plot recall
        plt.subplot(2, 2, 4)
        plt.plot(metrics["val_recall"], label="Validation Recall")
        plt.title("Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        metrics_path = os.path.join(self.work_dir, f"{prefix}metrics_curves.png")
        plt.savefig(metrics_path, dpi=150)
        plt.close()
        
        # Create figure for losses
        plt.figure(figsize=(12, 8))
        
        # Plot total loss
        plt.subplot(2, 3, 1)
        plt.plot(metrics["train_loss_total"], label="Train Loss")
        plt.plot(metrics["val_loss"], label="Validation Loss")
        plt.title("Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot KL loss
        plt.subplot(2, 3, 2)
        plt.plot(metrics["train_loss_kl"], label="KL Loss")
        plt.title("KL Divergence Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot CE loss
        plt.subplot(2, 3, 3)
        plt.plot(metrics["train_loss_ce"], label="CE Loss")
        plt.title("Cross-Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot feature loss
        plt.subplot(2, 3, 4)
        plt.plot(metrics["train_loss_feat"], label="Feature Loss")
        plt.title("Feature Alignment Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot attention loss
        plt.subplot(2, 3, 5)
        plt.plot(metrics["train_loss_attn"], label="Attention Loss")
        plt.title("Attention Map Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot intermediate loss
        plt.subplot(2, 3, 6)
        plt.plot(metrics["train_loss_inter"], label="Intermediate Loss")
        plt.title("Intermediate Layer Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        losses_path = os.path.join(self.work_dir, f"{prefix}loss_curves.png")
        plt.savefig(losses_path, dpi=150)
        plt.close()

def main():
    """Main function."""
    # Parse arguments
    arg = parse_args()
    
    # Run enhanced distillation
    trainer = EnhancedDistillTrainer(arg)
    trainer.start()

if __name__ == "__main__":
    main()
