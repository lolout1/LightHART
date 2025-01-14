import argparse
import yaml
import traceback
import random
import sys
import os
import time
import shutil
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, precision_recall_fscore_support, accuracy_score

# Scientific computing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Progress bar, metrics
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# Local imports
from Models.TeacherModel import TeacherModel
from Models.StudentModel import StudentModel
from utils.dataset import prepare_smartfallmm, filter_subjects
from Models.StudentTrans import LightTransformerStudent


####################################################
#          DistillationLoss for Binary KD
####################################################
class DistillationLoss(nn.Module):
    """
    Distillation Loss for binary classification tasks.
    Combines BCE loss with soft targets from teacher model.
    alpha: weight for BCE with hard labels (ground truth)
    (1-alpha): weight for KL divergence with teacher's soft predictions
    temperature: softening temperature for KL divergence
    """
    def __init__(self, alpha=0.5, temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits, labels):
        """
        Forward pass of distillation loss
        Args:
            student_logits: Raw logits from student model [B] or [B,1]
            teacher_logits: Raw logits from teacher model [B] or [B,1]
            labels: Ground truth labels [B]
        Returns:
            Total loss combining hard and soft targets
        """
        # Ensure inputs have correct shape
        if student_logits.dim() == 2:
            student_logits = student_logits.squeeze(1)
        if teacher_logits.dim() == 2:
            teacher_logits = teacher_logits.squeeze(1)
        
        # Hard label loss with ground truth
        hard_loss = self.bce(student_logits, labels.float())
        
        # Soft label loss with teacher predictions
        # Apply temperature scaling
        soft_student = F.log_softmax(torch.stack([student_logits, -student_logits], dim=1) / self.temperature, dim=1)
        soft_teacher = F.softmax(torch.stack([teacher_logits, -teacher_logits], dim=1) / self.temperature, dim=1)
        
        # KL divergence loss
        soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        
        return total_loss

####################################################
#        Helper Functions
####################################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "y", "1")

def import_class(name):
    try:
        mod_str, _sep, class_str = name.rpartition('.')
        __import__(mod_str)
        return getattr(sys.modules[mod_str], class_str)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing {name}: {str(e)}")

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

def process_args(args_dict):
    dict_keys = [
        'student_args',
        'teacher_args',
        'dataset_args',
        'train_feeder_args',
        'val_feeder_args',
        'test_feeder_args',
        'distill_args'
    ]
    for k in dict_keys:
        if k in args_dict and isinstance(args_dict[k], str):
            try:
                args_dict[k] = eval(args_dict[k])
            except:
                args_dict[k] = {}
    return args_dict

####################################################
#        Main Distiller Class
####################################################
class FallDetectionDistiller:
    def __init__(self, arg):
        self.arg = arg
        self.best_f1 = 0
        self.setup_environment()
        self.setup_models()
        self.load_data()
        self.setup_training_components()

    def setup_environment(self):
        # Create run directory
        base_dir = self.arg.work_dir
        run_dirs = [d for d in os.listdir(base_dir) if d.startswith('distill_run')]
        run_num = len(run_dirs) + 1
        self.run_dir = os.path.join(base_dir, f'distill_run{run_num}')
        os.makedirs(self.run_dir, exist_ok=True)
        
        # GPU selection
        self.device = (
            f'cuda:{self.arg.device[0]}'
            if torch.cuda.is_available() and len(self.arg.device) > 0
            else 'cpu'
        )
        os.makedirs(self.arg.work_dir, exist_ok=True)

    def get_teacher_weight_for_fold(self, fold):
        """Get teacher weight path for specific fold"""
        teacher_weights = {
            1: "/home/abheekp/Fall_Detection_KD_Multimodal/single_sensor/LightHART/exps/smartfall_har/kd/student/fold_1/TeacherModel_best_weights_f1_0.9353_loss_0.2529.pt",
            2: "/home/abheekp/Fall_Detection_KD_Multimodal/single_sensor/LightHART/exps/smartfall_har/kd/student/fold_2/TeacherModel_best_weights_f1_1.0000_loss_0.0002.pt",
            3: "/home/abheekp/Fall_Detection_KD_Multimodal/single_sensor/LightHART/exps/smartfall_har/kd/student/fold_3/TeacherModel_best_weights_f1_0.9840_loss_0.0270.pt",
            4: "/home/abheekp/Fall_Detection_KD_Multimodal/single_sensor/LightHART/exps/smartfall_har/kd/student/fold_4/TeacherModel_best_weights_f1_0.9569_loss_0.1406.pt",
            5: "/home/abheekp/Fall_Detection_KD_Multimodal/single_sensor/LightHART/exps/smartfall_har/kd/student/fold_5/TeacherModel_best_weights_f1_1.0000_loss_0.0122.pt"
        }
        return teacher_weights.get(fold)

    def setup_models(self, fold=None):
        # Teacher
        self.teacher = TeacherModel(**self.arg.teacher_args).to(self.device)
        teacher_weight = self.get_teacher_weight_for_fold(fold) if fold else self.arg.teacher_weight
        teacher_ckpt = torch.load(teacher_weight, map_location=self.device)
        
        if isinstance(teacher_ckpt, dict):
            if 'model_state_dict' in teacher_ckpt:
                self.teacher.load_state_dict(teacher_ckpt['model_state_dict'], strict=False)
            elif 'state_dict' in teacher_ckpt:
                self.teacher.load_state_dict(teacher_ckpt['state_dict'], strict=False)
            else:
                self.teacher.load_state_dict(teacher_ckpt, strict=False)
        else:
            self.teacher.load_state_dict(teacher_ckpt, strict=False)
        self.teacher.eval()

        # Student
        self.student = StudentModel(**self.arg.student_args).to(self.device)

        # Optionally load pre-trained student
        if hasattr(self.arg, 'weights') and self.arg.weights:
            student_w = torch.load(self.arg.weights, map_location=self.device)
            if isinstance(student_w, dict):
                if 'model_state_dict' in student_w:
                    self.student.load_state_dict(student_w['model_state_dict'], strict=False)
                elif 'state_dict' in student_w:
                    self.student.load_state_dict(student_w['state_dict'], strict=False)
                else:
                    self.student.load_state_dict(student_w, strict=False)
            else:
                self.student.load_state_dict(student_w, strict=False)

    def load_data(self):
        builder = prepare_smartfallmm(self.arg)
        # Example manual split
        test_subjects = [29, 30, 31]
        train_subjects = [32, 33, 34, 35, 36, 37, 38, 39, 43, 44, 45, 46]

        train_data = filter_subjects(builder, train_subjects)
        val_data = filter_subjects(builder, test_subjects)

        FeederClass = import_class(self.arg.feeder)

        self.data_loader = {}
        self.data_loader['train'] = DataLoader(
            dataset=FeederClass(**self.arg.train_feeder_args, dataset=train_data),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            pin_memory=True
        )

        if self.arg.include_val:
            self.data_loader['val'] = DataLoader(
                dataset=FeederClass(**self.arg.val_feeder_args, dataset=val_data),
                batch_size=self.arg.val_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                pin_memory=True
            )

    def setup_training_components(self):
        if self.arg.phase.lower() == 'train':
            # Distillation
            self.distill_criterion = DistillationLoss(**self.arg.distill_args)

            # BCE for val
            self.val_criterion = nn.BCELoss()

            self.optimizer = optim.AdamW(
                self.student.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )

            steps_per_epoch = len(self.data_loader['train'])
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.arg.base_lr,
                epochs=self.arg.num_epoch,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                div_factor=25,
                final_div_factor=1e4
            )
        else:
            self.distill_criterion = None
            self.val_criterion = None
            self.optimizer = None
            self.scheduler = None

    def train_epoch(self, epoch):
        self.student.train()
        loader = self.data_loader['train']

        epoch_loss = 0.0
        all_preds, all_targets = [], []

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True):
            inputs, targets, _ = batch
            acc_data = inputs['accelerometer'].to(self.device).float()
            skl_data = inputs['skeleton'].to(self.device).float()
            targets = targets.to(self.device).float()

            # Teacher
            with torch.no_grad():
                teacher_logits, _ = self.teacher(acc_data, skl_data)  # [B], [B,256]

            # Student
            student_logits, _ = self.student(acc_data)  # [B], [B,64]

            # Distillation loss => final output + features
            loss = self.distill_criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=targets
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()

            # predictions
            preds = (student_logits > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss /= len(loader)
        metrics = self.compute_metrics(all_preds, all_targets)
        metrics['loss'] = epoch_loss
        return metrics

    def validate(self, epoch):
        self.student.eval()
        loader = self.data_loader['val']

        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation", dynamic_ncols=True):
                inputs, targets, _ = batch
                acc_data = inputs['accelerometer'].to(self.device).float()
                skl_data = inputs['skeleton'].to(self.device).float()
                targets = targets.to(self.device).float()

                # Teacher
                teacher_logits, _ = self.teacher(acc_data, skl_data)

                # Student
                student_logits, _ = self.student(acc_data)

                # Distillation loss (same as train)
                loss = self.distill_criterion(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=targets
                )
                val_loss += loss.item()

                preds = (student_logits > 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        val_loss /= len(loader)
        metrics = self.compute_metrics(all_preds, all_targets)
        metrics['loss'] = val_loss
        return metrics

    def compute_metrics(self, preds, targets):
        preds_np = np.array(preds).flatten()
        targets_np = np.array(targets).flatten()
        return {
            'precision': precision_score(targets_np, preds_np, zero_division=0),
            'recall':    recall_score(targets_np, preds_np, zero_division=0),
            'f1':        f1_score(targets_np, preds_np, zero_division=0),
            'accuracy':  np.mean(preds_np == targets_np)
        }

    def save_model(self, fold_dir, epoch, metrics, is_best=False):
        """Save model checkpoint with detailed metrics"""
        lr = self.optimizer.param_groups[0]['lr']
        save_dict = {
            'epoch': epoch,
            'state_dict': self.student.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        model_name = (
            f"student_fold{fold_dir.split('_')[-1]}_"
            f"f1_{metrics['f1']:.4f}_"
            f"loss_{metrics['loss']:.4f}_"
            f"acc_{metrics['accuracy']:.4f}_"
            f"lr_{lr:.6f}.pt"
        )
        
        save_path = os.path.join(fold_dir, model_name)
        torch.save(save_dict, save_path)
        self.print_log(f"Saved model checkpoint to {save_path}")
        
        if is_best:
            best_path = os.path.join(fold_dir, 'best_model.pt')
            shutil.copy(save_path, best_path)
            self.print_log(f"Saved best model to {best_path}")

    def save_fold_results(self, fold, train_losses, val_losses, metrics, fold_dir, train_subjects, val_subjects, teacher_weight):
        """Save comprehensive results for each fold"""
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save training info
        info = {
            'fold': fold,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'teacher_weight': teacher_weight,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': metrics
        }
        torch.save(info, os.path.join(fold_dir, 'fold_info.pt'))
        
        # Plot and save loss curves
        plt.figure(figsize=(10, 6))
        epochs = range(len(train_losses))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - Fold {fold}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fold_dir, 'loss_curve.png'))
        plt.close()
        
        # Plot precision-recall curve for best epoch
        best_epoch = max(range(len(metrics)), key=lambda i: metrics[i]['f1'])
        best_metrics = metrics[best_epoch]
        
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot([best_metrics['precision']], [best_metrics['recall']], 'ro', label='Operating Point')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title(f'Precision-Recall Curve - Fold {fold} (Best Epoch)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fold_dir, 'pr_curve.png'))
        plt.close()

    def analyze_dataset_distribution(self, dataset, name=""):
        """Analyze fall vs non-fall distribution in the dataset"""
        labels = dataset['labels']
        total_samples = len(labels)
        falls = np.sum(labels)
        non_falls = total_samples - falls
        
        # Calculate percentages
        falls_percent = (falls / total_samples) * 100
        non_falls_percent = (non_falls / total_samples) * 100
        
        # Calculate ratios
        falls_ratio = falls / total_samples
        non_falls_ratio = non_falls / total_samples
        falls_to_non_falls = falls / non_falls if non_falls > 0 else float('inf')
        
        self.print_log(f"\nDataset Distribution ({name}):")
        self.print_log(f"Total Samples: {total_samples}")
        self.print_log(f"Falls: {falls} ({falls_percent:.2f}%)")
        self.print_log(f"Non-Falls: {non_falls} ({non_falls_percent:.2f}%)")
        self.print_log("\nRatios:")
        self.print_log(f"Falls:Total = {falls_ratio:.3f}")
        self.print_log(f"Non-Falls:Total = {non_falls_ratio:.3f}")
        self.print_log(f"Falls:Non-Falls = {falls_to_non_falls:.3f}")
        
        return {
            'total': total_samples,
            'falls': int(falls),
            'non_falls': int(non_falls),
            'falls_percent': falls_percent,
            'non_falls_percent': non_falls_percent,
            'falls_ratio': falls_ratio,
            'non_falls_ratio': non_falls_ratio,
            'falls_to_non_falls': falls_to_non_falls
        }

    def analyze_subject_distribution(self, subject):
        """Analyze fall vs non-fall distribution for a specific subject"""
        builder = prepare_smartfallmm(self.arg)
        builder.make_dataset([subject])  # Need to call make_dataset before accessing data
        subject_data = builder.data
        
        if 'labels' not in subject_data:
            return {
                'subject': subject,
                'total': 0,
                'falls': 0,
                'non_falls': 0,
                'falls_percent': 0,
                'non_falls_percent': 0
            }
        
        labels = subject_data['labels']
        total_samples = len(labels)
        falls = np.sum(labels)
        non_falls = total_samples - falls
        
        falls_percent = (falls / total_samples) * 100 if total_samples > 0 else 0
        non_falls_percent = (non_falls / total_samples) * 100 if total_samples > 0 else 0
        
        return {
            'subject': subject,
            'total': total_samples,
            'falls': int(falls),
            'non_falls': int(non_falls),
            'falls_percent': falls_percent,
            'non_falls_percent': non_falls_percent
        }

    def create_fixed_folds(self, subjects):
        """Create fixed folds for cross validation based on predefined assignments"""
        # Predefined fold assignments
        fold_assignments = [
            ([43, 35, 36], "Fold 1: 38.3% falls"),
            ([44, 34, 32], "Fold 2: 39.7% falls"),
            ([45, 37, 38], "Fold 3: 44.8% falls"),
            ([46, 29, 31], "Fold 4: 41.4% falls"),
            ([30, 39], "Fold 5: 43.3% falls")
        ]
        
        folds = []
        for fold_num, (val_subjects, description) in enumerate(fold_assignments, 1):
            # Get training subjects (all subjects not in validation)
            train_subjects = [s for s in subjects if s not in val_subjects]
            
            # Calculate fold statistics
            val_stats = {
                'falls': 0,
                'non_falls': 0,
                'total': 0
            }
            
            for subject in val_subjects:
                stats = self.analyze_subject_distribution(subject)
                val_stats['falls'] += stats['falls']
                val_stats['non_falls'] += stats['non_falls']
                val_stats['total'] += stats['total']
            
            # Calculate fall percentage
            if val_stats['total'] > 0:
                fall_percent = (val_stats['falls'] / val_stats['total']) * 100
            else:
                fall_percent = 0
            
            # Log fold information
            self.print_log(f"\nCreated {description}")
            self.print_log(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
            self.print_log(f"Training subjects ({len(train_subjects)}): {train_subjects}")
            self.print_log(f"Validation set statistics:")
            self.print_log(f"  Falls: {val_stats['falls']}")
            self.print_log(f"  Non-Falls: {val_stats['non_falls']}")
            self.print_log(f"  Fall percentage: {fall_percent:.1f}%")
            
            folds.append((train_subjects, val_subjects))
        
        return folds

    def start(self):
        """Start training process with 5-fold cross validation"""
        if self.arg.phase == 'train':
            try:
                # Initial subject-wise analysis
                print("\nAnalyzing Subject Distribution:")
                print("-" * 70)
                
                # Create folds using fixed assignments
                folds = self.create_fixed_folds(self.arg.subjects)
                fold_best_metrics = []  # Track best metrics for each fold
                
                # Train on each fold
                for fold, (train_subjects, val_subjects) in enumerate(folds, 1):
                    fold_dir = os.path.join(self.arg.work_dir, f'fold_{fold}')
                    os.makedirs(fold_dir, exist_ok=True)
                    
                    # Get teacher weight for this fold
                    teacher_weight = self.get_teacher_weight_for_fold(fold)
                    self.print_log(f"\nLoading teacher weights from: {teacher_weight}")
                    
                    # Reset models and metrics for this fold
                    self.setup_models(fold=fold)
                    
                    # Load data for this fold
                    builder = prepare_smartfallmm(self.arg)
                    train_data = filter_subjects(builder, train_subjects)
                    val_data = filter_subjects(builder, val_subjects)
                    
                    # Import feeder class
                    FeederClass = import_class(self.arg.feeder)
                    
                    self.data_loader = {}
                    self.data_loader['train'] = DataLoader(
                        dataset=FeederClass(**self.arg.train_feeder_args, dataset=train_data),
                        batch_size=self.arg.batch_size,
                        shuffle=True,
                        num_workers=self.arg.num_worker,
                        pin_memory=True
                    )
                    self.data_loader['val'] = DataLoader(
                        dataset=FeederClass(**self.arg.val_feeder_args, dataset=val_data),
                        batch_size=self.arg.val_batch_size,
                        shuffle=False,
                        num_workers=self.arg.num_worker,
                        pin_memory=True
                    )
                    
                    # Reset training components
                    self.setup_training_components()
                    
                    # Initialize best metrics for this fold
                    best_val_loss = float('inf')
                    best_f1 = 0
                    best_epoch = 0
                    best_state = None
                    
                    train_losses = []
                    val_losses = []
                    val_metrics = []
                    
                    for epoch in range(self.arg.num_epoch):
                        train_metrics = self.train_epoch(epoch)
                        val_metrics = self.validate(epoch)
                        
                        train_losses.append(train_metrics['loss'])
                        val_losses.append(val_metrics['loss'])
                        
                        # Update best metrics
                        if val_metrics['f1'] > best_f1:
                            best_f1 = val_metrics['f1']
                            best_val_loss = val_metrics['loss']
                            best_epoch = epoch
                            best_state = {
                                'epoch': epoch,
                                'state_dict': self.student.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'scheduler': self.scheduler.state_dict(),
                                'metrics': val_metrics
                            }
                            
                            # Save best model for this fold
                            save_path = os.path.join(fold_dir, f'best_model_fold{fold}.pt')
                            torch.save(best_state, save_path)
                            self.print_log(f"\nNew best model saved! Fold {fold}, F1: {best_f1:.4f}, Loss: {best_val_loss:.4f}")
                        
                        # Print epoch metrics
                        self.print_metrics('Train', epoch, train_metrics)
                        self.print_metrics('Val', epoch, val_metrics)
                    
                    # Store best metrics for this fold
                    fold_best_metrics.append({
                        'fold': fold,
                        'best_f1': best_f1,
                        'best_val_loss': best_val_loss,
                        'best_epoch': best_epoch,
                        'train_subjects': train_subjects,
                        'val_subjects': val_subjects
                    })
                    
                    # Save fold results
                    self.save_fold_results(fold, train_losses, val_losses, val_metrics, 
                                         fold_dir, train_subjects, val_subjects, teacher_weight)
                
                # Print summary of all folds
                print("\n" + "="*70)
                print("Cross-validation Results Summary")
                print("="*70)
                print(f"{'Fold':^6} {'Best Epoch':^12} {'Best F1':^10} {'Best Loss':^10}")
                print("-"*70)
                
                avg_f1 = 0
                avg_loss = 0
                for metrics in fold_best_metrics:
                    print(f"{metrics['fold']:^6d} {metrics['best_epoch']:^12d} "
                          f"{metrics['best_f1']:^10.4f} {metrics['best_val_loss']:^10.4f}")
                    avg_f1 += metrics['best_f1']
                    avg_loss += metrics['best_val_loss']
                
                avg_f1 /= len(fold_best_metrics)
                avg_loss /= len(fold_best_metrics)
                print("-"*70)
                print(f"{'Average':^6} {'-':^12} {avg_f1:^10.4f} {avg_loss:^10.4f}")
                print("="*70)
                
                # Save overall results
                torch.save({
                    'fold_best_metrics': fold_best_metrics,
                    'avg_f1': avg_f1,
                    'avg_loss': avg_loss
                }, os.path.join(self.arg.work_dir, 'cross_validation_results.pt'))
                
            except Exception as e:
                self.print_log("Error during training: " + str(e))
                self.print_log("Traceback: " + traceback.format_exc())

    def print_metrics(self, phase, epoch, metrics):
        """Print metrics in a neat format"""
        self.print_log(
            f"\nEpoch {epoch:3d} [{phase:5s}] "
            f"Loss: {metrics['loss']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

    def print_log(self, msg, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            msg = f"[ {localtime} ] {msg}"
        print(msg)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, "log.txt"), "a") as f:
                print(msg, file=f)

####################################################
#               Argument Parser
####################################################
def get_args():
    parser = argparse.ArgumentParser("Fall Detection Knowledge Distillation")
    parser.add_argument('--config', type=str, default='./config/smartfallmm/distill.yaml')
    parser.add_argument('--work-dir', type=str, default='work_dir')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--student_model', type=str, default=None)
    parser.add_argument('--teacher_model', type=str, default=None)
    parser.add_argument('--teacher_weight', type=str, default='')
    parser.add_argument('--student_args', type=str, default='{}')
    parser.add_argument('--teacher_args', type=str, default='{}')
    parser.add_argument('--dataset', type=str, default='smartfallmm')
    parser.add_argument('--dataset_args', type=str, default='{}')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--include_val', type=str2bool, default=True)
    parser.add_argument('--subjects', nargs='+', type=int, default=[])
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--distill_loss', type=str, default='distiller.DistillationLoss')
    parser.add_argument('--distill_args', type=str, default='{}')
    parser.add_argument('--feeder', type=str, default=None)
    parser.add_argument('--train_feeder_args', type=str, default='{}')
    parser.add_argument('--val_feeder_args', type=str, default='{}')
    parser.add_argument('--test_feeder_args', type=str, default='{}')
    parser.add_argument('--device', nargs='+', type=int, default=[0])
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--model_saved_name', type=str, default='student_model')
    parser.add_argument('--weights', type=str, default=None)
    return parser

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_args = yaml.safe_load(f)
        parser.set_defaults(**yaml_args)
        args = parser.parse_args()

    # Process nested dict arguments
    args_dict = vars(args)
    args_dict = process_args(args_dict)

    # Initialize seeds
    init_seed(args.seed)

    # Create work_dir
    os.makedirs(args.work_dir, exist_ok=True)

    # Copy config for reference
    if args.config and os.path.exists(args.config):
        shutil.copy(args.config, os.path.join(args.work_dir, os.path.basename(args.config)))

    # Start Distiller
    trainer = FallDetectionDistiller(arg=args)
    trainer.print_log("=== Training Configuration ===")
    trainer.print_log(f"Teacher model: {args.teacher_model}")
    trainer.print_log(f"Student model: {args.student_model}")
    trainer.print_log(f"Teacher weights: {args.teacher_weight}")
    trainer.print_log(f"Batch size: {args.batch_size}")
    trainer.print_log(f"Learning rate: {args.base_lr}")
    trainer.print_log(f"Number of epochs: {args.num_epoch}")
    trainer.print_log(f"Device: {args.device}")
    trainer.print_log(f"Working directory: {args.work_dir}")
    trainer.print_log("=" * 28)

    trainer.start()
