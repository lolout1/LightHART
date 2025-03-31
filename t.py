#!/usr/bin/env python3
import os
import time
import yaml
import json
import argparse
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, balanced_accuracy_score
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected')

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try: return getattr(sys.modules[mod_str], class_str)
    except AttributeError: raise ImportError(f'Class {class_str} cannot be found')

def get_args():
    parser = argparse.ArgumentParser(description='Lightweight Fall Detection Training')
    parser.add_argument('--config', default='config/lightweightTransformer.yaml', help='Config file path')
    parser.add_argument('--dataset', type=str, default='smartfallmm', help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='Training batch size')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N', help='Validation batch size')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N', help='Test batch size')
    parser.add_argument('--num-epoch', type=int, default=60, metavar='N', help='Training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Start epoch number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--base-lr', type=float, default=0.0005, metavar='LR', help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0004, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--model', default='Models.simple.FallDetectionTransformer', help='Model class path')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--model-args', default=None, help='Model arguments')
    parser.add_argument('--model-saved-name', type=str, default='lightweight_Transmodel', help='Model save name')
    parser.add_argument('--dataset-args', default=None, help='Dataset arguments')
    parser.add_argument('--subjects', nargs='+', type=int, default=[32,39,30,31,33,34,35,37,43,44,45,36,29], help='Subject IDs')
    parser.add_argument('--val-subjects', nargs='+', type=int, default=[38,46], help='Validation subject IDs')
    parser.add_argument('--permanent-train', nargs='+', type=int, default=[45,36,29], help='Always in training set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--work-dir', type=str, default='work_dir/lightweight', help='Working directory')
    parser.add_argument('--print-log', type=str2bool, default=True, help='Print and save logs')
    parser.add_argument('--phase', type=str, default='train', help='Train or evaluation')
    parser.add_argument('--num-worker', type=int, default=0, help='Data loader workers')
    return parser.parse_args()

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary, self.val_loss_summary, self.train_metrics, self.val_metrics = [], [], [], []
        self.best_f1, self.best_accuracy, self.patience_counter = 0, 0, 0
        self.available_subjects = [s for s in arg.subjects if s not in arg.val_subjects and s not in arg.permanent_train]
        self.data_loader, self.best_loss = dict(), float('inf')
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}'
        self.device = f'cuda:{self.arg.device}' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model and dataset args if needed
        if not hasattr(arg, 'model_args') or not arg.model_args:
            arg.model_args = {
                'num_layers': 2,
                'embed_dim': 32,
                'num_classes': 2,
                'seq_length': 64,
                'num_heads': 2,
                'dropout': 0.2,
                'use_batch_norm': True
            }
        elif isinstance(arg.model_args, str):
            try:
                arg.model_args = eval(arg.model_args)
            except:
                try:
                    arg.model_args = yaml.safe_load(arg.model_args)
                except:
                    pass
        
        if not hasattr(arg, 'dataset_args') or not arg.dataset_args:
            arg.dataset_args = {
                'mode': 'sliding_window',
                'max_length': 64,
                'task': 'fd',
                'modalities': ['accelerometer', 'gyroscope'],
                'age_group': ['young'],
                'sensors': ['watch'],
                'fusion_options': {
                    'enabled': False
                }
            }
        elif isinstance(arg.dataset_args, str):
            try:
                arg.dataset_args = eval(arg.dataset_args)
            except:
                try:
                    arg.dataset_args = yaml.safe_load(arg.dataset_args)
                except:
                    pass
        
        # Create model 
        self.Model = import_class(arg.model)
        self.model = self.Model(**arg.model_args).to(self.device)
        
        # Create work directory
        os.makedirs(self.arg.work_dir, exist_ok=True)
        
        # Print setup info
        self.print_log(f'Using device: {self.device}')
        self.print_log(f'Model: {arg.model}')
        self.print_log(f'# Parameters: {sum(p.numel() for p in self.model.parameters())}')
        
        # Generate test/train folds
        self.test_combinations = self.generate_test_combinations()
        self.print_log(f"Generated {len(self.test_combinations)} test/train combinations")
        
    def generate_test_combinations(self):
        test_candidates = [s for s in self.arg.subjects if s not in self.arg.permanent_train and s not in self.arg.val_subjects]
        combinations_list = []
        for test_pair in combinations(test_candidates, 2):
            train_set = self.arg.permanent_train + [s for s in test_candidates if s not in test_pair]
            combinations_list.append((list(train_set), list(test_pair)))
        return combinations_list
    
    def load_data(self, train_subjects, test_subjects=None):
        from utils.dataset import prepare_smartfallmm, split_by_subjects
        from Feeder.Make_Dataset import UTD_mm
        
        try:
            # Prepare dataset
            builder = prepare_smartfallmm(self.arg)
            
            # Load training data
            self.print_log(f"Loading training data for subjects: {train_subjects}")
            norm_train = split_by_subjects(builder, train_subjects, False)
            
            if 'accelerometer' not in norm_train or len(norm_train['accelerometer']) == 0: 
                self.print_log("ERROR: No accelerometer data found in training set")
                return False
            if 'gyroscope' not in norm_train or len(norm_train['gyroscope']) == 0: 
                self.print_log("ERROR: No gyroscope data found in training set")
                return False
            if 'labels' not in norm_train or len(norm_train['labels']) == 0: 
                self.print_log("ERROR: No labels found in training set")
                return False
            
            # Filter out samples without both sensors
            valid_indices = []
            for i in range(len(norm_train['accelerometer'])):
                if (i < len(norm_train['gyroscope']) and 
                    np.any(np.abs(norm_train['accelerometer'][i]) > 1e-6) and 
                    np.any(np.abs(norm_train['gyroscope'][i]) > 1e-6)):
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                self.print_log("ERROR: No valid samples with both sensors in training set")
                return False
                
            filtered_train = {}
            for key in norm_train:
                if key == 'labels':
                    filtered_train[key] = norm_train[key][valid_indices]
                elif isinstance(norm_train[key], np.ndarray) and len(norm_train[key]) > 0:
                    filtered_train[key] = norm_train[key][valid_indices]
                else:
                    filtered_train[key] = norm_train[key]
            
            # Create training dataloader
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=UTD_mm(filtered_train, batch_size=self.arg.batch_size),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                collate_fn=UTD_mm.custom_collate_fn)
            
            # Load validation data
            self.print_log(f"Loading validation data for subjects: {self.arg.val_subjects}")
            norm_val = split_by_subjects(builder, self.arg.val_subjects, False)
            
            # Filter val data
            valid_val_indices = []
            if 'accelerometer' in norm_val and 'gyroscope' in norm_val:
                for i in range(len(norm_val.get('accelerometer', []))):
                    if (i < len(norm_val['gyroscope']) and 
                        np.any(np.abs(norm_val['accelerometer'][i]) > 1e-6) and 
                        np.any(np.abs(norm_val['gyroscope'][i]) > 1e-6)):
                        valid_val_indices.append(i)
            
            filtered_val = {}
            for key in norm_val:
                if key == 'labels' and len(valid_val_indices) > 0:
                    filtered_val[key] = norm_val[key][valid_val_indices]
                elif isinstance(norm_val[key], np.ndarray) and len(norm_val[key]) > 0 and len(valid_val_indices) > 0:
                    filtered_val[key] = norm_val[key][valid_val_indices]
                else:
                    filtered_val[key] = norm_val[key]
            
            # Create validation dataloader
            if len(valid_val_indices) > 0:
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=UTD_mm(filtered_val, batch_size=self.arg.val_batch_size),
                    batch_size=self.arg.val_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    collate_fn=UTD_mm.custom_collate_fn)
            
            # Load test data if provided
            if test_subjects:
                self.print_log(f"Loading test data for subjects: {test_subjects}")
                norm_test = split_by_subjects(builder, test_subjects, False)
                
                # Filter test data
                valid_test_indices = []
                if 'accelerometer' in norm_test and 'gyroscope' in norm_test:
                    for i in range(len(norm_test.get('accelerometer', []))):
                        if (i < len(norm_test['gyroscope']) and 
                            np.any(np.abs(norm_test['accelerometer'][i]) > 1e-6) and 
                            np.any(np.abs(norm_test['gyroscope'][i]) > 1e-6)):
                            valid_test_indices.append(i)
                
                filtered_test = {}
                for key in norm_test:
                    if key == 'labels' and len(valid_test_indices) > 0:
                        filtered_test[key] = norm_test[key][valid_test_indices]
                    elif isinstance(norm_test[key], np.ndarray) and len(norm_test[key]) > 0 and len(valid_test_indices) > 0:
                        filtered_test[key] = norm_test[key][valid_test_indices]
                    else:
                        filtered_test[key] = norm_test[key]
                
                # Create test dataloader
                if len(valid_test_indices) > 0:
                    self.data_loader['test'] = torch.utils.data.DataLoader(
                        dataset=UTD_mm(filtered_test, batch_size=self.arg.test_batch_size),
                        batch_size=self.arg.test_batch_size,
                        shuffle=False,
                        num_workers=self.arg.num_worker,
                        collate_fn=UTD_mm.custom_collate_fn)
            
            return True
        except Exception as e:
            self.print_log(f"ERROR in load_data: {str(e)}")
            self.print_log(traceback.format_exc())
            return False
    
    def load_optimizer(self):
        optimizer_name = self.arg.optimizer.lower()
        if optimizer_name == "adam": 
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif optimizer_name == "adamw": 
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif optimizer_name == "sgd": 
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else: 
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def print_log(self, string, print_time=True):
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f: 
                print(string, file=f)
                
    def save_model_weights(self, path):
        # Save only the model state dictionary (weights)
        torch.save(self.model.state_dict(), path)
            
    def load_model_weights(self, path):
        try:
            # Create a new model instance and load state dict
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict)
            return True
        except Exception as e:
            self.print_log(f"Error loading model weights: {str(e)}")
            return False
    
    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        loss_value = []
        acc_value = []
        train_loss = 0
        accuracy = 0
        cnt = 0
        criterion = nn.CrossEntropyLoss()
        y_true, y_pred = [], []
        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch} (Train)")
        
        for batch_idx, (inputs, targets, _) in enumerate(process):
            # Skip batches without both required modalities
            if not isinstance(inputs, dict) or 'accelerometer' not in inputs or 'gyroscope' not in inputs:
                continue
                
            # Skip batches with invalid data
            acc_valid = torch.any(torch.abs(inputs['accelerometer']) > 1e-6)
            gyro_valid = torch.any(torch.abs(inputs['gyroscope']) > 1e-6)
            if not acc_valid or not gyro_valid:
                continue
                
            # Move tensors to device
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device).float()
            
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            predictions = torch.max(outputs, 1)[1]
            accuracy += (predictions == targets).sum().item()
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            cnt += targets.size(0)
            
            process.set_postfix({'loss': f"{train_loss/(batch_idx+1):.4f}", 'acc': f"{100.0*accuracy/cnt:.2f}%"})
            
        if cnt == 0:
            self.print_log("ERROR: No valid data in training epoch")
            return None
            
        train_loss /= len(loader)
        accuracy *= 100.0 / cnt
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision *= 100
        recall *= 100
        balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
        
        self.train_loss_summary.append(train_loss)
        self.train_metrics.append({
            'accuracy': accuracy, 
            'f1': f1, 
            'precision': precision, 
            'recall': recall, 
            'balanced_accuracy': balanced_acc
        })
        
        self.print_log(f'Epoch {epoch+1}/{self.arg.num_epoch} - Training Loss: {train_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.2f}%')
        
        # Evaluate on validation set
        val_metrics = self.eval(epoch, 'val')
        
        if val_metrics:
            val_f1 = val_metrics['f1']
            self.val_loss_summary.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            if val_f1 > self.best_f1:
                improvement = val_f1 - self.best_f1
                self.best_f1 = val_f1
                self.best_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                
                # Save with meaningful name including metrics
                save_name = f"{self.model_path}_f1_{self.best_f1:.2f}_acc_{self.best_accuracy:.2f}_epoch_{epoch+1}.pt"
                self.save_model_weights(save_name)
                self.print_log(f"New best model: F1 improved by {improvement:.2f} to {val_f1:.2f}, saved to {save_name}")
                
                # Also save as default model
                self.save_model_weights(f"{self.model_path}_best.pt")
            else:
                self.patience_counter += 1
                self.print_log(f"No F1 improvement for {self.patience_counter} epochs (patience: {self.arg.patience})")
                if self.patience_counter >= self.arg.patience:
                    return False  # Signal early stopping
        
        return True  # Continue training
    
    def eval(self, epoch, mode='val'):
        self.model.eval()
        if mode not in self.data_loader:
            self.print_log(f"No {mode} dataloader available")
            return None
            
        loader = self.data_loader[mode]
        loss_value = []
        loss = 0
        accuracy = 0
        cnt = 0
        criterion = nn.CrossEntropyLoss()
        y_true, y_pred = [], []
        process = tqdm(loader, desc=f"Epoch {epoch+1} ({mode.capitalize()})")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(process):
                # Skip batches without both required modalities
                if not isinstance(inputs, dict) or 'accelerometer' not in inputs or 'gyroscope' not in inputs:
                    continue
                    
                # Skip batches with invalid data
                acc_valid = torch.any(torch.abs(inputs['accelerometer']) > 1e-6)
                gyro_valid = torch.any(torch.abs(inputs['gyroscope']) > 1e-6)
                if not acc_valid or not gyro_valid:
                    continue
                
                # Move tensors to device
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(self.device).float()
                
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                batch_loss = criterion(outputs, targets)
                
                # Track metrics
                loss += batch_loss.item()
                predictions = torch.max(outputs, 1)[1]
                accuracy += (predictions == targets).sum().item()
                y_true.extend(targets.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())
                cnt += targets.size(0)
                
                process.set_postfix({'loss': f"{loss/(batch_idx+1):.4f}", 'acc': f"{100.0*accuracy/cnt:.2f}%"})
        
        if cnt == 0:
            self.print_log(f"No valid data in {mode} evaluation")
            return None
            
        loss /= len(loader)
        accuracy = 100.0 * accuracy / cnt
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision *= 100
        recall *= 100
        balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
        
        self.print_log(f'{mode.capitalize()} metrics: Loss={loss:.4f}, Acc={accuracy:.2f}%, F1={f1:.2f}%, Precision={precision:.2f}%, Recall={recall:.2f}%')
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{mode.capitalize()} Confusion Matrix")
        plt.colorbar()
        plt.xticks([0, 1], ['No Fall', 'Fall'])
        plt.yticks([0, 1], ['No Fall', 'Fall'])
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
                         
        plt.tight_layout()
        plt.savefig(f'{self.arg.work_dir}/cm_{mode}.png')
        plt.close()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'balanced_accuracy': balanced_acc,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
    def run_fold(self, train_subjects, test_subjects, fold_idx):
        fold_dir = os.path.join(self.arg.work_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Reset model and metrics
        self.train_loss_summary, self.val_loss_summary = [], []
        self.train_metrics, self.val_metrics = [], []
        self.best_f1, self.best_accuracy, self.patience_counter = 0, 0, 0
        
        # Create new model instance
        self.model = self.Model(**self.arg.model_args).to(self.device)
        
        # Log fold information
        self.print_log(f"\n{'='*80}")
        self.print_log(f"Fold {fold_idx}: Train={train_subjects}, Test={test_subjects}, Val={self.arg.val_subjects}")
        self.print_log(f"{'='*80}")
        
        # Save fold config
        with open(os.path.join(fold_dir, 'split.json'), 'w') as f:
            json.dump({
                'train': train_subjects,
                'test': test_subjects,
                'val': self.arg.val_subjects
            }, f, indent=2)
            
        # Load data for this fold
        if not self.load_data(train_subjects, test_subjects):
            self.print_log(f"Failed to load data for fold {fold_idx}")
            return None
            
        # Set up optimizer
        self.load_optimizer()
        
        # Train the model
        for epoch in range(self.arg.num_epoch):
            if not self.train(epoch):
                self.print_log(f"Early stopping at epoch {epoch+1}")
                break
                
        # Create training curves
        if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_loss_summary, label='Train Loss')
            plt.plot(self.val_loss_summary, label='Val Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(fold_dir, 'loss_curves.png'))
            plt.close()
            
        # Load best model for testing
        best_model_path = f"{self.model_path}_best.pt"
        if os.path.exists(best_model_path):
            # Create a new model instance and load state dict
            self.model = self.Model(**self.arg.model_args).to(self.device)
            if self.load_model_weights(best_model_path):
                self.print_log("Loaded best model for final evaluation")
            else:
                self.print_log("Failed to load best model - using current model state")
            
        # Final evaluation on test set
        test_metrics = self.eval(0, 'test')
        
        # Save metrics
        if test_metrics:
            test_metrics_json = {k: v for k, v in test_metrics.items() if k not in ['y_true', 'y_pred']}
            with open(os.path.join(fold_dir, 'test_metrics.json'), 'w') as f:
                json.dump(test_metrics_json, f, indent=2)
                
            # Copy best model to fold directory with metrics in filename
            if os.path.exists(best_model_path):
                import shutil
                save_name = f"lightweight_f1_{test_metrics['f1']:.2f}_acc_{test_metrics['accuracy']:.2f}.pt"
                dest_path = os.path.join(fold_dir, save_name)
                shutil.copy(best_model_path, dest_path)
                self.print_log(f"Copied best model to {dest_path}")
                
            return test_metrics
        return None
        
    def start(self):
        self.print_log(f"Starting training with device: {self.device}")
        self.print_log(f"Parameters: {vars(self.arg)}")
        
        # Initialize empty results lists
        fold_results = []
        test_configs = []
        
        # Run all folds
        for fold_idx, (train_subjects, test_subjects) in enumerate(self.test_combinations):
            self.print_log(f"\nRunning fold {fold_idx+1}/{len(self.test_combinations)}")
            
            try:
                test_metrics = self.run_fold(train_subjects, test_subjects, fold_idx+1)
                
                if test_metrics:
                    fold_result = {
                        'fold': fold_idx + 1,
                        'train_subjects': train_subjects,
                        'test_subjects': test_subjects,
                        'val_subjects': self.arg.val_subjects,
                        'accuracy': test_metrics['accuracy'],
                        'f1': test_metrics['f1'],
                        'precision': test_metrics['precision'],
                        'recall': test_metrics['recall'],
                        'balanced_accuracy': test_metrics['balanced_accuracy']
                    }
                    
                    fold_results.append(fold_result)
                    test_configs.append({
                        'config_id': fold_idx + 1,
                        'train_subjects': train_subjects,
                        'test_subjects': test_subjects,
                        'metrics': {k: v for k, v in test_metrics.items() if k not in ['y_true', 'y_pred']}
                    })
                    
                    # Save current results after each fold
                    with open(os.path.join(self.arg.work_dir, 'fold_results.json'), 'w') as f:
                        json.dump(fold_results, f, indent=2)
            except Exception as e:
                self.print_log(f"Error in fold {fold_idx+1}: {str(e)}")
                self.print_log(traceback.format_exc())
        
        # Calculate average metrics
        if fold_results:
            avg_metrics = {
                'accuracy': np.mean([fold['accuracy'] for fold in fold_results]),
                'f1': np.mean([fold['f1'] for fold in fold_results]),
                'precision': np.mean([fold['precision'] for fold in fold_results]),
                'recall': np.mean([fold['recall'] for fold in fold_results]),
                'balanced_accuracy': np.mean([fold['balanced_accuracy'] for fold in fold_results]),
                'std_accuracy': np.std([fold['accuracy'] for fold in fold_results]),
                'std_f1': np.std([fold['f1'] for fold in fold_results]),
                'std_precision': np.std([fold['precision'] for fold in fold_results]),
                'std_recall': np.std([fold['recall'] for fold in fold_results]),
                'std_balanced_accuracy': np.std([fold['balanced_accuracy'] for fold in fold_results])
            }
            
            # Save summary
            summary = {
                'average_metrics': avg_metrics,
                'test_configs': test_configs
            }
            
            with open(os.path.join(self.arg.work_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Create summary CSV
            csv_rows = []
            for fold in fold_results:
                csv_rows.append({
                    'fold': fold['fold'],
                    'test_subjects': str(fold['test_subjects']),
                    'accuracy': fold['accuracy'],
                    'f1': fold['f1'],
                    'precision': fold['precision'],
                    'recall': fold['recall'],
                    'balanced_accuracy': fold['balanced_accuracy']
                })
                
            pd.DataFrame(csv_rows).to_csv(os.path.join(self.arg.work_dir, 'summary.csv'), index=False)
            
            # Print final summary
            self.print_log("\n" + "="*80)
            self.print_log("Final average metrics across all folds:")
            self.print_log(f"Accuracy: {avg_metrics['accuracy']:.2f} ± {avg_metrics['std_accuracy']:.2f}%")
            self.print_log(f"F1 Score: {avg_metrics['f1']:.2f} ± {avg_metrics['std_f1']:.2f}%")
            self.print_log(f"Precision: {avg_metrics['precision']:.2f} ± {avg_metrics['std_precision']:.2f}%")
            self.print_log(f"Recall: {avg_metrics['recall']:.2f} ± {avg_metrics['std_recall']:.2f}%")
            self.print_log(f"Balanced Accuracy: {avg_metrics['balanced_accuracy']:.2f} ± {avg_metrics['std_balanced_accuracy']:.2f}%")
            self.print_log("="*80)
            
            # Save best model with performance in name
            best_fold_idx = np.argmax([fold['f1'] for fold in fold_results])
            best_fold = fold_results[best_fold_idx]
            self.print_log(f"\nBest performing fold: {best_fold['fold']}")
            self.print_log(f"Test subjects: {best_fold['test_subjects']}")
            self.print_log(f"F1 Score: {best_fold['f1']:.2f}%")
            
            best_model_name = f"lightweight_best_f1_{avg_metrics['f1']:.2f}_acc_{avg_metrics['accuracy']:.2f}.pt"
            best_fold_dir = os.path.join(self.arg.work_dir, f"fold_{best_fold['fold']}")
            
            for file in os.listdir(best_fold_dir):
                if file.endswith(".pt"):
                    src_path = os.path.join(best_fold_dir, file)
                    dst_path = os.path.join(self.arg.work_dir, best_model_name)
                    import shutil
                    shutil.copy(src_path, dst_path)
                    self.print_log(f"Copied best model to {dst_path}")
                    break
        else:
            self.print_log("No successful folds completed")

def main():
    args = get_args()
    
    # Load config
    if args.config is not None:
        with open(args.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                args.__dict__[k] = default_arg[k]
    
    # Setup random seed
    init_seed(args.seed)
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    # Start trainer
    trainer = Trainer(args)
    trainer.start()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
