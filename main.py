import traceback
from typing import List, Dict, Tuple, Union, Optional
import random
import sys
import os
import time
import shutil
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, balanced_accuracy_score
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from scipy.spatial.transform import Rotation
from utils.dataset import prepare_smartfallmm, split_by_subjects
from utils.imu_fusion import clear_filters, align_sensor_data, process_window_with_filter

MAX_THREADS = 30
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)

def get_args():
    parser = argparse.ArgumentParser(description='Fall Detection and Human Activity Recognition')
    parser.add_argument('--config', default='./config/smartfallmm/fusion_madgwick.yaml', help='Config file path')
    parser.add_argument('--dataset', type=str, default='smartfallmm', help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='Training batch size')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N', help='Validation batch size')
    parser.add_argument('--num-epoch', type=int, default=60, metavar='N', help='Training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Start epoch number')
    parser.add_argument('--weights-only', type=str2bool, default=False, help='Load only weights')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR', help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0004, help='Weight decay')
    parser.add_argument('--kfold', type=str2bool, default=False, help='Use cross validation')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--model', default=None, help='Model class path')
    parser.add_argument('--device', nargs='+', default=[0], type=int, help='CUDA device IDs')
    parser.add_argument('--model-args', default=None, help='Model arguments')
    parser.add_argument('--weights', type=str, help='Pretrained weights file')
    parser.add_argument('--model-saved-name', type=str, default='model', help='Model save name')
    parser.add_argument('--loss', default='loss.BCE', help='Loss function class')
    parser.add_argument('--loss-args', default="{}", type=str, help='Loss function arguments')
    parser.add_argument('--dataset-args', default=None, help='Dataset arguments')
    parser.add_argument('--subjects', nargs='+', type=int, help='Subject IDs')
    parser.add_argument('--feeder', default=None, help='DataLoader class')
    parser.add_argument('--train-feeder-args', default=None, help='Train loader arguments')
    parser.add_argument('--val-feeder-args', default=None, help='Validation loader arguments')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--work-dir', type=str, default='work_dir', help='Working directory')
    parser.add_argument('--print-log', type=str2bool, default=True, help='Print and save logs')
    parser.add_argument('--phase', type=str, default='train', help='Train or evaluation')
    parser.add_argument('--num-worker', type=int, default=0, help='Data loader workers')
    parser.add_argument('--multi-gpu', type=str2bool, default=True, help='Use multiple GPUs')
    parser.add_argument('--parallel-threads', type=int, default=4, help='Processing threads')
    parser.add_argument('--run-comparison', type=str2bool, default=False, help='Run filter comparison')
    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected')

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try: return getattr(sys.modules[mod_str], class_str)
    except AttributeError: raise ImportError(f'Class {class_str} cannot be found')

def setup_gpu_environment(args):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if isinstance(args.device, list) and len(args.device) > 0: devices = args.device
        elif num_gpus >= 2 and args.multi_gpu: devices = [0, 1]
        elif num_gpus == 1: devices = [0]
        else: devices = []
        gpu_list = ",".join(map(str, devices))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        return devices
    else: return []

class Trainer:
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary, self.val_loss_summary, self.train_metrics, self.val_metrics = [], [], [], []
        self.best_f1, self.best_accuracy, self.patience_counter = 0, 0, 0
        self.train_subjects, self.val_subject = [], None
        self.optimizer, self.norm_train, self.norm_val = None, None, None
        self.data_loader, self.best_loss = dict(), float('inf')
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}.pt'
        self.max_threads = min(arg.parallel_threads, MAX_THREADS)
        self.inertial_modality = [modality for modality in arg.dataset_args['modalities'] if modality != 'skeleton']
        self.has_gyro = 'gyroscope' in self.inertial_modality
        self.has_fusion = len(self.inertial_modality) > 1 or ('fusion_options' in arg.dataset_args and arg.dataset_args['fusion_options'].get('enabled', False))
        self.fuse = self.has_fusion
        self.filter_type = arg.dataset_args['fusion_options'].get('filter_type', 'madgwick') if 'fusion_options' in arg.dataset_args else "madgwick"
        self.filter_params = arg.dataset_args['fusion_options'] if 'fusion_options' in arg.dataset_args else {}
        self.use_cache = arg.dataset_args['fusion_options'].get('use_cache', False) if 'fusion_options' in arg.dataset_args else False
        self.cache_dir = arg.dataset_args['fusion_options'].get('cache_dir', 'processed_data') if 'fusion_options' in arg.dataset_args else 'processed_data'
        if self.use_cache: os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.arg.work_dir, exist_ok=True)
        self.save_config(arg.config, arg.work_dir)
        self.available_gpus = setup_gpu_environment(arg)
        arg.device = self.available_gpus if self.available_gpus else arg.device
        self.output_device = arg.device[0] if type(arg.device) is list and len(arg.device) > 0 else arg.device
        self.model = self.load_model(arg.model, arg.model_args) if self.arg.phase == 'train' else torch.load(self.arg.weights)
        if len(self.available_gpus) > 1 and arg.multi_gpu: self.model = nn.DataParallel(self.model, device_ids=self.available_gpus)
        self.load_loss()
        self.print_log(f'# Parameters: {self.count_parameters(self.model)}')
        self.print_log(f'Sensor modalities: {self.inertial_modality}')
        self.print_log(f'Using fusion: {self.fuse} with filter type: {self.filter_type}')
        self.print_log(f'Caching enabled: {self.use_cache}, dir: {self.cache_dir}')
        if self.available_gpus: self.print_log(f'Using GPUs: {self.available_gpus}')
        else: self.print_log('Using CPU for computation')

    def save_config(self, src_path, desc_path):
        config_file = src_path.rpartition("/")[-1]
        shutil.copy(src_path, f'{desc_path}/{config_file}')

    def count_parameters(self, model):
        total_size = 0
        for param in model.parameters(): total_size += param.nelement() * param.element_size()
        for buffer in model.buffers(): total_size += buffer.nelement() * buffer.element_size()
        return total_size

    def has_empty_value(self, *lists): return any(len(lst) == 0 for lst in lists)

    def load_model(self, model, model_args):
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        Model = import_class(model)
        model = Model(**model_args).to(device)
        return model

    def load_loss(self): self.criterion = torch.nn.CrossEntropyLoss()

    def load_weights(self):
        if isinstance(self.model, nn.DataParallel): self.model.module.load_state_dict(torch.load(self.model_path))
        else: self.model.load_state_dict(torch.load(self.model_path))

    def load_optimizer(self):
        optimizer_name = self.arg.optimizer.lower()
        if optimizer_name == "adam": self.optimizer = optim.Adam(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif optimizer_name == "adamw": self.optimizer = optim.AdamW(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif optimizer_name == "sgd": self.optimizer = optim.SGD(self.model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else: raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def distribution_viz(self, labels, work_dir, mode):
        values, count = np.unique(labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(values, count)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title(f'{mode.capitalize()} Label Distribution')
        plt.savefig(f'{work_dir}/{mode}_label_distribution.png')
        plt.close()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if self.arg.phase == 'train':
            builder = prepare_smartfallmm(self.arg)
            self.norm_train = split_by_subjects(builder, self.train_subjects, self.fuse)
            if self.has_empty_value(list(self.norm_train.values())):
                if 'accelerometer' not in self.norm_train or len(self.norm_train['accelerometer']) == 0: return False
                if 'labels' not in self.norm_train or len(self.norm_train['labels']) == 0: return False
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args, dataset=self.norm_train),
                batch_size=self.arg.batch_size, shuffle=True, num_workers=self.arg.num_worker)
            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
            
            self.norm_val = split_by_subjects(builder, self.val_subject, self.fuse)
            if 'accelerometer' not in self.norm_val or len(self.norm_val['accelerometer']) == 0 or 'labels' not in self.norm_val or len(self.norm_val['labels']) == 0:
                self.print_log("WARNING: Invalid validation data")
                return False
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args, dataset=self.norm_val),
                batch_size=self.arg.val_batch_size, shuffle=False, num_workers=self.arg.num_worker)
            self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')
            return True
        else:
            builder = prepare_smartfallmm(self.arg)
            try:
                self.norm_val = split_by_subjects(builder, self.val_subject, self.fuse)
                if 'accelerometer' not in self.norm_val or len(self.norm_val['accelerometer']) == 0: return False
                if 'labels' not in self.norm_val or len(self.norm_val['labels']) == 0: return False
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.val_feeder_args, dataset=self.norm_val),
                    batch_size=self.arg.val_batch_size, shuffle=False, num_workers=self.arg.num_worker)
                return True
            except Exception as e: return False

    def record_time(self): 
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string, print_time=True):
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f: print(string, file=f)

    def loss_viz(self, train_loss, val_loss):
        epochs = range(len(train_loss))
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_loss, 'b-', label="Training Loss")
        plt.plot(epochs, val_loss, 'r-', label="Validation Loss")
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'{self.arg.work_dir}/train_vs_val_loss.png')
        plt.close()

    def cm_viz(self, y_pred, y_true):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        class_labels = np.unique(y_true)
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels)
        plt.yticks(tick_marks, class_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(f'{self.arg.work_dir}/confusion_matrix.png')
        plt.close()

    def create_df(self, columns=['fold', 'val_subject', 'train_subjects', 'accuracy', 'f1_score', 'precision', 'recall']): 
        return pd.DataFrame(columns=columns)

    def train(self, epoch):
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        self.model.train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader=0.001, model=0.001, stats=0.001)
        acc_value, accuracy, cnt, train_loss = [], 0, 0, 0
        y_true, y_pred = [], []
        process = tqdm(loader, desc=f"Epoch {epoch+1}/{self.arg.num_epoch} (Train)")
        for batch_idx, (inputs, targets, idx) in enumerate(process):
            with torch.no_grad():
                acc_data = inputs['accelerometer'].to(device)
                targets = targets.to(device)
                gyro_data = inputs.get('gyroscope', None)
                if gyro_data is not None: gyro_data = gyro_data.to(device)
                quaternion = inputs.get('quaternion', None)
                if quaternion is not None: quaternion = quaternion.to(device)
            timer['dataloader'] += self.split_time()
            self.optimizer.zero_grad()
            if hasattr(self.model, 'forward_quaternion') and quaternion is not None: 
                logits = self.model.forward_quaternion(acc_data.float(), quaternion.float())
            elif gyro_data is not None and hasattr(self.model, 'forward_multi_sensor'): 
                logits = self.model.forward_multi_sensor(acc_data.float(), gyro_data.float())
            else: logits = self.model(acc_data.float())
            loss = self.criterion(logits, targets)
            loss.mean().backward()
            self.optimizer.step()
            timer['model'] += self.split_time()
            with torch.no_grad():
                train_loss += loss.mean().item()
                predictions = torch.argmax(F.log_softmax(logits, dim=1), 1)
                accuracy += (predictions == targets).sum().item()
                y_true.extend(targets.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())
            cnt += len(targets)
            timer['stats'] += self.split_time()
            process.set_postfix({'loss': f"{train_loss/(batch_idx+1):.4f}", 'acc': f"{100.0*accuracy/cnt:.2f}%"})
        
        train_loss /= len(loader)
        accuracy *= 100. / cnt
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision *= 100
        recall *= 100
        balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
        self.train_loss_summary.append(train_loss)
        acc_value.append(accuracy)
        train_metrics = {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall, 'balanced_accuracy': balanced_acc}
        self.train_metrics.append(train_metrics)
        proportion = {k: f'{int(round(v * 100 / sum(timer.values()))):02d}%' for k, v in timer.items()}
        self.print_log(f'Epoch {epoch+1}/{self.arg.num_epoch} - Training Loss: {train_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.2f}%')
        self.print_log(f'Time: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}, [Stats]{proportion["stats"]}')
        
        val_metrics = self.eval(epoch)
        val_f1 = val_metrics['f1']
        self.val_loss_summary.append(val_metrics['loss'])
        self.val_metrics.append(val_metrics)
        
        if val_f1 > self.best_f1:
            improvement = val_f1 - self.best_f1
            self.best_f1 = val_f1
            self.best_accuracy = val_metrics['accuracy']
            self.best_loss = val_metrics['loss']
            self.patience_counter = 0
            self.print_log(f"New best model: F1 improved by {improvement:.2f} to {val_f1:.2f}")
            try:
                if isinstance(self.model, nn.DataParallel): torch.save(deepcopy(self.model.module.state_dict()), self.model_path)
                else: torch.save(deepcopy(self.model.state_dict()), self.model_path)
            except Exception as e:
                self.print_log(f"Error saving model: {str(e)}")
                emergency_path = os.path.join(self.arg.work_dir, 'best_model_emergency.pt')
                try:
                    if isinstance(self.model, nn.DataParallel): torch.save(self.model.module.state_dict(), emergency_path)
                    else: torch.save(self.model.state_dict(), emergency_path)
                except: pass
        else:
            self.patience_counter += 1
            self.print_log(f"No F1 improvement for {self.patience_counter} epochs (patience: {self.arg.patience})")
            if self.patience_counter >= self.arg.patience:
                return {'early_stop': True}
        
        return {'early_stop': self.patience_counter >= self.arg.patience}

    def eval(self, epoch):
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        self.model.eval()
        self.print_log(f'Evaluating on validation fold (Epoch {epoch+1})')
        loss, cnt, accuracy = 0, 0, 0
        label_list, pred_list = [], []
        process = tqdm(self.data_loader['val'], desc=f"Epoch {epoch+1} (Val)")
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                acc_data = inputs['accelerometer'].to(device)
                targets = targets.to(device)
                gyro_data = inputs.get('gyroscope', None)
                if gyro_data is not None: gyro_data = gyro_data.to(device)
                quaternion = inputs.get('quaternion', None)
                if quaternion is not None: quaternion = quaternion.to(device)
                
                if hasattr(self.model, 'forward_quaternion') and quaternion is not None:
                    logits = self.model.forward_quaternion(acc_data.float(), quaternion.float())
                elif gyro_data is not None and hasattr(self.model, 'forward_multi_sensor'):
                    logits = self.model.forward_multi_sensor(acc_data.float(), gyro_data.float())
                else: logits = self.model(acc_data.float())
                
                batch_loss = self.criterion(logits, targets)
                loss += batch_loss.sum().item()
                predictions = torch.argmax(F.log_softmax(logits, dim=1), 1)
                accuracy += (predictions == targets).sum().item()
                label_list.extend(targets.cpu().tolist())
                pred_list.extend(predictions.cpu().tolist())
                cnt += len(targets)
                process.set_postfix({'loss': f"{loss/cnt:.4f}", 'acc': f"{100.0*accuracy/cnt:.2f}%"})
            
            if cnt > 0:
                loss /= cnt
                target = np.array(label_list)
                y_pred = np.array(pred_list)
                f1 = f1_score(target, y_pred, average='macro') * 100
                precision, recall, _, _ = precision_recall_fscore_support(target, y_pred, average='macro')
                precision *= 100
                recall *= 100
                balanced_acc = balanced_accuracy_score(target, y_pred) * 100
                accuracy *= 100. / cnt
                
                self.print_log(f'Val metrics: Loss={loss:.4f}, Acc={accuracy:.2f}%, F1={f1:.2f}, Precision={precision:.2f}%, Recall={recall:.2f}%, Balanced Acc={balanced_acc:.2f}%')
                try: self.cm_viz(y_pred, target)
                except Exception as e: self.print_log(f"Error creating confusion matrix: {str(e)}")
                
                return {'loss': loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'balanced_accuracy': balanced_acc}
            else:
                self.print_log("No validation samples processed")
                return {'loss': float('inf'), 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'balanced_accuracy': 0}

    def generate_filter_comparison(self):
        comparison_dir = os.path.join(self.arg.work_dir, 'filter_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        cv_summary_path = os.path.join(self.arg.work_dir, 'cv_summary.json')
        if not os.path.exists(cv_summary_path): return
        try:
            with open(cv_summary_path, 'r') as f: cv_summary = json.load(f)
            if 'filter_type' not in cv_summary: cv_summary['filter_type'] = self.filter_type
            filter_summary_path = os.path.join(comparison_dir, f'{self.filter_type}_summary.json')
            with open(filter_summary_path, 'w') as f: json.dump(cv_summary, f, indent=2)
        except Exception as e: self.print_log(f"Error saving filter comparison data: {str(e)}")
    
    def start(self):
        try:
            self.train_loss_summary, self.val_loss_summary, self.train_metrics, self.val_metrics = [], [], [], []
            self.best_accuracy, self.best_f1, self.best_loss, self.patience_counter = 0, 0, float('inf'), 0
            self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
            results = self.create_df()
            fold_assignments = []
            if hasattr(self.arg, 'kfold'):
                if isinstance(self.arg.kfold, dict) and self.arg.kfold.get('enabled', False):
                    if 'fold_assignments' in self.arg.kfold:
                        fold_assignments = self.arg.kfold.get('fold_assignments', [])
            if not fold_assignments and self.arg.subjects:
                all_subjects = self.arg.subjects.copy()
                num_folds = getattr(self.arg, 'num_folds', 5)
                if hasattr(self.arg, 'kfold') and isinstance(self.arg.kfold, dict):
                    num_folds = self.arg.kfold.get('num_folds', 5)
                np.random.seed(self.arg.seed)
                np.random.shuffle(all_subjects)
                fold_size = len(all_subjects) // num_folds
                for i in range(num_folds):
                    start_idx = i * fold_size
                    end_idx = start_idx + fold_size if i < num_folds - 1 else len(all_subjects)
                    fold_assignments.append(all_subjects[start_idx:end_idx])
            fold_metrics = []
            all_subjects = []
            for fold in fold_assignments:
                all_subjects.extend(fold)
            for fold_idx, val_fold in enumerate(fold_assignments):
                self.print_log(f"\n{'='*20} Fold {fold_idx+1}/{len(fold_assignments)} {'='*20}")
                self.best_loss, self.best_accuracy, self.best_f1, self.patience_counter = float('inf'), 0, 0, 0
                self.val_subject = val_fold
                self.train_subjects = []
                for i, fold in enumerate(fold_assignments):
                    if i != fold_idx:
                        self.train_subjects.extend(fold)
                self.print_log(f'Validation subjects: {self.val_subject}')
                self.print_log(f'Training subjects: {self.train_subjects}')
                fold_dir = os.path.join(self.arg.work_dir, f"fold_{fold_idx+1}")
                os.makedirs(fold_dir, exist_ok=True)
                fold_cache_dir = os.path.join(self.cache_dir, f"fold_{fold_idx+1}")
                os.makedirs(fold_cache_dir, exist_ok=True)
                clear_filters()
                self.model = self.load_model(self.arg.model, self.arg.model_args)
                if len(self.available_gpus) > 1 and self.arg.multi_gpu:
                    self.model = nn.DataParallel(self.model, device_ids=self.available_gpus)
                if not self.load_data():
                    self.print_log(f"ERROR: Failed to load data for fold {fold_idx+1}")
                    continue
                self.load_optimizer()
                self.model_path = os.path.join(fold_dir, f"{self.arg.model_saved_name}.pt")
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    train_result = self.train(epoch)
                    if train_result.get('early_stop', False):
                        self.print_log(f"Early stopping after {epoch+1} epochs (no F1 improvement)")
                        break
                if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                    try: 
                        self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                        shutil.copy(os.path.join(self.arg.work_dir, "train_vs_val_loss.png"), 
                                  os.path.join(fold_dir, "train_vs_val_loss.png"))
                    except Exception as e: self.print_log(f"Error creating loss visualization: {str(e)}")
                if os.path.exists(self.model_path):
                    val_model = self.load_model(self.arg.model, self.arg.model_args)
                    if len(self.available_gpus) > 1 and self.arg.multi_gpu:
                        val_model = nn.DataParallel(val_model, device_ids=self.available_gpus)
                    try:
                        if isinstance(val_model, nn.DataParallel): val_model.module.load_state_dict(torch.load(self.model_path))
                        else: val_model.load_state_dict(torch.load(self.model_path))
                        self.model = val_model
                    except Exception as e: self.print_log(f"WARNING: Could not load best model: {str(e)}")
                self.model.eval()
                final_metrics = self.eval(epoch=0)
                fold_result = {
                    'fold': fold_idx + 1,
                    'val_subjects': self.val_subject,
                    'train_subjects': self.train_subjects,
                    'accuracy': final_metrics['accuracy'],
                    'f1': final_metrics['f1'],
                    'precision': final_metrics['precision'],
                    'recall': final_metrics['recall'],
                    'balanced_accuracy': final_metrics['balanced_accuracy']
                }
                fold_metrics.append(fold_result)
                result_row = pd.Series({
                    'fold': fold_idx + 1,
                    'val_subject': str(self.val_subject),
                    'train_subjects': str(self.train_subjects),
                    'accuracy': round(final_metrics['accuracy'], 2),
                    'f1_score': round(final_metrics['f1'], 2),
                    'precision': round(final_metrics['precision'], 2),
                    'recall': round(final_metrics['recall'], 2)
                })
                results.loc[len(results)] = result_row
                with open(os.path.join(fold_dir, 'validation_results.json'), 'w') as f:
                    json.dump(final_metrics, f, indent=2)
                if os.path.exists(os.path.join(self.arg.work_dir, "confusion_matrix.png")):
                    shutil.copy(os.path.join(self.arg.work_dir, "confusion_matrix.png"), 
                              os.path.join(fold_dir, "confusion_matrix.png"))
                self.train_loss_summary, self.val_loss_summary = [], []
            if fold_metrics:
                avg_metrics = {
                    'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
                    'accuracy_std': np.std([m['accuracy'] for m in fold_metrics]),
                    'f1': np.mean([m['f1'] for m in fold_metrics]),
                    'f1_std': np.std([m['f1'] for m in fold_metrics]),
                    'precision': np.mean([m['precision'] for m in fold_metrics]),
                    'precision_std': np.std([m['precision'] for m in fold_metrics]),
                    'recall': np.mean([m['recall'] for m in fold_metrics]),
                    'recall_std': np.std([m['recall'] for m in fold_metrics]),
                    'balanced_accuracy': np.mean([m['balanced_accuracy'] for m in fold_metrics]),
                    'balanced_accuracy_std': np.std([m['balanced_accuracy'] for m in fold_metrics])
                }
                cv_summary = {'fold_metrics': fold_metrics, 'average_metrics': avg_metrics, 'filter_type': self.filter_type}
                summary_path = os.path.join(self.arg.work_dir, 'cv_summary.json')
                with open(summary_path, 'w') as f: json.dump(cv_summary, f, indent=2)
                self.print_log(f'\n===== Cross-Validation Results =====')
                self.print_log(f'Mean accuracy: {avg_metrics["accuracy"]:.2f}% ± {avg_metrics["accuracy_std"]:.2f}%')
                self.print_log(f'Mean F1 score: {avg_metrics["f1"]:.2f} ± {avg_metrics["f1_std"]:.2f}')
                self.print_log(f'Mean precision: {avg_metrics["precision"]:.2f}% ± {avg_metrics["precision_std"]:.2f}%')
                self.print_log(f'Mean recall: {avg_metrics["recall"]:.2f}% ± {avg_metrics["recall_std"]:.2f}%')
                self.print_log(f'Mean balanced accuracy: {avg_metrics["balanced_accuracy"]:.2f}% ± {avg_metrics["balanced_accuracy_std"]:.2f}%')
                if hasattr(self.arg, 'run_comparison') and self.arg.run_comparison:
                    self.generate_filter_comparison()
            results.to_csv(os.path.join(self.arg.work_dir, 'fold_scores.csv'), index=False)
        except Exception as e:
            self.print_log(f"ERROR in training workflow: {str(e)}")
            self.print_log(traceback.format_exc())
            if hasattr(self, 'model'):
                emergency_path = os.path.join(self.arg.work_dir, 'emergency_checkpoint.pt')
                try:
                    if isinstance(self.model, nn.DataParallel): torch.save(self.model.module.state_dict(), emergency_path)
                    else: torch.save(self.model.state_dict(), emergency_path)
                except Exception as save_error: self.print_log(f"Could not save emergency checkpoint: {str(save_error)}")

def main():
    parser = get_args()
    arg = parser.parse_args()
    if arg.config is not None:
        with open(arg.config, 'r') as f: default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(arg).keys()
        parser.add_argument('--test-feeder-args', default=None, help='Test loader arguments') 
        arg_after_add = parser.parse_args([])
        key = vars(arg_after_add).keys()
        for k in default_arg.keys():
            if k not in key: 
                parser.add_argument(f'--{k.replace("_", "-")}', default=None)
        parser.set_defaults(**default_arg)
        arg = parser.parse_args()
    init_seed(arg.seed)
    trainer = Trainer(arg)
    trainer.start()

if __name__ == '__main__': main()
