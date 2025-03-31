import tqdm
import traceback
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
from itertools import combinations
import math

MAX_THREADS = 40
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
FILTER_INSTANCES = {}

class FallDetectionTransformer(nn.Module):
    def __init__(self, acc_frames=64, num_classes=2, num_heads=4, acc_coords=3, 
                 num_layers=2, embed_dim=32, dropout=0.2, use_batch_norm=True, **kwargs):
        super().__init__()
        self.acc_frames = acc_frames
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        self.num_classes = num_classes
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(), nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )
        self.gyro_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(), nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )
        self.feature_dim = embed_dim * 2
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=num_heads, dim_feedforward=self.feature_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=num_layers, 
            norm=nn.LayerNorm(self.feature_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.classifier[-1].bias)
        fan_in = self.classifier[-1].weight.size(1)
        nn.init.normal_(self.classifier[-1].weight, 0, 1/math.sqrt(fan_in))
    
    def forward(self, acc_data, gyro_data):
        if acc_data.shape[1] == self.acc_coords and len(acc_data.shape) == 3:
            acc_data_conv = acc_data
        else:
            acc_data_conv = acc_data.transpose(1, 2)
            
        if gyro_data.shape[1] == self.acc_coords and len(gyro_data.shape) == 3:
            gyro_data_conv = gyro_data
        else:
            gyro_data_conv = gyro_data.transpose(1, 2)
        
        acc_features = self.acc_encoder(acc_data_conv)
        gyro_features = self.gyro_encoder(gyro_data_conv)
        
        acc_features = acc_features.transpose(1, 2)
        gyro_features = gyro_features.transpose(1, 2)
        
        fused_features = torch.cat([acc_features, gyro_features], dim=2)
        transformer_output = self.transformer(fused_features)
        pooled = torch.mean(transformer_output, dim=1)
        
        return self.classifier(pooled)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected')

def get_args():
    parser = argparse.ArgumentParser(description='Fall Detection and Human Activity Recognition')
    parser.add_argument('--config', default='./config/lightweightTransformer.yaml', help='Config file path')
    parser.add_argument('--dataset', type=str, default='smartfallmm', help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='Training batch size')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N', help='Validation batch size')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N', help='Test batch size')
    parser.add_argument('--num-epoch', type=int, default=60, metavar='N', help='Training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Start epoch number')
    parser.add_argument('--weights-only', type=str2bool, default=False, help='Load only weights')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--base-lr', type=float, default=0.0005, metavar='LR', help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0004, help='Weight decay')
    parser.add_argument('--kfold', type=str2bool, default=True, help='Use cross validation')
    parser.add_argument('--num-folds', type=int, default=6, help='Number of folds')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--model', default=None, help='Model class path')
    parser.add_argument('--device', nargs='+', default=[0], type=int, help='CUDA device IDs')
    parser.add_argument('--model-args', default=None, help='Model arguments')
    parser.add_argument('--weights', type=str, help='Pretrained weights file')
    parser.add_argument('--model-saved-name', type=str, default='model', help='Model save name')
    parser.add_argument('--loss', default='loss.BCE', help='Loss function class')
    parser.add_argument('--loss-args', default="{}", type=str, help='Loss function arguments')
    parser.add_argument('--dataset-args', default=None, help='Dataset arguments')
    parser.add_argument('--subjects', nargs='+', type=int, default=[32,39,30,31,33,34,35,37,43,44,45,36,29], help='Subject IDs')
    parser.add_argument('--val-subjects', nargs='+', type=int, default=[38,46], help='Validation subject IDs')
    parser.add_argument('--permanent-train', nargs='+', type=int, default=[45,36,29], help='Always in training set')
    parser.add_argument('--feeder', default=None, help='DataLoader class')
    parser.add_argument('--train-feeder-args', default=None, help='Train loader arguments')
    parser.add_argument('--val-feeder-args', default=None, help='Validation loader arguments')
    parser.add_argument('--test-feeder-args', default=None, help='Test loader arguments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--work-dir', type=str, default='work_dir', help='Working directory')
    parser.add_argument('--print-log', type=str2bool, default=True, help='Print and save logs')
    parser.add_argument('--phase', type=str, default='train', help='Train or evaluation')
    parser.add_argument('--num-worker', type=int, default=0, help='Data loader workers')
    parser.add_argument('--multi-gpu', type=str2bool, default=False, help='Use multiple GPUs')
    parser.add_argument('--parallel-threads', type=int, default=4, help='Processing threads')
    parser.add_argument('--run-comparison', type=str2bool, default=False, help='Run filter comparison')
    parser.add_argument('--filter-type', type=str, default='madgwick', help='Filter type to use')
    parser.add_argument('--rotate-tests', type=str2bool, default=False, help='Rotate through test combinations')
    parser.add_argument('--test-combinations', type=str2bool, default=False, help='Test all combinations')
    parser.add_argument('--export-edge', type=str2bool, default=False, help='Export model for edge deployment')
    return parser

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

def generate_cv_folds(subjects, val_subjects, permanent_train, max_test_subjects=2, num_folds=6):
    available_subjects = [s for s in subjects if s not in permanent_train and s not in val_subjects]
    max_possible_folds = len(available_subjects) // max_test_subjects
    if num_folds > max_possible_folds:
        print(f"Warning: Requested {num_folds} folds but only {max_possible_folds} are possible. Using {max_possible_folds} folds.")
        num_folds = max_possible_folds
    all_test_combinations = list(combinations(available_subjects, max_test_subjects))
    random.seed(42)
    random.shuffle(all_test_combinations)
    selected_folds = []
    used_subjects = set()
    for combo in all_test_combinations:
        if len(selected_folds) >= num_folds: break
        if any(subj in used_subjects for subj in combo): continue
        selected_folds.append(list(combo))
        used_subjects.update(combo)
    if len(selected_folds) < num_folds and max_test_subjects > 1:
        remaining_subjects = [s for s in available_subjects if s not in used_subjects]
        while len(selected_folds) < num_folds and remaining_subjects:
            next_subject = remaining_subjects.pop(0)
            selected_folds.append([next_subject])
            used_subjects.add(next_subject)
    fold_assignments = []
    for test_subjects in selected_folds:
        train_subjects = permanent_train.copy()
        for s in subjects:
            if s not in test_subjects and s not in val_subjects and s not in train_subjects:
                train_subjects.append(s)
        fold_assignments.append({
            'train': train_subjects,
            'test': test_subjects,
            'val': val_subjects
        })
    return fold_assignments

class Trainer:
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_f1 = 0
        self.best_accuracy = 0
        self.patience_counter = 0
        self.val_subjects = arg.val_subjects if hasattr(arg, 'val_subjects') else [38, 46]
        self.permanent_train = arg.permanent_train if hasattr(arg, 'permanent_train') else [45, 36, 29]
        self.available_subjects = [s for s in arg.subjects if s not in self.val_subjects and s not in self.permanent_train]
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = dict()
        self.best_loss = float('inf')
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}.pt'
        self.weights_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}_weights_only.pt'
        self.max_threads = min(getattr(arg, 'parallel_threads', 4), MAX_THREADS)
        if not hasattr(arg, 'dataset_args') or arg.dataset_args is None:
            arg.dataset_args = {
                'modalities': ['accelerometer', 'gyroscope'],
                'mode': 'sliding_window',
                'max_length': 64,
                'task': 'fd',
                'age_group': ['young'],
                'sensors': ['watch'],
                'fusion_options': {
                    'enabled': True,
                    'filter_type': getattr(arg, 'filter_type', 'madgwick'),
                    'process_per_window': True,
                    'preserve_filter_state': True,
                    'acc_threshold': 3.0,
                    'gyro_threshold': 1.0,
                    'visualize': False,
                    'save_aligned': True
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
        self.inertial_modality = [modality for modality in arg.dataset_args.get('modalities', ['accelerometer', 'gyroscope']) 
                                 if modality != 'skeleton']
        self.has_gyro = 'gyroscope' in self.inertial_modality
        fusion_options = arg.dataset_args.get('fusion_options', {})
        self.has_fusion = len(self.inertial_modality) > 1 or (fusion_options and fusion_options.get('enabled', False))
        self.fuse = self.has_fusion
        self.filter_type = getattr(arg, 'filter_type', None) or fusion_options.get('filter_type', 'madgwick')
        self.filter_params = fusion_options
        self.use_cache = fusion_options.get('use_cache', True)
        self.cache_dir = fusion_options.get('cache_dir', 'processed_data')
        self.cv_results = {
            'fold_metrics': [],
            'average_metrics': {},
            'filter_type': self.filter_type,
            'folds': []
        }
        self.enable_kfold = getattr(arg, 'kfold', True)
        self.num_folds = getattr(arg, 'num_folds', 6)
        if self.enable_kfold:
            self.fold_assignments = generate_cv_folds(
                arg.subjects,
                self.val_subjects,
                self.permanent_train,
                max_test_subjects=2,
                num_folds=self.num_folds
            )
            self.cv_results['folds'] = self.fold_assignments
        if self.use_cache:
            filter_cache_dir = os.path.join(self.cache_dir, self.filter_type)
            os.makedirs(filter_cache_dir, exist_ok=True)
        os.makedirs(self.arg.work_dir, exist_ok=True)
        if hasattr(arg, 'config') and arg.config:
            self.save_config(arg.config, arg.work_dir)
        self.available_gpus = setup_gpu_environment(arg)
        arg.device = self.available_gpus if self.available_gpus else getattr(arg, 'device', [0])
        self.output_device = arg.device[0] if isinstance(arg.device, list) and len(arg.device) > 0 else arg.device
        if not hasattr(arg, 'model') or not arg.model:
            self.model = FallDetectionTransformer(
                num_layers=3,
                embed_dim=48,
                acc_coords=3,
                num_classes=2,
                acc_frames=64,
                num_heads=8,
                dropout=0.3,
                use_batch_norm=True
            ).to(f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu')
        else:
            Model = import_class(arg.model)
            model_args = arg.model_args or {
                'num_layers': 3,
                'embed_dim': 48,
                'acc_coords': 3,
                'num_classes': 2,
                'acc_frames': 64,
                'num_heads': 8,
                'dropout': 0.3,
                'use_batch_norm': True
            }
            if isinstance(model_args, str):
                try:
                    model_args = eval(model_args)
                except:
                    try:
                        model_args = yaml.safe_load(model_args)
                    except:
                        pass
            self.model = Model(**model_args).to(f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu')
        self.load_loss()
        if self.fuse and self.filter_type != 'none':
            all_subjects = arg.subjects
            cache_dir = os.path.join(self.cache_dir, self.filter_type)
            if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
                self.print_log(f"Preprocessing all {len(all_subjects)} subjects with {self.filter_type} filter")
                from utils.imu_fusion import preprocess_all_subjects
                preprocess_all_subjects(all_subjects, self.filter_type, cache_dir, 
                                      arg.dataset_args.get('max_length', 64))
                self.print_log(f"Preprocessing complete")
        self.print_log(f'# Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        self.print_log(f'Sensor modalities: {self.inertial_modality}')
        self.print_log(f'Using fusion: {self.fuse} with filter type: {self.filter_type}')
        if self.available_gpus: 
            self.print_log(f'Using GPUs: {self.available_gpus}')
        else:
            self.print_log('Using CPU for computation')

    def save_config(self, src_path, desc_path):
        config_file = src_path.rpartition("/")[-1]
        shutil.copy(src_path, f'{desc_path}/{config_file}')

    def load_loss(self): 
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_weights(self, path=None):
        weights_path = path if path else self.model_path
        try:
            self.model.load_state_dict(torch.load(weights_path))
            return True
        except Exception as e:
            self.print_log(f"Error loading weights: {str(e)}")
            return False

    def save_weights(self, path=None, weights_only=False):
        try:
            if weights_only:
                torch.save(self.model.state_dict(), path if path else self.weights_path)
            else:
                torch.save(self.model, path if path else self.model_path)
            return True
        except Exception as e:
            self.print_log(f"Error saving weights: {str(e)}")
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

    def distribution_viz(self, labels, work_dir, mode):
        values, count = np.unique(labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(values, count)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title(f'{mode.capitalize()} Label Distribution')
        plt.savefig(f'{work_dir}/{mode}_label_distribution.png')
        plt.close()

    def load_data(self, train_subjects, test_subjects=None):
        try:
            if not hasattr(self.arg, 'feeder') or not self.arg.feeder:
                self.arg.feeder = 'Feeder.Make_Dataset.UTD_mm'
                self.print_log(f"Using default feeder: {self.arg.feeder}")   
            if not hasattr(self.arg, 'train_feeder_args') or not self.arg.train_feeder_args:
                self.arg.train_feeder_args = {'batch_size': self.arg.batch_size, 'drop_last': True}
            if not hasattr(self.arg, 'val_feeder_args') or not self.arg.val_feeder_args:
                self.arg.val_feeder_args = {'batch_size': self.arg.val_batch_size, 'drop_last': True}
            if not hasattr(self.arg, 'test_feeder_args') or not self.arg.test_feeder_args:
                test_batch_size = getattr(self.arg, 'test_batch_size', getattr(self.arg, 'val_batch_size', self.arg.batch_size))
                self.arg.test_feeder_args = {'batch_size': test_batch_size, 'drop_last': False}
            Feeder = import_class(self.arg.feeder)
            builder = prepare_smartfallmm(self.arg)
            self.print_log(f"Loading training data for subjects: {train_subjects}")
            self.norm_train = split_by_subjects(builder, train_subjects, self.fuse)
            if 'accelerometer' not in self.norm_train or len(self.norm_train['accelerometer']) == 0: 
                self.print_log("ERROR: No accelerometer data found in training set")
                return False
            if 'labels' not in self.norm_train or len(self.norm_train['labels']) == 0: 
                self.print_log("ERROR: No labels found in training set")
                return False
            try:
                self.data_loader['train'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.train_feeder_args, dataset=self.norm_train),
                    batch_size=self.arg.batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker,
                    collate_fn=getattr(Feeder, 'custom_collate_fn', None))
                self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
            except Exception as e:
                self.print_log(f"ERROR creating train dataloader: {str(e)}")
                self.print_log(traceback.format_exc())
                return False
            self.print_log(f"Loading validation data for subjects: {self.val_subjects}")
            self.norm_val = split_by_subjects(builder, self.val_subjects, self.fuse)
            if 'accelerometer' not in self.norm_val or len(self.norm_val['accelerometer']) == 0: 
                self.print_log("WARNING: No accelerometer data found in validation set")
            elif 'labels' not in self.norm_val or len(self.norm_val['labels']) == 0:
                self.print_log("WARNING: No labels found in validation set")
            else:
                try:
                    self.data_loader['val'] = torch.utils.data.DataLoader(
                        dataset=Feeder(**self.arg.val_feeder_args, dataset=self.norm_val),
                        batch_size=self.arg.val_batch_size,
                        shuffle=False,
                        num_workers=self.arg.num_worker,
                        collate_fn=getattr(Feeder, 'custom_collate_fn', None))
                    self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, 'val')
                except Exception as e:
                    self.print_log(f"ERROR creating validation dataloader: {str(e)}")
                    self.print_log(traceback.format_exc())
            if test_subjects:
                self.print_log(f"Loading test data for subjects: {test_subjects}")
                self.norm_test = split_by_subjects(builder, test_subjects, self.fuse)
                if 'accelerometer' not in self.norm_test or len(self.norm_test['accelerometer']) == 0: 
                    self.print_log("WARNING: No accelerometer data found in test set")
                elif 'labels' not in self.norm_test or len(self.norm_test['labels']) == 0:
                    self.print_log("WARNING: No labels found in test set")
                else:
                    try:
                        test_batch_size = getattr(self.arg, 'test_batch_size', getattr(self.arg, 'val_batch_size', self.arg.batch_size))
                        self.data_loader['test'] = torch.utils.data.DataLoader(
                            dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                            batch_size=test_batch_size,
                            shuffle=False,
                            num_workers=self.arg.num_worker,
                            collate_fn=getattr(Feeder, 'custom_collate_fn', None))
                        self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, 'test')
                    except Exception as e:
                        self.print_log(f"ERROR creating test dataloader: {str(e)}")
                        self.print_log(traceback.format_exc())
            return True
        except Exception as e:
            self.print_log(f"ERROR in load_data: {str(e)}")
            self.print_log(traceback.format_exc())
            return False

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
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f: 
                print(string, file=f)

    def loss_viz(self, train_loss, val_loss, save_path=None):
        epochs = range(len(train_loss))
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_loss, 'b-', label="Training Loss")
        plt.plot(epochs, val_loss, 'r-', label="Validation Loss")
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(save_path if save_path else f'{self.arg.work_dir}/train_vs_val_loss.png')
        plt.close()

    def cm_viz(self, y_pred, y_true, filename='confusion_matrix.png'):
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
        plt.savefig(f'{self.arg.work_dir}/{filename}')
        plt.close()

    def create_df(self, columns=['fold', 'train_subjects', 'test_subjects', 'accuracy', 'f1_score', 'precision', 'recall']): 
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
        if not self.filter_params.get('preserve_filter_state', True):
            global FILTER_INSTANCES
            FILTER_INSTANCES = {}
        for batch_idx, (data_tuple, targets, idx) in enumerate(process):
            try:
                acc_data, gyro_data = data_tuple
                if acc_data is None or gyro_data is None:
                    self.print_log(f"WARNING: Missing data in batch {batch_idx}")
                    continue
                if torch.any(torch.isnan(acc_data)) or torch.any(torch.isnan(gyro_data)):
                    self.print_log(f"WARNING: NaN values in sensor data in batch {batch_idx}")
                    continue
                if len(acc_data.shape) == 2:
                    batch_size = acc_data.shape[0]
                    acc_data = acc_data.view(batch_size, -1, 3)
                if len(gyro_data.shape) == 2:
                    batch_size = gyro_data.shape[0]
                    gyro_data = gyro_data.view(batch_size, -1, 3)
                acc_data = acc_data.to(device).float()
                gyro_data = gyro_data.to(device).float()
                targets = targets.to(device)
                timer['dataloader'] += self.split_time()
                self.optimizer.zero_grad()
                logits = self.model(acc_data, gyro_data)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()
                timer['model'] += self.split_time()
                with torch.no_grad():
                    train_loss += loss.item()
                    predictions = torch.argmax(F.softmax(logits, dim=1), 1)
                    accuracy += (predictions == targets).sum().item()
                    y_true.extend(targets.cpu().tolist())
                    y_pred.extend(predictions.cpu().tolist())
                cnt += len(targets)
                timer['stats'] += self.split_time()
                process.set_postfix({'loss': f"{train_loss/(batch_idx+1):.4f}", 'acc': f"{100.0*accuracy/cnt:.2f}%"})
            except Exception as e:
                self.print_log(f"ERROR in training batch {batch_idx}: {str(e)}")
                self.print_log(traceback.format_exc())
                continue
        if cnt == 0:
            self.print_log("ERROR: No samples processed in training epoch")
            return {'early_stop': True}
        train_loss /= max(1, len(loader))
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
        val_metrics = self.eval(epoch, 'val')
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
                self.save_weights(self.model_path, weights_only=False)
                self.save_weights(self.weights_path, weights_only=True)
            except Exception as e:
                self.print_log(f"Error saving model: {str(e)}")
                emergency_path = os.path.join(self.arg.work_dir, 'best_model_emergency.pt')
                try:
                    self.save_weights(emergency_path, weights_only=False)
                except: 
                    pass
        else:
            self.patience_counter += 1
            self.print_log(f"No F1 improvement for {self.patience_counter} epochs (patience: {self.arg.patience})")
            if self.patience_counter >= self.arg.patience:
                return {'early_stop': True}
        return {'early_stop': self.patience_counter >= self.arg.patience}

    def eval(self, epoch, mode='val'):
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        self.model.eval()
        self.print_log(f'Evaluating on {mode} set (Epoch {epoch+1})')
        if mode not in self.data_loader:
            self.print_log(f"ERROR: No dataloader available for {mode} set")
            return {'loss': float('inf'), 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'balanced_accuracy': 0}
        loader = self.data_loader[mode]
        loss, cnt, accuracy = 0, 0, 0
        label_list, pred_list = [], []
        process = tqdm(loader, desc=f"Epoch {epoch+1} ({mode.capitalize()})")
        with torch.no_grad():
            for batch_idx, (data_tuple, targets, idx) in enumerate(process):
                try:
                    acc_data, gyro_data = data_tuple
                    if acc_data is None or gyro_data is None:
                        self.print_log(f"WARNING: Missing data in {mode} batch {batch_idx}")
                        continue
                    if torch.any(torch.isnan(acc_data)) or torch.any(torch.isnan(gyro_data)):
                        self.print_log(f"WARNING: NaN values in sensor data in {mode} batch {batch_idx}")
                        continue
                    if len(acc_data.shape) == 2:
                        batch_size = acc_data.shape[0]
                        acc_data = acc_data.view(batch_size, -1, 3)
                    if len(gyro_data.shape) == 2:
                        batch_size = gyro_data.shape[0]
                        gyro_data = gyro_data.view(batch_size, -1, 3)
                    acc_data = acc_data.to(device).float()
                    gyro_data = gyro_data.to(device).float()
                    targets = targets.to(device)
                    logits = self.model(acc_data, gyro_data)
                    batch_loss = self.criterion(logits, targets)
                    loss += batch_loss.item()
                    predictions = torch.argmax(F.softmax(logits, dim=1), 1)
                    accuracy += (predictions == targets).sum().item()
                    label_list.extend(targets.cpu().tolist())
                    pred_list.extend(predictions.cpu().tolist())
                    cnt += len(targets)
                    process.set_postfix({'loss': f"{loss/(batch_idx+1):.4f}", 'acc': f"{100.0*accuracy/cnt:.2f}%"})
                except Exception as e:
                    self.print_log(f"ERROR in {mode} batch {batch_idx}: {str(e)}")
                    self.print_log(traceback.format_exc())
                    continue
            if cnt > 0:
                loss /= max(1, len(loader))
                target = np.array(label_list)
                y_pred = np.array(pred_list)
                if len(np.unique(target)) > 1:
                    f1 = f1_score(target, y_pred, average='macro') * 100
                    precision, recall, _, _ = precision_recall_fscore_support(target, y_pred, average='macro')
                    precision *= 100
                    recall *= 100
                    balanced_acc = balanced_accuracy_score(target, y_pred) * 100
                else:
                    f1 = 0.0
                    precision = 0.0
                    recall = 0.0
                    balanced_acc = 0.0
                    self.print_log(f"WARNING: Only one class found in {mode} set: {np.unique(target)}")
                accuracy = 100.0 * accuracy / cnt
                self.print_log(f'{mode.capitalize()} metrics: Loss={loss:.4f}, Acc={accuracy:.2f}%, F1={f1:.2f}, Precision={precision:.2f}%, Recall={recall:.2f}%, Balanced Acc={balanced_acc:.2f}%')
                cm_filename = f'confusion_matrix_{mode}.png'
                try: 
                    self.cm_viz(y_pred, target, cm_filename)
                except Exception as e: 
                    self.print_log(f"Error creating confusion matrix: {str(e)}")
                return {'loss': loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'balanced_accuracy': balanced_acc}
            else:
                self.print_log(f"No {mode} samples processed")
                return {'loss': float('inf'), 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'balanced_accuracy': 0}

    def save_cv_summary(self):
        if not self.cv_results['fold_metrics']:
            return
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        avg_metrics = {}
        for metric in metrics:
            values = [fold[metric] for fold in self.cv_results['fold_metrics']]
            avg_metrics[metric] = float(np.mean(values))
            avg_metrics[f"{metric}_std"] = float(np.std(values))
        self.cv_results['average_metrics'] = avg_metrics
        self.cv_results['filter_type'] = self.filter_type
        summary_path = os.path.join(self.arg.work_dir, 'cv_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.cv_results, f, indent=2)
        self.print_log(f"Cross-validation summary saved to {summary_path}")
    
    def run_fold(self, fold_idx, train_subjects, test_subjects, fold_dir):
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_accuracy = 0
        self.best_f1 = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.print_log(f"Creating new model instance for fold {fold_idx+1}")
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        self.model = FallDetectionTransformer(
            num_layers=3,
            embed_dim=48,
            acc_coords=3,
            num_classes=2,
            acc_frames=64,
            num_heads=8,
            dropout=0.3,
            use_batch_norm=True
        ).to(device)
        if not self.load_data(train_subjects, test_subjects):
            self.print_log(f"ERROR: Failed to load data for fold {fold_idx+1}")
            return None
        self.load_optimizer()
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            train_result = self.train(epoch)
            if train_result.get('early_stop', False):
                self.print_log(f"Early stopping after {epoch+1} epochs (no F1 improvement)")
                break
        if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
            try: 
                loss_path = os.path.join(fold_dir, f"train_vs_val_loss_fold_{fold_idx+1}.png")
                self.loss_viz(self.train_loss_summary, self.val_loss_summary, loss_path)
            except Exception as e: 
                self.print_log(f"Error creating loss visualization: {str(e)}")
        if os.path.exists(self.model_path):
            try:
                self.load_weights(self.model_path)
                self.print_log("Loaded best model for final evaluation")
            except Exception as e: 
                self.print_log(f"WARNING: Could not load best model: {str(e)}")
        self.model.eval()
        test_metrics = self.eval(0, 'test')
        with open(os.path.join(fold_dir, f'fold_{fold_idx+1}_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        for file in ['confusion_matrix_test.png', 'confusion_matrix_val.png']:
            if os.path.exists(os.path.join(self.arg.work_dir, file)):
                shutil.copy(os.path.join(self.arg.work_dir, file), 
                          os.path.join(fold_dir, f"{file.split('.')[0]}_fold_{fold_idx+1}.png"))
        if getattr(self.arg, 'export_edge', False):
            try:
                self.export_edge_model(os.path.join(fold_dir, f"edge_model_fold_{fold_idx+1}.tflite"))
            except Exception as e:
                self.print_log(f"Error exporting edge model: {str(e)}")
        return test_metrics
    
    def export_edge_model(self, filepath):
        try:
            import ai_edge_torch
            self.model.eval()
            use_cuda = torch.cuda.is_available()
            device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
            acc = torch.randn(1, 64, 3).to(device)
            gyro = torch.randn(1, 64, 3).to(device)
            nhwc_model = ai_edge_torch.to_channel_last_io(self.model, args=[0, 1])
            acc_nhwc = acc.permute(0, 2, 1)  # [1, 64, 3] -> [1, 3, 64]
            gyro_nhwc = gyro.permute(0, 2, 1)  # [1, 64, 3] -> [1, 3, 64]
            edge_model = ai_edge_torch.convert(nhwc_model, (acc_nhwc, gyro_nhwc))
            edge_model.export(filepath)
            self.print_log(f"Exported edge model to {filepath}")
            return True
        except ImportError:
            self.print_log("Could not import ai_edge_torch. Install with: pip install ai-edge-torch")
            return False
        except Exception as e:
            self.print_log(f"Error during edge model export: {str(e)}")
            self.print_log(traceback.format_exc())
            return False
    
    def start(self):
        try:
            self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
            global FILTER_INSTANCES
            FILTER_INSTANCES = {}
            if self.enable_kfold:
                self.print_log(f"\n===== Running {len(self.fold_assignments)}-fold cross validation =====")
                fold_results_df = self.create_df()
                cv_dir = os.path.join(self.arg.work_dir, "cv_results")
                os.makedirs(cv_dir, exist_ok=True)
                for fold_idx, fold in enumerate(self.fold_assignments):
                    train_subjects = fold['train']
                    test_subjects = fold['test']
                    self.print_log(f"\n{'='*20} Fold {fold_idx+1}/{len(self.fold_assignments)} {'='*20}")
                    self.print_log(f"Train: {train_subjects}")
                    self.print_log(f"Test: {test_subjects}")
                    self.print_log(f"Validation: {self.val_subjects}")
                    fold_dir = os.path.join(cv_dir, f"fold_{fold_idx+1}")
                    os.makedirs(fold_dir, exist_ok=True)
                    FILTER_INSTANCES = {}
                    fold_metrics = self.run_fold(fold_idx, train_subjects, test_subjects, fold_dir)
                    if fold_metrics:
                        result_row = {
                            'fold': fold_idx + 1,
                            'train_subjects': str(train_subjects),
                            'test_subjects': str(test_subjects),
                            'accuracy': round(fold_metrics['accuracy'], 2),
                            'f1_score': round(fold_metrics['f1'], 2),
                            'precision': round(fold_metrics['precision'], 2),
                            'recall': round(fold_metrics['recall'], 2)
                        }
                        fold_results_df.loc[len(fold_results_df)] = result_row
                        self.cv_results['fold_metrics'].append(fold_metrics)
                fold_results_df.to_csv(os.path.join(cv_dir, 'fold_results.csv'), index=False)
                self.save_cv_summary()
                avg_metrics = self.cv_results.get('average_metrics', {})
                if avg_metrics:
                    self.print_log(f'\n===== Cross-validation Results Summary ({self.filter_type}) =====')
                    self.print_log(f'Mean accuracy: {avg_metrics.get("accuracy", 0):.2f}% ± {avg_metrics.get("accuracy_std", 0):.2f}%')
                    self.print_log(f'Mean F1 score: {avg_metrics.get("f1", 0):.2f} ± {avg_metrics.get("f1_std", 0):.2f}')
                    self.print_log(f'Mean precision: {avg_metrics.get("precision", 0):.2f}% ± {avg_metrics.get("precision_std", 0):.2f}%')
                    self.print_log(f'Mean recall: {avg_metrics.get("recall", 0):.2f}% ± {avg_metrics.get("recall_std", 0):.2f}%')
                    self.print_log(f'Mean balanced accuracy: {avg_metrics.get("balanced_accuracy", 0):.2f}% ± {avg_metrics.get("balanced_accuracy_std", 0):.2f}%')
            else:
                train_subjects = [s for s in self.arg.subjects if s not in self.val_subjects]
                if not self.load_data(train_subjects):
                    self.print_log("ERROR: Failed to load data")
                    return
                self.load_optimizer()
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    train_result = self.train(epoch)
                    if train_result.get('early_stop', False):
                        self.print_log(f"Early stopping after {epoch+1} epochs (no F1 improvement)")
                        break
                if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                    try: 
                        self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                    except Exception as e: 
                        self.print_log(f"Error creating loss visualization: {str(e)}")
                if os.path.exists(self.model_path):
                    try:
                        self.load_weights(self.model_path)
                        self.print_log("Loaded best model for final evaluation")
                    except Exception as e: 
                        self.print_log(f"WARNING: Could not load best model: {str(e)}")
                self.model.eval()
                final_metrics = self.eval(0, 'val')
                self.print_log(f'\n===== Final Results =====')
                self.print_log(f'Accuracy: {final_metrics["accuracy"]:.2f}%')
                self.print_log(f'F1 score: {final_metrics["f1"]:.2f}')
                self.print_log(f'Precision: {final_metrics["precision"]:.2f}%')
                self.print_log(f'Recall: {final_metrics["recall"]:.2f}%')
                self.print_log(f'Balanced accuracy: {final_metrics["balanced_accuracy"]:.2f}%')
                with open(os.path.join(self.arg.work_dir, 'final_results.json'), 'w') as f:
                    json.dump(final_metrics, f, indent=2)
                if getattr(self.arg, 'export_edge', False):
                    self.export_edge_model(os.path.join(self.arg.work_dir, "edge_model.tflite"))
        except Exception as e:
            self.print_log(f"ERROR in training workflow: {str(e)}")
            self.print_log(traceback.format_exc())
            if hasattr(self, 'model'):
                emergency_path = os.path.join(self.arg.work_dir, 'emergency_checkpoint.pt')
                try:
                    self.save_weights(emergency_path, weights_only=False)
                except Exception as save_error: 
                    self.print_log(f"Could not save emergency checkpoint: {str(save_error)}")

def main():
    parser = get_args()
    arg = parser.parse_args()
    if arg.config is not None:
        with open(arg.config, 'r') as f: 
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(arg).keys()
        for k in default_arg.keys():
            if k not in key: 
                arg.__dict__[k] = default_arg[k]
    init_seed(arg.seed)
    trainer = Trainer(arg)
    trainer.start()

if __name__ == '__main__': 
    main()
