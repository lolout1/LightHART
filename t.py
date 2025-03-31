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

MAX_THREADS = 40
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
FILTER_INSTANCES = {}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected')

def get_args():
    parser = argparse.ArgumentParser(description='Fall Detection and Human Activity Recognition')
    parser.add_argument('--config', default='./config/smartfallmm/fusion_madgwick.yaml', help='Config file path')
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

def get_or_create_filter(subject_id, action_id, filter_type, filter_params=None):
    from utils.imu_fusion import MadgwickFilter, KalmanFilter, ExtendedKalmanFilter
    key = f"{subject_id}_{action_id}_{filter_type}"
    if key not in FILTER_INSTANCES:
        print(f"Creating new filter instance for {key}")
        if filter_type == 'madgwick':
            beta = filter_params.get('beta', 0.1) if filter_params else 0.1
            FILTER_INSTANCES[key] = MadgwickFilter(beta=beta)
        elif filter_type == 'kalman':
            process_noise = filter_params.get('process_noise', 5e-5) if filter_params else 5e-5
            measurement_noise = filter_params.get('measurement_noise', 0.1) if filter_params else 0.1
            FILTER_INSTANCES[key] = KalmanFilter(process_noise=process_noise, measurement_noise=measurement_noise)
        elif filter_type == 'ekf':
            process_noise = filter_params.get('process_noise', 1e-5) if filter_params else 1e-5
            measurement_noise = filter_params.get('measurement_noise', 0.05) if filter_params else 0.05
            FILTER_INSTANCES[key] = ExtendedKalmanFilter(process_noise=process_noise, measurement_noise=measurement_noise)
        else:
            FILTER_INSTANCES[key] = MadgwickFilter()
    return FILTER_INSTANCES[key]

def process_sequence_with_filter(acc_data, gyro_data, timestamps=None, subject_id=0, action_id=0, 
                              filter_type='madgwick', filter_params=None, use_cache=True, 
                              cache_dir="processed_data", window_id=0):
    if filter_type == 'none':
        return np.zeros((len(acc_data), 4))
        
    cache_key = f"S{subject_id:02d}A{action_id:02d}W{window_id:04d}_{filter_type}"
    cache_path = os.path.join(cache_dir, f"{cache_key}.npz")
    
    if use_cache and os.path.exists(cache_path):
        try:
            cached_data = np.load(cache_path)
            return cached_data['quaternion']
        except Exception as e:
            print(f"Warning: Failed to load cached data: {str(e)}")
    
    orientation_filter = get_or_create_filter(subject_id, action_id, filter_type, filter_params)
    quaternions = np.zeros((len(acc_data), 4))
    
    for i in range(len(acc_data)):
        acc = acc_data[i]
        gyro = gyro_data[i]
        timestamp = timestamps[i] if timestamps is not None else None
        
        gravity_direction = np.array([0, 0, 9.81])
        if i > 0:
            r = Rotation.from_quat([quaternions[i-1, 1], quaternions[i-1, 2], quaternions[i-1, 3], quaternions[i-1, 0]])
            gravity_direction = r.inv().apply([0, 0, 9.81])
            
        acc_with_gravity = acc + gravity_direction
        norm = np.linalg.norm(acc_with_gravity)
        if norm > 1e-6:
            acc_with_gravity = acc_with_gravity / norm
            
        q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
        quaternions[i] = q
    
    if use_cache:
        cache_dir_path = os.path.dirname(cache_path)
        if not os.path.exists(cache_dir_path):
            os.makedirs(cache_dir_path, exist_ok=True)
        try:
            np.savez_compressed(cache_path, 
                               quaternion=quaternions, 
                               window_id=window_id,
                               subject_id=subject_id,
                               action_id=action_id)
        except Exception as e:
            print(f"Warning: Failed to cache results: {str(e)}")
    
    return quaternions

def extract_window_data(inputs, window_idx, subject_id, action_id, filter_type, filter_params, 
                       use_cache=True, cache_dir="processed_data"):
    acc_data = inputs['accelerometer']
    gyro_data = inputs.get('gyroscope', None)
    timestamps = inputs.get('timestamps', None)
    
    if gyro_data is None or len(gyro_data) == 0:
        return {'quaternion': np.zeros((len(acc_data), 4))}
    
    window_quat = process_sequence_with_filter(
        acc_data, gyro_data, timestamps, 
        subject_id, action_id, 
        filter_type, filter_params,
        use_cache, cache_dir, window_idx
    )
    
    return {'quaternion': window_quat}

def generate_test_combinations(full_subjects, permanent_train=None, val_subjects=None, max_test_subjects=2):
    if permanent_train is None:
        permanent_train = []
    if val_subjects is None:
        val_subjects = []
        
    available_subjects = [s for s in full_subjects if s not in permanent_train and s not in val_subjects]
    all_combos = []
    
    for test_set in combinations(available_subjects, max_test_subjects):
        train_set = permanent_train + [s for s in available_subjects if s not in test_set]
        all_combos.append((list(train_set), list(test_set)))
    
    return all_combos

class Trainer:
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary, self.val_loss_summary, self.train_metrics, self.val_metrics = [], [], [], []
        self.best_f1, self.best_accuracy, self.patience_counter = 0, 0, 0
        self.val_subjects = arg.val_subjects if hasattr(arg, 'val_subjects') else [38, 46]
        self.permanent_train = arg.permanent_train if hasattr(arg, 'permanent_train') else [45, 36, 29]
        self.available_subjects = [s for s in arg.subjects if s not in self.val_subjects and s not in self.permanent_train]
        self.optimizer, self.norm_train, self.norm_val, self.norm_test = None, None, None, None
        self.data_loader, self.best_loss = dict(), float('inf')
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
        
        self.test_summary = {'test_configs': [], 'average_metrics': {}}
        self.enable_kfold = getattr(arg, 'kfold', False)
        
        if self.enable_kfold and hasattr(arg, 'dataset_args') and arg.dataset_args and 'kfold' in arg.dataset_args:
            self.kfold_config = arg.dataset_args.get('kfold', {})
        else:
            self.kfold_config = {"enabled": False}
        
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
            arg.model = 'Models.fusion_transformer.FusionTransModel'
            self.print_log(f"Using default model: {arg.model}")
        
        if not hasattr(arg, 'model_args') or not arg.model_args:
            arg.model_args = {
                'num_layers': 3,
                'embed_dim': 48,
                'acc_coords': 3,
                'quat_coords': 4,
                'num_classes': 2,
                'acc_frames': 64,
                'mocap_frames': 64,
                'num_heads': 8,
                'fusion_type': 'concat',
                'dropout': 0.3,
                'use_batch_norm': True,
                'feature_dim': 144
            }
        elif isinstance(arg.model_args, str):
            try:
                arg.model_args = eval(arg.model_args)
            except:
                try:
                    arg.model_args = yaml.safe_load(arg.model_args)
                except:
                    pass
        
        self.model = self.load_model(arg.model, arg.model_args) if self.arg.phase == 'train' else torch.load(getattr(arg, 'weights', self.model_path))
            
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
                
        self.print_log(f'# Parameters: {self.count_parameters(self.model)}')
        self.print_log(f'Sensor modalities: {self.inertial_modality}')
        self.print_log(f'Using fusion: {self.fuse} with filter type: {self.filter_type}')
        
        if self.available_gpus: 
            self.print_log(f'Using GPUs: {self.available_gpus}')
        else:
            self.print_log('Using CPU for computation')
        
        if getattr(arg, 'rotate_tests', False) or getattr(arg, 'test_combinations', False):
            self.test_train_combinations = generate_test_combinations(
                arg.subjects, 
                self.permanent_train, 
                self.val_subjects,
                max_test_subjects=2
            )
            self.print_log(f"Generated {len(self.test_train_combinations)} test/train combinations")

    def save_config(self, src_path, desc_path):
        config_file = src_path.rpartition("/")[-1]
        shutil.copy(src_path, f'{desc_path}/{config_file}')

    def count_parameters(self, model):
        total_size = 0
        for param in model.parameters(): 
            total_size += param.nelement() * param.element_size()
        for buffer in model.buffers(): 
            total_size += buffer.nelement() * buffer.element_size()
        return total_size

    def has_empty_value(self, *lists): 
        return any(len(lst) == 0 for lst in lists)

    def load_model(self, model, model_args):
        use_cuda = torch.cuda.is_available()
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        Model = import_class(model)
        
        if not model_args or not isinstance(model_args, dict):
            model_args = {
                'num_layers': 3,
                'embed_dim': 48,
                'acc_coords': 3,
                'quat_coords': 4,
                'num_classes': 2,
                'acc_frames': 64,
                'mocap_frames': 64,
                'num_heads': 8,
                'fusion_type': 'concat',
                'dropout': 0.3,
                'use_batch_norm': True,
                'feature_dim': 144
            }
            
        model = Model(**model_args).to(device)
        return model

    def load_loss(self): 
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_weights(self, path=None):
            self.model.load_state_dict(torch.load(weights_path))

    def save_weights(self, path=None, weights_only=False):
        if weights_only:
                torch.save(self.model.state_dict(), path if path else self.weights_path)
        else:
                torch.save(self.model, path if path else self.model_path)

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

    def create_df(self, columns=['test_config', 'train_subjects', 'test_subjects', 'accuracy', 'f1_score', 'precision', 'recall']): 
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
            self.print_log("Reset filter instances for new epoch")
        
        for batch_idx, (inputs, targets, idx) in enumerate(process):
            try:
                if not isinstance(inputs, dict):
                    self.print_log(f"WARNING: Inputs is not a dictionary in batch {batch_idx}")
                    continue
                
                if 'accelerometer' not in inputs or inputs['accelerometer'] is None:
                    self.print_log(f"WARNING: Missing accelerometer data in batch {batch_idx}")
                    continue
                
                if 'gyroscope' not in inputs or inputs['gyroscope'] is None:
                    self.print_log(f"WARNING: Missing gyroscope data in batch {batch_idx}")
                    continue
                
                # Extract tensors from inputs
                acc_data = inputs['accelerometer']
                gyro_data = inputs['gyroscope']
                
                # Validate data
                if torch.any(torch.isnan(acc_data)) or torch.any(torch.isnan(gyro_data)):
                    self.print_log(f"WARNING: NaN values in sensor data in batch {batch_idx}")
                    continue
                
                # Ensure proper dimensions for channel-last format [batch, seq, channels]
                if len(acc_data.shape) == 2:
                    batch_size = acc_data.shape[0]
                    acc_data = acc_data.view(batch_size, -1, 3)
                elif acc_data.shape[1] == 3 and len(acc_data.shape) == 3:
                    # If in channel-first format [batch, channels, seq], transpose
                    acc_data = acc_data.transpose(1, 2)
                
                if len(gyro_data.shape) == 2:
                    batch_size = gyro_data.shape[0]
                    gyro_data = gyro_data.view(batch_size, -1, 3)
                elif gyro_data.shape[1] == 3 and len(gyro_data.shape) == 3:
                    # If in channel-first format [batch, channels, seq], transpose
                    gyro_data = gyro_data.transpose(1, 2)
                
                # Move tensors to device
                acc_data = acc_data.to(device).float()
                gyro_data = gyro_data.to(device).float()
                targets = targets.to(device)
                print("Targets: ",targets)
                print("Acc :", acc_data, gyro_data)
                timer['dataloader'] += self.split_time()
                self.optimizer.zero_grad()
                
                # Forward pass with the module directly to avoid DataParallel wrapper issues
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
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                try:
                    # Validate inputs
                    if not isinstance(inputs, dict):
                        self.print_log(f"WARNING: Inputs is not a dictionary in {mode} batch {batch_idx}")
                        continue
                    
                    if 'accelerometer' not in inputs or inputs['accelerometer'] is None:
                        self.print_log(f"WARNING: Missing accelerometer data in {mode} batch {batch_idx}")
                        continue
                    
                    if 'gyroscope' not in inputs or inputs['gyroscope'] is None:
                        self.print_log(f"WARNING: Missing gyroscope data in {mode} batch {batch_idx}")
                        continue
                    
                    # Extract tensors from inputs
                    acc_data = inputs['accelerometer']
                    gyro_data = inputs['gyroscope']
                    
                    # Validate data
                    if torch.any(torch.isnan(acc_data)) or torch.any(torch.isnan(gyro_data)):
                        self.print_log(f"WARNING: NaN values in sensor data in {mode} batch {batch_idx}")
                        continue
                    
                    # Ensure proper dimensions for channel-last format [batch, seq, channels]
                    if len(acc_data.shape) == 2:
                        batch_size = acc_data.shape[0]
                        acc_data = acc_data.view(batch_size, -1, 3)
                    elif acc_data.shape[1] == 3 and len(acc_data.shape) == 3:
                        # If in channel-first format [batch, channels, seq], transpose
                        acc_data = acc_data.transpose(1, 2)
                    
                    if len(gyro_data.shape) == 2:
                        batch_size = gyro_data.shape[0]
                        gyro_data = gyro_data.view(batch_size, -1, 3)
                    elif gyro_data.shape[1] == 3 and len(gyro_data.shape) == 3:
                        # If in channel-first format [batch, channels, seq], transpose
                        gyro_data = gyro_data.transpose(1, 2)
                    
                    # Move tensors to device
                    acc_data = acc_data.to(device).float()
                    gyro_data = gyro_data.to(device).float()
                    targets = targets.to(device)
                    
                    # Forward pass with the module directly to avoid DataParallel wrapper issues
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

    def save_test_summary(self):
        if not self.test_summary['test_configs']:
            return
            
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        avg_metrics = {}
        
        for metric in metrics:
            values = [config['metrics'][metric] for config in self.test_summary['test_configs']]
            avg_metrics[metric] = float(np.mean(values))
            avg_metrics[f"{metric}_std"] = float(np.std(values))
            
        self.test_summary['average_metrics'] = avg_metrics
        self.test_summary['filter_type'] = self.filter_type
        
        summary_path = os.path.join(self.arg.work_dir, 'test_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self.test_summary, f, indent=2)
        self.print_log(f"Test summary saved to {summary_path}")
    
    def run_fold(self, train_subjects, test_subjects, fold_dir):
        self.train_loss_summary, self.val_loss_summary, self.train_metrics, self.val_metrics = [], [], [], []
        self.best_accuracy, self.best_f1, self.best_loss, self.patience_counter = 0, 0, float('inf'), 0
        
        self.print_log(f"Creating new model instance for fold")
        model_args = self.arg.model_args
        
        if not self.load_data(train_subjects, test_subjects):
            self.print_log("ERROR: Failed to load data for fold")
            return None
        
        self.model = self.load_model(self.arg.model, model_args)
        self.model = self.load_model(arg.model, arg.model_args) if self.arg.phase == 'train' else torch.load(getattr(arg, 'weights', self.model_path))   
        self.load_optimizer()
        
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
        
        with open(os.path.join(fold_dir, 'validation_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        for file in ['confusion_matrix_test.png', 'confusion_matrix_val.png']:
            if os.path.exists(os.path.join(self.arg.work_dir, file)):
                shutil.copy(os.path.join(self.arg.work_dir, file), 
                          os.path.join(fold_dir, file))
        
        return test_metrics
    
    def start(self):
        try:
            self.train_loss_summary, self.val_loss_summary, self.train_metrics, self.val_metrics = [], [], [], []
            self.best_accuracy, self.best_f1, self.best_loss, self.patience_counter = 0, 0, float('inf'), 0
            self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
            
            global FILTER_INSTANCES
            FILTER_INSTANCES = {}
            
            if self.arg.rotate_tests or self.arg.test_combinations:
                test_results_df = self.create_df()
                
                for config_idx, (train_subjects, test_subjects) in enumerate(self.test_train_combinations):
                    self.print_log(f"\n{'='*20} Test Configuration {config_idx+1}/{len(self.test_train_combinations)} {'='*20}")
                    self.print_log(f"Train: {train_subjects}")
                    self.print_log(f"Test: {test_subjects}")
                    
                    config_dir = os.path.join(self.arg.work_dir, f"test_config_{config_idx+1}")
                    os.makedirs(config_dir, exist_ok=True)
                    
                    FILTER_INSTANCES = {}
                    
                    test_metrics = self.run_fold(train_subjects, test_subjects, config_dir)
                    
                    if test_metrics:
                        result_row = {
                            'test_config': config_idx + 1,
                            'train_subjects': str(train_subjects),
                            'test_subjects': str(test_subjects),
                            'accuracy': round(test_metrics['accuracy'], 2),
                            'f1_score': round(test_metrics['f1'], 2),
                            'precision': round(test_metrics['precision'], 2),
                            'recall': round(test_metrics['recall'], 2)
                        }
                        
                        test_results_df.loc[len(test_results_df)] = result_row
                        
                        self.test_summary['test_configs'].append({
                            'config_id': config_idx + 1,
                            'train_subjects': train_subjects,
                            'test_subjects': test_subjects,
                            'metrics': test_metrics
                        })
                    
                    if not self.arg.test_combinations:
                        break
                
                self.save_test_summary()
                test_results_df.to_csv(os.path.join(self.arg.work_dir, 'test_results.csv'), index=False)
                
                avg_metrics = self.test_summary.get('average_metrics', {})
                if avg_metrics:
                    self.print_log(f'\n===== Test Results Summary ({self.filter_type}) =====')
                    self.print_log(f'Mean accuracy: {avg_metrics.get("accuracy", 0):.2f}% ± {avg_metrics.get("accuracy_std", 0):.2f}%')
                    self.print_log(f'Mean F1 score: {avg_metrics.get("f1", 0):.2f} ± {avg_metrics.get("f1_std", 0):.2f}')
                    self.print_log(f'Mean precision: {avg_metrics.get("precision", 0):.2f}% ± {avg_metrics.get("precision_std", 0):.2f}%')
                    self.print_log(f'Mean recall: {avg_metrics.get("recall", 0):.2f}% ± {avg_metrics.get("recall_std", 0):.2f}%')
                    self.print_log(f'Mean balanced accuracy: {avg_metrics.get("balanced_accuracy", 0):.2f}% ± {avg_metrics.get("balanced_accuracy_std", 0):.2f}%')
            
            elif self.enable_kfold and self.kfold_config and self.kfold_config.get('enabled', False):
                self.print_log(f"\n===== Running {self.kfold_config.get('num_folds', 5)}-fold cross validation =====")
                fold_assignments = self.kfold_config.get('fold_assignments', [])
                
                if not fold_assignments:
                    self.print_log("ERROR: No fold assignments configured")
                    return
                
                results = []
                for fold_idx, test_subjects in enumerate(fold_assignments):
                    self.print_log(f"\n{'='*20} Fold {fold_idx+1}/{len(fold_assignments)} {'='*20}")
                    
                    FILTER_INSTANCES = {}
                    
                    train_subjects = self.permanent_train.copy()
                    for s in self.arg.subjects:
                        if s not in test_subjects and s not in self.val_subjects and s not in train_subjects:
                            train_subjects.append(s)
                    
                    self.print_log(f"Train: {train_subjects}")
                    self.print_log(f"Test: {test_subjects}")
                    self.print_log(f"Validation: {self.val_subjects}")
                    
                    fold_dir = os.path.join(self.arg.work_dir, f"fold_{fold_idx+1}")
                    os.makedirs(fold_dir, exist_ok=True)
                    
                    fold_metrics = self.run_fold(train_subjects, test_subjects, fold_dir)
                    
                    if fold_metrics:
                        self.test_summary['test_configs'].append({
                            'fold_id': fold_idx + 1,
                            'train_subjects': train_subjects,
                            'test_subjects': test_subjects,
                            'metrics': fold_metrics
                        })
                        results.append(fold_metrics)
                
                if results:
                    self.save_test_summary()
                    self.print_log(f"\n===== Cross-validation Results Summary ({self.filter_type}) =====")
                    avg_metrics = self.test_summary.get('average_metrics', {})
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
                
                self.test_summary['test_configs'].append({
                    'config_id': 1,
                    'train_subjects': train_subjects,
                    'test_subjects': self.val_subjects,
                    'metrics': final_metrics
                })
                
                self.test_summary['average_metrics'] = {
                    'accuracy': final_metrics['accuracy'],
                    'f1': final_metrics['f1'],
                    'precision': final_metrics['precision'],
                    'recall': final_metrics['recall'],
                    'balanced_accuracy': final_metrics['balanced_accuracy']
                }
                
                self.save_test_summary()
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
