import traceback
import random
import sys
import os
import time
import datetime
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from utils.dataset import prepare_smartfallmm, split_by_subjects
from utils.callbacks import EarlyStopping
from collections import Counter
from copy import deepcopy
import argparse
import yaml
import logging
from scipy.signal import find_peaks, butter, sosfilt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FallDetection')

def get_args():
    parser = argparse.ArgumentParser(description='Fall Detection')
    parser.add_argument('--config', default='./config/smartfallmm/student.yaml')
    parser.add_argument('--dataset', type=str, default='utd')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=8)
    parser.add_argument('--num-epoch', type=int, default=70)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--model', default=None)
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--model-args', default=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--model-saved-name', type=str, default='test')
    parser.add_argument('--loss', default='loss.BCE')
    parser.add_argument('--loss-args', default="{}", type=str)
    parser.add_argument('--dataset-args', default=str)
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--feeder', default=None)
    parser.add_argument('--train-feeder-args', default=str)
    parser.add_argument('--val-feeder-args', default=str)
    parser.add_argument('--test_feeder_args', default=str)
    parser.add_argument('--include-val', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--work-dir', type=str, default='simple')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str)
    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

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
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

THRESHOLD = 0.5

class CustomEarlyStop:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_f1 = 0
        self.early_stop = False
    
    def __call__(self, loss, metrics=None):
        f1 = metrics[1] if metrics else 0
        
        if self.best_loss is None:
            self.best_loss = loss
            if metrics:
                self.best_f1 = f1
            return
        
        if metrics and f1 > self.best_f1 + self.min_delta:
            self.best_f1 = f1
            self.counter = 0
            return
            
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class SignalProcessor:
    def __init__(self, window_size=128, stride=64, target_freq=25, butter_cutoff=7.5):
        self.window_size = window_size
        self.stride = stride
        self.target_freq = target_freq
        self.butter_cutoff = butter_cutoff
        self.sos = self._get_butterworth_coeffs()
    
    def _get_butterworth_coeffs(self):
        nyquist = 0.5 * self.target_freq
        normal_cutoff = self.butter_cutoff / nyquist
        return butter(4, normal_cutoff, btype='low', analog=False, output='sos')
    
    def normalize_signal(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-8] = 1.0
        
        return (data - mean) / std
    
    def filter_signal(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = sosfilt(self.sos, data[:, i])
        
        return filtered_data
    
    def detect_peaks(self, data, height=1.5, distance=50):
        if data.ndim > 1:
            magnitudes = np.sqrt(np.sum(data**2, axis=1))
        else:
            magnitudes = np.abs(data)
            
        peaks, _ = find_peaks(magnitudes, height=height, distance=distance)
        return peaks
    
    def extract_windows_around_peaks(self, data, is_fall=False):
        if not is_fall:
            return self.sliding_window(data)
        
        peaks = self.detect_peaks(data, height=1.5, distance=self.window_size//2)
        
        if len(peaks) == 0:
            return self.sliding_window(data)
        
        windows = []
        for peak in peaks:
            start = max(0, peak - self.window_size//2)
            end = start + self.window_size
            
            if end > len(data):
                start = max(0, len(data) - self.window_size)
                end = len(data)
            
            if end - start == self.window_size:
                windows.append(data[start:end])
        
        if not windows:
            return self.sliding_window(data)
            
        return np.array(windows)
    
    def sliding_window(self, data):
        if len(data) < self.window_size:
            padding = np.zeros((self.window_size - len(data), data.shape[1]))
            padded_data = np.vstack([data, padding])
            return np.array([padded_data])
        
        windows = []
        for start in range(0, len(data) - self.window_size + 1, self.stride):
            windows.append(data[start:start + self.window_size])
            
        return np.array(windows)
    
    def add_magnitude_feature(self, data):
        windows = []
        for window in data:
            magnitude = np.sqrt(np.sum(window[:, :3]**2, axis=1)).reshape(-1, 1)
            enhanced = np.concatenate([window, magnitude], axis=1)
            windows.append(enhanced)
        return np.array(windows)
    
    def process_all(self, data, is_fall=False):
        filtered_data = self.filter_signal(data)
        normalized_data = self.normalize_signal(filtered_data)
        
        windows = self.extract_windows_around_peaks(normalized_data, is_fall)
        enhanced_windows = self.add_magnitude_feature(windows)
        
        return enhanced_windows

class Trainer():
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.train_f1_summary = []
        self.val_f1_summary = []
        self.best_loss = float('inf')
        self.best_f1 = 0
        self.test_accuracy = 0
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_auc = 0
        self.train_subjects = []
        self.val_subject = [38, 46]
        self.test_subject = None
        self.optimizer = None
        self.scheduler = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = dict()
        self.early_stop = CustomEarlyStop(patience=25, min_delta=0.0005)
        self.inertial_modality = [modality for modality in arg.dataset_args['modalities'] if modality != 'skeleton']
        self.fuse = len(self.inertial_modality) > 1
        self.fixed_train_subjects = [45, 36, 29]
        self.all_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44] + self.fixed_train_subjects
        self.test_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
        self.signal_processor = SignalProcessor(window_size=128, stride=64)
        
        if os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.arg.work_dir, exist_ok=True)
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}'
        self.save_config(self.arg.config, self.arg.work_dir)
        
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            self.load_pretrained_model()
        
        self.include_val = arg.include_val
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size: {num_params / (1024 ** 2):.2f} MB')
        
    def load_pretrained_model(self):
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        if use_cuda and torch.cuda.device_count() > self.output_device:
            self.model = torch.load(self.arg.weights)
            self.device = f'cuda:{self.output_device}'
        else:
            self.model = torch.load(self.arg.weights, map_location='cpu')
            self.device = 'cpu'

    def add_avg_df(self, results):
        try:
            numeric_cols = results.select_dtypes(include=[np.number]).columns
            averages = {col: results[col].mean() for col in numeric_cols}
            averages['test_subject'] = 'Average'
            for col in numeric_cols:
                averages[col] = round(averages[col], 2)
            avg_row = pd.DataFrame([averages])
            return pd.concat([results, avg_row], ignore_index=True)
        except:
            return results

    def save_config(self, src_path, desc_path):
        shutil.copy(src_path, f'{desc_path}/{src_path.rpartition("/")[-1]}')

    def cal_weights(self):
        try:
            label_count = Counter(self.norm_train['labels'])
            if 0 in label_count and 1 in label_count and label_count[1] > 0:
                weight_ratio = min(5.0, max(1.0, label_count[0] / max(1, label_count[1])))
                self.pos_weights = torch.Tensor([weight_ratio])
            else:
                self.pos_weights = torch.Tensor([1.0])
            self.pos_weights = self.pos_weights.to(self.device)
        except:
            self.pos_weights = torch.Tensor([1.0]).to(self.device)

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
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        if use_cuda and self.output_device < torch.cuda.device_count():
            device = f'cuda:{self.output_device}'
            self.device = device
        else:
            device = 'cpu'
            self.device = device
            
        Model = import_class(model)
        model = Model(**model_args).to(device)
        return model

    def load_loss(self):
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

    def load_weights(self, model_path):
        try:
            device_str = self.device
            model_dict = torch.load(model_path, map_location=device_str)
            self.model.load_state_dict(model_dict)
        except:
            logger.error(f"Error loading weights from {model_path}")

    def load_optimizer(self, parameters):
        if self.arg.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(parameters, lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(parameters, lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(parameters, lr=self.arg.base_lr, momentum=0.9, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")
            
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-6
        )

    def distribution_viz(self, labels, work_dir, mode):
        try:
            values, count = np.unique(labels, return_counts=True)
            plt.figure(figsize=(10, 6))
            plt.bar(x=values, height=count)
            plt.xlabel('Labels')
            plt.ylabel('Count')
            plt.title(f'{mode.capitalize()} Label Distribution')
            plt.savefig(os.path.join(work_dir, f'{mode}_Label_Distribution.png'))
            plt.close()
        except:
            pass

    def process_dataset(self, data, labels):
        try:
            acc_data = data.get('accelerometer', None)
            if acc_data is None:
                return data
                
            fall_indices = np.where(labels == 1)[0]
            non_fall_indices = np.where(labels == 0)[0]
            
            processed_windows = []
            processed_labels = []
            
            # Process fall data with peak detection
            for idx in fall_indices:
                windows = self.signal_processor.process_all(acc_data[idx], is_fall=True)
                processed_windows.extend(windows)
                processed_labels.extend([1] * len(windows))
            
            # Process non-fall data with sliding windows
            for idx in non_fall_indices:
                windows = self.signal_processor.process_all(acc_data[idx], is_fall=False)
                processed_windows.extend(windows)
                processed_labels.extend([0] * len(windows))
            
            if len(processed_windows) == 0:
                return data
                
            processed_windows = np.array(processed_windows)
            processed_labels = np.array(processed_labels)
            
            # Verify all windows have the same shape
            expected_shape = (128, 4)
            valid_indices = []
            for i, window in enumerate(processed_windows):
                if window.shape == expected_shape:
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                return data
                
            processed_windows = processed_windows[valid_indices]
            processed_labels = processed_labels[valid_indices]
            
            # Create dummy skeleton data to match the accelerometer data
            dummy_skl = np.zeros((len(processed_windows), 128, 32, 3))
            
            return {
                'accelerometer': processed_windows,
                'labels': processed_labels,
                'skeleton': dummy_skl
            }
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            return data

    def load_data(self):
        try:
            Feeder = import_class(self.arg.feeder)
            builder = prepare_smartfallmm(self.arg)
            
            if self.arg.phase == 'train':
                raw_train = split_by_subjects(builder, self.train_subjects, self.fuse)
                raw_val = split_by_subjects(builder, self.val_subject, self.fuse)
                
                self.norm_train = self.process_dataset(raw_train, raw_train['labels'])
                self.norm_val = self.process_dataset(raw_val, raw_val['labels'])
                
                if self.has_empty_value(list(self.norm_val.values())):
                    logger.error("Empty validation data")
                    return False

                self.data_loader['train'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.train_feeder_args, dataset=self.norm_train),
                    batch_size=self.arg.batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker)
                
                self.cal_weights()
                self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, f'train_{self.test_subject[0]}')
                
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.val_feeder_args, dataset=self.norm_val),
                    batch_size=self.arg.batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker)
            
            raw_test = split_by_subjects(builder, self.test_subject, self.fuse)
            self.norm_test = self.process_dataset(raw_test, raw_test['labels'])
                                             
            if self.has_empty_value(list(self.norm_test.values())):
                logger.error("Empty test data")
                return False
                
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset=self.norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker)
                
            self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, f'test_{self.test_subject[0]}')

            train_counts = np.bincount(self.norm_train['labels'].astype(int))
            val_counts = np.bincount(self.norm_val['labels'].astype(int))
            test_counts = np.bincount(self.norm_test['labels'].astype(int))
            
            self.print_log(f"Data loaded - Train: {len(self.norm_train['labels'])} samples")
            self.print_log(f"Val: {len(self.norm_val['labels'])} samples")
            self.print_log(f"Test: {len(self.norm_test['labels'])} samples")
            
            self.print_log(f"Train class distribution - Non-falls: {train_counts[0]}, Falls: {train_counts[1] if len(train_counts) > 1 else 0}")
            self.print_log(f"Val class distribution - Non-falls: {val_counts[0]}, Falls: {val_counts[1] if len(val_counts) > 1 else 0}")
            self.print_log(f"Test class distribution - Non-falls: {test_counts[0]}, Falls: {test_counts[1] if len(test_counts) > 1 else 0}")
                
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            traceback.print_exc()
            return False

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string, print_time=True):
        logger.info(string)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(string, file=f)

    def loss_viz(self, train_loss, val_loss):
        try:
            epochs = range(len(train_loss))
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_loss, 'b', label="Training Loss")
            plt.plot(epochs, val_loss, 'r', label="Validation Loss")
            plt.title(f'Train Vs Val Loss for {self.test_subject[0]}')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(self.arg.work_dir, f'trainvsval_{self.test_subject[0]}.png'))
            plt.close()
        except:
            pass

    def metrics_viz(self, train_metrics, val_metrics, metric_name):
        try:
            epochs = range(len(train_metrics))
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_metrics, 'b', label=f"Training {metric_name}")
            plt.plot(epochs, val_metrics, 'r', label=f"Validation {metric_name}")
            plt.title(f'Train Vs Val {metric_name} for {self.test_subject[0]}')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel(metric_name)
            plt.savefig(os.path.join(self.arg.work_dir, f'{metric_name.lower()}_{self.test_subject[0]}.png'))
            plt.close()
        except:
            pass

    def cm_viz(self, y_pred, y_true, subject_id):
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar()
            plt.xticks([0, 1], ['Non-Fall', 'Fall'])
            plt.yticks([0, 1], ['Non-Fall', 'Fall'])
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.title(f"Confusion Matrix - Subject {subject_id}")
            
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.arg.work_dir, f'confusion_matrix_{subject_id}.png'))
            plt.close()
        except:
            pass

    def create_df(self, columns=['test_subject', 'accuracy', 'f1_score', 'precision', 'recall', 'auc']):
        return pd.DataFrame(columns=columns)

    def cal_prediction(self, logits):
        return (torch.sigmoid(logits) > THRESHOLD).int().squeeze(1)

    def cal_metrics(self, targets, predictions):
        try:
            targets = np.array(targets)
            predictions = np.array(predictions)
            
            if len(targets) == 0 or len(predictions) == 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0
                
            if len(np.unique(targets)) < 2 or len(np.unique(predictions)) < 2:
                accuracy = 100.0 if np.all(targets == predictions) else 0.0
                return accuracy, 0.0, 0.0, 0.0, 0.0
                
            f1 = f1_score(targets, predictions)
            precision = precision_score(targets, predictions, zero_division=0)
            recall = recall_score(targets, predictions, zero_division=0)
            accuracy = accuracy_score(targets, predictions)
            
            try:
                auc_score = roc_auc_score(targets, predictions)
            except:
                auc_score = 0.0
                
            return accuracy * 100, f1 * 100, recall * 100, precision * 100, auc_score * 100
        except:
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def train(self, epoch):
        self.model.train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader=0.001, model=0.001, stats=0.001)
        label_list = []
        pred_list = []
        cnt = 0
        train_loss = 0

        process = tqdm(loader, ncols=80)

        for batch_idx, (inputs, targets, idx) in enumerate(process):
            try:
                with torch.no_grad():
                    acc_data = inputs['accelerometer'].to(self.device)
                    skl_data = inputs['skeleton'].to(self.device)
                    targets = targets.to(self.device)

                timer['dataloader'] += self.split_time()

                self.optimizer.zero_grad()
                logits, _ = self.model(acc_data.float(), skl_data.float())
                loss = self.criterion(logits.squeeze(1), targets.float())
                loss.backward()
                self.optimizer.step()

                timer['model'] += self.split_time()
                with torch.no_grad():
                    train_loss += loss.mean().item()
                    preds = self.cal_prediction(logits)
                    label_list.extend(targets.tolist())
                    pred_list.extend(preds.tolist())

                cnt += len(targets)
                timer['stats'] += self.split_time()
            except Exception as e:
                logger.error(f"Error in training step: {str(e)}")
                continue

        train_loss /= cnt if cnt > 0 else 1
        accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)

        self.train_loss_summary.append(train_loss)
        self.train_f1_summary.append(f1)
        
        proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values())))) for k, v in timer.items()}
        
        self.print_log(
            f'Epoch {epoch}: Training Loss: {train_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%')
        self.print_log(f'Time consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')
        
        val_loss, val_metrics = self.eval(epoch, loader_name='val')
        self.val_loss_summary.append(val_loss)
        self.val_f1_summary.append(val_metrics[1])
        
        self.scheduler.step(val_metrics[1])
        self.early_stop(val_loss, val_metrics)
        
        if val_metrics[1] > self.best_f1:
            self.best_f1 = val_metrics[1]
            self.best_loss = val_loss
            try:
                torch.save(deepcopy(self.model.state_dict()), f'{self.model_path}_{self.test_subject[0]}.pth')
                self.print_log(f'Weights Saved: F1 improved to {self.best_f1:.2f}%')
            except:
                pass

    def eval(self, epoch, loader_name='val', result_file=None):
        if result_file:
            f_r = open(result_file, 'w', encoding='utf-8')
        self.model.eval()

        self.print_log(f'Eval epoch: {epoch}')

        loss = 0
        cnt = 0
        label_list = []
        pred_list = []
        prob_list = []

        process = tqdm(self.data_loader[loader_name], ncols=80)
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                try:
                    acc_data = inputs['accelerometer'].to(self.device)
                    skl_data = inputs['skeleton'].to(self.device)
                    targets = targets.to(self.device)

                    logits, _ = self.model(acc_data.float(), skl_data.float())
                    batch_loss = self.criterion(logits.squeeze(1), targets.float())
                    loss += batch_loss.sum().item()
                    
                    probs = torch.sigmoid(logits.squeeze(1))
                    preds = (probs > THRESHOLD).int()
                    
                    label_list.extend(targets.tolist())
                    pred_list.extend(preds.tolist())
                    prob_list.extend(probs.tolist())
                    cnt += len(targets)
                except Exception as e:
                    logger.error(f"Error in evaluation step: {str(e)}")
                    continue
                
            loss /= cnt if cnt > 0 else 1
            accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)

        if result_file:
            for i, (pred, true, prob) in enumerate(zip(pred_list, label_list, prob_list)):
                f_r.write(f"{pred} => {true} (confidence: {prob:.4f})\n")
            f_r.close()
                
        self.print_log(f'{loader_name.capitalize()} Loss: {loss:.4f}, ' +
                      f'Acc: {accuracy:.2f}%, F1: {f1:.2f}%, ' +
                      f'Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%')
            
        if loader_name == 'test':
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_recall = recall
            self.test_precision = precision
            self.test_auc = auc_score
            
            self.cm_viz(pred_list, label_list, self.test_subject[0])
            
        return loss, (accuracy, f1, recall, precision, auc_score)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters: \n{}\n'.format(str(vars(self.arg))))
            results = self.create_df()
            
            for test_subject in self.test_subjects:
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.train_f1_summary = []
                self.val_f1_summary = []
                self.best_loss = float('inf')
                self.best_f1 = 0
                
                self.test_subject = [test_subject]
                self.train_subjects = self.fixed_train_subjects + [s for s in self.test_subjects if s != test_subject]
                
                self.print_log(f"===== Starting fold with test subject {test_subject} =====")
                self.print_log(f"Train subjects: {self.train_subjects}")
                self.print_log(f"Val subjects: {self.val_subject}")
                
                self.model = self.load_model(self.arg.model, self.arg.model_args)
                self.print_log(f'Model Parameters: {self.count_parameters(self.model)}')
                
                if not self.load_data():
                    self.print_log(f"Skipping subject {test_subject} due to data loading issues")
                    continue

                self.load_optimizer(self.model.parameters())
                self.load_loss()
                
                self.early_stop = CustomEarlyStop(patience=25, min_delta=0.0005)
                
                try:
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        self.train(epoch)
                        if self.early_stop.early_stop:
                            self.print_log(f"Early stopping at epoch {epoch}")
                            break
                except Exception as e:
                    logger.error(f"Error during training: {str(e)}")
                    continue
                        
                try:
                    model_path = f'{self.model_path}_{self.test_subject[0]}.pth'
                    if os.path.exists(model_path):
                        self.model = self.load_model(self.arg.model, self.arg.model_args)
                        self.load_weights(model_path)
                        self.model.eval()
                        self.print_log(f'===== Testing Subject {self.test_subject[0]} =====')
                        self.eval(epoch=0, loader_name='test', 
                                 result_file=f'{self.arg.work_dir}/predictions_{self.test_subject[0]}.txt')
                        
                        self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                        self.metrics_viz(self.train_f1_summary, self.val_f1_summary, 'F1')
                        
                        subject_result = pd.DataFrame([{
                            'test_subject': str(self.test_subject[0]),
                            'accuracy': round(self.test_accuracy, 2),
                            'f1_score': round(self.test_f1, 2),
                            'precision': round(self.test_precision, 2),
                            'recall': round(self.test_recall, 2),
                            'auc': round(self.test_auc, 2)
                        }])
                        results = pd.concat([results, subject_result], ignore_index=True)
                    else:
                        self.print_log(f"No weights found for subject {self.test_subject[0]}, skipping evaluation")
                except Exception as e:
                    logger.error(f"Error during evaluation: {str(e)}")
                    continue
                
            try:
                if len(results) > 0:
                    results = self.add_avg_df(results)
                    results.to_csv(f'{self.arg.work_dir}/overall_scores.csv', index=False)
                    self.print_log("\nOverall Results:")
                    self.print_log(results.to_string())
                else:
                    self.print_log("No results to save")
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
        else:
            self.test_subject = self.arg.subjects[:1]
            if self.load_data():
                self.print_log(f"Testing model on subject {self.test_subject[0]}")
                self.eval(epoch=0, loader_name='test', 
                         result_file=f'{self.arg.work_dir}/predictions_{self.test_subject[0]}.txt')
                
                results = pd.DataFrame([{
                    'test_subject': str(self.test_subject[0]),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2)
                }])
                results.to_csv(f'{self.arg.work_dir}/test_results.csv', index=False)

if __name__ == "__main__":
    parser = get_args()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WRONG ARG: {k}')
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    init_seed(arg.seed)
    try:
        trainer = Trainer(arg)
        trainer.start()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        traceback.print_exc()
