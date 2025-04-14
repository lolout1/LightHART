from collections import defaultdict, Counter
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.preprocessing import StandardScaler
import torch
from utils.accelerometer_processor import AccelerometerProcessor

def csvloader(file_path, **kwargs):
    import pandas as pd
    file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
    if 'skeleton' in file_path: 
        cols = 96
    else: 
        cols = 3
    activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
    return activity_data

def matloader(file_path, **kwargs):
    from scipy.io import loadmat
    key = kwargs.get('key', None)
    assert key in ['d_iner', 'd_skel'], f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', **kwargs):
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.processor = AccelerometerProcessor(target_freq=50, window_size=max_length)
        
    def load_file(self, file_path):
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        return data
    
    def _import_loader(self, file_path):
        file_type = file_path.split('.')[-1]
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        return LOADER_MAP[file_type]
    
    def process_accelerometer_data(self, file_path, label):
        acc_data = self.load_file(file_path)
        acc_data = self._butterworth_filter(acc_data, cutoff=7.5, fs=30)
        
        # For selecting subwindow with highest variance
        if acc_data.shape[0] > 300:
            acc_data = self._select_subwindow(acc_data)
            
        windows = self.processor.segment_windows(acc_data)
        enhanced_windows = self.processor.enhance_features(windows)
        labels = np.full(len(enhanced_windows), label)
        
        return enhanced_windows, labels
    
    def _butterworth_filter(self, data, cutoff=7.5, fs=30, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)
    
    def _select_subwindow(self, data):
        magnitude = np.linalg.norm(data, axis=1)
        df = pd.DataFrame({"values": magnitude})
        df["variance"] = df["values"].rolling(window=125).var()
        max_idx = df["variance"].idxmax()
        final_start = max(0, max_idx-100)
        final_end = min(len(data), max_idx + 100)
        return data[final_start:final_end, :]
    
    def make_dataset(self, subjects, fuse):
        self.data = defaultdict(list)
        self.fuse = fuse
        
        acc_data_all = []
        labels_all = []
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)  # Binary fall detection
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                
                if 'accelerometer' in trial.files:
                    try:
                        windows, window_labels = self.process_accelerometer_data(
                            trial.files['accelerometer'], label
                        )
                        acc_data_all.append(windows)
                        labels_all.append(window_labels)
                    except Exception as e:
                        print(f"Error processing {trial.files['accelerometer']}: {e}")
        
        if acc_data_all:
            self.data['accelerometer'] = np.concatenate(acc_data_all, axis=0)
            self.data['labels'] = np.concatenate(labels_all, axis=0)
            
            # Create dummy skeleton data for compatibility
            dummy_skl = np.zeros((self.data['accelerometer'].shape[0], 
                                  self.data['accelerometer'].shape[1], 32, 3))
            self.data['skeleton'] = dummy_skl
        
        return self.data
    
    def normalization(self):
        for key, value in self.data.items():
            if key != 'labels':
                num_samples, length = value.shape[:2]
                norm_data = StandardScaler().fit_transform(value.reshape(num_samples*length, -1))
                self.data[key] = norm_data.reshape(num_samples, length, -1)
        return self.data
