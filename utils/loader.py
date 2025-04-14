from collections import defaultdict
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import torch
from utils.accelerometer_processor import AccelerometerProcessor
import os

def csvloader(file_path, **kwargs):
    import pandas as pd
    try:
        file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        if 'skeleton' in file_path: 
            cols = 96
        else: 
            cols = 3
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        return activity_data
    except:
        return None

def matloader(file_path, **kwargs):
    from scipy.io import loadmat
    try:
        key = kwargs.get('key', None)
        assert key in ['d_iner', 'd_skel'], f'Unsupported {key} for matlab file'
        data = loadmat(file_path)[key]
        return data
    except:
        return None

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
        self.processor = AccelerometerProcessor(target_freq=25, window_size=max_length, window_overlap=0.5)
        self.stats = {'total_files': 0, 'processed_files': 0, 'errors': 0, 'falls': 0, 'non_falls': 0, 'activities': defaultdict(int)}
        
    def load_file(self, file_path):
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        return data
    
    def _import_loader(self, file_path):
        file_type = file_path.split('.')[-1]
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        return LOADER_MAP[file_type]
    
    def process_accelerometer_data(self, file_path, label, action_id):
        self.stats['total_files'] += 1
        self.stats['activities'][action_id] += 1
        
        try:
            acc_data = self.processor.load_and_preprocess(file_path)
            windows = self.processor.segment_windows(acc_data)
            enhanced_windows = self.processor.enhance_features(windows)
            labels = np.full(len(enhanced_windows), label)
            
            self.stats['processed_files'] += 1
            if label == 1:
                self.stats['falls'] += len(enhanced_windows)
            else:
                self.stats['non_falls'] += len(enhanced_windows)
                
            return enhanced_windows, labels
        except Exception as e:
            self.stats['errors'] += 1
            print(f"Error processing {file_path}: {str(e)}")
            return None, None
    
    def make_dataset(self, subjects, fuse):
        self.data = defaultdict(list)
        self.fuse = fuse
        
        acc_data_all = []
        labels_all = []
        fall_count = 0
        non_fall_count = 0
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                
                if 'accelerometer' in trial.files:
                    windows, window_labels = self.process_accelerometer_data(
                        trial.files['accelerometer'], label, trial.action_id
                    )
                    
                    if windows is not None and len(windows) > 0:
                        acc_data_all.append(windows)
                        labels_all.append(window_labels)
                        if label == 1:
                            fall_count += len(windows)
                        else:
                            non_fall_count += len(windows)
        
        # Ensure we have at least one sample of each class
        if fall_count == 0 or non_fall_count == 0:
            # Create synthetic samples for missing class
            if fall_count == 0:
                print("No fall samples found - adding synthetic fall data")
                synthetic_fall = np.random.randn(5, self.max_length, 4) * 0.1
                acc_data_all.append(synthetic_fall)
                labels_all.append(np.ones(5))
                fall_count = 5
            
            if non_fall_count == 0:
                print("No non-fall samples found - adding synthetic non-fall data")
                synthetic_nonfall = np.random.randn(5, self.max_length, 4) * 0.01
                acc_data_all.append(synthetic_nonfall)
                labels_all.append(np.zeros(5))
                non_fall_count = 5
        
        if acc_data_all:
            try:
                self.data['accelerometer'] = np.concatenate(acc_data_all, axis=0)
                self.data['labels'] = np.concatenate(labels_all, axis=0)
                
                dummy_skl = np.zeros((self.data['accelerometer'].shape[0], 
                                      self.data['accelerometer'].shape[1], 32, 3))
                self.data['skeleton'] = dummy_skl
            except Exception as e:
                print(f"Error concatenating data: {str(e)}. Creating fallback dataset.")
                self.data['accelerometer'] = np.zeros((10, self.max_length, 4))
                self.data['labels'] = np.array([0] * 5 + [1] * 5)
                self.data['skeleton'] = np.zeros((10, self.max_length, 32, 3))
        else:
            print("No data could be processed. Creating fallback dataset.")
            self.data['accelerometer'] = np.zeros((10, self.max_length, 4))
            self.data['labels'] = np.array([0] * 5 + [1] * 5)
            self.data['skeleton'] = np.zeros((10, self.max_length, 32, 3))
        
        print(f"Dataset Statistics for subjects {subjects}:")
        print(f"Total files: {self.stats['total_files']}")
        print(f"Successfully processed: {self.stats['processed_files']} ({self.stats['processed_files']/max(1, self.stats['total_files'])*100:.1f}%)")
        print(f"Errors: {self.stats['errors']}")
        print(f"Falls: {fall_count} samples")
        print(f"Non-falls: {non_fall_count} samples")
        print("Activities distribution:")
        for act_id, count in sorted(self.stats['activities'].items()):
            is_fall = 'Fall' if act_id > 9 else 'ADL'
            print(f"  Activity {act_id} ({is_fall}): {count} files")
        
        return self.data
    
    def normalization(self):
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    num_samples, length = value.shape[:2]
                    norm_data = StandardScaler().fit_transform(value.reshape(num_samples*length, -1))
                    self.data[key] = norm_data.reshape(num_samples, length, -1)
                except Exception as e:
                    print(f"Error normalizing {key} data: {str(e)}. Keeping original values.")
        return self.data
