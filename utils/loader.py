import os
import numpy as np
import pandas as pd
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import traceback
import time
from tqdm import tqdm

from utils.imu_fusion import (
    align_sensor_data, save_aligned_data, process_window_with_filter, 
    process_sequential_windows, create_filter_id, clear_filters, get_filter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loader")

MAX_THREADS = 30
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(MAX_THREADS)
dataset_locks = defaultdict(threading.Lock)

def csvloader(file_path, **kwargs):
    try:
        try:
            file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        except:
            file_data = pd.read_csv(file_path, index_col=False, header=None, sep=';').dropna().bfill()
        
        if 'skeleton' in file_path:
            cols = 96
        else:
            if file_data.shape[1] > 4:
                cols = file_data.shape[1] - 3
                file_data = file_data.iloc[:, 3:]
            else:
                cols = 3
        
        if file_data.shape[1] < cols:
            missing_cols = cols - file_data.shape[1]
            for i in range(missing_cols):
                file_data[f'missing_{i}'] = 0
        
        if file_data.shape[0] > 2:
            activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        else:
            activity_data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
        
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise

def matloader(file_path, **kwargs):
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        raise ValueError(f'Unsupported {key} for matlab file')
    from scipy.io import loadmat
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def pad_sequence_numpy(sequence, max_sequence_length, input_shape):
    import torch
    import torch.nn.functional as F
    
    shape = list(input_shape)
    shape[0] = max_sequence_length
    
    if len(sequence) > max_sequence_length:
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
        pooled = F.avg_pool1d(sequence_tensor, kernel_size=len(sequence)//max_sequence_length, stride=len(sequence)//max_sequence_length)
        sequence = pooled.permute(0, 2, 1).squeeze(0).numpy()
    
    result = np.zeros(shape, sequence.dtype)
    result[:len(sequence)] = sequence[:max_sequence_length]
    
    return result

def align_sequence(data):
    acc_data = data.get('accelerometer')
    gyro_data = data.get('gyroscope')
    
    if acc_data is None or gyro_data is None or len(acc_data) == 0 or len(gyro_data) == 0:
        return data
    
    aligned_acc, aligned_gyro, common_times = align_sensor_data(acc_data, gyro_data)
    
    if len(aligned_acc) == 0 or len(aligned_gyro) == 0:
        if len(acc_data) > 5 and len(gyro_data) > 5:
            logger.warning("Sensor alignment failed, attempting to find partial overlapping windows")
            
            min_window_size = 32
            best_window_size = 0
            best_start_acc = 0
            best_start_gyro = 0
            
            for acc_start in range(0, len(acc_data) - min_window_size, min_window_size // 2):
                for gyro_start in range(0, len(gyro_data) - min_window_size, min_window_size // 2):
                    window_size = min(len(acc_data) - acc_start, len(gyro_data) - gyro_start)
                    
                    if window_size > best_window_size:
                        best_window_size = window_size
                        best_start_acc = acc_start
                        best_start_gyro = gyro_start
            
            if best_window_size >= min_window_size:
                aligned_data = {
                    'accelerometer': acc_data[best_start_acc:best_start_acc + best_window_size],
                    'gyroscope': gyro_data[best_start_gyro:best_start_gyro + best_window_size],
                    'aligned_timestamps': np.linspace(0, best_window_size / 30.0, best_window_size)
                }
                return aligned_data
            
            logger.warning("Could not find suitable overlapping windows, skipping file")
        return data
    
    aligned_data = {
        'accelerometer': aligned_acc,
        'gyroscope': aligned_gyro,
        'aligned_timestamps': common_times
    }
    
    skeleton_data = data.get('skeleton')
    if skeleton_data is not None:
        skeleton_times = np.linspace(0, len(skeleton_data)/30.0, len(skeleton_data))
        aligned_skeleton = np.zeros((len(common_times), skeleton_data.shape[1], skeleton_data.shape[2]))
        
        for joint in range(skeleton_data.shape[1]):
            for coord in range(skeleton_data.shape[2]):
                aligned_skeleton[:, joint, coord] = np.interp(common_times, skeleton_times, skeleton_data[:, joint, coord])
        
        aligned_data['skeleton'] = aligned_skeleton
    
    return aligned_data

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', fusion_options=None, **kwargs):
        if mode not in ['avg_pool', 'sliding_window']:
            raise ValueError(f"Unsupported processing method {mode}")
        
        self.dataset = dataset
        self.data = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.fusion_options = fusion_options or {}
        
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        os.makedirs(os.path.join(self.aligned_data_dir, "accelerometer"), exist_ok=True)
        os.makedirs(os.path.join(self.aligned_data_dir, "gyroscope"), exist_ok=True)
        
        if fusion_options:
            self.fusion_enabled = fusion_options.get('enabled', False)
            self.filter_type = fusion_options.get('filter_type', 'madgwick')
            self.stateful = not fusion_options.get('process_per_window', False)
            self.window_stride = fusion_options.get('window_stride', 32)
            self.filter_params = fusion_options
            self.is_linear_acc = True  # CSV files contain linear acceleration
            
            logger.info(f"Fusion options: enabled={self.fusion_enabled}, filter_type={self.filter_type}, stateful={self.stateful}")
        else:
            self.fusion_enabled = False
            self.filter_type = 'madgwick'
            self.stateful = True
            self.window_stride = 32
            self.filter_params = {}
            self.is_linear_acc = True
    
    def load_file(self, file_path):
        try:
            file_type = file_path.split('.')[-1]
            if file_type not in ['csv', 'mat']:
                raise ValueError(f"Unsupported file type {file_type}")
            
            loader = LOADER_MAP[file_type]
            data = loader(file_path, **self.kwargs)
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def process(self, data, label, fuse=False, filter_type='madgwick', filter_params=None, trial_id=None):
        if self.mode == 'avg_pool':
            processed_data = {}
            
            for modality, modality_data in data.items():
                if modality != 'labels':
                    processed_data[modality] = pad_sequence_numpy(
                        sequence=modality_data,
                        max_sequence_length=self.max_length,
                        input_shape=modality_data.shape
                    )
            
            processed_data['labels'] = np.array([label])
            
            if fuse and 'accelerometer' in processed_data and 'gyroscope' in processed_data:
                try:
                    timestamps = processed_data.get('aligned_timestamps', None)
                    filter_id = create_filter_id(
                        subject_id=trial_id[0] if trial_id else 0,
                        action_id=trial_id[1] if trial_id else 0,
                        trial_id=trial_id[2] if trial_id and len(trial_id) > 2 else None,
                        filter_type=filter_type
                    )
                    
                    result = process_window_with_filter(
                        processed_data['accelerometer'],
                        processed_data['gyroscope'],
                        timestamps,
                        filter_type=filter_type,
                        filter_id=filter_id,
                        reset_filter=not self.stateful,
                        is_linear_acc=self.is_linear_acc,
                        filter_params=filter_params
                    )
                    
                    processed_data['quaternion'] = result['quaternion']
                except Exception as e:
                    logger.error(f"Fusion processing failed: {str(e)}")
                    processed_data['quaternion'] = np.zeros((self.max_length, 4))
                    processed_data['quaternion'][:, 0] = 1.0
            
            if 'quaternion' not in processed_data:
                processed_data['quaternion'] = np.zeros((self.max_length, 4))
                processed_data['quaternion'][:, 0] = 1.0
        else:
            filter_id = create_filter_id(
                subject_id=trial_id[0] if trial_id else 0,
                action_id=trial_id[1] if trial_id else 0,
                trial_id=trial_id[2] if trial_id and len(trial_id) > 2 else None,
                filter_type=filter_type
            )
            
            processed_data = process_sequential_windows(
                data=data,
                window_size=self.max_length,
                stride=self.window_stride,
                label=label,
                filter_type=filter_type,
                filter_params=filter_params,
                base_filter_id=filter_id,
                stateful=self.stateful,
                is_linear_acc=self.is_linear_acc
            )
        
        return processed_data

    def make_dataset(self, subjects, fuse=False, filter_type=None, visualize=False, save_aligned=False):
        self.data = {}
        self.fuse = fuse
        
        if filter_type is None and hasattr(self, 'filter_type'):
            filter_type = self.filter_type
        else:
            filter_type = 'madgwick'
            
        if hasattr(self, 'fusion_options'): 
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)
        
        clear_filters()
        processed_count = 0
        
        subject_action_groups = {}
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue
            
            key = (trial.subject_id, trial.action_id)
            if key not in subject_action_groups:
                subject_action_groups[key] = []
            
            subject_action_groups[key].append(trial)
        
        logger.info(f"Processing {len(self.dataset.matched_trials)} trials with filter_type={filter_type}, fuse={fuse}")
        
        for (subject_id, action_id), trials in subject_action_groups.items():
            base_filter_id = create_filter_id(subject_id, action_id, filter_type=filter_type)
            
            for trial in trials:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                
                trial_id = (trial.subject_id, trial.action_id, trial.sequence_number)
                trial_data = {}
                
                for modality_name, file_path in trial.files.items():
                    try:
                        unimodal_data = self.load_file(file_path)
                        trial_data[modality_name] = unimodal_data
                    except:
                        continue
                
                if 'accelerometer' not in trial_data or 'gyroscope' not in trial_data:
                    logger.warning(f"Skipping trial S{trial.subject_id}A{trial.action_id}T{trial.sequence_number} - missing required sensors")
                    continue
                
                if len(trial_data['accelerometer']) < 5 or len(trial_data['gyroscope']) < 5:
                    logger.warning(f"Skipping trial S{trial.subject_id}A{trial.action_id}T{trial.sequence_number} - sensor data too short")
                    continue
                
                trial_data = align_sequence(trial_data)
                
                if 'accelerometer' not in trial_data or 'gyroscope' not in trial_data:
                    logger.warning(f"Skipping trial S{trial.subject_id}A{trial.action_id}T{trial.sequence_number} - could not align sensors")
                    continue
                
                if len(trial_data['accelerometer']) < 5 or len(trial_data['gyroscope']) < 5:
                    logger.warning(f"Skipping trial S{trial.subject_id}A{trial.action_id}T{trial.sequence_number} - aligned sensor data too short")
                    continue
                
                if save_aligned:
                    aligned_acc = trial_data.get('accelerometer')
                    aligned_gyro = trial_data.get('gyroscope')
                    aligned_timestamps = trial_data.get('aligned_timestamps')
                    
                    if aligned_acc is not None and aligned_gyro is not None and len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                        save_aligned_data(
                            trial.subject_id,
                            trial.action_id,
                            trial.sequence_number,
                            aligned_acc,
                            aligned_gyro,
                            aligned_timestamps if aligned_timestamps is not None else np.arange(len(aligned_acc))
                        )
                
                processed_data = self.process(
                    trial_data, 
                    label, 
                    fuse, 
                    filter_type, 
                    self.filter_params if hasattr(self, 'filter_params') else None,
                    trial_id
                )
                
                if processed_data is not None:
                    if 'labels' in processed_data and len(processed_data['labels']) > 0:
                        self._add_trial_data(processed_data)
                        processed_count += 1
        
        logger.info(f"Processed {processed_count} trials")
        
        for key in self.data:
            if len(self.data[key]) > 0:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {str(e)}")
                    del self.data[key]
        
        if 'quaternion' not in self.data and 'accelerometer' in self.data:
            acc_shape = self.data['accelerometer'].shape
            self.data['quaternion'] = np.zeros((acc_shape[0], acc_shape[1], 4))
            self.data['quaternion'][..., 0] = 1.0
        
        clear_filters()
        return self.data

    def _add_trial_data(self, trial_data):
        for modality, modality_data in trial_data.items():
            if modality not in self.data:
                self.data[modality] = []
            
            if isinstance(modality_data, np.ndarray) and modality_data.size > 0:
                self.data[modality].append(modality_data)

    def normalization(self):
        from sklearn.preprocessing import StandardScaler
        
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion'] and len(value.shape) >= 2:
                        num_samples, length = value.shape[:2]
                        reshaped_data = value.reshape(num_samples * length, -1)
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                        self.data[key] = norm_data.reshape(value.shape)
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        self.data[key] = StandardScaler().fit_transform(value)
                except Exception as e:
                    logger.error(f"Error normalizing {key}: {str(e)}")
        
        return self.data
