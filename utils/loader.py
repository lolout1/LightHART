import os
import numpy as np
import pandas as pd
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import find_peaks
from collections import defaultdict
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any
import time
from tqdm import tqdm

from utils.imu_fusion import (
    align_sensor_data, save_aligned_data, process_window_with_filter, 
    process_sequential_windows, create_filter_id, clear_filters, get_filter
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("loader")

MAX_THREADS = 30
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(MAX_THREADS)

def csvloader(file_path, **kwargs):
    try:
        try: file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        except: file_data = pd.read_csv(file_path, index_col=False, header=None, sep=';').dropna().bfill()
        
        if 'skeleton' in file_path: cols = 96
        else:
            if file_data.shape[1] > 4:
                cols = file_data.shape[1] - 3
                file_data = file_data.iloc[:, 3:]
            else: cols = 3
        
        if file_data.shape[1] < cols:
            missing_cols = cols - file_data.shape[1]
            for i in range(missing_cols): file_data[f'missing_{i}'] = 0
        
        activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32) if file_data.shape[0] > 2 else file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise

def matloader(file_path, **kwargs):
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']: raise ValueError(f'Unsupported {key} for matlab file')
    from scipy.io import loadmat
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def avg_pool(sequence, window_size=5, stride=1, max_length=512, shape=None):
    import torch.nn.functional as F
    import torch
    
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
    
    stride = ((sequence_tensor.shape[2] // max_length) + 1) if max_length < sequence_tensor.shape[2] else 1
    pooled = F.avg_pool1d(sequence_tensor, kernel_size=window_size, stride=stride)
    pooled_np = pooled.squeeze(0).numpy().transpose(1, 0)
    result = pooled_np.reshape(-1, *shape[1:])
    
    return result

def pad_sequence_numpy(sequence, max_sequence_length, input_shape):
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    actual_length = min(len(pooled_sequence), max_sequence_length)
    new_sequence[:actual_length] = pooled_sequence[:actual_length]
    return new_sequence

def align_sequence(data):
    acc_data = data.get('accelerometer')
    gyro_data = data.get('gyroscope')
    if acc_data is None or gyro_data is None: return data
    
    aligned_acc, aligned_gyro, common_times = align_sensor_data(acc_data, gyro_data)
    if len(aligned_acc) == 0 or len(aligned_gyro) == 0:
        logger.warning("Sensor alignment failed, using original data")
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

def estimate_orientation_from_linear_acc_gyro(acc_data, gyro_data, window_size=64):
    quaternions = np.zeros((window_size, 4))
    quaternions[:, 0] = 1.0
    
    current_q = np.array([1.0, 0.0, 0.0, 0.0])
    alpha = 0.02
    dt = 1.0/30.0
    
    for i in range(min(len(acc_data), window_size)):
        gyro = gyro_data[i]
        linear_acc = acc_data[i]
        
        if i > 0:
            w, x, y, z = quaternions[i-1]
            prev_rot = Rotation.from_quat([x, y, z, w])
            gravity_sensor_frame = prev_rot.inv().apply([0, 0, 9.81])
            acc_with_gravity = linear_acc + gravity_sensor_frame
        else:
            acc_with_gravity = linear_acc + np.array([0, 0, 9.81])
        
        acc_norm = np.linalg.norm(acc_with_gravity)
        if acc_norm > 1e-6: acc_normalized = acc_with_gravity / acc_norm
        else: acc_normalized = np.array([0, 0, 1])
        
        gravity_ref = np.array([0, 0, 1])
        rotation_axis = np.cross(gravity_ref, acc_normalized)
        axis_norm = np.linalg.norm(rotation_axis)
        
        if axis_norm > 1e-6:
            rotation_axis /= axis_norm
            angle = np.arccos(np.clip(np.dot(gravity_ref, acc_normalized), -1.0, 1.0))
            acc_q = np.zeros(4)
            acc_q[0] = np.cos(angle/2)
            acc_q[1:4] = rotation_axis * np.sin(angle/2)
        else:
            acc_q = np.array([1, 0, 0, 0]) if acc_normalized[2] >= 0 else np.array([0, 1, 0, 0])
        
        if i > 0:
            w, x, y, z = quaternions[i-1]
            gyro_q_dot = 0.5 * np.array([
                -x*gyro[0] - y*gyro[1] - z*gyro[2],
                w*gyro[0] + y*gyro[2] - z*gyro[1],
                w*gyro[1] - x*gyro[2] + z*gyro[0],
                w*gyro[2] + x*gyro[1] - y*gyro[0]
            ])
            gyro_q = quaternions[i-1] + gyro_q_dot * dt
            gyro_q_norm = np.linalg.norm(gyro_q)
            if gyro_q_norm > 1e-10: gyro_q /= gyro_q_norm
        else: gyro_q = acc_q.copy()
        
        q = np.zeros(4)
        dot = np.sum(gyro_q * acc_q)
        if dot < 0: acc_q, dot = -acc_q, -dot
        
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995: q = gyro_q + alpha * (acc_q - gyro_q)
        else:
            theta_0 = np.arccos(dot)
            theta = theta_0 * alpha
            sin_theta = np.sin(theta)
            sin_theta_0 = np.sin(theta_0)
            
            s_gyro = np.cos(theta) - dot * sin_theta / sin_theta_0
            s_acc = sin_theta / sin_theta_0
            
            q = s_gyro * gyro_q + s_acc * acc_q
        
        q_norm = np.linalg.norm(q)
        quaternions[i] = q / q_norm if q_norm > 1e-10 else np.array([1, 0, 0, 0])
    
    return quaternions

def process_sequential_windows(data, window_size=64, stride=32, label=None, filter_type='madgwick', 
                             filter_params=None, base_filter_id=None, stateful=True, is_linear_acc=True):
    if 'accelerometer' not in data or data['accelerometer'] is None or 'gyroscope' not in data or data['gyroscope'] is None:
        return {'labels': np.array([label])} if label is not None else {'labels': np.array([])}
    
    acc_data = data['accelerometer']
    gyro_data = data['gyroscope']
    timestamps = data.get('aligned_timestamps', None)
    
    if len(acc_data) < window_size // 2:
        return {'labels': np.array([label])} if label is not None else {'labels': np.array([])}
    
    num_windows = max(1, (len(acc_data) - window_size) // stride + 1)
    windows = defaultdict(list)
    
    if label is not None: windows['labels'] = [label] * num_windows
    
    filter_id = base_filter_id if base_filter_id is not None else f"sequential_{filter_type}"
    
    for i in range(num_windows):
        start = i * stride
        end = min(start + window_size, len(acc_data))
        
        if end - start < window_size // 2: continue
        
        acc_window = np.zeros((window_size, acc_data.shape[1]))
        actual_length = end - start
        acc_window[:actual_length] = acc_data[start:end]
        
        gyro_window = np.zeros((window_size, gyro_data.shape[1]))
        actual_gyro_length = min(actual_length, len(gyro_data) - start)
        if actual_gyro_length > 0: gyro_window[:actual_gyro_length] = gyro_data[start:start + actual_gyro_length]
        
        ts_window = None
        if timestamps is not None:
            ts_window = np.zeros(window_size)
            actual_ts_length = min(actual_length, len(timestamps) - start)
            if actual_ts_length > 0: ts_window[:actual_ts_length] = timestamps[start:start + actual_ts_length]
        
        reset = not stateful or i == 0
        result = process_window_with_filter(
            acc_window, gyro_window, ts_window, 
            filter_type=filter_type, filter_id=filter_id, 
            reset_filter=reset, is_linear_acc=is_linear_acc
        )
        
        windows['accelerometer'].append(acc_window)
        windows['gyroscope'].append(gyro_window)
        windows['quaternion'].append(result['quaternion'])
    
    for key in windows:
        if key != 'labels' and len(windows[key]) > 0: windows[key] = np.array(windows[key])
    if 'labels' in windows and isinstance(windows['labels'], list): windows['labels'] = np.array(windows['labels'])
    
    return windows

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', fusion_options=None, **kwargs):
        if mode not in ['avg_pool', 'sliding_window']: raise ValueError(f"Unsupported processing method {mode}")
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
            self.stateful = not fusion_options.get('reset_per_window', False)
            self.window_stride = fusion_options.get('window_stride', 32)
            self.filter_params = fusion_options
            self.is_linear_acc = True
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
            if file_type not in ['csv', 'mat']: raise ValueError(f"Unsupported file type {file_type}")
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
                        is_linear_acc=self.is_linear_acc
                    )
                    
                    processed_data['quaternion'] = result['quaternion']
                except Exception as e:
                    logger.error(f"Fusion processing failed: {str(e)}")
                    processed_data['quaternion'] = np.zeros((self.max_length, 4))
            
            if 'quaternion' not in processed_data:
                processed_data['quaternion'] = np.zeros((self.max_length, 4))
                processed_data['quaternion'][:, 0] = 1.0
            
            return processed_data
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
        
        if filter_type is None and hasattr(self, 'filter_type'): filter_type = self.filter_type
        else: filter_type = 'madgwick'
            
        if hasattr(self, 'fusion_options'): 
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)
        
        clear_filters()
        processed_count = 0
        
        subject_action_groups = {}
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects: continue
            key = (trial.subject_id, trial.action_id)
            if key not in subject_action_groups: subject_action_groups[key] = []
            subject_action_groups[key].append(trial)
        
        logger.info(f"Processing {len(self.dataset.matched_trials)} trials with filter_type={filter_type}, fuse={fuse}")
        
        for (subject_id, action_id), trials in subject_action_groups.items():
            base_filter_id = create_filter_id(subject_id, action_id, filter_type=filter_type)
            
            for trial in trials:
                if self.task == 'fd': label = int(trial.action_id > 9)
                elif self.task == 'age': label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else: label = trial.action_id - 1
                
                trial_id = (trial.subject_id, trial.action_id, trial.sequence_number)
                trial_data = {}
                
                for modality_name, file_path in trial.files.items():
                    try:
                        unimodal_data = self.load_file(file_path)
                        trial_data[modality_name] = unimodal_data
                    except: continue
                
                if fuse and ('accelerometer' not in trial_data or 'gyroscope' not in trial_data): 
                    logger.warning(f"Skipping trial S{trial.subject_id}A{trial.action_id}T{trial.sequence_number} - missing required sensors")
                    continue
                
                trial_data = align_sequence(trial_data)
                
                if save_aligned:
                    aligned_acc = trial_data.get('accelerometer')
                    aligned_gyro = trial_data.get('gyroscope')
                    aligned_timestamps = trial_data.get('aligned_timestamps')
                    
                    if aligned_acc is not None and aligned_gyro is not None:
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
                try: self.data[key] = np.concatenate(self.data[key], axis=0)
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
            if modality not in self.data: self.data[modality] = []
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
