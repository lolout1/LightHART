import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("loader")

MAX_THREADS = 22
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(MAX_THREADS)
filter_registry = {}

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

LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

def hybrid_interpolate(x, y, x_new, threshold=2.0, window_size=5):
    if len(x) < 2 or len(y) < 2:
        return None
    try:
        dy, dx = np.diff(y), np.diff(x)
        rates = np.abs(dy / np.maximum(dx, 1e-10))
        if len(rates) >= window_size:
            rates = savgol_filter(rates, window_size, 2)
        rapid_changes = rates > threshold
        if not np.any(rapid_changes):
            try:
                from scipy.interpolate import CubicSpline
                return CubicSpline(x, y)(x_new)
            except:
                from scipy.interpolate import interp1d
                return interp1d(x, y, bounds_error=False, fill_value='extrapolate')(x_new)
        if np.all(rapid_changes):
            from scipy.interpolate import interp1d
            return interp1d(x, y, bounds_error=False, fill_value='extrapolate')(x_new)
        from scipy.interpolate import interp1d, CubicSpline
        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        try:
            spline_interp = CubicSpline(x, y)
        except:
            return linear_interp(x_new)
        y_interp = np.zeros_like(x_new, dtype=float)
        segments, segment_start = [], None
        for i in range(len(rapid_changes)):
            if rapid_changes[i] and segment_start is None:
                segment_start = i
            elif not rapid_changes[i] and segment_start is not None:
                segments.append((segment_start, i))
                segment_start = None
        if segment_start is not None:
            segments.append((segment_start, len(rapid_changes)))
        linear_mask = np.zeros_like(x_new, dtype=bool)
        buffer = 0.05
        for start_idx, end_idx in segments:
            t_start = max(x[start_idx] - buffer, x[0])
            t_end = min(x[min(end_idx, len(x)-1)] + buffer, x[-1])
            linear_mask |= (x_new >= t_start) & (x_new <= t_end)
        if np.any(linear_mask):
            y_interp[linear_mask] = linear_interp(x_new[linear_mask])
        if np.any(~linear_mask):
            y_interp[~linear_mask] = spline_interp(x_new[~linear_mask])
        return y_interp
    except:
        try:
            from scipy.interpolate import interp1d
            return interp1d(x, y, bounds_error=False, fill_value='extrapolate')(x_new)
        except:
            return None

def align_sensor_data(acc_data, gyro_data, time_tolerance=0.01):
    if acc_data is None or gyro_data is None or len(acc_data) == 0 or len(gyro_data) == 0:
        return None, None, None
    if isinstance(acc_data, np.ndarray) and isinstance(gyro_data, np.ndarray):
        acc_times, gyro_times = np.arange(len(acc_data)), np.arange(len(gyro_data))
        if len(acc_data) == len(gyro_data):
            return acc_data, gyro_data, acc_times
        min_len = min(len(acc_data), len(gyro_data))
        return acc_data[:min_len], gyro_data[:min_len], acc_times[:min_len]
    try:
        if isinstance(acc_data.iloc[0, 0], str):
            acc_times = pd.to_datetime(acc_data.iloc[:, 0]).values
        else:
            acc_times = acc_data.iloc[:, 0].values
        if isinstance(gyro_data.iloc[0, 0], str):
            gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).values
        else:
            gyro_times = gyro_data.iloc[:, 0].values
    except:
        logger.error("Error extracting timestamps from data")
        return None, None, None
    start_time, end_time = max(acc_times[0], gyro_times[0]), min(acc_times[-1], gyro_times[-1])
    acc_in_range = (acc_times >= start_time) & (acc_times <= end_time)
    gyro_in_range = (gyro_times >= start_time) & (gyro_times <= end_time)
    if not np.any(acc_in_range) or not np.any(gyro_in_range):
        return None, None, None
    acc_data_filtered = acc_data.iloc[acc_in_range, 1:4].values
    gyro_data_filtered = gyro_data.iloc[gyro_in_range, 1:4].values
    if len(acc_data_filtered) < 3 or len(gyro_data_filtered) < 3:
        return None, None, None
    common_times = np.linspace(start_time, end_time, min(len(acc_data_filtered), len(gyro_data_filtered)))
    aligned_acc, aligned_gyro = np.zeros((len(common_times), 3)), np.zeros((len(common_times), 3))
    for axis in range(3):
        acc_interp = hybrid_interpolate(acc_times[acc_in_range], acc_data_filtered[:, axis], common_times)
        gyro_interp = hybrid_interpolate(gyro_times[gyro_in_range], gyro_data_filtered[:, axis], common_times)
        if acc_interp is None or gyro_interp is None:
            return None, None, None
        aligned_acc[:, axis], aligned_gyro[:, axis] = acc_interp, gyro_interp
    return aligned_acc, aligned_gyro, common_times

def save_aligned_data(subject_id, action_id, trial_id, acc_data, gyro_data, timestamps=None, save_dir="data/aligned"):
    if acc_data is None or gyro_data is None or len(acc_data) == 0 or len(gyro_data) == 0:
        return False
    os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
    os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
    if timestamps is not None:
        os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
    filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
    with file_semaphore:
        np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
        np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
        if timestamps is not None:
            np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)
    return True

def apply_adaptive_filter(acc_data, cutoff_freq=2.0, fs=30.0):
    if acc_data is None or len(acc_data) == 0:
        return None
    data_length = acc_data.shape[0]
    if data_length < 15:
        filtered_data = np.zeros_like(acc_data)
        for i in range(acc_data.shape[1]):
            if data_length > 2:
                filtered_data[:, i] = np.convolve(acc_data[:, i], np.ones(3)/3, mode='same')
            else:
                filtered_data[:, i] = acc_data[:, i]
        return filtered_data
    filtered_data = np.zeros_like(acc_data)
    padlen = min(data_length - 1, 10)
    try:
        for i in range(acc_data.shape[1]):
            b, a = butter(4, cutoff_freq / (fs/2), btype='low')
            filtered_data[:, i] = filtfilt(b, a, acc_data[:, i], padlen=padlen)
    except:
        filtered_data = acc_data.copy()
    return filtered_data

def avg_pool(sequence, window_size=5, stride=1, max_length=512, shape=None):
    import torch
    import torch.nn.functional as F
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
    stride = ((sequence_tensor.shape[2] // max_length) + 1 if max_length < sequence_tensor.shape[2] else 1)
    pooled = F.avg_pool1d(sequence_tensor, kernel_size=window_size, stride=stride)
    pooled_np = pooled.squeeze(0).numpy().transpose(1, 0)
    return pooled_np.reshape(-1, *shape[1:])

def pad_sequence_numpy(sequence, max_sequence_length, input_shape):
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def align_sequence(data):
    acc_data = data.get('accelerometer')
    gyro_data = data.get('gyroscope')
    if acc_data is None or gyro_data is None:
        return None
    aligned_acc, aligned_gyro, common_times = align_sensor_data(acc_data, gyro_data)
    if aligned_acc is None or aligned_gyro is None or common_times is None:
        logger.warning("Sensor alignment failed")
        return None
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
                aligned_skeleton[:, joint, coord] = np.interp(
                    common_times,
                    skeleton_times,
                    skeleton_data[:, joint, coord]
                )
        aligned_data['skeleton'] = aligned_skeleton
    return aligned_data

def fix_gravity(acc_data):
    if acc_data is None or len(acc_data) == 0:
        return None
    gravity_dir = np.array([0, 0, 9.81])
    return acc_data + gravity_dir

def get_stateful_filter(subject_id, action_id, trial_id, filter_type, reset=False, params=None):
    filter_id = f"S{subject_id}A{action_id}T{trial_id}_{filter_type}"
    from utils.imu_fusion import get_filter
    return get_filter(filter_id, filter_type, params, reset=reset)

def process_sequence_with_filter(acc_data, gyro_data, timestamps=None, 
                               filter_id=None, filter_type='madgwick', reset_filter=False):
    from utils.imu_fusion import process_imu_data
    result = process_imu_data(
        acc_data, gyro_data, timestamps, 
        filter_type=filter_type, 
        filter_id=filter_id, 
        reset_filter=reset_filter
    )
    return result

def selective_sliding_window(data, window_size, stride=10, label=0, 
                           fuse=False, filter_type='madgwick', 
                           trial_id=None, use_stateful=True):
    if data is None:
        return {'labels': np.array([label])}
    acc_data = data.get('accelerometer')
    gyro_data = data.get('gyroscope')
    ts_data = data.get('aligned_timestamps')
    
    if acc_data is None or gyro_data is None:
        logger.warning("Missing required sensor data, skipping trial")
        return {'labels': np.array([label])}
    
    if len(acc_data) < window_size or len(gyro_data) < window_size:
        logger.warning("Insufficient data length, skipping trial")
        return {'labels': np.array([label])}
    
    num_windows = max(1, (len(acc_data) - window_size) // stride + 1)
    windowed_data = defaultdict(list)
    
    for i in range(num_windows):
        start = i * stride
        end = min(start + window_size, len(acc_data))
        
        if end - start < window_size:
            continue
        
        window_acc = acc_data[start:end]
        window_gyro = gyro_data[start:end]
        window_ts = ts_data[start:end] if ts_data is not None else None
        
        if len(window_acc) < window_size:
            padded_acc = np.zeros((window_size, window_acc.shape[1]))
            padded_acc[:len(window_acc)] = window_acc
            window_acc = padded_acc
            
        if len(window_gyro) < window_size:
            padded_gyro = np.zeros((window_size, window_gyro.shape[1]))
            padded_gyro[:len(window_gyro)] = window_gyro
            window_gyro = padded_gyro
            
        if window_ts is not None and len(window_ts) < window_size:
            padded_ts = np.zeros(window_size)
            padded_ts[:len(window_ts)] = window_ts
            window_ts = padded_ts
        
        windowed_data['accelerometer'].append(window_acc)
        windowed_data['gyroscope'].append(window_gyro)
        if window_ts is not None:
            windowed_data['timestamps'].append(window_ts)
        
        if fuse:
            window_filter_id = f"{trial_id}_window_{i}" if use_stateful else None
            quat_result = process_sequence_with_filter(
                window_acc, window_gyro, window_ts,
                filter_id=window_filter_id,
                filter_type=filter_type,
                reset_filter=(i == 0)
            )
            
            if quat_result is not None and 'quaternion' in quat_result:
                window_quat = quat_result['quaternion']
                if len(window_quat) < window_size:
                    padded_quat = np.zeros((window_size, 4))
                    padded_quat[:len(window_quat)] = window_quat
                    window_quat = padded_quat
                windowed_data['quaternion'].append(window_quat)
            else:
                windowed_data['quaternion'].append(np.zeros((window_size, 4)))
    
    for key in windowed_data:
        if len(windowed_data[key]) > 0:
            windowed_data[key] = np.array(windowed_data[key])
    
    windows_created = len(windowed_data.get('accelerometer', []))
    windowed_data['labels'] = np.repeat(label, windows_created)
    
    return windowed_data

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
        self.fusion_options = fusion_options or {}
        
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        os.makedirs(os.path.join(self.aligned_data_dir, "accelerometer"), exist_ok=True)
        os.makedirs(os.path.join(self.aligned_data_dir, "gyroscope"), exist_ok=True)
        
        if fusion_options:
            self.fusion_enabled = fusion_options.get('enabled', False)
            self.filter_type = fusion_options.get('filter_type', 'madgwick')
            self.stateful = not fusion_options.get('reset_per_window', False)
            self.window_stride = fusion_options.get('window_stride', 10)
            self.filter_params = fusion_options
            logger.info(f"Fusion options: enabled={self.fusion_enabled}, filter_type={self.filter_type}, stateful={self.stateful}")
        else:
            self.fusion_enabled = False
            self.filter_type = 'madgwick'
            self.stateful = True
            self.window_stride = 10
            self.filter_params = {}
    
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
        if data is None:
            return None
            
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
                timestamps = processed_data.get('aligned_timestamps', None)
                filter_id = f"{trial_id}_{filter_type}" if trial_id else None
                quat_result = process_sequence_with_filter(
                    processed_data['accelerometer'],
                    processed_data['gyroscope'],
                    timestamps,
                    filter_id=filter_id,
                    filter_type=filter_type,
                    reset_filter=not self.stateful
                )
                
                if quat_result is not None:
                    processed_data.update(quat_result)
                else:
                    processed_data['quaternion'] = np.zeros((self.max_length, 4))
            
            if 'quaternion' not in processed_data:
                processed_data['quaternion'] = np.zeros((self.max_length, 4))
            
            return processed_data
        else:
            processed_data = selective_sliding_window(
                data=data,
                window_size=self.max_length,
                stride=self.window_stride,
                label=label,
                fuse=fuse,
                filter_type=filter_type,
                trial_id=trial_id,
                use_stateful=self.stateful
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
        
        # Reset filter registry at the start
        from utils.imu_fusion import clear_filters
        clear_filters()
        
        processed_count = 0
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue
            
            if self.task == 'fd':
                label = int(trial.action_id > 9)
            elif self.task == 'age':
                label = int(trial.subject_id < 29 or trial.subject_id > 46)
            else:
                label = trial.action_id - 1
                
            trial_id = f"S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}"
            trial_data = {}
            
            # Load all modalities
            for modality_name, file_path in trial.files.items():
                try:
                    unimodal_data = self.load_file(file_path)
                    trial_data[modality_name] = unimodal_data
                except Exception as e:
                    logger.warning(f"Skipping {modality_name} file due to error: {str(e)}")
            
            # Skip trial if missing key modalities
            if 'accelerometer' not in trial_data or 'gyroscope' not in trial_data:
                logger.warning(f"Trial {trial_id} missing required sensors, skipping")
                continue
            
            # Align and process data
            aligned_data = align_sequence(trial_data)
            if aligned_data is None:
                logger.warning(f"Data alignment failed for trial {trial_id}, skipping")
                continue
            
            # Save aligned data if requested
            if save_aligned:
                save_aligned_data(
                    trial.subject_id,
                    trial.action_id,
                    trial.sequence_number,
                    aligned_data.get('accelerometer'),
                    aligned_data.get('gyroscope'),
                    aligned_data.get('aligned_timestamps')
                )
            
            # Process trial data
            processed_data = self.process(
                aligned_data, 
                label, 
                fuse, 
                filter_type, 
                self.filter_params,
                trial_id
            )
            
            # Add processed data to dataset
            if processed_data is not None and 'labels' in processed_data:
                self._add_trial_data(processed_data)
                processed_count += 1
        
        logger.info(f"Processed {processed_count} trials")
        
        # Concatenate all data
        for key in self.data:
            if len(self.data[key]) > 0:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {str(e)}")
                    del self.data[key]
        
        # Ensure quaternion data exists
        if 'quaternion' not in self.data and 'accelerometer' in self.data:
            acc_shape = self.data['accelerometer'].shape
            self.data['quaternion'] = np.zeros((acc_shape[0], acc_shape[1], 4))
        
        # Clean up filter registry
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
