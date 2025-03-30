import tqdm
import os
import time
import traceback
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import find_peaks, savgol_filter
import threading
from collections import defaultdict

from utils.imu_fusion import (
    process_imu_data,
    save_aligned_sensor_data,
    align_sensor_data,
    MAX_THREADS,
    thread_pool,
    file_semaphore,
    MadgwickFilter,
    KalmanFilter,
    ExtendedKalmanFilter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loader")
filter_cache = {}

def csvloader(file_path: str, **kwargs) -> np.ndarray:
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

def matloader(file_path: str, **kwargs) -> np.ndarray:
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        raise ValueError(f"Unsupported {key} for matlab file")
    try:
        from scipy.io import loadmat
        data = loadmat(file_path)[key]
        return data
    except Exception as e:
        logger.error(f"Error loading MAT {file_path}: {str(e)}")
        raise

LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

def get_filter_instance(subject_id, action_id, filter_type, reset=False):
    global filter_cache
    cache_key = f"{subject_id}_{action_id}_{filter_type}"
    
    if reset or cache_key not in filter_cache:
        if filter_type == 'madgwick':
            filter_instance = MadgwickFilter(beta=0.1)
        elif filter_type == 'kalman':
            filter_instance = KalmanFilter()
        elif filter_type == 'ekf':
            filter_instance = ExtendedKalmanFilter()
        else:
            filter_instance = MadgwickFilter()
        
        filter_cache[cache_key] = filter_instance
    
    return filter_cache[cache_key]

def stateful_process_imu_data(acc_data, gyro_data, timestamps=None, filter_type='madgwick', trial_id=None, reset_filter=False, return_features=False):
    if trial_id is None:
        return process_imu_data(acc_data, gyro_data, timestamps, filter_type, return_features)
    
    orientation_filter = get_filter_instance(trial_id, 0, filter_type, reset=reset_filter)
    
    try:
        quaternions = []
        
        for i in range(len(acc_data)):
            acc = acc_data[i]
            gyro = gyro_data[i]
            timestamp = timestamps[i] if timestamps is not None else None
            
            gravity_direction = np.array([0, 0, 9.81])
            if i > 0 and len(quaternions) > 0:
                from scipy.spatial.transform import Rotation
                last_q = quaternions[-1]
                r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
                
            acc_with_gravity = acc + gravity_direction
            acc_with_gravity = acc_with_gravity / np.linalg.norm(acc_with_gravity)
            
            q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
            quaternions.append(q)
        
        results = {'quaternion': np.array(quaternions)}
        
        if return_features:
            from utils.imu_fusion import extract_features_from_window
            features = extract_features_from_window({
                'quaternion': np.array(quaternions),
                'accelerometer': acc_data,
                'gyroscope': gyro_data
            })
            results['fusion_features'] = features
        
        return results
        
    except Exception as e:
        logger.error(f"Error in stateful IMU processing: {str(e)}")
        return {'quaternion': np.zeros((len(acc_data), 4))}

def avg_pool(sequence: np.ndarray, window_size: int = 5, stride: int = 1, max_length: int = 512, shape: Optional[Tuple] = None) -> np.ndarray:
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    import torch.nn.functional as F
    import torch
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
    if max_length < sequence_tensor.shape[2]:
        stride = ((sequence_tensor.shape[2] // max_length) + 1)
    else:
        stride = 1
    pooled = F.avg_pool1d(sequence_tensor, kernel_size=window_size, stride=stride)
    pooled_np = pooled.squeeze(0).numpy().transpose(1, 0)
    result = pooled_np.reshape(-1, *shape[1:])
    return result

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, input_shape: np.ndarray) -> np.ndarray:
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    actual_length = min(len(pooled_sequence), max_sequence_length)
    new_sequence[:actual_length] = pooled_sequence[:actual_length]
    return new_sequence

def sliding_window(data: np.ndarray, clearing_time_index: int, max_time: int, sub_window_size: int, stride_size: int) -> np.ndarray:
    if clearing_time_index < sub_window_size - 1:
        raise AssertionError("Clearing value needs to be greater or equal to (window size - 1)")
    
    start = clearing_time_index - sub_window_size + 1
    if max_time >= data.shape[0] - sub_window_size:
        max_time = max_time - sub_window_size + 1
    
    sub_windows = (
        start +
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time, step=stride_size), 0).T
    )
    
    return data[sub_windows]

def load_cached_processed(subject_id, action_id, filter_type, cache_dir="processed_data"):
    trial_id = f"S{subject_id:02d}A{action_id:02d}"
    cache_file = os.path.join(cache_dir, f"S{subject_id:02d}", f"{trial_id}.npz")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        data = np.load(cache_file)
        return {
            'accelerometer': data['accelerometer'],
            'gyroscope': data['gyroscope'],
            'quaternion': data['quaternion'],
            'timestamps': data['timestamps'] if 'timestamps' in data else None
        }
    except Exception as e:
        logger.error(f"Error loading cached data for {trial_id}: {e}")
        return None

def align_sequence(data):
    try:
        acc_data = data.get('accelerometer')
        gyro_data = data.get('gyroscope')
        skeleton_data = data.get('skeleton')
        
        if acc_data is None or gyro_data is None:
            return data
        
        aligned_acc, aligned_gyro, common_times = align_sensor_data(acc_data, gyro_data)
        
        if len(aligned_acc) == 0 or len(aligned_gyro) == 0:
            return {}
        
        aligned_data = {
            'accelerometer': aligned_acc,
            'gyroscope': aligned_gyro,
            'aligned_timestamps': common_times
        }
        
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
    
    except Exception as e:
        logger.error(f"Error in sequence alignment: {str(e)}")
        return {}

def _extract_window(data, start, end, window_size, fuse, filter_type='madgwick', trial_id=None, use_stateful=True, is_first_window=False):
    window_data = {}
    
    for modality, modality_data in data.items():
        if modality != 'labels' and modality_data is not None and len(modality_data) > 0:
            try:
                if modality == 'aligned_timestamps':
                    if len(modality_data.shape) == 1:
                        window_data_array = modality_data[start:min(end, len(modality_data))]
                        if len(window_data_array) < window_size:
                            padded = np.zeros(window_size, dtype=window_data_array.dtype)
                            padded[:len(window_data_array)] = window_data_array
                            window_data_array = padded
                    else:
                        window_data_array = modality_data[start:min(end, len(modality_data)), :]
                        if window_data_array.shape[0] < window_size:
                            padded = np.zeros((window_size, window_data_array.shape[1]), dtype=window_data_array.dtype)
                            padded[:window_data_array.shape[0]] = window_data_array
                            window_data_array = padded
                else:
                    window_data_array = modality_data[start:min(end, len(modality_data)), :]
                    if window_data_array.shape[0] < window_size:
                        padded = np.zeros((window_size, window_data_array.shape[1]), dtype=window_data_array.dtype)
                        padded[:window_data_array.shape[0]] = window_data_array
                        window_data_array = padded
                
                window_data[modality] = window_data_array
            
            except Exception as e:
                if modality == 'accelerometer':
                    window_data[modality] = np.zeros((window_size, 3))
                elif modality == 'gyroscope':
                    window_data[modality] = np.zeros((window_size, 3))
                elif modality == 'quaternion':
                    window_data[modality] = np.zeros((window_size, 4))
                else:
                    window_data[modality] = None

    if fuse and 'accelerometer' in window_data and 'gyroscope' in window_data:
        try:
            acc_window = window_data['accelerometer']
            gyro_window = window_data['gyroscope']
            timestamps = None
            
            if 'aligned_timestamps' in window_data:
                timestamps = window_data['aligned_timestamps']
                if len(timestamps.shape) > 1:
                    timestamps = timestamps[:, 0] if timestamps.shape[1] > 0 else None
            
            if use_stateful and trial_id is not None:
                fusion_results = stateful_process_imu_data(
                    acc_data=acc_window,
                    gyro_data=gyro_window,
                    timestamps=timestamps,
                    filter_type=filter_type,
                    trial_id=trial_id,
                    reset_filter=is_first_window,
                    return_features=False
                )
            else:
                fusion_results = process_imu_data(
                    acc_data=acc_window,
                    gyro_data=gyro_window,
                    timestamps=timestamps,
                    filter_type=filter_type,
                    return_features=False
                )
            
            window_data['quaternion'] = fusion_results.get('quaternion', np.zeros((window_size, 4)))
            
            if 'fusion_features' in fusion_results:
                window_data['fusion_features'] = fusion_results['fusion_features']
        
        except Exception as e:
            window_data['quaternion'] = np.zeros((window_size, 4))
    else:
        window_data['quaternion'] = np.zeros((window_size, 4))
    
    if 'quaternion' not in window_data or window_data['quaternion'] is None:
        window_data['quaternion'] = np.zeros((window_size, 4))
    elif window_data['quaternion'].shape[0] != window_size:
        temp = np.zeros((window_size, 4))
        if window_data['quaternion'].shape[0] < window_size:
            temp[:window_data['quaternion'].shape[0]] = window_data['quaternion']
        else:
            temp = window_data['quaternion'][:window_size]
        window_data['quaternion'] = temp
    
    return window_data

def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int, peaks: Union[List[int], np.ndarray], label: int, fuse: bool, filter_type: str = 'madgwick', trial_id=None, use_stateful=True) -> Dict[str, np.ndarray]:
    windowed_data = defaultdict(list)
    
    has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and len(data['gyroscope']) > 0
    has_acc = 'accelerometer' in data and data['accelerometer'] is not None and len(data['accelerometer']) > 0
    
    if not has_acc:
        return {'labels': np.array([label])}
    
    if fuse and not has_gyro:
        fuse = False
    
    max_workers = min(30, len(peaks)) if len(peaks) > 0 else 1
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for peak_idx, peak in enumerate(peaks):
            start = max(0, peak - window_size // 2)
            end = min(len(data['accelerometer']), start + window_size)
            
            if end - start < window_size // 2:
                continue
            
            futures.append((
                peak_idx,
                peak,
                executor.submit(
                    _extract_window,
                    data,
                    start,
                    end,
                    window_size,
                    fuse,
                    filter_type,
                    trial_id,
                    use_stateful,
                    peak_idx == 0
                )
            ))
        
        from tqdm import tqdm
        windows_created = 0
        collection_iterator = tqdm(futures, desc="Processing windows") if len(futures) > 5 else futures
        
        for peak_idx, peak, future in collection_iterator:
            try:
                window_data = future.result()
                
                if fuse and ('quaternion' not in window_data or window_data['quaternion'] is None):
                    window_data['quaternion'] = np.zeros((window_size, 4))
                
                for modality, modality_window in window_data.items():
                    if modality_window is not None:
                        windowed_data[modality].append(modality_window)
                
                windows_created += 1
            except Exception as e:
                logger.error(f"Error processing window at peak {peak}: {str(e)}")
    
    for modality in windowed_data:
        if modality != 'labels' and len(windowed_data[modality]) > 0:
            try:
                windowed_data[modality] = np.array(windowed_data[modality])
            except Exception as e:
                if modality == 'quaternion':
                    windowed_data[modality] = np.zeros((windows_created, window_size, 4))
    
    windowed_data['labels'] = np.repeat(label, windows_created)
    
    if fuse and ('quaternion' not in windowed_data or len(windowed_data['quaternion']) == 0):
        if 'accelerometer' in windowed_data and len(windowed_data['accelerometer']) > 0:
            num_windows = len(windowed_data['accelerometer'])
            windowed_data['quaternion'] = np.zeros((num_windows, window_size, 4))
    
    return windowed_data


def preprocess_all_subjects(subjects, filter_type, output_dir, max_length=64):
    logger.info(f"Preprocessing all subjects with {filter_type} filter")
    
    from utils.dataset import SmartFallMM
    from tqdm.auto import tqdm  # Correct import
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = SmartFallMM(
        root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'),
        fusion_options={'filter_type': filter_type}
    )
    
    dataset.pipe_line(
        age_group=['young'],
        modalities=['accelerometer', 'gyroscope'],
        sensors=['watch']
    )
    
    for subject_id in tqdm(subjects, desc=f"Preprocessing ({filter_type})"):
        subject_dir = os.path.join(output_dir, f"S{subject_id:02d}")
        os.makedirs(subject_dir, exist_ok=True)
        
        subject_trials = [trial for trial in dataset.matched_trials if trial.subject_id == subject_id]
        
        for trial in tqdm(subject_trials, desc=f"Subject {subject_id}", leave=False):
            action_id = trial.action_id
            trial_id = f"S{subject_id:02d}A{action_id:02d}"
            
            trial_data = {}
            try:
                if not ('accelerometer' in trial.files and 'gyroscope' in trial.files):
                    continue
                    
                for modality_name, file_path in trial.files.items():
                    if modality_name in ['accelerometer', 'gyroscope']:
                        try:
                            file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
                            if file_data.shape[1] > 4:
                                cols = file_data.shape[1] - 3
                                file_data = file_data.iloc[:, 3:]
                            else:
                                cols = 3
                            
                            if file_data.shape[0] > 2:
                                data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
                            else:
                                data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
                            
                            trial_data[modality_name] = data
                        except:
                            continue
                
                if 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                    acc_data = trial_data['accelerometer']
                    gyro_data = trial_data['gyroscope']
                    
                    aligned_acc, aligned_gyro, timestamps = align_sensor_data(acc_data, gyro_data)
                    
                    if len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                        result = process_imu_data(
                            aligned_acc, aligned_gyro, timestamps,
                            filter_type=filter_type, 
                            trial_id=trial_id, 
                            reset_filter=True
                        )
                        
                        output_file = os.path.join(subject_dir, f"{trial_id}.npz")
                        np.savez_compressed(
                            output_file,
                            accelerometer=aligned_acc,
                            gyroscope=aligned_gyro,
                            quaternion=result['quaternion'],
                            timestamps=timestamps,
                            filter_type=filter_type
                        )
            except Exception as e:
                logger.error(f"Error processing trial {trial_id}: {str(e)}")
                continue
    
    logger.info(f"Preprocessing complete for all subjects with {filter_type} filter")
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
        
        for dir_name in ["accelerometer", "gyroscope", "skeleton"]:
            os.makedirs(os.path.join(self.aligned_data_dir, dir_name), exist_ok=True)
        
        if fusion_options:
            self.fusion_enabled = fusion_options.get('enabled', False)
            self.filter_type = fusion_options.get('filter_type', 'madgwick')
            self.use_stateful = fusion_options.get('process_per_window', False) == False
        else:
            self.fusion_enabled = False
            self.filter_type = 'madgwick'
            self.use_stateful = False
    
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

    def process(self, data, label, fuse=False, filter_type='madgwick', visualize=False, trial_id=None):
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
                    
                    if self.use_stateful and trial_id is not None:
                        fusion_result = stateful_process_imu_data(
                            acc_data=processed_data['accelerometer'],
                            gyro_data=processed_data['gyroscope'],
                            timestamps=timestamps,
                            filter_type=filter_type,
                            trial_id=trial_id,
                            reset_filter=True,
                            return_features=False
                        )
                    else:
                        fusion_result = process_imu_data(
                            acc_data=processed_data['accelerometer'],
                            gyro_data=processed_data['gyroscope'],
                            timestamps=timestamps,
                            filter_type=filter_type,
                            return_features=False
                        )
                    
                    processed_data.update(fusion_result)
                
                except Exception as e:
                    processed_data['quaternion'] = np.zeros((self.max_length, 4))
            
            if 'quaternion' not in processed_data:
                processed_data['quaternion'] = np.zeros((self.max_length, 4))
            
            return processed_data
        
        else:
            sqrt_sum = np.sqrt(np.sum(data['accelerometer']**2, axis=1))
            
            if label == 1:
                peaks, _ = find_peaks(sqrt_sum, height=12, distance=10)
            else:
                peaks, _ = find_peaks(sqrt_sum, height=10, distance=20)
            
            processed_data = selective_sliding_window(
                data=data,
                window_size=self.max_length,
                peaks=peaks,
                label=label,
                fuse=fuse,
                filter_type=filter_type,
                trial_id=trial_id,
                use_stateful=self.use_stateful
            )
            
            return processed_data

    def _add_trial_data(self, trial_data):
        for modality, modality_data in trial_data.items():
            if isinstance(modality_data, np.ndarray) and modality_data.size > 0:
                if modality not in self.data:
                    self.data[modality] = []
                self.data[modality].append(modality_data)

    def _len_check(self, d):
        return all(len(v) > 0 for v in d.values())

    def _process_trial(self, trial, label, fuse, filter_type, visualize, save_aligned=False):
        try:
            trial_id = f"S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}"
            
            if fuse and filter_type != 'none':
                cache_dir = self.fusion_options.get('cache_dir', 'processed_data')
                cache_dir = os.path.join(cache_dir, filter_type)
                cached_data = load_cached_processed(trial.subject_id, trial.action_id, filter_type, cache_dir)
                
                if cached_data is not None:
                    processed_data = {}
                    for modality, data in cached_data.items():
                        if modality != 'timestamps' and data is not None and len(data) > 0:
                            processed_data[modality] = pad_sequence_numpy(
                                sequence=data,
                                max_sequence_length=self.max_length,
                                input_shape=data.shape
                            )
                    
                    processed_data['labels'] = np.array([label])
                    return processed_data
            
            trial_data = {}
            
            if not ('accelerometer' in trial.files and 'gyroscope' in trial.files):
                return None
                
            for modality_name, file_path in trial.files.items():
                try:
                    unimodal_data = self.load_file(file_path)
                    trial_data[modality_name] = unimodal_data
                except Exception as e:
                    return None
            
            trial_data = align_sequence(trial_data)
            
            if not trial_data or not ('accelerometer' in trial_data and 'gyroscope' in trial_data):
                return None
            
            if save_aligned:
                aligned_acc = trial_data.get('accelerometer')
                aligned_gyro = trial_data.get('gyroscope')
                aligned_skl = trial_data.get('skeleton')
                aligned_timestamps = trial_data.get('aligned_timestamps')
                
                if aligned_acc is not None and aligned_gyro is not None:
                    save_aligned_sensor_data(
                        trial.subject_id,
                        trial.action_id,
                        trial.sequence_number,
                        aligned_acc,
                        aligned_gyro,
                        aligned_skl,
                        aligned_timestamps if aligned_timestamps is not None else np.arange(len(aligned_acc))
                    )
            
            processed_data = self.process(
                trial_data, 
                label, 
                fuse, 
                filter_type, 
                visualize,
                trial_id
            )
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Trial processing failed: {str(e)}")
            return None

    def make_dataset(self, subjects: List[int], fuse: bool, filter_type: str = 'madgwick',
                    visualize: bool = False, save_aligned: bool = False):
        self.data = {}
        self.fuse = fuse
        
        if hasattr(self, 'fusion_options'):
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)
            
        from concurrent.futures import ThreadPoolExecutor, as_completed
        future_to_trial = {}
        
        with ThreadPoolExecutor(max_workers=min(8, len(self.dataset.matched_trials))) as executor:
            count = 0
            processed_count = 0
            skipped_count = 0
            
            for trial in self.dataset.matched_trials:
                if trial.subject_id not in subjects:
                    continue
                
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                
                future = executor.submit(
                    self._process_trial,
                    trial,
                    label,
                    fuse,
                    filter_type,
                    False,
                    save_aligned
                )
                future_to_trial[future] = trial
            
            from tqdm import tqdm
            for future in tqdm(as_completed(future_to_trial), total=len(future_to_trial), desc="Processing trials"):
                trial = future_to_trial[future]
                count += 1
                
                try:
                    trial_data = future.result()
                    if trial_data is not None and self._len_check(trial_data):
                        self._add_trial_data(trial_data)
                        processed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    skipped_count += 1
        
        for key in self.data:
            values = self.data[key]
            if all(isinstance(x, np.ndarray) for x in values):
                try:
                    self.data[key] = np.concatenate(values, axis=0)
                except Exception as e:
                    logger.error(f"Error concatenating {key} data: {str(e)}")
            else:
                logger.warning(f"Cannot concatenate {key} data - mixed types")
        
        if 'quaternion' not in self.data and 'accelerometer' in self.data:
            acc_shape = self.data['accelerometer'].shape
            self.data['quaternion'] = np.zeros((acc_shape[0], acc_shape[1], 4))
        
        global filter_cache
        filter_cache.clear()

    def normalization(self) -> Dict[str, np.ndarray]:
        from sklearn.preprocessing import StandardScaler
        
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion', 'linear_acceleration'] and len(value.shape) >= 2:
                        num_samples, length = value.shape[:2]
                        reshaped_data = value.reshape(num_samples * length, -1)
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                        self.data[key] = norm_data.reshape(value.shape)
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        self.data[key] = StandardScaler().fit_transform(value)
                except Exception as e:
                    logger.error(f"Error normalizing {key} data: {str(e)}")
        
        return self.data
