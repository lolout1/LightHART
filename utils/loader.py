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

from utils.imu_fusion import (
    align_sensor_data, save_aligned_data, process_imu_data, 
    process_windows_with_filter, create_filter_id, clear_filters
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("loader")

MAX_THREADS = 30
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(MAX_THREADS)

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

def avg_pool(sequence, window_size=5, stride=1, max_length=512, shape=None):
    import torch.nn.functional as F
    import torch
    
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
    
    if max_length < sequence_tensor.shape[2]:
        stride = ((sequence_tensor.shape[2] // max_length) + 1)
    else:
        stride = 1
        
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
    
    if acc_data is None or gyro_data is None:
        return data
    
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
                aligned_skeleton[:, joint, coord] = np.interp(
                    common_times,
                    skeleton_times,
                    skeleton_data[:, joint, coord]
                )
        
        aligned_data['skeleton'] = aligned_skeleton
    
    return aligned_data


def estimate_orientation_from_acc(acc_data: np.ndarray, window_size: int = 64) -> np.ndarray:
    """
    Estimate quaternion orientation from accelerometer data only (gravity direction).
    
    When gyroscope data is unavailable, this function creates orientation estimates
    based solely on accelerometer gravity direction.
    
    Args:
        acc_data: Accelerometer data array
        window_size: Size of output window
        
    Returns:
        Array of quaternions representing orientation
    """
    quaternions = np.zeros((window_size, 4))
    quaternions[:, 0] = 1.0  # Initialize with identity quaternion
    
    # Process each sample
    for i in range(min(len(acc_data), window_size)):
        acc = acc_data[i]
        acc_norm = np.linalg.norm(acc)
        
        # Skip if acceleration is too small
        if acc_norm < 1e-6:
            continue
            
        # Normalize to get direction
        acc_normalized = acc / acc_norm
        
        # Reference gravity vector (pointing down)
        gravity = np.array([0, 0, 1])
        
        # Get rotation axis (cross product of vectors)
        rotation_axis = np.cross(gravity, acc_normalized)
        axis_norm = np.linalg.norm(rotation_axis)
        
        # If vectors are aligned (or opposite), use default quaternion
        if axis_norm < 1e-6:
            if acc_normalized[2] < 0:  # If pointing opposite to gravity
                quaternions[i] = np.array([0, 1, 0, 0])  # 180° rotation around X
            continue
            
        # Normalize rotation axis
        rotation_axis /= axis_norm
        
        # Calculate rotation angle
        angle = np.arccos(np.clip(np.dot(gravity, acc_normalized), -1.0, 1.0))
        
        # Create quaternion from axis-angle
        quaternions[i, 0] = np.cos(angle / 2)
        quaternions[i, 1:4] = rotation_axis * np.sin(angle / 2)
        
        # Normalize quaternion
        q_norm = np.linalg.norm(quaternions[i])
        if q_norm > 1e-10:
            quaternions[i] /= q_norm
    
    return quaternions

def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int, peaks: Union[List[int], np.ndarray],
                           label: int, fuse: bool, filter_type: str = 'madgwick', 
                           trial_id=None, use_stateful=True, stride: int = 10,
                           filter_params: Optional[Dict] = None, 
                           base_filter_id: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Process selected windows using sliding window approach with peak detection.
    
    Args:
        data: Dictionary containing sensor data by modality
        window_size: Size of each window
        peaks: List of peak indices to center windows on
        label: Activity label
        fuse: Whether to apply sensor fusion
        filter_type: Filter type ('madgwick', 'kalman', 'ekf', 'ukf')
        trial_id: Trial identifier for stateful filtering
        use_stateful: Whether to maintain filter state between windows
        stride: Stride between windows
        filter_params: Parameters for the specific filter
        base_filter_id: Base identifier for filter instance management
        
    Returns:
        Dictionary of processed windowed data
    """
    start_time = time.time()
    logger.info(f"Creating {len(peaks)} windows with {filter_type} fusion (stride={stride})")
    
    # Handle None values
    if filter_params is None:
        filter_params = {}
    
    if base_filter_id is None and trial_id is not None:
        base_filter_id = f"trial_{trial_id}"
    elif base_filter_id is None:
        base_filter_id = f"label_{label}_type_{filter_type}"
    
    windowed_data = defaultdict(list)
    
    # Check available modalities
    has_acc = 'accelerometer' in data and data['accelerometer'] is not None and len(data['accelerometer']) > 0
    has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and len(data['gyroscope']) > 0
    
    # Require accelerometer data
    if not has_acc:
        logger.error("Missing accelerometer data - required for processing")
        return {'labels': np.array([label])}
    
    # Handle missing gyroscope gracefully
    if fuse and not has_gyro:
        logger.warning("Gyroscope data unavailable - will perform acc-only processing")
        fuse = False
    
    windows_created = 0
    
    for window_idx, peak in enumerate(tqdm(peaks, desc="Processing windows")):
        # Apply stride for window positioning
        start = max(0, peak - window_size // 2)
        end = min(len(data['accelerometer']), start + window_size)
        
        # Skip if window is too small
        if end - start < window_size // 2:
            logger.debug(f"Skipping window at peak {peak}: too small ({end-start} < {window_size//2})")
            continue
        
        try:
            window_data = {}
            
            # Process accelerometer data
            acc_window = data['accelerometer'][start:min(end, len(data['accelerometer'])), :]
            if acc_window.shape[0] < window_size:
                padded_acc = np.zeros((window_size, acc_window.shape[1]), dtype=acc_window.dtype)
                padded_acc[:acc_window.shape[0]] = acc_window
                acc_window = padded_acc
            window_data['accelerometer'] = acc_window
            
            # Process gyroscope if available
            if has_gyro:
                gyro_window = data['gyroscope'][start:min(end, len(data['gyroscope'])), :]
                if gyro_window.shape[0] < window_size:
                    padded_gyro = np.zeros((window_size, gyro_window.shape[1]), dtype=gyro_window.dtype)
                    padded_gyro[:gyro_window.shape[0]] = gyro_window
                    gyro_window = padded_gyro
                window_data['gyroscope'] = gyro_window
            else:
                # Create zero gyroscope data if fusion is requested
                window_data['gyroscope'] = np.zeros((window_size, 3)) if fuse else None
            
            # Process timestamps if available
            if 'aligned_timestamps' in data:
                timestamps = data['aligned_timestamps'][start:min(end, len(data['aligned_timestamps']))]
                if isinstance(timestamps, np.ndarray):
                    if len(timestamps.shape) > 1:
                        timestamps = timestamps[:, 0] if timestamps.shape[1] > 0 else None
                    if len(timestamps) < window_size:
                        padded_ts = np.zeros(window_size, dtype=timestamps.dtype) if len(timestamps) > 0 else np.zeros(window_size)
                        padded_ts[:len(timestamps)] = timestamps
                        timestamps = padded_ts
                window_data['aligned_timestamps'] = timestamps
            
            # Apply sensor fusion
            if fuse and has_gyro:
                try:
                    # Create unique filter ID combining base ID and window info
                    filter_instance_id = f"{base_filter_id}_{window_idx}" if use_stateful else None
                    
                    if use_stateful and filter_instance_id is not None:
                        fusion_results = stateful_process_imu_data(
                            acc_data=window_data['accelerometer'],
                            gyro_data=window_data['gyroscope'],
                            timestamps=window_data.get('aligned_timestamps', None),
                            filter_type=filter_type,
                            filter_params=filter_params,
                            trial_id=filter_instance_id,
                            reset_filter=(window_idx == 0),
                            return_features=False
                        )
                    else:
                        fusion_results = process_imu_data(
                            acc_data=window_data['accelerometer'],
                            gyro_data=window_data['gyroscope'],
                            timestamps=window_data.get('aligned_timestamps', None),
                            filter_type=filter_type,
                            filter_params=filter_params,
                            return_features=False
                        )
                    window_data['quaternion'] = fusion_results.get('quaternion', np.zeros((window_size, 4)))
                except Exception as e:
                    logger.error(f"Error in fusion processing: {str(e)}")
                    window_data['quaternion'] = np.zeros((window_size, 4))
            else:
                # For acc-only, estimate orientation from gravity direction
                try:
                    window_data['quaternion'] = estimate_orientation_from_acc(
                        window_data['accelerometer'], window_size=window_size
                    )
                except Exception as e:
                    logger.error(f"Error in acc-only orientation estimation: {str(e)}")
                    window_data['quaternion'] = np.zeros((window_size, 4))
            
            # Add valid data to windows collection
            for modality, modality_window in window_data.items():
                if modality_window is not None:
                    windowed_data[modality].append(modality_window)
            
            windows_created += 1
            
            # Apply stride - skip next (stride-1) peaks
            if stride > 1 and window_idx < len(peaks) - 1:
                next_idx = window_idx + 1
                while next_idx < len(peaks) and peaks[next_idx] < peak + stride:
                    next_idx += 1
                
        except Exception as e:
            logger.error(f"Error processing window at peak {peak}: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Handle empty results
    if windows_created == 0:
        logger.warning(f"No valid windows created for label {label}")
        return {'labels': np.array([label])}
    
    # Convert window lists to arrays
    for modality in windowed_data:
        if modality != 'labels' and len(windowed_data[modality]) > 0:
            try:
                windowed_data[modality] = np.array(windowed_data[modality])
                logger.debug(f"Created {modality} windows with shape {windowed_data[modality].shape}")
            except Exception as e:
                logger.error(f"Error converting {modality} to array: {str(e)}")
                if modality == 'quaternion':
                    windowed_data[modality] = np.zeros((windows_created, window_size, 4))
    
    # Create labels array
    windowed_data['labels'] = np.repeat(label, windows_created)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Created {windows_created} windows in {elapsed_time:.2f}s")
    
    return windowed_data

class ModalityFile:
    def __init__(self, subject_id, action_id, sequence_number, file_path):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

class Modality:
    def __init__(self, name):
        self.name = name
        self.files = []
    
    def add_file(self, subject_id, action_id, sequence_number, file_path):
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)

class MatchedTrial:
    def __init__(self, subject_id, action_id, sequence_number):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files = {}
    
    def add_file(self, modality_name, file_path):
        self.files[modality_name] = file_path

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
            self.stateful = not fusion_options.get('reset_per_window', False)
            self.use_fixed_windows = fusion_options.get('use_fixed_windows', True)
            self.window_stride = fusion_options.get('window_stride', 10)
            self.filter_params = fusion_options
            logger.info(f"Fusion options: enabled={self.fusion_enabled}, filter_type={self.filter_type}, stateful={self.stateful}")
        else:
            self.fusion_enabled = False
            self.filter_type = 'madgwick'
            self.stateful = True
            self.use_fixed_windows = True
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
                    
                    fusion_result = process_imu_data(
                        acc_data=processed_data['accelerometer'],
                        gyro_data=processed_data['gyroscope'],
                        timestamps=timestamps,
                        filter_type=filter_type,
                        filter_id=filter_id,
                        reset_filter=not self.stateful,
                        return_features=False
                    )
                    
                    processed_data.update(fusion_result)
                except Exception as e:
                    logger.error(f"Fusion processing failed: {str(e)}")
                    processed_data['quaternion'] = np.zeros((self.max_length, 4))
            
            if 'quaternion' not in processed_data:
                processed_data['quaternion'] = np.zeros((self.max_length, 4))
            
            return processed_data
        else:
            filter_id = create_filter_id(
                subject_id=trial_id[0] if trial_id else 0,
                action_id=trial_id[1] if trial_id else 0,
                trial_id=trial_id[2] if trial_id and len(trial_id) > 2 else None,
                filter_type=filter_type
            )
            
            processed_data = selective_sliding_window(
                data=data,
                window_size=self.max_length,
                stride=self.window_stride if hasattr(self, 'window_stride') else 10,
                label=label,
                fuse=fuse,
                filter_type=filter_type,
                filter_params=filter_params,
                base_filter_id=filter_id,
                stateful=self.stateful if hasattr(self, 'stateful') else True,
                use_fixed_windows=self.use_fixed_windows if hasattr(self, 'use_fixed_windows') else True
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
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue
            
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
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {str(e)}")
                    del self.data[key]
        
        if 'quaternion' not in self.data and 'accelerometer' in self.data:
            acc_shape = self.data['accelerometer'].shape
            self.data['quaternion'] = np.zeros((acc_shape[0], acc_shape[1], 4))
        
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
