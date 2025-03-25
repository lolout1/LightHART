from typing import Any, List
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np 
import logging
from scipy.signal import find_peaks
from utils.imu_fusion import process_imu_data, create_filter_id

logger = logging.getLogger("processor")

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
            logger.warning(f"File has fewer columns than expected: {file_data.shape[1]} < {cols}")
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

def matloader(file_path: str, **kwargs):
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

def ensure_3d_vector(v, default_value=0.0):
    if v is None:
        return np.array([default_value, default_value, default_value])
    
    v_array = np.asarray(v)
    if v_array.size == 0:
        return np.array([default_value, default_value, default_value])
    
    if v_array.shape[-1] == 3:
        return v_array
    
    if v_array.ndim == 1:
        if v_array.size == 1:
            return np.array([v_array[0], default_value, default_value])
        if v_array.size == 2:
            return np.array([v_array[0], v_array[1], default_value])
        return v_array[:3]
    
    if v_array.ndim == 2 and v_array.shape[0] == 1:
        if v_array.shape[1] == 1:
            return np.array([v_array[0, 0], default_value, default_value])
        if v_array.shape[1] == 2:
            return np.array([v_array[0, 0], v_array[0, 1], default_value])
        return v_array[0, :3]
    
    return np.array([default_value, default_value, default_value])

def avg_pool(sequence: np.array, window_size: int = 5, stride: int = 1, 
             max_length: int = 512, shape: int = None) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    stride = ((sequence.shape[2] // max_length) + 1 if max_length < sequence.shape[2] else 1)
    sequence = F.avg_pool1d(sequence, kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1, 0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, input_shape: np.array) -> np.ndarray:
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def fixed_sliding_window(data: np.ndarray, window_size: int = 64, stride: int = 10,
                      base_filter_id: str = None, filter_type: str = 'madgwick', 
                      stateful: bool = True) -> np.array:
    from utils.imu_fusion import process_windows_with_filter
    
    if len(data.shape) == 1:
        padded_data = np.zeros((len(data), 3))
        padded_data[:, 0] = data
        data = padded_data
    elif data.shape[1] < 3:
        padded_data = np.zeros((data.shape[0], 3))
        padded_data[:, :data.shape[1]] = data
        data = padded_data
    
    # Ensure data is long enough for at least one window
    if len(data) < window_size:
        window_data = np.zeros((1, window_size, data.shape[1]))
        window_data[0, :len(data)] = data
        return window_data
    
    # Calculate how many windows we'll create
    num_windows = max(1, (len(data) - window_size) // stride + 1)
    windows = []
    
    # Extract windows with fixed stride
    for i in range(num_windows):
        start = i * stride
        end = min(start + window_size, len(data))
        
        # Skip small windows at the end
        if end - start < window_size // 2:
            continue
            
        window_data = np.zeros((window_size, data.shape[1]))
        actual_length = end - start
        window_data[:actual_length] = data[start:end]
        windows.append(window_data)
    
    if not windows:
        return np.zeros((0, window_size, data.shape[1]))
    
    windows_array = np.array(windows)
    
    if base_filter_id is not None:
        gyro_windows = np.zeros_like(windows_array)
        window_data = {
            'accelerometer': windows_array,
            'gyroscope': gyro_windows
        }
        processed = process_windows_with_filter(
            window_data, 
            filter_type=filter_type,
            base_filter_id=base_filter_id,
            reset_per_window=not stateful
        )
        if 'quaternion' in processed:
            windows_array = np.concatenate([windows_array, processed['quaternion']], axis=2)
    
    return windows_array

def peak_based_window(data: np.ndarray, window_size: int = 64, 
                    height: float = 1.3, distance: int = 75,
                    base_filter_id: str = None, filter_type: str = 'madgwick', 
                    stateful: bool = True) -> np.array:
    from utils.imu_fusion import process_windows_with_filter
    
    if len(data.shape) == 1:
        padded_data = np.zeros((len(data), 3))
        padded_data[:, 0] = data
        data = padded_data
    elif data.shape[1] < 3:
        padded_data = np.zeros((data.shape[0], 3))
        padded_data[:, :data.shape[1]] = data
        data = padded_data
    
    # Get acceleration magnitude
    sqrt_sum = np.sqrt(np.sum(data**2, axis=1))
    
    # Find peaks with consistent parameters for both falls and ADLs
    peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
    
    # If no peaks found, use middle of data or highest acceleration point
    if len(peaks) == 0:
        if len(sqrt_sum) > window_size:
            max_idx = np.argmax(sqrt_sum)
            peaks = np.array([max_idx])
        else:
            peaks = np.array([len(sqrt_sum) // 2])
    
    windows = []
    
    for peak in peaks:
        start = max(0, peak - window_size // 2)
        end = min(len(data), start + window_size)
        
        if end - start < window_size // 2:
            continue
            
        window_data = np.zeros((window_size, data.shape[1]))
        actual_length = end - start
        window_data[:actual_length] = data[start:end]
        windows.append(window_data)
    
    if not windows:
        return np.zeros((0, window_size, data.shape[1]))
    
    windows_array = np.array(windows)
    
    if base_filter_id is not None:
        gyro_windows = np.zeros_like(windows_array)
        window_data = {
            'accelerometer': windows_array,
            'gyroscope': gyro_windows
        }
        processed = process_windows_with_filter(
            window_data, 
            filter_type=filter_type,
            base_filter_id=base_filter_id,
            reset_per_window=not stateful
        )
        if 'quaternion' in processed:
            windows_array = np.concatenate([windows_array, processed['quaternion']], axis=2)
    
    return windows_array

class Processor(ABC):
    def __init__(self, file_path: str, mode: str, max_length: int = 64, label: int = None, **kwargs):
        assert mode in ['sliding_window', 'avg_pool'], f'Processing mode: {mode} is undefined'
        self.label = label 
        self.mode = mode
        self.max_length = max_length
        self.data = []
        self.file_path = file_path
        self.input_shape = []
        self.kwargs = kwargs
        self.stateful = kwargs.get('stateful', True)
        self.filter_type = kwargs.get('filter_type', 'madgwick')
        self.window_stride = kwargs.get('window_stride', 10)
        self.use_fixed_windows = kwargs.get('use_fixed_windows', True)
        
        parts = file_path.split('/')[-1].split('.')[0].split('_')
        if len(parts) >= 3:
            parts = parts[0].split('A')
            if len(parts) >= 2:
                try:
                    self.subject_id = int(parts[0][1:])
                    self.action_id = int(parts[1])
                    self.base_filter_id = create_filter_id(self.subject_id, self.action_id, filter_type=self.filter_type)
                except:
                    self.base_filter_id = None
            else:
                self.base_filter_id = None
        else:
            self.base_filter_id = None

    def set_input_shape(self, sequence: np.ndarray) -> List[int]:
        self.input_shape = sequence.shape

    def _import_loader(self, file_path: str):
        file_type = file_path.split('.')[-1]
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        return LOADER_MAP[file_type]
    
    def load_file(self, file_path: str):
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        self.set_input_shape(data)
        return data

    def process(self, data):
        if self.mode == 'avg_pool':
            if len(data.shape) == 1 or (len(data.shape) > 1 and data.shape[1] < 3):
                valid_data = []
                for i in range(len(data)):
                    try:
                        valid_data.append(ensure_3d_vector(data[i]))
                    except:
                        valid_data.append(np.zeros(3))
                data = np.array(valid_data)
                self.input_shape = data.shape
            
            data = pad_sequence_numpy(
                sequence=data, 
                max_sequence_length=self.max_length,
                input_shape=self.input_shape
            )
        else:  # sliding_window
            if self.use_fixed_windows:
                # Use consistent fixed stride windows for both training and real-time inference
                data = fixed_sliding_window(
                    data,
                    window_size=self.max_length,
                    stride=self.window_stride,
                    base_filter_id=self.base_filter_id,
                    filter_type=self.filter_type,
                    stateful=self.stateful
                )
            else:
                # Use peak-based windowing with consistent parameters
                height = 1.3  # Compromise between 1.4 (fall) and 1.2 (ADL)
                distance = 75  # Compromise between 50 (fall) and 100 (ADL)
                
                data = peak_based_window(
                    data,
                    window_size=self.max_length,
                    height=height,
                    distance=distance,
                    base_filter_id=self.base_filter_id,
                    filter_type=self.filter_type,
                    stateful=self.stateful
                )
        return data
