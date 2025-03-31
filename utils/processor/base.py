from typing import Any, List
from abc import ABC, abstractmethod
from scipy.io import loadmat
import pandas as pd
import numpy as np 
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
import logging

logger = logging.getLogger("processor")

# Add these functions to the top of utils/processor/base.py

def fix_monotonic_timestamps(timestamps):
    """Ensure timestamps are strictly increasing by correcting duplicates and inversions"""
    if len(timestamps) <= 1:
        return timestamps
        
    fixed_timestamps = np.array(timestamps, dtype=np.float64)
    min_increment = 1e-6  # microsecond increment for duplicates
    
    # Identify duplicates and backward jumps
    for i in range(1, len(fixed_timestamps)):
        if fixed_timestamps[i] <= fixed_timestamps[i-1]:
            # Fix timestamp by adding increment to previous timestamp
            fixed_timestamps[i] = fixed_timestamps[i-1] + min_increment
    
    return fixed_timestamps

def last_value_resample(data, timestamps, target_timestamps):
    """
    Resample data using last-value-forward approach (Android-compatible)
    This method only uses past information, suitable for real-time processing
    """
    if len(data) == 0 or len(timestamps) == 0:
        return np.zeros((len(target_timestamps), data.shape[1] if len(data.shape) > 1 else 1))
        
    # Ensure timestamps are monotonically increasing
    timestamps = fix_monotonic_timestamps(timestamps)
    
    # Convert to numpy arrays if they aren't already
    data = np.array(data)
    timestamps = np.array(timestamps)
    target_timestamps = np.array(target_timestamps)
    
    # Initialize output array
    if len(data.shape) > 1:
        resampled = np.zeros((len(target_timestamps), data.shape[1]))
    else:
        resampled = np.zeros(len(target_timestamps))
    
    # Find the first valid data point
    if target_timestamps[0] < timestamps[0]:
        # For timestamps before the first data point, use the first value
        idx = 0
    else:
        # Find the first data point that occurred before or at the first target timestamp
        idx = np.searchsorted(timestamps, target_timestamps[0], side='right') - 1
        idx = max(0, idx)
    
    # If the resampling target starts before data, use the first data point for initial values
    if idx == 0:
        if len(data.shape) > 1:
            resampled[0] = data[0]
        else:
            resampled[0] = data[0]
    
    # Resample each target timestamp
    last_idx = idx
    for i, target_time in enumerate(target_timestamps):
        # Find the last measurement that occurred before or at the target time
        while last_idx + 1 < len(timestamps) and timestamps[last_idx + 1] <= target_time:
            last_idx += 1
            
        # Use the last valid measurement
        if len(data.shape) > 1:
            resampled[i] = data[last_idx]
        else:
            resampled[i] = data[last_idx]
    
    return resampled

# Replace the selective_sliding_window function with this Android-compatible version
def selective_sliding_window(data: np.ndarray, length: int, window_size: int, stride_size: int, height: float, distance: int) -> np.array:
    try:
        # Ensure data is at least 2D with 3 columns
        if len(data.shape) == 1:
            padded_data = np.zeros((len(data), 3))
            padded_data[:, 0] = data
            data = padded_data
        elif data.shape[1] < 3:
            padded_data = np.zeros((data.shape[0], 3))
            padded_data[:, :data.shape[1]] = data
            data = padded_data
        
        # Get the magnitude of acceleration
        sqrt_sum = np.sqrt(np.sum(data**2, axis=1))
        
        # Find peaks - using a simpler peak detection suitable for Android real-time
        # This avoids the scipy dependency which might be challenging for Android
        peaks = []
        for i in range(distance, len(sqrt_sum) - distance):
            if sqrt_sum[i] > height:
                # Check if it's a local maximum
                local_max = True
                for j in range(i - distance, i):
                    if sqrt_sum[j] > sqrt_sum[i]:
                        local_max = False
                        break
                for j in range(i + 1, i + distance + 1):
                    if j < len(sqrt_sum) and sqrt_sum[j] > sqrt_sum[i]:
                        local_max = False
                        break
                
                if local_max:
                    peaks.append(i)
        
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
            
        return windows
    except Exception as e:
        logger.error(f"Error in selective_sliding_window: {str(e)}")
        return []

# Replace the sliding_window function with this Android-compatible version
def sliding_window(data: np.ndarray, window_size: int, stride_size: int) -> np.array:
    """
    Extract sliding windows from data
    
    Args:
        data: Input data [samples, features]
        window_size: Number of samples per window
        stride_size: Number of samples to advance between windows
        
    Returns:
        Array of windowed data [num_windows, window_size, features]
    """
    if len(data) < window_size:
        return np.array([])
        
    # Calculate number of windows
    num_windows = (len(data) - window_size) // stride_size + 1
    
    if num_windows <= 0:
        return np.array([data[:window_size]])
    
    # Extract windows
    windows = []
    for i in range(num_windows):
        start_idx = i * stride_size
        end_idx = start_idx + window_size
        
        if end_idx <= len(data):
            windows.append(data[start_idx:end_idx])
    
    return np.array(windows)

# Override the pad_sequence_numpy function
def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, input_shape: np.array) -> np.ndarray:
    """
    Pad or truncate sequence to fixed length without complex interpolation
    
    Args:
        sequence: Input sequence
        max_sequence_length: Target length
        input_shape: Original shape
        
    Returns:
        Padded/truncated sequence
    """
    shape = list(input_shape)
    shape[0] = max_sequence_length
    
    # Simple solution that's compatible with Android real-time processing
    new_sequence = np.zeros(shape, sequence.dtype)
    
    # If sequence is too long, truncate
    if len(sequence) >= max_sequence_length:
        new_sequence = sequence[:max_sequence_length]
    # If sequence is too short, pad with zeros
    else:
        new_sequence[:len(sequence)] = sequence
    
    return new_sequence

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
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    stride = ((sequence.shape[2] // max_length) + 1 if max_length < sequence.shape[2] else 1)
    sequence = F.avg_pool1d(sequence, kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1, 0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence

class Processor(ABC):
    def __init__(self, file_path: str, mode: str, max_length: str, label: int, **kwargs):
        assert mode in ['sliding_window', 'avg_pool'], f'Processing mode: {mode} is undefined'
        self.label = label 
        self.mode = mode
        self.max_length = max_length
        self.data = []
        self.file_path = file_path
        self.input_shape = []
        self.kwargs = kwargs

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
            # Ensure data has 3 columns for accelerometer/gyroscope
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
            if self.label == 1:  # Fall
                data = selective_sliding_window(
                    data, 
                    length=self.input_shape[0],
                    window_size=self.max_length, 
                    stride_size=10, 
                    height=1.4, 
                    distance=50
                )
            else:  # ADL
                data = selective_sliding_window(
                    data, 
                    length=self.input_shape[0],
                    window_size=self.max_length, 
                    stride_size=10, 
                    height=1.2, 
                    distance=100
                )
        return data
