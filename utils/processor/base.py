from typing import Any, List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    '''Function to fitter noise '''
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0) 

def sliding_window(data: np.ndarray, clearing_time_index: int, max_time: int,
                  sub_window_size: int = 64, stride_size: int = 10) -> np.ndarray:
    """
    Create sliding windows with improved handling of short sequences
    
    Args:
        data: Input sequence data array
        clearing_time_index: Initial offset (window_size - 1)
        max_time: Maximum sequence length to process
        sub_window_size: Size of each window
        stride_size: Steps between windows
        
    Returns:
        Windowed data array with shape (n_windows, sub_window_size, ...)
    """
    # Handle sequences shorter than window size
    if len(data) < sub_window_size:
        pad_length = sub_window_size - len(data)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        padded_data = np.pad(data, 
                            ((pad_left, pad_right), (0, 0)), 
                            mode='reflect')
        return np.expand_dims(padded_data, 0)
    
    # Adjust max_time if needed
    if max_time > data.shape[0] - sub_window_size:
        max_time = data.shape[0] - sub_window_size
    
    # Calculate window starts
    start = clearing_time_index - sub_window_size + 1
    window_starts = np.arange(0, max_time, stride_size)
    
    # Add final window if needed
    final_start = data.shape[0] - sub_window_size
    if final_start > window_starts[-1]:
        window_starts = np.append(window_starts, final_start)
    
    # Create window indices
    sub_windows = (
        start +
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(window_starts, 0).T
    )
    
    # Handle any out-of-bounds indices
    sub_windows = np.clip(sub_windows, 0, len(data) - 1)
    
    return data[sub_windows]

def avg_pool(sequence: np.ndarray, window_size: int = 5, stride: int = 1,
            max_length: int = 512, shape: Optional[Tuple] = None) -> np.ndarray:
    """Average pooling with shape preservation"""
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    
    # Adjust stride for target length
    if max_length < sequence.shape[2]:
        stride = (sequence.shape[2] // max_length) + 1
    
    sequence = F.avg_pool1d(sequence, kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1, 0)
    
    return sequence.reshape(-1, *shape[1:])

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int,
                      input_shape: np.ndarray) -> np.ndarray:
    """Pad sequence to uniform length using average pooling"""
    shape = list(input_shape)
    shape[0] = max_sequence_length
    
    # Pool sequence if longer than max length
    pooled = avg_pool(sequence, max_length=max_sequence_length, shape=input_shape)
    
    # Create zero-padded output
    padded = np.zeros(shape, sequence.dtype)
    padded[:len(pooled)] = pooled
    
    return padded

def csvloader(file_path: str, **kwargs) -> Optional[np.ndarray]:
    '''
    Loads csv data
    '''
    file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
    num_col = file_data.shape[1]
    num_extra_col = num_col % 3
    cols_to_select = num_col - num_extra_col
    activity_data = file_data.iloc[2:, -cols_to_select:].to_numpy(dtype=np.float32)
    return activity_data

def matloader(file_path: str, **kwargs) -> Optional[np.ndarray]:
    """Load MATLAB files with key validation"""
    key = kwargs.get('key', None)
    if not key:
        raise ValueError("Key required for MATLAB file loading")
    
    try:
        data = loadmat(file_path)[key]
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

# File extension to loader mapping
LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

class Processor(ABC):
    """Base processor for sensor data"""
    def __init__(self, file_path: str, mode: str, max_length: int,
                 window_size: int = 64, stride_size: int = 10, **kwargs):
        assert mode in ['sliding_window', 'avg_pool']
        self.mode = mode
        self.max_length = max_length
        self.file_path = file_path
        self.window_size = window_size
        self.stride_size = stride_size
        self.kwargs = kwargs
        self.input_shape = None

    def process(self) -> Optional[np.ndarray]:
        """Process data with improved error handling"""
        data = self._load_data()
        if data is None:
            return None
            
        self._set_input_shape(data)
        
        try:
            if self.mode == 'sliding_window':
                data = sliding_window(
                    data=data,
                    clearing_time_index=self.window_size-1,
                    max_time=len(data),
                    sub_window_size=self.window_size,
                    stride_size=self.stride_size
                )
            else:
                data = pad_sequence_numpy(
                    sequence=data,
                    max_sequence_length=self.max_length,
                    input_shape=self.input_shape
                )
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return None
            
        return data

    def _set_input_shape(self, sequence: np.ndarray) -> None:
        """Store sequence shape"""
        self.input_shape = sequence.shape

    def _load_data(self) -> Optional[np.ndarray]:
        """Load data using mapped loader"""
        ext = self.file_path.split('.')[-1]
        if ext not in LOADER_MAP:
            raise ValueError(f"Unsupported file type: {ext}")
            
        return LOADER_MAP[ext](self.file_path, **self.kwargs)