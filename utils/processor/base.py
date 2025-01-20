from typing import Any, List, Dict, Optional, Tuple
from abc import ABC
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
import warnings

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """Function to filter noise."""
    if data is None or len(data) == 0:
        return data
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    required_padlen = 3 * max(len(a), len(b))
    if data.shape[0] <= required_padlen:
        padlen = data.shape[0] - 1
        if padlen < 1:
            return data
        return filtfilt(b, a, data, axis=0, padlen=padlen)
    else:
        return filtfilt(b, a, data, axis=0)

def sliding_window(data: np.ndarray,
                   clearing_time_index: int,
                   max_time: int,
                   sub_window_size: int = 128,
                   stride_size: int = 32) -> np.ndarray:
    """
    Creates sliding windows [n_windows, 96, n_feats].
    Detailed logging of final #windows is done in Processor.process.
    """
    if len(data) < sub_window_size:
        # We do NOT do partial windows here. Return single padded or skip.
        # Actual logic is in Processor.process.
        pad_length = sub_window_size - len(data)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        padded_data = np.pad(data,
                             ((pad_left, pad_right), (0, 0)),
                             mode='edge')
        return np.expand_dims(padded_data, 0)
    
    if max_time > data.shape[0] - sub_window_size:
        max_time = data.shape[0] - sub_window_size

    start = clearing_time_index - sub_window_size + 1
    window_starts = np.arange(0, max_time, stride_size)
    final_start = data.shape[0] - sub_window_size
    if final_start > window_starts[-1]:
        window_starts = np.append(window_starts, final_start)
    
    sub_windows = (
        start +
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(window_starts, 0).T
    )
    sub_windows = np.clip(sub_windows, 0, len(data) - 1)
    return data[sub_windows]

def avg_pool(sequence: np.ndarray,
             window_size: int = 5,
             stride: int = 1,
             max_length: int = 512,
             shape: Optional[Tuple] = None) -> np.ndarray:
    """Average pooling with shape preservation."""
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    
    if max_length < sequence.shape[2]:
        stride = (sequence.shape[2] // max_length) + 1
    
    sequence = F.avg_pool1d(sequence, kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1, 0)
    return sequence.reshape(-1, *shape[1:])

def pad_sequence_numpy(sequence: np.ndarray,
                       max_sequence_length: int,
                       input_shape: np.ndarray) -> np.ndarray:
    """Pad sequence to uniform length using average pooling."""
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled = avg_pool(sequence, max_length=max_sequence_length, shape=input_shape)
    padded = np.zeros(shape, sequence.dtype)
    padded[:len(pooled)] = pooled
    return padded

def csvloader(file_path: str, **kwargs) -> Optional[np.ndarray]:
    """Loads CSV data."""
    file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
    num_col = file_data.shape[1]
    num_extra_col = num_col % 3
    cols_to_select = num_col - num_extra_col
    activity_data = file_data.iloc[2:, -cols_to_select:].to_numpy(dtype=np.float32)
    return activity_data

def matloader(file_path: str, **kwargs) -> Optional[np.ndarray]:
    """Load MATLAB files with key validation."""
    key = kwargs.get('key', None)
    if not key:
        raise ValueError("Key required for MATLAB file loading")
    try:
        data = loadmat(file_path)[key]
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

class Processor(ABC):
    """Base processor for sensor data."""
    def __init__(self,
                 file_path: str,
                 mode: str,
                 max_length: int,
                 window_size: int = 128,
                 stride_size: int = 10,
                 **kwargs):
        assert mode in ['sliding_window', 'avg_pool']
        self.mode = mode
        self.max_length = max_length
        self.file_path = file_path
        self.window_size = window_size
        self.stride_size = stride_size
        self.kwargs = kwargs
        self.input_shape = None

    def process(self) -> Optional[np.ndarray]:
        """
        Process data, ignoring or padding sequences < 128:
        - if length < 96 => log & skip 
        - if 96 <= length < 128 => pad to 128 (repeat last val)
        - else => normal sliding window
        """
        data = self._load_data()
        if data is None:
            return None

        raw_len = len(data)
        print(f"[Processor] File: {self.file_path}, Number of raw samples: {raw_len}")

        if raw_len < 96:
            print(f"[Processor] Skipping file {self.file_path} because it has < 96 samples.")
            return None
        elif raw_len < self.window_size:
            # pad to 128
            pad_needed = self.window_size - raw_len
            print(f"[Processor] Padding file {self.file_path} from {raw_len} to 128 using last val.")
            last_vals = data[-1:].repeat(pad_needed, axis=0)
            data = np.concatenate([data, last_vals], axis=0)

        self._set_input_shape(data)
        
        try:
            if self.mode == 'sliding_window':
                # sub_window_size=128
                out_data = sliding_window(
                    data=data,
                    clearing_time_index=self.window_size - 1,
                    max_time=len(data),
                    sub_window_size=self.window_size,
                    stride_size=self.stride_size
                )
                print(f"[Processor] Created {out_data.shape[0]} windows from file {self.file_path}.")
                return out_data
            else:
                out_data = pad_sequence_numpy(
                    sequence=data,
                    max_sequence_length=self.max_length,
                    input_shape=self.input_shape
                )
                print(f"[Processor] Single sample shape {out_data.shape} from file {self.file_path}.")
                return out_data
        except Exception as e:
            print(f"Processing error in file {self.file_path}: {str(e)}")
            return None

    def _set_input_shape(self, sequence: np.ndarray) -> None:
        """Store sequence shape."""
        self.input_shape = sequence.shape

    def _load_data(self) -> Optional[np.ndarray]:
        """Load data using mapped loader."""
        ext = self.file_path.split('.')[-1]
        if ext not in LOADER_MAP:
            raise ValueError(f"Unsupported file type: {ext}")
        return LOADER_MAP[ext](self.file_path, **self.kwargs)
