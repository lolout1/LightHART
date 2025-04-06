from typing import Any, List
from abc import ABC, abstractmethod
from scipy.io import loadmat
import pandas as pd
import numpy as np 
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, butter, filtfilt

def csvloader(file_path: str, **kwargs):
    file_data = pd.read_csv(file_path, index_col=False, header = 0).dropna().bfill()
    num_col = file_data.shape[1]
    num_extra_col = num_col % 3
    cols_to_select = num_col - num_extra_col
    activity_data = file_data.iloc[2:, -3:].to_numpy(dtype=np.float32)
    return activity_data

def matloader(file_path: str, **kwargs):
    key = kwargs.get('key',None)
    assert key in ['d_iner' , 'd_skel'] , f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {
    'csv' : csvloader, 
    'mat' : matloader
}

def avg_pool(sequence : np.array, window_size : int = 5, stride :int =1, 
             max_length : int = 512 , shape : int = None) -> np.ndarray:
    shape = sequence.shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis = 0).transpose(0,2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    stride =  ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    sequence = F.avg_pool1d(sequence,kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1,0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, 
                       input_shape: np.array) -> np.ndarray:
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length = max_sequence_length, shape = input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def bandpass_filter(data, cutoff_low=0.5, cutoff_high=10.0, fs=30.0, order=2):
    nyq = 0.5 * fs
    low = cutoff_low / nyq
    high = cutoff_high / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = np.zeros_like(data)
    for axis in range(data.shape[1]):
        filtered_data[:, axis] = filtfilt(b, a, data[:, axis])
    return filtered_data

def improved_sliding_window(data, window_size=128, stride=32):
    if data is None or len(data) < window_size // 2:
        return []
    acc_magnitude = np.sqrt(np.sum(data**2, axis=1))
    height = np.mean(acc_magnitude) + 2.0 * np.std(acc_magnitude)
    distance = max(window_size // 4, 32)
    peaks, _ = find_peaks(acc_magnitude, height=height, distance=distance)
    if len(peaks) == 0:
        peaks = [np.argmax(acc_magnitude)]
    windows = []
    for peak in peaks:
        half_window = window_size // 2
        start = max(0, peak - half_window)
        end = min(len(data), start + window_size)
        if end - start < window_size:
            if start == 0:
                end = min(len(data), window_size)
            else:
                start = max(0, end - window_size)
        if end - start == window_size:
            windows.append(data[start:end])
    if not windows:
        for start in range(0, len(data) - window_size + 1, stride):
            windows.append(data[start:start + window_size])
    return windows

class Processor(ABC):
    def __init__(self, file_path:str, mode : str, max_length: str, label: int, **kwargs):
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

    def _import_loader(self, file_path:str) -> np.array:
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
            data = pad_sequence_numpy(sequence=data, max_sequence_length=self.max_length,
                                      input_shape=self.input_shape)
        else:
            data = improved_sliding_window(data=data, window_size=self.max_length, stride=10)
        return data
