# utils/processor/base.py

from typing import Any, List
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
import datetime
import re

########################################
# 1) Time2Vec
########################################
class Time2Vec(nn.Module):
    """
    Minimal Time2Vec layer that transforms each scalar time t into R^(out_channels).

    The transformation is defined as:
      T2V(t) = [ alpha_0 * t + beta_0,
                 sin(alpha_1 * t + beta_1),
                 ...,
                 sin(alpha_(out_channels-1) * t + beta_(out_channels-1)) ]

    If your sensor has 3 channels (x, y, z) and you set time2vec_dim to 8,
    the final per-sample feature dimension is 3 + 8 = 11.
    """
    def __init__(self, out_channels=8):
        super().__init__()
        self.out_channels = out_channels
        
        # The first channel is a linear transformation
        self.lin_weight = nn.Parameter(torch.randn(1))
        self.lin_bias   = nn.Parameter(torch.randn(1))
        
        # The remaining (out_channels - 1) are sinusoidal transformations
        self.per_weight = nn.Parameter(torch.randn(self.out_channels - 1))
        self.per_bias   = nn.Parameter(torch.randn(self.out_channels - 1))

    def forward(self, t):
        """
        Args:
          t: A tensor of shape (N, 1) containing scalar time values (in seconds).

        Returns:
          A tensor of shape (N, out_channels).
        """
        t_lin = self.lin_weight * t + self.lin_bias  # (N, 1)
        alpha = self.per_weight.unsqueeze(0)         # (1, out_channels-1)
        beta  = self.per_bias.unsqueeze(0)           # (1, out_channels-1)
        t_per = torch.sin(alpha * t + beta)          # (N, out_channels-1)
        return torch.cat([t_lin, t_per], dim=-1)


########################################
# 2) CSV + Timestamp Parse
########################################
def parse_watch_csv(file_path: str):
    """
    Parse a watch (or phone) accelerometer/gyroscope CSV file. The file is expected to contain:
      * a timestamp in its first column
      * sensor values (x, y, z) in the next three columns
    The CSV can be comma- or semicolon-separated, with or without a header.

    Returns a NumPy array of shape (N, 4): [time_in_seconds, x, y, z].
    """
    df = pd.read_csv(file_path, header=None, sep=None, engine='python')
    df = df.dropna(how='all').reset_index(drop=True)
    
    # If the first row looks like a header, drop it.
    first_val = str(df.iloc[0, 0]).lower()
    if re.search("time", first_val) or re.search("stamp", first_val):
        df = df.drop(0).reset_index(drop=True)
    
    if df.shape[1] < 4:
        raise ValueError(f"{file_path} must have >=4 columns: time_string,x,y,z")
    
    time_strs = df.iloc[:, 0].astype(str).values
    dt_series = pd.to_datetime(time_strs, errors='coerce')
    
    if dt_series.isnull().all():
        # Timestamps cannot be parsed => assume numeric or epoch
        try:
            times = time_strs.astype(float)
            if times[0] > 1e10:
                times = times / 1000.0
            times = times - times[0]
        except Exception as e:
            raise ValueError(f"Could not convert time strings to float: {e}")
    else:
        # Use first parsed time as reference
        base_time = dt_series[0]
        times = np.array([(ts - base_time).total_seconds() for ts in dt_series])
    
    sensor_data = df.iloc[:, 1:4].values.astype(np.float32)
    out = np.column_stack([times, sensor_data])
    return out

########################################
# 3) 4s Windows by Actual Time
########################################
def sliding_windows_by_time(arr, window_size_sec=4.0, stride_sec=1.0):  ### CHANGED (was 2.0)
    """
    For an array 'arr' of shape (N, 4): [time, x, y, z] sorted by time,
    returns a list of sub-arrays for windows of 'window_size_sec' length
    with a stride of 'stride_sec'. The number of rows in each window can vary.

    Now defaulted to 4-second windows (was 2 seconds).
    """
    if len(arr) == 0:
        return []
    min_t = arr[0, 0]
    max_t = arr[-1, 0]
    t_start = min_t
    windows = []
    while t_start + window_size_sec <= max_t + 1e-9:
        mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_start + window_size_sec)
        sub = arr[mask]
        if len(sub) > 0:
            windows.append(sub)
        t_start += stride_sec
    
    ### ADDED Debug statement
    print(f"[DEBUG] Created {len(windows)} windows from array of shape {arr.shape}")
    return windows

def apply_time2vec_to_window(window_array, t2v: Time2Vec):
    """
    Given a window_array of shape (n, 4): [time, x, y, z],
    remove the time column and concatenate the sensor values
    with the Time2Vec embedding of the time values.

    The output shape is (n, 3 + t2v.out_channels).
    """
    if len(window_array) == 0:
        return np.zeros((0, 3 + t2v.out_channels), dtype=np.float32)
    times = window_array[:, 0]
    feats = window_array[:, 1:]
    t_tensor = torch.from_numpy(times).float().unsqueeze(-1)  # shape (n, 1)
    with torch.no_grad():
        t_embed = t2v(t_tensor).numpy()  # shape (n, out_channels)
    out = np.concatenate([feats, t_embed], axis=-1)
    return out


########################################
# 4) LOADER_MAP (unchanged)
########################################
def csvloader(file_path: str, **kwargs):
    file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
    activity_data = file_data.iloc[2:, -3:].to_numpy(dtype=np.float32)
    return activity_data

def matloader(file_path: str, **kwargs):
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        raise ValueError(f"Unsupported key {key} for .mat file")
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

########################################
# 5) Pooled/Padded Approach (unchanged)
########################################
def avg_pool(sequence: np.array, window_size: int = 5, stride: int = 1,
             max_length: int = 512, shape: int = None) -> np.ndarray:
    shape = sequence.shape
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

def sliding_window(data: np.ndarray, clearing_time_index: int, max_time: int,
                   sub_window_size: int, stride_size: int) -> np.ndarray:
    assert clearing_time_index >= sub_window_size - 1, "clearing_value must be >= (window_size - 1)"
    start = clearing_time_index - sub_window_size + 1
    if max_time >= data.shape[0] - sub_window_size:
        max_time = max_time - sub_window_size + 1
    sub_windows = (
        start
        + np.expand_dims(np.arange(sub_window_size), 0)
        + np.expand_dims(np.arange(max_time, step=stride_size), 0).T
    )
    return data[sub_windows]

########################################
# 6) Processor Class
########################################
class Processor(ABC):
    """
    The Processor class handles reading a file and then processing it into
    windows. If mode='variable_time', it uses parse_watch_csv and then
    applies sliding time windows + Time2Vec.
    """
    def __init__(self, file_path: str, mode: str, max_length: int, **kwargs):
        assert mode in ['sliding_window', 'avg_pool', 'variable_time'], f"Processing mode={mode} not recognized."
        self.mode = mode
        self.max_length = max_length
        self.data = []
        self.file_path = file_path
        self.input_shape = []
        self.kwargs = kwargs

        self.time2vec_dim = kwargs.get('time2vec_dim', 8)
        # Window size & stride for 'variable_time' mode
        self.window_size_sec = kwargs.get('window_size_sec', 4.0)  ### CHANGED default to 4.0
        self.stride_sec = kwargs.get('stride_sec', 1.0)

        self.t2v = Time2Vec(out_channels=self.time2vec_dim)

    def set_input_shape(self, sequence: np.ndarray) -> List[int]:
        self.input_shape = sequence.shape
        return self.input_shape

    def _import_loader(self, file_path: str):
        file_type = file_path.split('.')[-1].lower()
        if file_type not in ['csv', 'mat']:
            raise ValueError(f"Unsupported file type: {file_type}")
        return LOADER_MAP[file_type]

    def load_file(self):
        if self.mode == 'variable_time':
            data = parse_watch_csv(self.file_path)
        else:
            loader = self._import_loader(self.file_path)
            data = loader(self.file_path, **self.kwargs)
        self.set_input_shape(data)
        return data

    def process(self, data):
        if self.mode == 'avg_pool':
            data = pad_sequence_numpy(sequence=data, max_sequence_length=self.max_length, input_shape=self.input_shape)
        elif self.mode == 'sliding_window':
            data = sliding_window(
                data=data,
                clearing_time_index=self.max_length - 1,
                max_time=self.input_shape[0],
                sub_window_size=self.max_length,
                stride_size=10
            )
        elif self.mode == 'variable_time':
            all_windows = sliding_windows_by_time(data, window_size_sec=self.window_size_sec, stride_sec=self.stride_sec)
            embedded = []
            for w in all_windows:
                w_embed = apply_time2vec_to_window(w, self.t2v)
                embedded.append(w_embed)
            return embedded
        return data
