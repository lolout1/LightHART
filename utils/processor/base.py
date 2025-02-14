#!/usr/bin/env python
# File: utils/processor/base.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

###############################################################################
# Time2Vec (used in teacher model, not builder)
###############################################################################
class Time2Vec(nn.Module):
    def __init__(self, out_channels=8):
        super().__init__()
        self.out_channels = out_channels
        self.lin_weight = nn.Parameter(torch.randn(1))
        self.lin_bias   = nn.Parameter(torch.randn(1))
        if out_channels > 1:
            self.per_weight = nn.Parameter(torch.randn(out_channels - 1))
            self.per_bias   = nn.Parameter(torch.randn(out_channels - 1))
        else:
            self.per_weight = None
            self.per_bias   = None

    def forward(self, t):
        # t => shape (N, 1)
        t_lin = self.lin_weight * t + self.lin_bias
        if self.per_weight is not None:
            alpha = self.per_weight.unsqueeze(0)
            beta  = self.per_bias.unsqueeze(0)
            t_per = torch.sin(alpha * t + beta)
            return torch.cat([t_lin, t_per], dim=-1)
        else:
            return t_lin

###############################################################################
# Original sliding window function for variable-length windows (e.g. skeleton)
###############################################################################
def sliding_windows_by_time(arr, window_size_sec=4.0, stride_sec=0.5):
    """
    Slides a window of duration `window_size_sec` over the data with a stride of 0.5 seconds.
    This function is used for variable-length windows (e.g. skeleton data).
    """
    if arr.shape[0] == 0:
        return []
    min_t = arr[0, 0]
    max_t = arr[-1, 0]
    windows = []
    t_start = min_t
    while t_start + window_size_sec <= max_t + 1e-9:
        mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_start + window_size_sec)
        sub = arr[mask]
        if sub.shape[0] > 0:
            windows.append(sub)
        t_start += stride_sec
    return windows

###############################################################################
# New function: Fixed window size for accelerometer data without altering original timestamps
###############################################################################
def sliding_windows_by_time_fixed(arr, window_size_sec=4.0, stride_sec=0.5, fixed_count=128, file_path=""):
    """
    Slides a 4-second window with a 0.5-second stride over the accelerometer data.
    For each window, if there are at least `fixed_count` samples, it uniformly subsamples the window to exactly
    fixed_count samples (preserving the original timestamps). If the window has fewer than fixed_count samples,
    the window is discarded. Detailed debug messages are printed.

    Parameters:
      arr            : np.ndarray with the first column as time (e.g. accelerometer data: time, x, y, z).
      window_size_sec: Duration of the window (default 4.0 sec).
      stride_sec     : Stride (set to 0.5 sec by default).
      fixed_count    : Desired number of samples per window (default 128).
      file_path      : The file path of the data (for logging purposes).

    Returns:
      A list of windows, each with shape (fixed_count, d) where d is the number of columns.
    """
    if arr.shape[0] == 0:
        return []
    min_t = arr[0, 0]
    max_t = arr[-1, 0]
    windows = []
    t_start = min_t
    while t_start + window_size_sec <= max_t + 1e-9:
        mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_start + window_size_sec)
        sub = arr[mask]
        original_count = sub.shape[0]
        if original_count < fixed_count:
            # Discard window if there are not enough samples
            print(f"[DEBUG] Discarding window in file '{file_path}' from {t_start:.3f} to {t_start+window_size_sec:.3f}: only {original_count} samples (required {fixed_count}).")
        else:
            # Uniformly subsample indices to select fixed_count samples without distorting original timestamps.
            indices = np.linspace(0, original_count - 1, fixed_count).astype(int)
            fixed_window = sub[indices]
            windows.append(fixed_window)
            print(f"[DEBUG] Accepted window in file '{file_path}' from {t_start:.3f} to {t_start+window_size_sec:.3f}: original samples = {original_count}, subsampled to {fixed_count}.")
        t_start += stride_sec
    return windows

###############################################################################
# Utility functions for file parsing
###############################################################################
def parse_watch_csv(file_path: str):
    df = pd.read_csv(file_path, header=None, sep=None, engine='python')
    df = df.dropna(how='all').reset_index(drop=True)

    first_val = str(df.iloc[0, 0]).lower()
    if re.search("time", first_val) or re.search("stamp", first_val):
        df = df.drop(0).reset_index(drop=True)

    if df.shape[1] < 4:
        raise ValueError(f"[parse_watch_csv] {file_path} must have >=4 columns")

    time_strs = df.iloc[:, 0].astype(str).values
    dt_series = pd.to_datetime(time_strs, errors='coerce')
    if dt_series.isnull().all():
        try:
            times = time_strs.astype(float)
            if times[0] > 1e10:
                times = times / 1000.0
            times = times - times[0]
        except:
            raise ValueError(f"[parse_watch_csv] cannot parse time for {file_path}")
    else:
        base_time = dt_series[0]
        times = np.array([(ts - base_time).total_seconds() for ts in dt_series])
    sensor_data = df.iloc[:, 1:4].values.astype(np.float32)
    return np.column_stack([times, sensor_data])

def create_skeleton_timestamps(skel_array: np.ndarray, fps=30.0):
    n = skel_array.shape[0]
    tvals = np.arange(n) / fps
    tvals = tvals.reshape(-1, 1)
    return np.hstack([tvals, skel_array])

###############################################################################
# Processor class: handles file processing and windowing
###############################################################################
class Processor(nn.Module):
    def __init__(self, file_path: str, mode: str, max_length: int, **kwargs):
        super().__init__()
        self.file_path = file_path
        self.mode = mode
        self.max_length = max_length
        self.kwargs = kwargs

        self.window_size_sec = kwargs.pop('window_size_sec', 4.0)
        self.stride_sec = kwargs.pop('stride_sec', 0.5)
        self.input_shape = None

    def set_input_shape(self, arr: np.ndarray):
        self.input_shape = arr.shape

    def load_file(self, is_skeleton=False):
        try:
            if self.mode == 'variable_time':
                if is_skeleton:
                    df = pd.read_csv(self.file_path, header=None, comment='#', engine='python')
                    if any("Unnamed" in str(c) for c in df.columns):
                        print(f"[WARN] 'Unnamed' columns => skip file {self.file_path}")
                        return np.zeros((0, 0), dtype=np.float32)
                    df = df.dropna(how='all').fillna(0)
                    raw = df.values.astype(np.float32)
                    data = create_skeleton_timestamps(raw, fps=30.0)
                else:
                    data = parse_watch_csv(self.file_path)
            else:
                df = pd.read_csv(self.file_path, header=None, comment='#', engine='python')
                df = df.dropna(how='all').fillna(0)
                data = df.values.astype(np.float32)
            self.set_input_shape(data)
            return data
        except Exception as e:
            print(f"[WARN] Could not process file {self.file_path}, error: {e}. Skipping.")
            return np.zeros((0, 0), dtype=np.float32)

    def process(self, data: np.ndarray):
        if self.mode == 'variable_time':
            # For accelerometer data (4 columns), use fixed windowing with subsampling (or discarding if too short)
            if data.shape[1] == 4:
                windows = sliding_windows_by_time_fixed(
                    data,
                    window_size_sec=self.window_size_sec,
                    stride_sec=self.stride_sec,
                    fixed_count=128,
                    file_path=self.file_path
                )
            else:
                # For skeleton data, use the original sliding window (variable-length)
                windows = sliding_windows_by_time(
                    data,
                    window_size_sec=self.window_size_sec,
                    stride_sec=self.stride_sec
                )
            return windows
        else:
            return data

