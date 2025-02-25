# File: utils/processor/base.py

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

###############################################################################
# Utility Functions
###############################################################################

def parse_watch_csv(file_path: str):
    """
    Loads watch CSV => shape (N,4): [time_elapsed, x, y, z].
    If first col is datetime, convert to seconds from first sample.
    If first col is already numeric, subtract first value so it becomes time_elapsed=0..N.
    """
    df = pd.read_csv(file_path, header=None, sep=None, engine='python')
    df = df.dropna(how='all').reset_index(drop=True)

    # Sometimes row 0 has 'time' or 'timestamp'
    first_val = str(df.iloc[0, 0]).lower()
    if re.search("time", first_val) or re.search("stamp", first_val):
        df = df.drop(0).reset_index(drop=True)

    if df.shape[1] < 4:
        raise ValueError(f"[parse_watch_csv] {file_path} needs >=4 columns: time, x, y, z")

    time_strs = df.iloc[:, 0].astype(str).values
    dt_series = pd.to_datetime(time_strs, errors='coerce')

    if dt_series.isnull().all():
        # Possibly numeric timestamps
        times = time_strs.astype(float)
        if times[0] > 1e10:  # e.g. epoch ms
            times = times / 1000.0
        times = times - times[0]  # time elapsed from first sample
    else:
        # Datetime => convert to seconds from first sample
        base_time = dt_series[0]
        times = np.array([(ts - base_time).total_seconds() for ts in dt_series])

    # x,y,z
    sensor_data = df.iloc[:, 1:4].values.astype(np.float32)
    return np.column_stack([times, sensor_data])  # shape (N,4)


def create_skeleton_timestamps(skel_array: np.ndarray, fps=30.0):
    """
    For skeleton arrays => shape (num_frames, feats).
    Add a time col => shape (num_frames, 1+feats).
    """
    n = skel_array.shape[0]
    tvals = np.arange(n) / fps
    tvals = tvals.reshape(-1, 1)
    return np.hstack([tvals, skel_array])


def sliding_windows_by_time(arr, window_size_sec=4.0, stride_sec=1.0):
    """
    Variable-length windows (no resampling). For skeleton data => returns list of sub-arrays.
    Each sub-array => shape(#frames_in_that_window, 1+...).
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


def sliding_windows_by_time_fixed(arr, window_size_sec=4.0, stride_sec=1.0,
                                  fixed_count=128, file_path=""):
    """
    For accelerometer => shape(N,4): [time, x, y, z].
    Slide 4s => uniform-subsample to exactly 'fixed_count' points if possible;
    if sub-window has < fixed_count, discard.
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
            # discard
            pass
        else:
            idx = np.linspace(0, original_count - 1, fixed_count).astype(int)
            sub_128 = sub[idx]
            windows.append(sub_128)

        t_start += stride_sec

    return windows

###############################################################################
# Processor class
###############################################################################

class Processor(nn.Module):
    """
    For each file => parse -> produce a list of sub-windows (each shape(128,4) if watch,
    or variable-length if skeleton).
    """
    def __init__(self, file_path: str, mode: str, max_length: int, **kwargs):
        super().__init__()
        self.file_path = file_path
        self.mode = mode
        self.max_length = max_length
        self.window_size_sec = kwargs.pop('window_size_sec', 4.0)
        self.stride_sec = kwargs.pop('stride_sec', 1.0)

    def load_file(self, is_skeleton=False):
        """
        If is_skeleton => load CSV => shape(N, feats), add time col => shape(N, 1+feats).
        Else => parse watch CSV => shape(N, 4) => [time, x, y, z].
        """
        try:
            if self.mode == 'variable_time':
                if is_skeleton:
                    df = pd.read_csv(self.file_path, header=None).dropna(how='all').fillna(0)
                    raw = df.values.astype(np.float32)
                    data = create_skeleton_timestamps(raw, fps=30.0)
                else:
                    data = parse_watch_csv(self.file_path)
            else:
                # fallback
                df = pd.read_csv(self.file_path, header=None).dropna(how='all').fillna(0)
                data = df.values.astype(np.float32)

            return data
        except Exception as e:
            print(f"[WARN] Could not load {self.file_path}, error={e}")
            return np.zeros((0, 0), dtype=np.float32)

    def process(self, data: np.ndarray):
        """
        If watch => sliding_windows_by_time_fixed => 4s->128 samples.
        If skeleton => sliding_windows_by_time => variable length.
        """
        if self.mode == 'variable_time':
            if data.shape[1] == 4:
                # watch => shape(N,4)
                windows = sliding_windows_by_time_fixed(
                    data,
                    window_size_sec=self.window_size_sec,
                    stride_sec=self.stride_sec,
                    fixed_count=128,
                    file_path=self.file_path
                )
            else:
                # skeleton => shape(N,1+3J) or something
                windows = sliding_windows_by_time(
                    data,
                    window_size_sec=self.window_size_sec,
                    stride_sec=self.stride_sec
                )
            return windows
        else:
            return [data]  # fallback => single window

