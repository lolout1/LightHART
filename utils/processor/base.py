import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

###############################################################################
# CSV Parsing Utilities
###############################################################################

def parse_watch_csv(file_path: str):
    df = pd.read_csv(file_path, header=None, sep=None, engine='python')
    df = df.dropna(how='all').reset_index(drop=True)

    first_val = str(df.iloc[0, 0]).lower()
    if re.search("time|stamp", first_val):
        df = df.drop(0).reset_index(drop=True)

    if df.shape[1] < 4:
        raise ValueError(f"[parse_watch_csv] {file_path} needs >=4 columns: time, x, y, z")

    time_strs = df.iloc[:, 0].astype(str).values
    dt_series = pd.to_datetime(time_strs, errors='coerce')

    if dt_series.isnull().all():
        times = time_strs.astype(float)
        if times[0] > 1e10:
            times /= 1000.0
        times -= times[0]
    else:
        base_time = dt_series[0]  # Fixed indexing issue here
        times = np.array([(ts - base_time).total_seconds() for ts in dt_series])

    sensor_data = df.iloc[:, 1:4].values.astype(np.float32)
    return np.column_stack([times, sensor_data]).astype(np.float32)


def create_skeleton_timestamps(skel_array: np.ndarray, fps=30.0):
    num_frames = skel_array.shape[0]
    tvals = np.arange(num_frames, dtype=np.float32) / fps
    return np.hstack([tvals.reshape(-1, 1), skel_array])


###############################################################################
# Sliding Window Utilities
###############################################################################

def sliding_windows_by_time(arr, window_size_sec=4.0, stride_sec=1.0):
    if arr.shape[0] == 0:
        return []
    min_t = arr[0, 0]
    max_t = arr[-1, 0]
    windows = []
    t_start = min_t

    while t_start + window_size_sec <= max_t + 1e-9:
        mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_start + window_size_sec)
        sub = arr[mask]
        if sub.size:
            windows.append(sub)
        t_start += stride_sec
    return windows


def sliding_windows_by_time_fixed(arr, window_size_sec=4.0, stride_sec=1.0, fixed_count=128):
    if arr.shape[0] == 0:
        return []

    min_t, max_t = arr[0, 0], arr[-1, 0]
    windows, t_start = [], min_t

    while t_start + window_size_sec <= max_t + 1e-9:
        mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_start + window_size_sec)
        sub = arr[mask]
        if len(sub) >= fixed_count:
            idx = np.linspace(0, len(sub)-1, fixed_count).astype(int)
            windows.append(sub[idx])
        t_start += stride_sec
    return windows


###############################################################################
# Robust Alignment Utilities
###############################################################################

def robust_align_modalities(imu_data, skel_data, imu_timestamps=None, skel_fps=30.0, method='dtw', min_points=5):
    if imu_data.size == 0 or skel_data.size == 0:
        return (np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros((0,)))

    if imu_timestamps is None:
        imu_timestamps = np.arange(len(imu_data), dtype=np.float32) / 50.0  # default IMU fps=50Hz

    if skel_data.shape[1] == 97:  # time included
        skel_times, skel_data = skel_data[:, 0], skel_data[:, 1:]
    else:
        skel_times = np.arange(len(skel_data), dtype=np.float32) / skel_fps

    if method == 'dtw':
        try:
            from dtaidistance import dtw
            imu_mag = np.linalg.norm(imu_data[:, :3], axis=1)
            skel_mag = np.linalg.norm(skel_data, axis=1)

            imu_norm = (imu_mag - imu_mag.mean()) / (imu_mag.std() + 1e-9)
            skel_norm = (skel_mag - skel_mag.mean()) / (skel_mag.std() + 1e-9)

            path = dtw.warping_path(imu_norm, skel_norm)
            imu_idx, skel_idx = zip(*path)

            aligned_imu, aligned_skel, aligned_ts = imu_data[list(imu_idx)], skel_data[list(skel_idx)], imu_timestamps[list(imu_idx)]

            if len(aligned_imu) < min_points:
                raise ValueError("Insufficient aligned points.")
            return aligned_imu, aligned_skel, aligned_ts

        except Exception as e:
            print(f"[WARN] DTW failed ({e}), fallback interpolation.")
            method = 'interpolation'

    if method == 'interpolation':
        t_start, t_end = max(imu_timestamps[0], skel_times[0]), min(imu_timestamps[-1], skel_times[-1])
        if t_end <= t_start:
            return (np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros((0,)))

        mask = (imu_timestamps >= t_start) & (imu_timestamps <= t_end)
        aligned_ts, aligned_imu = imu_timestamps[mask], imu_data[mask]

        from scipy.interpolate import interp1d
        interp_funcs = [interp1d(skel_times, skel_data[:, i], fill_value='extrapolate') for i in range(skel_data.shape[1])]
        aligned_skel = np.stack([f(aligned_ts) for f in interp_funcs], axis=1)

        if len(aligned_ts) < min_points:
            return (np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros((0,)))

        return aligned_imu, aligned_skel, aligned_ts

    min_len = min(len(imu_data), len(skel_data))
    return imu_data[:min_len], skel_data[:min_len], imu_timestamps[:min_len]


###############################################################################
# Processor Class
###############################################################################

class Processor(nn.Module):
    def __init__(self, file_path, mode='variable_time', max_length=128, window_size_sec=4.0, stride_sec=1.0):
        super().__init__()
        self.file_path, self.mode, self.max_length = file_path, mode, max_length
        self.window_size_sec, self.stride_sec = window_size_sec, stride_sec

    def load_file(self, is_skeleton=False):
        try:
            if self.mode == 'variable_time':
                if is_skeleton:
                    data = pd.read_csv(self.file_path, header=None).dropna(how='all').fillna(0).values.astype(np.float32)
                    data = create_skeleton_timestamps(data, fps=30.0)
                else:
                    data = parse_watch_csv(self.file_path)
            else:
                data = pd.read_csv(self.file_path, header=None).dropna(how='all').fillna(0).values.astype(np.float32)
            return data
        except Exception as e:
            print(f"[WARN] load_file failed ({self.file_path}): {e}")
            return np.zeros((0, 0), dtype=np.float32)

    def process(self, data):
        if data.size == 0:
            return []
        if self.mode == 'variable_time':
            if data.shape[1] == 4:
                return sliding_windows_by_time_fixed(data, self.window_size_sec, self.stride_sec, 128)
            else:
                return sliding_windows_by_time(data, self.window_size_sec, self.stride_sec)
        return [data]

