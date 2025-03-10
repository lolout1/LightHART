# utils/processor/base.py

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from utils.imu_fusion import StandardKalmanIMU, ExtendedKalmanIMU, UnscentedKalmanIMU, robust_align_modalities

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


def parse_gyro_csv(file_path: str):
    """
    Parse gyroscope CSV file, similar to accelerometer
    Returns array with shape (N,4): [time_elapsed, gx, gy, gz]
    """
    return parse_watch_csv(file_path)  # Same format as accelerometer


def create_skeleton_timestamps(skel_array: np.ndarray, fps=30.0):
    """
    For skeleton arrays => shape (num_frames, feats).
    Add a time col => shape (num_frames, 1+feats).
    """
    n = skel_array.shape[0]
    tvals = np.arange(n) / fps
    tvals = tvals.reshape(-1, 1)
    return np.hstack([tvals, skel_array])


def fuse_accel_gyro(accel_data, gyro_data, filter_type='ekf', timestamps=None):
    """
    Fuse accelerometer and gyroscope data using selected Kalman filter
    
    Args:
        accel_data: Accelerometer data, shape (N, 3) or (N, 4) with time
        gyro_data: Gyroscope data, shape (N, 3) or (N, 4) with time
        filter_type: Type of filter ('standard', 'ekf', 'ukf')
        timestamps: Optional array of timestamps, shape (N,)
        
    Returns:
        Fused IMU data with orientation, shape (N, features)
    """
    # Extract accelerometer and gyroscope values
    if accel_data.shape[1] == 4:
        timestamps = accel_data[:, 0]
        accel_xyz = accel_data[:, 1:4]
    else:
        accel_xyz = accel_data
        
    if gyro_data.shape[1] == 4:
        gyro_xyz = gyro_data[:, 1:4]
    else:
        gyro_xyz = gyro_data
    
    # Select and initialize the appropriate filter
    if filter_type.lower() == 'standard':
        filter = StandardKalmanIMU()
    elif filter_type.lower() == 'ekf':
        filter = ExtendedKalmanIMU()
    elif filter_type.lower() == 'ukf':
        filter = UnscentedKalmanIMU()
    else:
        raise ValueError(f"Unknown filter type: {filter_type}. Must be 'standard', 'ekf', or 'ukf'.")
    
    # Process the data
    fused_data = filter.process_sequence(accel_xyz, gyro_xyz, timestamps)
    
    return fused_data


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
    Enhanced processor with IMU fusion capabilities
    """
    def __init__(self, file_path: str, mode: str, max_length: int, **kwargs):
        super().__init__()
        self.file_path = file_path
        self.mode = mode
        self.max_length = max_length
        self.window_size_sec = kwargs.pop('window_size_sec', 4.0)
        self.stride_sec = kwargs.pop('stride_sec', 1.0)
        self.imu_fusion = kwargs.pop('imu_fusion', None)
        self.align_method = kwargs.pop('align_method', 'dtw')
        self.input_shape = None
        
    def set_input_shape(self, data):
        """Set input shape based on data"""
        self.input_shape = data.shape

    def load_file(self, is_skeleton=False, is_gyroscope=False):
        """
        If is_skeleton => load CSV => shape(N, feats), add time col => shape(N, 1+feats).
        If is_gyroscope => parse gyro CSV => shape(N, 4) => [time, gx, gy, gz].
        Else => parse watch CSV => shape(N, 4) => [time, x, y, z].
        """
        try:
            if self.mode == 'variable_time':
                if is_skeleton:
                    df = pd.read_csv(self.file_path, header=None).dropna(how='all').fillna(0)
                    raw = df.values.astype(np.float32)
                    data = create_skeleton_timestamps(raw, fps=30.0)
                elif is_gyroscope:
                    data = parse_gyro_csv(self.file_path)
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

    def load_and_fuse_imu(self, accel_path, gyro_path, filter_type='ekf'):
        """
        Load accelerometer and gyroscope data and apply IMU fusion
        
        Args:
            accel_path: Path to accelerometer data file
            gyro_path: Path to gyroscope data file
            filter_type: Type of Kalman filter to use ('standard', 'ekf', 'ukf')
            
        Returns:
            Fused IMU data with orientation features
        """
        # Load accelerometer and gyroscope data
        accel_data = parse_watch_csv(accel_path)
        gyro_data = parse_gyro_csv(gyro_path)
        
        # Check if data is valid
        if accel_data.shape[0] == 0 or gyro_data.shape[0] == 0:
            print(f"[WARN] Empty accelerometer or gyroscope data for {accel_path} and {gyro_path}")
            return np.zeros((0, 0), dtype=np.float32)
        
        # Extract timestamps from accelerometer data
        timestamps = accel_data[:, 0]
        
        # Ensure gyroscope data is aligned with accelerometer data
        # Interpolate gyroscope data to match accelerometer timestamps
        from scipy.interpolate import interp1d
        
        # Create interpolator only if timestamps differ
        if not np.array_equal(timestamps, gyro_data[:, 0]):
            gyro_interp = interp1d(
                gyro_data[:, 0],
                gyro_data[:, 1:],
                axis=0,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            # Interpolate gyroscope data at accelerometer timestamps
            gyro_aligned = gyro_interp(timestamps)
        else:
            gyro_aligned = gyro_data[:, 1:]
        
        # Apply IMU fusion
        fused_data = fuse_accel_gyro(
            accel_data[:, 1:],  # Skip time column
            gyro_aligned,
            filter_type=filter_type,
            timestamps=timestamps
        )
        
        # Add timestamps column back to fused data
        fused_with_time = np.column_stack([timestamps, fused_data])
        
        return fused_with_time

    def process(self, data: np.ndarray, is_fused=False):
        """
        If watch => sliding_windows_by_time_fixed => 4s->128 samples.
        If skeleton => sliding_windows_by_time => variable length.
        
        For fused data, use the same approach as watch data.
        """
        if self.mode == 'variable_time':
            if data.shape[1] == 4 or is_fused:
                # watch or fused data
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

    def align_modalities(self, accel_data, skel_data):
        """
        Align accelerometer data with skeleton data
        
        Args:
            accel_data: Accelerometer data with timestamps
            skel_data: Skeleton data with timestamps
            
        Returns:
            Aligned accelerometer and skeleton data
        """
        if accel_data.shape[0] == 0 or skel_data.shape[0] == 0:
            return [], []
            
        # Extract timestamps from accelerometer
        accel_timestamps = accel_data[:, 0]
        
        # Align using robust method
        aligned_accel, aligned_skel, aligned_timestamps = robust_align_modalities(
            accel_data[:, 1:],  # skip time column
            skel_data[:, 1:],    # skip time column
            accel_timestamps,
            method=self.align_method
        )
        
        # Add timestamps back
        aligned_accel_with_time = np.column_stack([aligned_timestamps, aligned_accel])
        aligned_skel_with_time = np.column_stack([aligned_timestamps, aligned_skel])
        
        return aligned_accel_with_time, aligned_skel_with_time
