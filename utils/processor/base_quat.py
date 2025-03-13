"""
Base quaternion processing utilities for IMU data.
"""

import re
import numpy as np
import pandas as pd
import math

def parse_watch_csv(file_path: str):
    """
    Parse watch CSV with variable sampling rate.
    
    Args:
        file_path: Path to CSV file with format [time, x, y, z]
        
    Returns:
        Array of shape (n, 4) with [time_elapsed, x, y, z]
    """
    try:
        df = pd.read_csv(file_path, header=None, sep=None, engine='python')
        df = df.dropna(how='all').reset_index(drop=True)
        
        # Check if first row contains headers
        first_val = str(df.iloc[0, 0]).lower()
        if re.search("time", first_val) or re.search("stamp", first_val):
            df = df.drop(0).reset_index(drop=True)
        
        # Check for minimum columns
        if df.shape[1] < 4:
            print(f"Warning: {file_path} has fewer than 4 columns - expected [time, x, y, z]")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Try to parse timestamps
        time_strs = df.iloc[:, 0].astype(str).values
        dt_series = pd.to_datetime(time_strs, errors='coerce')
        
        if dt_series.isnull().all():
            # Possibly numeric timestamps
            try:
                times = time_strs.astype(float)
                if times[0] > 1e10:  # Epoch milliseconds
                    times = times / 1000.0
                times = times - times[0]  # Time elapsed from first sample
            except:
                print(f"Warning: Could not parse timestamps in {file_path}, using indices")
                times = np.arange(len(time_strs), dtype=np.float32)
        else:
            # Convert datetime to seconds from first sample
            base_time = dt_series[0]
            times = np.array([(ts - base_time).total_seconds() for ts in dt_series])
        
        # Parse x, y, z columns
        try:
            x = df.iloc[:, 1].astype(float).values
            y = df.iloc[:, 2].astype(float).values
            z = df.iloc[:, 3].astype(float).values
            sensor_data = np.column_stack([x, y, z])
        except:
            print(f"Warning: Could not parse x,y,z data in {file_path}")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Combine time and sensor data
        return np.column_stack([times, sensor_data]).astype(np.float32)
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return np.zeros((0, 4), dtype=np.float32)

def create_skeleton_timestamps(skel_array: np.ndarray, fps=30.0):
    """
    Add time column to skeleton array.
    
    Args:
        skel_array: Array of shape (n, feats) with skeleton data
        fps: Frames per second for skeleton data
        
    Returns:
        Array of shape (n, 1+feats) with added time column
    """
    n = skel_array.shape[0]
    tvals = np.arange(n) / fps
    tvals = tvals.reshape(-1, 1)
    return np.hstack([tvals, skel_array])

def sliding_windows_by_time_fixed(arr, window_size_sec=4.0, stride_sec=1.0, fixed_count=128, min_count=96):
    """
    Create fixed-length windows by sliding through time-stamped data.
    
    Args:
        arr: Array of shape (n, 4+) with time column at index 0
        window_size_sec: Window size in seconds
        stride_sec: Stride in seconds
        fixed_count: Fixed number of samples to output per window
        min_count: Minimum number of samples required per window
        
    Returns:
        List of arrays, each of shape (fixed_count, 4+)
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
        
        if original_count >= min_count:
            if original_count != fixed_count:
                # Resample to fixed_count points
                idx = np.linspace(0, original_count - 1, fixed_count).astype(int)
                sub_fixed = sub[idx]
                windows.append(sub_fixed)
            else:
                windows.append(sub)
        
        t_start += stride_sec
    
    return windows

def sliding_windows_by_time(arr, window_size_sec=4.0, stride_sec=1.0, min_samples=5):
    """
    Create variable-length windows by sliding through time-stamped data.
    
    Args:
        arr: Array of shape (n, 4+) with time column at index 0
        window_size_sec: Window size in seconds
        stride_sec: Stride in seconds
        min_samples: Minimum number of samples required per window
        
    Returns:
        List of arrays, each preserving all samples in the window
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
        
        if sub.shape[0] >= min_samples:
            windows.append(sub)
        
        t_start += stride_sec
    
    return windows

def robust_align_modalities(imu_data, skel_data, imu_ts, skel_fps=30.0, method='dtw', wrist_idx=9):
    """
    Align IMU (watch) and skeleton data using DTW or other methods.
    
    Args:
        imu_data: Array of shape (n_imu, 4+) with IMU data (time at index 0)
        skel_data: Array of shape (n_skel, 1+96) with skeleton data (time at index 0)
        imu_ts: Array of shape (n_imu,) with IMU timestamps
        skel_fps: Frame rate of skeleton data (default: 30.0 Hz)
        method: Alignment method ('dtw', 'simple')
        wrist_idx: Index of wrist joint to align with watch data (default: 9)
        
    Returns:
        Tuple of (aligned_imu, aligned_skel, aligned_timestamps)
    """
    if imu_data.shape[0] == 0 or skel_data.shape[0] == 0:
        return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
    
    # Extract skeleton wrist position for alignment
    skel_cols = skel_data.shape[1]
    joint_cols = 3  # x, y, z per joint
    
    if skel_cols >= 96 + 1:  # Time column + 96 joint coordinates (32 joints * 3)
        wrist_start_col = 1 + wrist_idx * joint_cols  # +1 for time column
        wrist_end_col = wrist_start_col + joint_cols
        
        # Extract wrist position from skeleton data
        wrist_pos = skel_data[:, wrist_start_col:wrist_end_col]
        
        # Calculate wrist velocity (magnitude)
        wrist_vel = np.zeros(skel_data.shape[0])
        for i in range(1, skel_data.shape[0]):
            dt = (skel_data[i, 0] - skel_data[i-1, 0]) if skel_data.shape[1] > 96 else 1.0/skel_fps
            if dt == 0:
                dt = 1.0/skel_fps
            wrist_vel[i] = np.linalg.norm(wrist_pos[i] - wrist_pos[i-1]) / dt
    else:
        # Fallback if skeleton data doesn't match expected format
        print("Warning: Skeleton data doesn't have expected number of columns for wrist extraction")
        wrist_vel = np.ones(skel_data.shape[0])
    
    # Calculate IMU total acceleration magnitude
    imu_mag = np.linalg.norm(imu_data[:, 1:4], axis=1)
    
    if method == 'dtw':
        try:
            from dtaidistance import dtw
            
            # Normalize sequences
            norm_imu_mag = (imu_mag - np.mean(imu_mag)) / (np.std(imu_mag) + 1e-6)
            norm_wrist_vel = (wrist_vel - np.mean(wrist_vel)) / (np.std(wrist_vel) + 1e-6)
            
            # Calculate DTW path
            path = dtw.warping_path(norm_imu_mag, norm_wrist_vel)
            
            # Extract aligned indices
            imu_idx, skel_idx = zip(*path)
            
            # Remove duplicates while preserving order
            imu_unique_idx = []
            skel_unique_idx = []
            seen_imu = set()
            seen_skel = set()
            
            for i, s in zip(imu_idx, skel_idx):
                if i not in seen_imu:
                    imu_unique_idx.append(i)
                    seen_imu.add(i)
                if s not in seen_skel:
                    skel_unique_idx.append(s)
                    seen_skel.add(s)
            
            # Create aligned arrays
            aligned_imu = imu_data[imu_unique_idx]
            aligned_skel = skel_data[skel_unique_idx]
            aligned_ts = imu_ts[imu_unique_idx]
            
            print(f"DTW alignment: IMU {imu_data.shape[0]} -> {aligned_imu.shape[0]}, " +
                  f"Skeleton {skel_data.shape[0]} -> {aligned_skel.shape[0]}")
            
            return aligned_imu, aligned_skel, aligned_ts
            
        except Exception as e:
            print(f"DTW alignment failed: {e}, falling back to simple alignment")
            method = 'simple'
    
    if method == 'simple':
        # Simple approach: use the smaller length
        length = min(imu_data.shape[0], skel_data.shape[0])
        if length < 10:
            return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
        
        aligned_imu = imu_data[:length]
        aligned_skel = skel_data[:length]
        aligned_ts = imu_ts[:length]
        
        print(f"Simple alignment: truncated to {length} samples")
        
        return aligned_imu, aligned_skel, aligned_ts
