"""
Base quaternion processing utilities for IMU data.
Provides functions for processing IMU data with quaternion-based orientation.
"""

import re
import numpy as np
import pandas as pd
import math
import os
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt

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
                times = df.iloc[:, 0].astype(float).values
                if times[0] > 1e10:  # Epoch milliseconds
                    times = times / 1000.0
                times = times - times[0]  # Time elapsed from first sample
            except:
                print(f"Warning: Could not parse timestamps in {file_path}, using indices")
                times = np.arange(len(time_strs), dtype=np.float32)
        else:
            # Convert datetime to seconds from first sample
            base_time = dt_series.iloc[0]
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

def preprocess_imu_data(accel_data, gyro_data):
    """
    Preprocess IMU data before Kalman filtering.

    Args:
        accel_data: Accelerometer data with time column
        gyro_data: Gyroscope data with time column

    Returns:
        Tuple of processed (accel_data, gyro_data)
    """
    if accel_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        return accel_data, gyro_data

    # 1. Check timestamps match
    accel_time = accel_data[:, 0]
    gyro_time = gyro_data[:, 0]

    # If sample rates differ, interpolate to match accelerometer timestamps
    if not np.array_equal(accel_time, gyro_time):
        if gyro_data.shape[0] > 1:
            gyro_interp = interp1d(
                gyro_time,
                gyro_data[:, 1:],
                axis=0,
                bounds_error=False,
                fill_value="extrapolate"
            )

            gyro_values = gyro_interp(accel_time)
            gyro_data = np.column_stack([accel_time, gyro_values])

    # 2. Apply median filter to reduce noise
    accel_values = accel_data[:, 1:]
    gyro_values = gyro_data[:, 1:]

    if accel_values.shape[0] >= 5:  # Need at least 5 samples for a reasonable filter
        filtered_accel = np.zeros_like(accel_values)
        filtered_gyro = np.zeros_like(gyro_values)

        for i in range(accel_values.shape[1]):
            filtered_accel[:, i] = medfilt(accel_values[:, i], kernel_size=3)

        for i in range(gyro_values.shape[1]):
            filtered_gyro[:, i] = medfilt(gyro_values[:, i], kernel_size=3)

        accel_data = np.column_stack([accel_time, filtered_accel])
        gyro_data = np.column_stack([accel_time, filtered_gyro])

    return accel_data, gyro_data

def interpolate_missing_values(data):
    """
    Interpolate missing values in time series data.

    Args:
        data: Array with time in first column

    Returns:
        Array with interpolated values
    """
    if data.shape[0] <= 1:
        return data

    # Check for large gaps in time
    times = data[:, 0]
    dt = np.diff(times)
    median_dt = np.median(dt)

    # If gaps are more than 5x the median dt, interpolate
    gaps = np.where(dt > 5 * median_dt)[0]
    if len(gaps) == 0:
        return data

    # Interpolate each gap
    new_data = []
    for i in range(len(gaps) + 1):
        if i == 0:
            start_idx = 0
        else:
            start_idx = gaps[i-1] + 1

        if i == len(gaps):
            end_idx = len(times)
        else:
            end_idx = gaps[i] + 1

        segment = data[start_idx:end_idx]
        new_data.append(segment)

        if i < len(gaps):
            # Create interpolation segment
            t_start = times[gaps[i]]
            t_end = times[gaps[i] + 1]

            # How many points to insert
            n_points = int((t_end - t_start) / median_dt) - 1
            if n_points > 0:
                t_new = np.linspace(t_start, t_end, n_points + 2)[1:-1]

                # Interpolate feature values
                interpolated = np.zeros((n_points, data.shape[1]))
                interpolated[:, 0] = t_new

                for j in range(1, data.shape[1]):
                    interpolated[:, j] = np.interp(
                        t_new,
                        [t_start, t_end],
                        [data[gaps[i], j], data[gaps[i] + 1, j]]
                    )

                new_data.append(interpolated)

    # Combine all segments
    return np.vstack(new_data)

def robust_align_modalities(imu_data, skel_data, imu_ts, skel_fps=30.0, method='dtw', wrist_idx=9):
    """
    Align IMU (watch) and skeleton data using DTW or other methods.

    Args:
        imu_data: Array of shape (n_imu, 3) with IMU data
        skel_data: Array of shape (n_skel, 96) with skeleton data
        imu_ts: Array of shape (n_imu,) with IMU timestamps
        skel_fps: Frame rate of skeleton data (default: 30.0 Hz)
        method: Alignment method ('dtw', 'interpolation', 'crop')
        wrist_idx: Index of wrist joint to align with watch data (default: 9)

    Returns:
        Tuple of (aligned_imu, aligned_skel, aligned_timestamps)
    """
    if imu_data.shape[0] == 0 or skel_data.shape[0] == 0:
        return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)

    # Extract skeleton wrist position for alignment
    skel_cols = skel_data.shape[1]
    joint_cols = 3  # x, y, z per joint

    if skel_cols >= 96:  # 96 joint coordinates (32 joints * 3)
        wrist_start_col = wrist_idx * joint_cols
        wrist_end_col = wrist_start_col + joint_cols

        # Extract wrist position from skeleton data
        wrist_pos = skel_data[:, wrist_start_col:wrist_end_col]

        # Calculate wrist velocity (magnitude)
        wrist_vel = np.zeros(skel_data.shape[0])
        for i in range(1, skel_data.shape[0]):
            dt = 1.0/skel_fps
            wrist_vel[i] = np.linalg.norm(wrist_pos[i] - wrist_pos[i-1]) / dt
    else:
        # Fallback if skeleton data doesn't match expected format
        print("Warning: Skeleton data doesn't have expected number of columns for wrist extraction")
        wrist_vel = np.ones(skel_data.shape[0])

    # Calculate IMU total acceleration magnitude
    imu_mag = np.linalg.norm(imu_data, axis=1)

    # Create skeleton timestamps
    skel_ts = np.arange(skel_data.shape[0]) / skel_fps

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

            return aligned_imu, aligned_skel, aligned_ts

        except Exception as e:
            print(f"DTW alignment failed: {e}, falling back to interpolation")
            method = 'interpolation'

    if method == 'interpolation':
        try:
            # Interpolate skeleton data to IMU timestamps
            skel_interp = interp1d(
                skel_ts,
                skel_data,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate"
            )

            # Get common time range
            t_min = max(skel_ts[0], imu_ts[0])
            t_max = min(skel_ts[-1], imu_ts[-1])

            # Filter IMU data to common range
            valid_mask = (imu_ts >= t_min) & (imu_ts <= t_max)
            if np.sum(valid_mask) < 10:  # Not enough overlap
                return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)

            valid_ts = imu_ts[valid_mask]
            valid_imu = imu_data[valid_mask]

            # Interpolate skeleton to IMU timestamps
            interp_skel = skel_interp(valid_ts)

            return valid_imu, interp_skel, valid_ts

        except Exception as e:
            print(f"Interpolation failed: {e}, falling back to crop")
            method = 'crop'

    if method == 'crop':
        # Simple approach: use the smaller length
        length = min(imu_data.shape[0], skel_data.shape[0])
        if length < 10:
            return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)

        aligned_imu = imu_data[:length]
        aligned_skel = skel_data[:length]
        aligned_ts = imu_ts[:length]

        return aligned_imu, aligned_skel, aligned_ts

    # If we reach here, alignment failed
    print(f"Alignment failed with method {method}")
    return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)

def extract_orientation_features(quat_orientation):
    """
    Extract additional features from quaternion orientation.

    Args:
        quat_orientation: Quaternion orientation array (N, 4)

    Returns:
        Array of orientation features (N, 7): [roll, pitch, yaw, quat_w, quat_x, quat_y, quat_z]
    """
    n_samples = quat_orientation.shape[0]
    orientation_features = np.zeros((n_samples, 7))

    for i in range(n_samples):
        # Convert quaternion [w, x, y, z] to scipy [x, y, z, w]
        q_scipy = [quat_orientation[i, 1], quat_orientation[i, 2],
                  quat_orientation[i, 3], quat_orientation[i, 0]]

        # Get Euler angles
        r = R.from_quat(q_scipy)
        euler = r.as_euler('xyz')

        # Store roll, pitch, yaw
        orientation_features[i, :3] = euler

        # Store quaternion
        orientation_features[i, 3:] = quat_orientation[i]

    return orientation_features

def get_device_orientation(accel, gravity=9.81):
    """
    Estimate device orientation from accelerometer data.

    Args:
        accel: Linear acceleration data (N, 3)
        gravity: Gravity constant

    Returns:
        Quaternion orientation (N, 4)
    """
    n_samples = accel.shape[0]
    quat_orientation = np.zeros((n_samples, 4))
    quat_orientation[:, 0] = 1.0  # Initialize to identity quaternion [w=1, x=0, y=0, z=0]

    # Estimate gravity direction (assuming device is mostly stationary)
    gravity_dir = np.mean(accel, axis=0)
    gravity_mag = np.linalg.norm(gravity_dir)

    if gravity_mag > 0.1 * gravity:
        # Normalize gravity direction
        gravity_dir = gravity_dir / gravity_mag

        # Global up vector
        up = np.array([0, 0, 1])

        # Calculate rotation axis
        rotation_axis = np.cross(gravity_dir, up)
        axis_mag = np.linalg.norm(rotation_axis)

        if axis_mag > 1e-6:
            # Normalize rotation axis
            rotation_axis = rotation_axis / axis_mag

            # Calculate rotation angle
            angle = np.arccos(np.dot(gravity_dir, up))

            # Create quaternion
            quat = np.zeros(4)
            quat[0] = np.cos(angle / 2)  # w
            quat[1:4] = rotation_axis * np.sin(angle / 2)  # x, y, z

            # Use this quaternion for all samples
            quat_orientation[:] = quat

    return quat_orientation

class QuaternionProcessor:
    """
    Processor for quaternion-based orientation data.
    """

    def __init__(self, file_path=None, mode='variable_time', max_length=128,
                 window_size_sec=4.0, stride_sec=1.0, filter_type='ekf',
                 align_method='dtw', wrist_idx=9):
        """
        Initialize quaternion processor.

        Args:
            file_path: Path to data file
            mode: Processing mode ('variable_time', 'fixed')
            max_length: Maximum sequence length
            window_size_sec: Window size in seconds
            stride_sec: Stride in seconds
            filter_type: Kalman filter type ('standard', 'ekf', 'ukf')
            align_method: Method for aligning IMU and skeleton ('dtw', 'interpolation', 'crop')
            wrist_idx: Index of wrist joint for skeleton alignment
        """
        self.file_path = file_path
        self.mode = mode
        self.max_length = max_length
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.filter_type = filter_type
        self.align_method = align_method
        self.wrist_idx = wrist_idx

        # Default filter parameters (these can be calibrated)
        self.filter_params = {
            'process_noise': 0.01,
            'measurement_noise': 0.1,
            'gyro_bias_noise': 0.01
        }

    def load_file(self, is_skeleton=False, is_gyroscope=False):
        """
        Load data file.

        Args:
            is_skeleton: Whether the file contains skeleton data
            is_gyroscope: Whether the file contains gyroscope data

        Returns:
            Loaded data array
        """
        if not self.file_path or not os.path.exists(self.file_path):
            return np.zeros((0, 4), dtype=np.float32)

        try:
            if is_skeleton:
                # Load skeleton data
                df = pd.read_csv(self.file_path, header=None).dropna(how='all').fillna(0)
                data = df.values.astype(np.float32)

                # Add time column if not present
                if data.shape[1] == 96:  # No time column
                    data = create_skeleton_timestamps(data)
            else:
                # Load IMU data
                data = parse_watch_csv(self.file_path)

            return data
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            return np.zeros((0, 4), dtype=np.float32)

    def process_with_filter(self, accel_data, gyro_data, filter_params=None):
        """
        Process IMU data with Kalman filter.

        Args:
            accel_data: Accelerometer data with time column
            gyro_data: Gyroscope data with time column
            filter_params: Optional filter parameters

        Returns:
            Fused data with orientation
        """
        from utils.imu_fusion import StandardKalmanIMU, ExtendedKalmanIMU, UnscentedKalmanIMU

        if accel_data.shape[0] == 0 or gyro_data.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Preprocess data
        accel_data, gyro_data = preprocess_imu_data(accel_data, gyro_data)

        # Get timestamps and values
        timestamps = accel_data[:, 0]
        accel_xyz = accel_data[:, 1:4]
        gyro_xyz = gyro_data[:, 1:4]

        # Use provided or default filter parameters
        if filter_params is None:
            filter_params = self.filter_params

        # Create filter
        if self.filter_type == 'standard':
            filter_obj = StandardKalmanIMU(**filter_params)
        elif self.filter_type == 'ekf':
            filter_obj = ExtendedKalmanIMU(**filter_params)
        else:  # ukf
            filter_obj = UnscentedKalmanIMU(**filter_params)

        # Process sequence
        fused_data = filter_obj.process_sequence(accel_xyz, gyro_xyz, timestamps)

        # Add time column
        return np.column_stack([timestamps, fused_data])

    def create_windows(self, data, is_skeleton=False):
        """
        Create windows from data.

        Args:
            data: Array with time in first column
            is_skeleton: Whether the data is skeleton

        Returns:
            List of window arrays
        """
        if data.shape[0] == 0:
            return []

        if is_skeleton:
            # Variable-length windows for skeleton
            return sliding_windows_by_time(
                data,
                window_size_sec=self.window_size_sec,
                stride_sec=self.stride_sec
            )
        else:
            # Fixed-length windows for IMU
            return sliding_windows_by_time_fixed(
                data,
                window_size_sec=self.window_size_sec,
                stride_sec=self.stride_sec,
                fixed_count=self.max_length
            )

    def process(self, accel_data, gyro_data=None, skel_data=None):
        """
        Process data with quaternion orientation.

        Args:
            accel_data: Accelerometer data with time column
            gyro_data: Optional gyroscope data with time column
            skel_data: Optional skeleton data with time column

        Returns:
            Dictionary with processed data and windows
        """
        result = {}

        # Process with Kalman filter if gyro data available
        if gyro_data is not None and gyro_data.shape[0] > 0:
            fused_data = self.process_with_filter(accel_data, gyro_data)
            result['fused_imu'] = fused_data
            imu_windows = self.create_windows(fused_data)
            result['fused_imu_windows'] = imu_windows
        else:
            # Cannot do proper fusion without gyro, use simple orientation estimate
            accel_xyz = accel_data[:, 1:4]
            quat = get_device_orientation(accel_xyz)

            # Extract orientation features
            orient_features = extract_orientation_features(quat)

            # Combine with accel data
            combined = np.column_stack([
                accel_data,
                orient_features,
                np.linalg.norm(accel_xyz, axis=1, keepdims=True)  # Add magnitude
            ])

            result['accel_with_orientation'] = combined
            accel_windows = self.create_windows(combined)
            result['accel_windows'] = accel_windows

        # Process skeleton if available
        if skel_data is not None and skel_data.shape[0] > 0:
            skel_windows = self.create_windows(skel_data, is_skeleton=True)
            result['skeleton_windows'] = skel_windows

            # Align modalities if both IMU and skeleton available
            if gyro_data is not None and gyro_data.shape[0] > 0:
                # Get IMU data without time column
                imu_data = fused_data[:, 1:]
                imu_ts = fused_data[:, 0]

                # Get skeleton data without time column
                skel_values = skel_data[:, 1:] if skel_data.shape[1] > 96 else skel_data

                # Align
                aligned_imu, aligned_skel, aligned_ts = robust_align_modalities(
                    imu_data,
                    skel_values,
                    imu_ts,
                    method=self.align_method,
                    wrist_idx=self.wrist_idx
                )

                if aligned_imu.shape[0] > 0 and aligned_skel.shape[0] > 0:
                    # Add time column back
                    aligned_imu_with_time = np.column_stack([aligned_ts, aligned_imu])
                    aligned_skel_with_time = np.column_stack([aligned_ts, aligned_skel])

                    result['aligned_imu'] = aligned_imu_with_time
                    result['aligned_skeleton'] = aligned_skel_with_time

                    # Create windows from aligned data
                    aligned_imu_windows = self.create_windows(aligned_imu_with_time)
                    aligned_skel_windows = self.create_windows(aligned_skel_with_time, is_skeleton=True)

                    # Ensure same number of windows
                    min_windows = min(len(aligned_imu_windows), len(aligned_skel_windows))
                    result['aligned_imu_windows'] = aligned_imu_windows[:min_windows]
                    result['aligned_skeleton_windows'] = aligned_skel_windows[:min_windows]

        return result
