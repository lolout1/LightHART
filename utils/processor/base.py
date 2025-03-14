import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IMUProcessor")

###############################################################################
# CSV Parsing Utilities with Robust Error Handling
###############################################################################

def parse_watch_csv(file_path: str):
    """
    Parse CSV file with timestamps and sensor data.
    
    Args:
        file_path: Path to CSV file with format [time, x, y, z]
        
    Returns:
        Array of shape (n, 4) with [time_elapsed, x, y, z] or empty array if file is invalid
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return np.zeros((0, 4), dtype=np.float32)
            
        # Read CSV file
        df = pd.read_csv(file_path, header=None, sep=None, engine='python')
        df = df.dropna(how='all').reset_index(drop=True)
        
        # Check if empty
        if df.empty:
            logger.warning(f"Empty file: {file_path}")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Check if first row contains headers
        if len(df) > 0:
            first_val = str(df.iloc[0, 0]).lower()
            if re.search("time", first_val) or re.search("stamp", first_val):
                df = df.iloc[1:].reset_index(drop=True)
        
        # Ensure minimum columns
        if df.shape[1] < 4:
            logger.warning(f"Insufficient columns in {file_path}: found {df.shape[1]}, need at least 4")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Extract time column
        time_strings = df.iloc[:, 0].values
        
        # Try to convert to datetime first
        try:
            # Convert to datetime and calculate seconds elapsed
            times = pd.to_datetime(time_strings)
            base_time = times[0]
            elapsed_times = [(t - base_time).total_seconds() for t in times]
            times_array = np.array(elapsed_times, dtype=np.float32)
        except Exception:
            # If datetime fails, try numeric parsing
            try:
                times_array = df.iloc[:, 0].astype(float).values
                # Check if large values (likely milliseconds epoch time)
                if times_array[0] > 1e10:
                    times_array = times_array / 1000.0
                # Make time relative to first sample
                times_array = times_array - times_array[0]
            except Exception as e:
                logger.warning(f"Failed to parse timestamps in {file_path}: {e}")
                return np.zeros((0, 4), dtype=np.float32)
        
        # Extract x, y, z columns
        try:
            sensor_values = df.iloc[:, 1:4].astype(float).values
        except Exception as e:
            logger.warning(f"Failed to parse sensor values in {file_path}: {e}")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Combine time and xyz data
        result = np.column_stack([times_array, sensor_values]).astype(np.float32)
        logger.info(f"Successfully parsed {file_path}: {result.shape[0]} samples")
        return result
        
    except Exception as e:
        logger.warning(f"Error parsing {file_path}: {e}")
        return np.zeros((0, 4), dtype=np.float32)


def create_skeleton_timestamps(skel_array: np.ndarray, fps=30.0):
    """Add time column to skeleton array based on assumed frame rate."""
    num_frames = skel_array.shape[0]
    tvals = np.arange(num_frames, dtype=np.float32) / fps
    return np.hstack([tvals.reshape(-1, 1), skel_array])


###############################################################################
# Sliding Window Utilities
###############################################################################

def sliding_windows_by_time(arr, window_size_sec=4.0, stride_sec=1.0, min_samples=5):
    """Create variable-length sliding windows based on timestamps in first column."""
    if arr.shape[0] < min_samples:
        return []
        
    min_t = arr[0, 0]
    max_t = arr[-1, 0]
    windows = []
    t_start = min_t

    while t_start + window_size_sec <= max_t + 1e-9:
        mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_start + window_size_sec)
        sub = arr[mask]
        if len(sub) >= min_samples:
            windows.append(sub)
        t_start += stride_sec
    return windows


def sliding_windows_by_time_fixed(arr, window_size_sec=4.0, stride_sec=1.0, fixed_count=128, min_samples=64):
    """Create fixed-length sliding windows with uniform temporal sampling."""
    if arr.shape[0] < min_samples:
        return []

    min_t, max_t = arr[0, 0], arr[-1, 0]
    windows, t_start = [], min_t

    while t_start + window_size_sec <= max_t + 1e-9:
        mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_start + window_size_sec)
        sub = arr[mask]
        if len(sub) >= min_samples:
            # Uniformly sample points to get exactly fixed_count samples
            idx = np.linspace(0, len(sub)-1, fixed_count).astype(int)
            windows.append(sub[idx])
        t_start += stride_sec
    return windows


###############################################################################
# Robust Alignment Utilities
###############################################################################

def extract_wrist_trajectory(skeleton_data, wrist_idx=9, joint_dim=3):
    """
    Extract wrist joint trajectory from skeleton data.
    
    Args:
        skeleton_data: Skeleton data of shape (n_frames, n_features)
        wrist_idx: Index of wrist joint
        joint_dim: Number of dimensions per joint (default: 3 for x,y,z)
        
    Returns:
        Wrist trajectory of shape (n_frames, joint_dim)
    """
    if skeleton_data.shape[1] < (wrist_idx + 1) * joint_dim:
        logger.warning(f"Skeleton data has insufficient dimensions for wrist_idx={wrist_idx}")
        return np.zeros((skeleton_data.shape[0], joint_dim))
    
    start_idx = wrist_idx * joint_dim
    end_idx = start_idx + joint_dim
    return skeleton_data[:, start_idx:end_idx]


def robust_align_modalities(imu_data, skel_data, imu_timestamps=None, skel_fps=30.0, 
                           method='dtw', wrist_idx=9, min_points=5):
    """
    Align IMU and skeleton modalities with robust fallback options.
    
    Args:
        imu_data: IMU data (accelerometer or gyroscope values)
        skel_data: Skeleton data
        imu_timestamps: IMU timestamps (optional)
        skel_fps: Skeleton frames per second
        method: Alignment method ('dtw', 'interpolation', 'crop')
        wrist_idx: Index of wrist joint in skeleton data
        min_points: Minimum number of points required for alignment
    
    Returns:
        Tuple of (aligned_imu, aligned_skel, aligned_timestamps)
    """
    # Handle empty inputs
    if imu_data.size == 0 or skel_data.size == 0:
        logger.warning("Empty data provided for alignment")
        return (np.zeros((0, max(3, imu_data.shape[1]) if imu_data.size > 0 else 3)), 
                np.zeros((0, skel_data.shape[1]) if skel_data.size > 0 else 96), 
                np.zeros(0))

    # Generate timestamps if not provided
    if imu_timestamps is None:
        imu_timestamps = np.arange(len(imu_data), dtype=np.float32) / 50.0  # default IMU fps=50Hz

    # Extract time from skeleton if present (column 0) or generate
    if skel_data.shape[1] == 97:  # time included
        skel_times, skel_data = skel_data[:, 0], skel_data[:, 1:]
    else:
        skel_times = np.arange(len(skel_data), dtype=np.float32) / skel_fps

    if method == 'dtw':
        try:
            from dtaidistance import dtw
            
            # Extract features for alignment
            # For IMU: use acceleration magnitude
            imu_mag = np.linalg.norm(imu_data[:, :min(3, imu_data.shape[1])], axis=1)
            
            # For skeleton: extract wrist trajectory if possible
            wrist_traj = extract_wrist_trajectory(skel_data, wrist_idx)
            
            # Calculate wrist velocity
            wrist_vel = np.zeros(len(wrist_traj))
            if len(wrist_traj) > 1:
                dt = 1.0/skel_fps
                for i in range(1, len(wrist_traj)):
                    wrist_vel[i] = np.linalg.norm(wrist_traj[i] - wrist_traj[i-1]) / dt
            
            # If wrist extraction failed, use overall skeleton magnitude
            if np.all(wrist_vel == 0):
                skel_mag = np.linalg.norm(skel_data, axis=1)
                feature_skel = skel_mag
            else:
                feature_skel = wrist_vel
            
            # Normalize sequences for DTW
            imu_norm = (imu_mag - np.mean(imu_mag)) / (np.std(imu_mag) + 1e-9)
            skel_norm = (feature_skel - np.mean(feature_skel)) / (np.std(feature_skel) + 1e-9)

            # Calculate DTW path
            path = dtw.warping_path(imu_norm, skel_norm)
            imu_idx, skel_idx = zip(*path)

            # Remove duplicates while preserving order
            imu_unique, skel_unique = [], []
            seen_imu, seen_skel = set(), set()
            
            for i, s in zip(imu_idx, skel_idx):
                if i not in seen_imu:
                    imu_unique.append(i)
                    seen_imu.add(i)
                if s not in seen_skel:
                    skel_unique.append(s)
                    seen_skel.add(s)
            
            # Get aligned data
            aligned_imu = imu_data[imu_unique]
            aligned_skel = skel_data[skel_unique]
            aligned_ts = imu_timestamps[imu_unique]

            if len(aligned_imu) < min_points:
                logger.warning(f"DTW alignment produced insufficient points: {len(aligned_imu)} < {min_points}")
                method = 'interpolation'  # Fall back to interpolation
            else:
                logger.info(f"DTW alignment successful: {len(aligned_imu)} points")
                return aligned_imu, aligned_skel, aligned_ts

        except Exception as e:
            logger.warning(f"DTW alignment failed: {e}, falling back to interpolation")
            method = 'interpolation'

    if method == 'interpolation':
        try:
            # Find common time range
            t_start = max(imu_timestamps[0], skel_times[0])
            t_end = min(imu_timestamps[-1], skel_times[-1])
            
            if t_end <= t_start:
                logger.warning("No overlapping time range between modalities")
                return (np.zeros((0, imu_data.shape[1])), 
                        np.zeros((0, skel_data.shape[1])), 
                        np.zeros(0))

            # Filter IMU data to common range
            mask = (imu_timestamps >= t_start) & (imu_timestamps <= t_end)
            if np.sum(mask) < min_points:
                logger.warning(f"Insufficient IMU points in common time range: {np.sum(mask)} < {min_points}")
                method = 'crop'  # Fall back to cropping
            else:
                aligned_ts = imu_timestamps[mask]
                aligned_imu = imu_data[mask]

                # Interpolate skeleton data to IMU timestamps
                from scipy.interpolate import interp1d
                
                # Create interpolation functions for each skeleton column
                interp_funcs = []
                for i in range(skel_data.shape[1]):
                    interp_funcs.append(
                        interp1d(skel_times, skel_data[:, i], 
                                bounds_error=False, fill_value="extrapolate")
                    )
                
                # Apply interpolation
                aligned_skel = np.zeros((len(aligned_ts), skel_data.shape[1]))
                for i, func in enumerate(interp_funcs):
                    aligned_skel[:, i] = func(aligned_ts)
                
                logger.info(f"Interpolation alignment successful: {len(aligned_ts)} points")
                return aligned_imu, aligned_skel, aligned_ts

        except Exception as e:
            logger.warning(f"Interpolation failed: {e}, falling back to basic alignment")
            method = 'crop'
    
    # Basic crop alignment as final fallback
    logger.info("Using basic crop alignment as fallback")
    min_len = min(len(imu_data), len(skel_data))
    if min_len < min_points:
        logger.warning(f"Insufficient samples after cropping: {min_len} < {min_points}")
        return (np.zeros((0, imu_data.shape[1])), 
                np.zeros((0, skel_data.shape[1])), 
                np.zeros(0))
    
    return imu_data[:min_len], skel_data[:min_len], imu_timestamps[:min_len]


###############################################################################
# Trial Matching Functionality
###############################################################################

def match_trials(accel_files, gyro_files, skel_files):
    """
    Match trials across modalities based on filenames.
    
    Args:
        accel_files: List of accelerometer file paths
        gyro_files: List of gyroscope file paths
        skel_files: List of skeleton file paths
        
    Returns:
        List of (subject_id, action_id, trial_num, accel_path, gyro_path, skel_path) tuples
    """
    # Extract trial info from filenames
    def extract_info(filepath):
        if not filepath:
            return None
            
        filename = os.path.basename(filepath)
        match = re.match(r"S(\d+)A(\d+)T(\d+)\.csv", filename)
        if match:
            subj = int(match.group(1))
            act = int(match.group(2))
            trial = int(match.group(3))
            return (subj, act, trial, filepath)
        return None
    
    # Create dictionaries of trial info
    accel_dict = {}
    for f in accel_files:
        info = extract_info(f)
        if info:
            accel_dict[(info[0], info[1], info[2])] = f
    
    gyro_dict = {}
    for f in gyro_files:
        info = extract_info(f)
        if info:
            gyro_dict[(info[0], info[1], info[2])] = f
    
    skel_dict = {}
    for f in skel_files:
        info = extract_info(f)
        if info:
            skel_dict[(info[0], info[1], info[2])] = f
    
    # Match trials that have at least accelerometer data
    matched_trials = []
    for key in accel_dict:
        subj, act, trial = key
        accel_path = accel_dict[key]
        gyro_path = gyro_dict.get(key, None)
        skel_path = skel_dict.get(key, None)
        
        matched_trials.append((subj, act, trial, accel_path, gyro_path, skel_path))
    
    # Sort by subject, action, trial
    matched_trials.sort()
    logger.info(f"Matched {len(matched_trials)} trials")
    
    return matched_trials


###############################################################################
# Processor Class
###############################################################################

class Processor(nn.Module):
    def __init__(self, file_path='', mode='variable_time', max_length=128, window_size_sec=4.0, stride_sec=1.0):
        super().__init__()
        self.file_path = file_path
        self.mode = mode
        self.max_length = max_length
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec

    def load_file(self, is_skeleton=False, is_gyroscope=False):
        """Load file based on type and mode."""
        try:
            if not self.file_path or not os.path.exists(self.file_path):
                logger.warning(f"File path invalid or not found: {self.file_path}")
                return np.zeros((0, 4 if not is_skeleton else 97), dtype=np.float32)
                
            if self.mode == 'variable_time':
                if is_skeleton:
                    # Load skeleton data
                    df = pd.read_csv(self.file_path, header=None)
                    data = df.values.astype(np.float32)
                    
                    # Add time column if not present
                    if data.shape[1] == 96:  # No time column
                        data = create_skeleton_timestamps(data, fps=30.0)
                else:
                    # Load IMU data
                    data = parse_watch_csv(self.file_path)
            else:
                # Fixed mode
                df = pd.read_csv(self.file_path, header=None)
                data = df.values.astype(np.float32)
                
            logger.info(f"Loaded {self.file_path}: shape={data.shape}")
            return data
            
        except Exception as e:
            logger.warning(f"Error loading {self.file_path}: {e}")
            return np.zeros((0, 4 if not is_skeleton else 97), dtype=np.float32)

    def process(self, data, is_fused=False):
        """Process data into windows."""
        if data.size == 0:
            logger.warning("Empty data, no windows created")
            return []
            
        if self.mode == 'variable_time':
            if data.shape[1] == 4 or is_fused:
                # For IMU or fused data, create fixed-length windows
                windows = sliding_windows_by_time_fixed(
                    data, 
                    self.window_size_sec, 
                    self.stride_sec, 
                    self.max_length
                )
            else:
                # For skeleton data, create variable-length windows
                windows = sliding_windows_by_time(
                    data, 
                    self.window_size_sec, 
                    self.stride_sec
                )
        else:
            # Single window for fixed mode
            windows = [data]
            
        logger.info(f"Created {len(windows)} windows")
        return windows
