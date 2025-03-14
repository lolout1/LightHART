# utils/processor/base.py

import os
import numpy as np
import pandas as pd
import torch
import logging
import re
from typing import Tuple, List, Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IMUProcessor")

def parse_watch_csv(file_path: str) -> np.ndarray:
    """
    Parse watch CSV file into array with time and IMU data.
    Handles various timestamp formats and ensures consistent output.
    
    Args:
        file_path: Path to CSV file with format [time, x, y, z]
        
    Returns:
        Array of shape (n, 4) with [time_elapsed, x, y, z] or empty array if invalid
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Read CSV file without using pandas.read_csv's date parsing
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Filter out empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            logger.warning(f"Empty file: {file_path}")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Check if first line might be a header
        first_line = lines[0].lower()
        if 'time' in first_line or 'stamp' in first_line:
            lines = lines[1:]
        
        if not lines:
            logger.warning(f"No data after removing header: {file_path}")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Parse each line manually
        data = []
        timestamps = []
        
        for line in lines:
            # Split by common delimiters
            parts = re.split(r'[,;\t]', line)
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) < 4:
                continue  # Skip lines with insufficient columns
            
            # Parse timestamp
            time_str = parts[0]
            
            # Parse sensor values
            try:
                x, y, z = map(float, parts[1:4])
                
                # Add to data
                timestamps.append(time_str)
                data.append([x, y, z])
            except (ValueError, IndexError):
                continue  # Skip invalid lines
        
        if not data:
            logger.warning(f"No valid data lines found: {file_path}")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Convert timestamps to seconds elapsed
        elapsed_times = []
        
        # Try datetime parsing first
        try:
            datetime_objs = pd.to_datetime(timestamps)
            start_time = datetime_objs[0]
            
            for dt in datetime_objs:
                elapsed_times.append((dt - start_time).total_seconds())
        except:
            # Try numeric conversion
            try:
                numeric_times = [float(ts) for ts in timestamps]
                
                # Check if likely milliseconds (epoch time)
                if np.mean(numeric_times) > 1e10:
                    numeric_times = [t / 1000.0 for t in numeric_times]
                
                # Make relative to first timestamp
                start_time = numeric_times[0]
                elapsed_times = [t - start_time for t in numeric_times]
            except:
                # Fall back to sequence numbers
                logger.warning(f"Could not parse timestamps in {file_path}, using sequence numbers")
                elapsed_times = list(range(len(data)))
        
        # Combine into final array
        result = np.zeros((len(data), 4), dtype=np.float32)
        result[:, 0] = elapsed_times
        result[:, 1:4] = data
        
        logger.info(f"Successfully parsed {file_path}: {len(data)} samples")
        return result
        
    except Exception as e:
        logger.warning(f"Error parsing {file_path}: {e}")
        return np.zeros((0, 4), dtype=np.float32)

def create_skeleton_timestamps(skel_array: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """
    Add time column to skeleton array based on fixed frame rate.
    
    Args:
        skel_array: Array of shape (n_frames, n_features) with skeleton data
        fps: Frame rate in frames per second (default: 30.0)
        
    Returns:
        Array of shape (n_frames, 1+n_features) with added time column
    """
    n_frames = skel_array.shape[0]
    time_column = (np.arange(n_frames, dtype=np.float32) / fps).reshape(-1, 1)
    return np.hstack([time_column, skel_array])

def sliding_windows_by_time(arr: np.ndarray, window_size_sec: float = 4.0, 
                          stride_sec: float = 1.0, min_samples: int = 5) -> List[np.ndarray]:
    """
    Create variable-length sliding windows based on timestamps in first column.
    
    Args:
        arr: Array with time in first column
        window_size_sec: Window size in seconds
        stride_sec: Stride in seconds
        min_samples: Minimum number of samples required for a valid window
        
    Returns:
        List of window arrays
    """
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

def sliding_windows_by_time_fixed(arr: np.ndarray, window_size_sec: float = 4.0, 
                                stride_sec: float = 1.0, fixed_count: int = 128, 
                                min_samples: int = 64) -> List[np.ndarray]:
    """
    Create fixed-length sliding windows with uniform temporal sampling.
    
    Args:
        arr: Array with time in first column
        window_size_sec: Window size in seconds
        stride_sec: Stride in seconds
        fixed_count: Target number of samples per window
        min_samples: Minimum number of samples required for a valid window
        
    Returns:
        List of window arrays with fixed sample count
    """
    if arr.shape[0] < min_samples:
        return []

    min_t, max_t = arr[0, 0], arr[-1, 0]
    windows, t_start = [], min_t

    while t_start + window_size_sec <= max_t + 1e-9:
        mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_start + window_size_sec)
        sub = arr[mask]
        if len(sub) >= min_samples:
            # Uniformly sample points to get exactly fixed_count samples
            if len(sub) != fixed_count:
                idx = np.linspace(0, len(sub)-1, fixed_count).astype(int)
                windows.append(sub[idx])
            else:
                windows.append(sub)
        t_start += stride_sec
    return windows

def resample_to_fixed_rate(arr: np.ndarray, target_fps: float = 30.0) -> np.ndarray:
    """
    Resample time series to a fixed frame rate using linear interpolation.
    
    Args:
        arr: Array with time in first column
        target_fps: Target frame rate in Hz
        
    Returns:
        Resampled array with uniform time steps
    """
    if arr.shape[0] < 2:
        return arr
        
    # Extract time column and values
    times = arr[:, 0]
    values = arr[:, 1:]
    
    # Calculate start and end times
    start_time = times[0]
    end_time = times[-1]
    duration = end_time - start_time
    
    # Create new uniform time points
    n_frames = int(np.ceil(duration * target_fps))
    if n_frames < 2:
        return arr  # Can't resample with less than 2 frames
        
    new_times = np.linspace(start_time, end_time, n_frames)
    
    # Interpolate each feature column
    new_values = np.zeros((n_frames, values.shape[1]), dtype=values.dtype)
    
    for i in range(values.shape[1]):
        # Use linear interpolation for each column
        from scipy.interpolate import interp1d
        interp_func = interp1d(times, values[:, i], 
                              bounds_error=False, 
                              fill_value=(values[0, i], values[-1, i]))
        new_values[:, i] = interp_func(new_times)
    
    # Combine into new array
    resampled = np.zeros((n_frames, arr.shape[1]), dtype=arr.dtype)
    resampled[:, 0] = new_times
    resampled[:, 1:] = new_values
    
    return resampled

def robust_align_modalities(imu_data: np.ndarray, skel_data: np.ndarray, 
                          imu_timestamps: Optional[np.ndarray] = None,
                          wrist_idx: int = 9, method: str = 'dtw') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust alignment of IMU and skeleton data with multiple fallback methods.
    
    Args:
        imu_data: IMU data (accelerometer/gyroscope values) shape (n, 3)
        skel_data: Skeleton data shape (m, 96)
        imu_timestamps: IMU timestamps (optional)
        wrist_idx: Index of wrist joint in skeleton
        method: Alignment method ('dtw', 'interpolation', 'resample', 'crop')
        
    Returns:
        Tuple of (aligned_imu, aligned_skel, aligned_timestamps)
    """
    # Handle empty inputs
    if imu_data.size == 0 or skel_data.size == 0:
        logger.warning("Empty data provided for alignment")
        empty_shape = (0, imu_data.shape[1] if imu_data.size > 0 else 3)
        return np.zeros(empty_shape), np.zeros((0, skel_data.shape[1] if skel_data.size > 0 else 96)), np.zeros(0)
    
    # Generate IMU timestamps if not provided (assume 50Hz)
    if imu_timestamps is None:
        imu_timestamps = np.arange(len(imu_data)) / 50.0
    
    # Generate skeleton timestamps at 30fps
    skel_timestamps = np.arange(len(skel_data)) / 30.0
    
    # If chosen method is 'resample', resample both to 30fps
    if method == 'resample':
        logger.info("Resampling IMU and skeleton data to fixed 30fps")
        try:
            # Create temporary arrays with timestamps
            imu_with_time = np.column_stack([imu_timestamps, imu_data])
            skel_with_time = np.column_stack([skel_timestamps, skel_data])
            
            # Resample IMU to 30fps
            resampled_imu = resample_to_fixed_rate(imu_with_time, target_fps=30.0)
            
            # Skeleton is already 30fps, just verify time steps
            if np.std(np.diff(skel_timestamps)) < 0.01:  # Already uniform
                resampled_skel = skel_with_time
            else:
                resampled_skel = resample_to_fixed_rate(skel_with_time, target_fps=30.0)
            
            # Find common time range
            min_time = max(resampled_imu[0, 0], resampled_skel[0, 0])
            max_time = min(resampled_imu[-1, 0], resampled_skel[-1, 0])
            
            # Extract common parts
            imu_mask = (resampled_imu[:, 0] >= min_time) & (resampled_imu[:, 0] <= max_time)
            skel_mask = (resampled_skel[:, 0] >= min_time) & (resampled_skel[:, 0] <= max_time)
            
            aligned_imu = resampled_imu[imu_mask, 1:]
            aligned_timestamps = resampled_imu[imu_mask, 0]
            aligned_skel = resampled_skel[skel_mask, 1:]
            
            # If lengths differ slightly due to floating point, adjust
            min_len = min(len(aligned_imu), len(aligned_skel))
            if min_len > 10:  # Ensure sufficient samples
                return aligned_imu[:min_len], aligned_skel[:min_len], aligned_timestamps[:min_len]
            else:
                logger.warning(f"Insufficient aligned samples after resampling: {min_len}")
                # Fall through to other methods
        except Exception as e:
            logger.warning(f"Resampling failed: {e}, trying other methods")
            # Fall through to other methods
    
    # Try DTW if requested
    if method in ['dtw', 'auto']:
        try:
            from dtaidistance import dtw
            
            # Extract wrist coordinates for alignment
            joint_dim = 3  # x, y, z per joint
            wrist_start = wrist_idx * joint_dim
            wrist_end = wrist_start + joint_dim
            
            if skel_data.shape[1] >= wrist_end:
                # Extract wrist movement
                wrist_pos = skel_data[:, wrist_start:wrist_end]
                
                # Compute wrist velocity
                wrist_vel = np.zeros(skel_data.shape[0])
                for i in range(1, len(wrist_vel)):
                    wrist_vel[i] = np.linalg.norm(wrist_pos[i] - wrist_pos[i-1]) / (1/30.0)  # 30fps
            else:
                # Fallback: use first 3 columns or overall movement
                logger.warning(f"Skeleton dimension {skel_data.shape[1]} too small for wrist extraction at index {wrist_idx}")
                if skel_data.shape[1] >= 3:
                    wrist_vel = np.linalg.norm(skel_data[:, :3], axis=1)
                else:
                    wrist_vel = np.linalg.norm(skel_data, axis=1)
            
            # IMU magnitude
            imu_mag = np.linalg.norm(imu_data, axis=1)
            
            # Normalize for DTW
            norm_imu = (imu_mag - np.mean(imu_mag)) / (np.std(imu_mag) + 1e-9)
            norm_wrist = (wrist_vel - np.mean(wrist_vel)) / (np.std(wrist_vel) + 1e-9)
            
            # DTW alignment
            path = dtw.warping_path(norm_imu, norm_wrist)
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
            
            # Create aligned arrays
            aligned_imu = imu_data[imu_unique]
            aligned_skel = skel_data[skel_unique]
            aligned_timestamps = imu_timestamps[imu_unique]
            
            if len(aligned_imu) >= 10:  # Ensure sufficient samples
                logger.info(f"DTW alignment successful: {len(aligned_imu)} points")
                return aligned_imu, aligned_skel, aligned_timestamps
            else:
                logger.warning(f"DTW alignment produced insufficient points: {len(aligned_imu)}")
                # Fall through to other methods
        except Exception as e:
            logger.warning(f"DTW alignment failed: {e}, trying interpolation")
    
    # Try interpolation
    try:
        # Find common time range
        t_min = max(imu_timestamps[0], skel_timestamps[0])
        t_max = min(imu_timestamps[-1], skel_timestamps[-1])
        
        if t_max <= t_min:
            logger.warning("No overlapping time range between modalities")
            return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
        
        # Filter IMU data to common range
        imu_mask = (imu_timestamps >= t_min) & (imu_timestamps <= t_max)
        filtered_imu = imu_data[imu_mask]
        filtered_timestamps = imu_timestamps[imu_mask]
        
        if len(filtered_timestamps) < 10:
            logger.warning(f"Insufficient IMU points in common time range: {len(filtered_timestamps)}")
            # Fall through to crop method
        else:
            # Interpolate skeleton to IMU timestamps
            from scipy.interpolate import interp1d
            
            interp_funcs = []
            for i in range(skel_data.shape[1]):
                interp_funcs.append(
                    interp1d(skel_timestamps, skel_data[:, i], 
                           bounds_error=False, fill_value="extrapolate")
                )
            
            # Apply interpolation
            interp_skel = np.zeros((len(filtered_timestamps), skel_data.shape[1]))
            for i, func in enumerate(interp_funcs):
                interp_skel[:, i] = func(filtered_timestamps)
            
            logger.info(f"Interpolation alignment successful: {len(filtered_timestamps)} points")
            return filtered_imu, interp_skel, filtered_timestamps
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}, falling back to crop alignment")
    
    # Last resort: simple crop alignment
    logger.info("Using basic crop alignment as fallback")
    min_len = min(len(imu_data), len(skel_data))
    
    if min_len < 10:  # Ensure sufficient samples
        logger.warning(f"Insufficient samples after cropping: {min_len}")
        return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
    
    return imu_data[:min_len], skel_data[:min_len], imu_timestamps[:min_len]

class Processor:
    """Base processor for handling sensor data loading and window creation."""
    
    def __init__(self, file_path='', mode='variable_time', max_length=128, 
                window_size_sec=4.0, stride_sec=1.0, resample_fps=None):
        """
        Initialize processor.
        
        Args:
            file_path: Path to data file
            mode: Processing mode ('variable_time', 'fixed', 'resample')
            max_length: Maximum sequence length
            window_size_sec: Window size in seconds
            stride_sec: Stride in seconds
            resample_fps: If specified, resample data to this frame rate
        """
        self.file_path = file_path
        self.mode = mode
        self.max_length = max_length
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.resample_fps = resample_fps

    def load_file(self, is_skeleton=False, is_gyroscope=False):
        """
        Load data file with robust error handling.
        
        Args:
            is_skeleton: Whether loading skeleton data
            is_gyroscope: Whether loading gyroscope data (unused)
            
        Returns:
            Loaded data array
        """
        try:
            if not self.file_path or not os.path.exists(self.file_path):
                logger.warning(f"File path invalid or not found: {self.file_path}")
                return np.zeros((0, 4 if not is_skeleton else 97), dtype=np.float32)
                
            if is_skeleton:
                # Load skeleton data
                try:
                    df = pd.read_csv(self.file_path, header=None)
                    data = df.values.astype(np.float32)
                    
                    # Add time column if not present
                    if data.shape[1] == 96:  # No time column
                        data = create_skeleton_timestamps(data, fps=30.0)
                    
                except Exception as e:
                    logger.warning(f"Error loading skeleton data: {e}")
                    return np.zeros((0, 97), dtype=np.float32)
            else:
                # Load IMU data
                data = parse_watch_csv(self.file_path)
            
            # Resample if requested
            if self.resample_fps is not None and data.shape[0] > 1:
                data = resample_to_fixed_rate(data, target_fps=self.resample_fps)
                
            logger.info(f"Loaded {self.file_path}: shape={data.shape}")
            return data
            
        except Exception as e:
            logger.warning(f"Error loading {self.file_path}: {e}")
            shape = (0, 97) if is_skeleton else (0, 4)
            return np.zeros(shape, dtype=np.float32)

    def process(self, data, is_fused=False):
        """
        Process data into windows.
        
        Args:
            data: Data array with time in first column
            is_fused: Whether data is already fused
            
        Returns:
            List of window arrays
        """
        if data.size == 0:
            logger.warning("Empty data, no windows created")
            return []
            
        if self.mode == 'variable_time':
            if data.shape[1] <= 4 or is_fused:  # IMU or fused data (fixed length)
                windows = sliding_windows_by_time_fixed(
                    data, 
                    self.window_size_sec, 
                    self.stride_sec, 
                    self.max_length
                )
            else:  # Skeleton (variable length)
                windows = sliding_windows_by_time(
                    data, 
                    self.window_size_sec, 
                    self.stride_sec
                )
        elif self.mode == 'resample':
            # First resample to fixed rate
            resampled = resample_to_fixed_rate(data, target_fps=30.0)
            # Then create windows
            windows = sliding_windows_by_time(
                resampled,
                self.window_size_sec,
                self.stride_sec
            )
        else:  # 'fixed' mode
            # Single window
            windows = [data]
            
        logger.info(f"Created {len(windows)} windows")
        return windows
