'''
Dataset Builder for SmartFallMM

This module handles loading, preprocessing, and alignment of multi-modal sensor data
for human activity recognition and fall detection. It provides robust support for
sensor fusion using the Madgwick filter algorithm.
'''
import os
import time
import traceback
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.io import loadmat
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from tqdm import tqdm
import threading

# Import IMU fusion module for thread pool management
from utils.imu_fusion import (
    process_imu_data, 
    extract_features_from_window,
    MadgwickFilter,
    save_aligned_sensor_data
)

# Configure logging
log_dir = "debug_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "loader.log"),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("loader")

# Add console handler for more immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Thread pool configuration
MAX_WORKER_THREADS = 8  # For general file operations
MAX_FUSION_THREADS = 4  # For sensor fusion processing
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)
file_semaphore = threading.Semaphore(MAX_FUSION_THREADS)

def update_thread_configuration(max_files: int, threads_per_file: int):
    """
    Update the thread pool configuration for parallel processing.
    
    Args:
        max_files: Maximum number of files to process in parallel
        threads_per_file: Number of threads to dedicate to each file
    """
    global MAX_WORKER_THREADS, MAX_FUSION_THREADS, thread_pool, file_semaphore
    
    # Ensure reasonable values
    max_files = max(1, min(max_files, 16))
    threads_per_file = max(1, min(threads_per_file, 8))
    
    new_total = max_files * threads_per_file
    
    # Only update if configuration changed significantly
    if abs(new_total - MAX_WORKER_THREADS) > 2:
        # Shutdown existing thread pool
        thread_pool.shutdown(wait=True)
        
        # Update configuration
        MAX_WORKER_THREADS = new_total
        MAX_FUSION_THREADS = max_files
        
        # Create new thread pool
        thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)
        file_semaphore = threading.Semaphore(MAX_FUSION_THREADS)
        
        logger.info(f"Updated thread configuration: {max_files} files × {threads_per_file} threads = {MAX_WORKER_THREADS} total")

def cleanup_resources():
    """Clean up thread pool resources properly to avoid hanging processes."""
    global thread_pool
    try:
        logger.info("Cleaning up thread pool resources")
        thread_pool.shutdown(wait=False)
        thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Load sensor data from CSV files with robust error handling.
    
    Handles different CSV formats including:
    - Standard inertial files (time, x, y, z)
    - Meta sensor files (epoch, time, elapsed time, x, y, z)
    - Skeleton files (96 columns of joint coordinates)
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments
        
    Returns:
        Numpy array containing loaded data
    """
    logger.debug(f"Loading CSV file: {file_path}")
    try:
        # Try with comma delimiter first
        try:
            file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        except Exception:
            # If that fails, try with semicolon delimiter
            logger.debug(f"Trying semicolon delimiter for {file_path}")
            file_data = pd.read_csv(file_path, index_col=False, header=None, sep=';').dropna().bfill()

        # Determine number of columns based on file type
        if 'skeleton' in file_path:
            cols = 96  # Skeleton data has 32 joints × 3 coordinates
        else:
            # Check if this is a meta sensor file
            if file_data.shape[1] > 4:
                # Meta sensor format: epoch, time, elapsed time, x, y, z
                cols = file_data.shape[1] - 3
                file_data = file_data.iloc[:, 3:]
            else:
                cols = 3  # Standard inertial data has 3 axes (x, y, z)

        # Handle case where file has fewer columns than expected
        if file_data.shape[1] < cols:
            logger.warning(f"File has fewer columns than expected: {file_data.shape[1]} < {cols}")
            # Add zero columns if needed
            missing_cols = cols - file_data.shape[1]
            for i in range(missing_cols):
                file_data[f'missing_{i}'] = 0

        # Extract data, skipping header rows if present
        if file_data.shape[0] > 2:
            activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        else:
            activity_data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)

        # Check for NaN values and replace with zeros
        if np.isnan(activity_data).any():
            logger.warning(f"NaN values found in {file_path}, replacing with zeros")
            activity_data = np.nan_to_num(activity_data, nan=0.0)
            
        # Check for infinite values and replace with large numbers
        if np.isinf(activity_data).any():
            logger.warning(f"Infinite values found in {file_path}, replacing with large finite values")
            activity_data = np.nan_to_num(activity_data, posinf=1e6, neginf=-1e6)

        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return empty array with appropriate shape
        if 'skeleton' in file_path:
            return np.zeros((1, 96), dtype=np.float32)
        else:
            return np.zeros((1, 3), dtype=np.float32)

def matloader(file_path: str, **kwargs) -> np.ndarray:
    '''
    Load data from MATLAB (.mat) files with robust error handling.

    Args:
        file_path: Path to the MAT file
        **kwargs: Additional arguments including 'key' to specify which data to load

    Returns:
        Numpy array with the loaded data
    '''
    logger.debug(f"Loading MAT file: {file_path}")

    try:
        # Check for valid key
        key = kwargs.get('key', None)
        if key not in ['d_iner', 'd_skel']:
            logger.error(f"Unsupported key for MatLab file: {key}")
            raise ValueError(f"Unsupported {key} for matlab file")

        # Load data from the specified key
        data = loadmat(file_path)[key]
        logger.debug(f"Loaded MAT data with shape: {data.shape}")
        
        # Check for NaN or infinite values
        if np.isnan(data).any() or np.isinf(data).any():
            logger.warning(f"NaN or infinite values found in {file_path}, replacing with valid values")
            data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
            
        return data
    except Exception as e:
        logger.error(f"Error loading MAT {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return empty array with appropriate shape for the key
        if key == 'd_iner':
            return np.zeros((1, 3), dtype=np.float32)
        else:  # d_skel
            return np.zeros((1, 96), dtype=np.float32)

# Map file extensions to appropriate loaders
LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

def align_sequence(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Align data from different sensors to a common time frame.
    
    This function handles both timestamp-based alignment and simple concatenation
    for fixed-rate data without timestamps.
    
    Args:
        data: Dictionary containing sensor data arrays from different modalities
        
    Returns:
        Dictionary with aligned data arrays and added timestamps
    """
    logger.debug("Aligning sequence data across modalities")
    
    try:
        # Check if we have all required modalities
        has_accel = 'accelerometer' in data and data['accelerometer'] is not None and len(data['accelerometer']) > 0
        has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and len(data['gyroscope']) > 0
        
        if not has_accel:
            logger.warning("Missing accelerometer data, alignment failed")
            return data
            
        # Extract sensor data and check validity
        acc_data = data['accelerometer']
        gyro_data = data['gyroscope'] if has_gyro else None
        
        if has_gyro and (len(acc_data) < 5 or len(gyro_data) < 5):
            logger.warning(f"Insufficient data for alignment: acc={len(acc_data)}, gyro={len(gyro_data) if gyro_data is not None else 0}")
            return data
        
        # Check if we already have timestamps in the accelerometer data
        # This assumes first column is time if data has more than 3 columns
        has_acc_timestamps = acc_data.shape[1] > 3
        has_gyro_timestamps = has_gyro and gyro_data.shape[1] > 3
        
        # If either sensor has timestamps, perform alignment
        if has_acc_timestamps or has_gyro_timestamps:
            logger.debug("Timestamp-based alignment")
            
            # Extract timestamps and sensor values
            if has_acc_timestamps:
                acc_times = acc_data[:, 0]
                acc_values = acc_data[:, 1:4]
            else:
                # Create synthetic timestamps at 30Hz
                acc_times = np.linspace(0, len(acc_data) / 30.0, len(acc_data))
                acc_values = acc_data
                
            if has_gyro and has_gyro_timestamps:
                gyro_times = gyro_data[:, 0]
                gyro_values = gyro_data[:, 1:4]
            elif has_gyro:
                # Create synthetic timestamps at 30Hz
                gyro_times = np.linspace(0, len(gyro_data) / 30.0, len(gyro_data))
                gyro_values = gyro_data
            
            # Find common time range
            if has_gyro:
                start_time = max(acc_times[0], gyro_times[0])
                end_time = min(acc_times[-1], gyro_times[-1])
                
                # Check for valid overlap
                if start_time >= end_time:
                    logger.warning("No time overlap between accelerometer and gyroscope data")
                    return data
                
                # Create common time grid at 50Hz
                sample_rate = 50.0
                duration = end_time - start_time
                num_samples = int(duration * sample_rate)
                
                if num_samples < 5:
                    logger.warning(f"Overlap too short: {duration:.2f}s")
                    return data
                
                common_times = np.linspace(start_time, end_time, num_samples)
                
                # Initialize arrays for interpolated data
                aligned_acc = np.zeros((num_samples, 3))
                aligned_gyro = np.zeros((num_samples, 3))
                
                # Perform linear interpolation for each axis
                for axis in range(3):
                    aligned_acc[:, axis] = np.interp(
                        common_times, 
                        acc_times, 
                        acc_values[:, axis]
                    )
                    
                    aligned_gyro[:, axis] = np.interp(
                        common_times, 
                        gyro_times, 
                        gyro_values[:, axis]
                    )
                
                # Update data dictionary with aligned data
                data['accelerometer'] = aligned_acc
                data['gyroscope'] = aligned_gyro
                data['aligned_timestamps'] = common_times
                
                logger.debug(f"Aligned {num_samples} samples from accelerometer and gyroscope")
            else:
                # Only accelerometer available, no actual alignment needed
                data['aligned_timestamps'] = acc_times
        else:
            # No timestamps available, create synthetic ones
            logger.debug("Creating synthetic timestamps for fixed-rate data")
            
            # Create timestamps at 30Hz
            acc_length = len(acc_data)
            data['aligned_timestamps'] = np.linspace(0, acc_length / 30.0, acc_length)
        
        return data
        
    except Exception as e:
        logger.error(f"Error during sequence alignment: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return original data if alignment fails
        return data

def avg_pool(sequence: np.ndarray, window_size: int = 5, stride: int = 1,
            max_length: int = 512, shape: Optional[Tuple] = None) -> np.ndarray:
    '''
    Applies average pooling to smooth data and reduce sequence length.

    Args:
        sequence: Input data sequence
        window_size: Size of pooling window
        stride: Stride for pooling
        max_length: Maximum target length
        shape: Shape of the input (used for reshaping)

    Returns:
        Pooled data sequence
    '''
    logger.debug(f"Applying avg_pool with window_size={window_size}, stride={stride}, max_length={max_length}")

    start_time = time.time()
    
    try:
        # Check for empty input
        if sequence is None or len(sequence) == 0:
            logger.warning("Empty sequence provided to avg_pool")
            return np.zeros((max_length, 3) if shape is None else shape)

        # Store original shape and reshape for pooling
        shape = sequence.shape if shape is None else shape
        
        # Handle 1D input
        if len(shape) == 1:
            sequence = sequence.reshape(-1, 1)
            
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)

        # Convert to torch tensor for F.avg_pool1d
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        # Calculate appropriate stride to achieve target length
        if max_length < sequence_tensor.shape[2]:
            stride = ((sequence_tensor.shape[2] // max_length) + 1)
            logger.debug(f"Adjusted stride to {stride} for max_length={max_length}")
        else:
            stride = 1

        # Apply pooling
        pooled = F.avg_pool1d(sequence_tensor, kernel_size=window_size, stride=stride)

        # Convert back to numpy and reshape
        pooled_np = pooled.squeeze(0).numpy().transpose(1, 0)
        
        # Handle reshaping edge cases
        if len(shape) > 1:
            try:
                result = pooled_np.reshape(-1, *shape[1:])
            except Exception:
                # If reshaping fails, use zero padding
                logger.warning(f"Reshaping failed, using zero padding. Pooled shape: {pooled_np.shape}, Target shape: (-1, {shape[1:]})")
                result = np.zeros((pooled_np.shape[0], *shape[1:]))
                result[:, :pooled_np.shape[1]] = pooled_np
        else:
            result = pooled_np

        elapsed_time = time.time() - start_time
        logger.debug(f"avg_pool complete: input shape {shape} → output shape {result.shape} in {elapsed_time:.4f}s")

        return result
        
    except Exception as e:
        logger.error(f"Error in avg_pool: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return zeros with target shape
        if shape is None:
            return np.zeros((max_length, 3))
        else:
            return np.zeros((max_length, *shape[1:]))

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int,
                      input_shape: np.ndarray) -> np.ndarray:
    '''
    Pads or truncates a sequence to a specified length with robust error handling.

    Args:
        sequence: Input data sequence
        max_sequence_length: Target sequence length
        input_shape: Shape of the input (used for reshaping)

    Returns:
        Padded/truncated sequence of uniform length
    '''
    logger.debug(f"Padding sequence to length {max_sequence_length}")
    
    try:
        # Check for empty input
        if sequence is None or len(sequence) == 0:
            logger.warning("Empty sequence provided to pad_sequence_numpy")
            new_shape = list(input_shape)
            new_shape[0] = max_sequence_length
            return np.zeros(new_shape)

        # Create target shape
        shape = list(input_shape)
        shape[0] = max_sequence_length

        # Apply pooling if needed
        pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)

        # Create zero-padded array of target shape
        new_sequence = np.zeros(shape, sequence.dtype)

        # Fill with pooled data up to available length
        actual_length = min(len(pooled_sequence), max_sequence_length)
        new_sequence[:actual_length] = pooled_sequence[:actual_length]

        logger.debug(f"Padding complete: shape {input_shape} → {new_sequence.shape}")

        return new_sequence
        
    except Exception as e:
        logger.error(f"Error in pad_sequence_numpy: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return zeros with target shape
        new_shape = list(input_shape)
        new_shape[0] = max_sequence_length
        return np.zeros(new_shape)

def sliding_window(data: np.ndarray, clearing_time_index: int, max_time: int,
                  sub_window_size: int, stride_size: int) -> np.ndarray:
    '''
    Extracts sliding windows from a time series with robust error handling.

    Args:
        data: Input data array
        clearing_time_index: Minimum index to start windows from
        max_time: Maximum time index to consider
        sub_window_size: Size of each window
        stride_size: Stride between consecutive windows

    Returns:
        Array of sliding windows
    '''
    logger.debug(f"Creating sliding windows with window_size={sub_window_size}, stride={stride_size}")
    
    try:
        # Check for empty input
        if data is None or len(data) == 0:
            logger.warning("Empty data provided to sliding_window")
            return np.zeros((1, sub_window_size, data.shape[1] if data is not None and len(data.shape) > 1 else 3))

        # Validate parameters
        if clearing_time_index < sub_window_size - 1:
            logger.warning(f"Invalid clearing_time_index: {clearing_time_index} < {sub_window_size - 1}")
            clearing_time_index = sub_window_size - 1

        # Calculate starting index
        start = clearing_time_index - sub_window_size + 1

        # Adjust max_time if needed to prevent out-of-bounds access
        if max_time >= data.shape[0] - sub_window_size:
            max_time = data.shape[0] - sub_window_size
            logger.debug(f"Adjusted max_time to {max_time}")
            
        # Handle case where max_time is too small
        if max_time < 0:
            logger.warning(f"max_time ({max_time}) is negative, using 0")
            max_time = 0

        # Handle case where start is too large
        if start >= max_time:
            logger.warning(f"start ({start}) >= max_time ({max_time}), using max_time-1")
            start = max(0, max_time - 1)

        # Generate indices for all windows
        sub_windows = (
            start +
            np.expand_dims(np.arange(sub_window_size), 0) +
            np.expand_dims(np.arange(start, max_time, step=stride_size), 0).T
        )

        # Extract windows using the indices
        result = data[sub_windows]
        logger.debug(f"Created {result.shape[0]} windows from data with shape {data.shape}")

        return result
        
    except Exception as e:
        logger.error(f"Error in sliding_window: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return single window of zeros
        if data is not None and len(data.shape) > 1:
            return np.zeros((1, sub_window_size, data.shape[1]))
        else:
            return np.zeros((1, sub_window_size, 3))

def hybrid_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray,
                      threshold: float = 2.0, window_size: int = 5) -> np.ndarray:
    """
    Hybrid interpolation that intelligently switches between cubic spline and linear
    interpolation based on the rate of change in the data.

    Args:
        x: Original x coordinates (timestamps)
        y: Original y coordinates (sensor values)
        x_new: New x coordinates for interpolation
        threshold: Rate of change threshold to switch methods (g/s for accelerometer)
        window_size: Window size for smoothing rate calculation

    Returns:
        Interpolated y values at x_new points
    """
    # Ensure we have enough data points for interpolation
    if len(x) < 2 or len(y) < 2:
        logger.warning("Not enough points for interpolation")
        return np.full_like(x_new, y[0] if len(y) > 0 else 0.0)

    try:
        # Calculate first differences to estimate rate of change
        dy = np.diff(y)
        dx = np.diff(x)
        
        # Avoid division by zero
        nonzero_dx = np.maximum(dx, 1e-10)
        rates = np.abs(dy / nonzero_dx)

        # Smooth the rates to avoid switching too frequently
        if len(rates) >= window_size:
            try:
                rates = savgol_filter(rates, window_size, 2)
            except Exception:
                # Fall back to moving average if savgol fails
                kernel = np.ones(window_size) / window_size
                rates = np.convolve(rates, kernel, mode='same')

        # Create mask for rapid changes
        rapid_changes = rates > threshold

        # If no rapid changes detected, use cubic spline for everything
        if not np.any(rapid_changes):
            logger.debug("Using cubic spline interpolation for entire signal")
            try:
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(x, y)
                return cs(x_new)
            except Exception as e:
                logger.warning(f"Cubic spline failed: {e}, falling back to linear")
                from scipy.interpolate import interp1d
                linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
                return linear_interp(x_new)

        # If all changes are rapid, use linear for everything
        if np.all(rapid_changes):
            logger.debug("Using linear interpolation for entire signal")
            from scipy.interpolate import interp1d
            linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            return linear_interp(x_new)

        # Otherwise, we need a hybrid approach
        logger.debug(f"Using hybrid interpolation: {np.sum(rapid_changes)}/{len(rapid_changes)} points have rapid changes")

        # Create interpolators for both methods
        from scipy.interpolate import interp1d, CubicSpline
        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        try:
            spline_interp = CubicSpline(x, y)
        except Exception as e:
            logger.warning(f"Cubic spline failed: {e}, using linear for all points")
            return linear_interp(x_new)

        # Find segments with rapid changes
        y_interp = np.zeros_like(x_new, dtype=float)
        segments = []

        # Group consecutive points with rapid changes into segments
        segment_start = None
        for i in range(len(rapid_changes)):
            if rapid_changes[i] and segment_start is None:
                segment_start = i
            elif not rapid_changes[i] and segment_start is not None:
                segments.append((segment_start, i))
                segment_start = None

        # Add the last segment if it exists
        if segment_start is not None:
            segments.append((segment_start, len(rapid_changes)))

        # Create mask for points that need linear interpolation
        linear_mask = np.zeros_like(x_new, dtype=bool)

        # Mark regions around rapid changes (with buffer)
        buffer = 0.05  # 50ms buffer
        for start_idx, end_idx in segments:
            # Convert indices to timestamps with buffer
            t_start = max(x[start_idx] - buffer, x[0])
            t_end = min(x[min(end_idx, len(x)-1)] + buffer, x[-1])

            # Mark points in the region
            linear_mask |= (x_new >= t_start) & (x_new <= t_end)

        # Apply appropriate interpolation to each region
        if np.any(linear_mask):
            y_interp[linear_mask] = linear_interp(x_new[linear_mask])

        if np.any(~linear_mask):
            y_interp[~linear_mask] = spline_interp(x_new[~linear_mask])

        return y_interp

    except Exception as e:
        logger.error(f"Hybrid interpolation failed: {e}")
        logger.error(traceback.format_exc())
        
        # Fallback to simple linear interpolation
        try:
            from scipy.interpolate import interp1d
            linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            return linear_interp(x_new)
        except Exception:
            # Ultimate fallback: return zeros
            return np.zeros_like(x_new)

def _extract_window(data, start, end, window_size, fuse, filter_type='madgwick'):
    """
    Helper function to extract a window from data with proper fusion handling.
    Designed to be run in a separate thread for parallel processing.
    
    Args:
        data: Dictionary of sensor data arrays
        start: Start index for window
        end: End index for window
        window_size: Target window size
        fuse: Whether to apply fusion
        filter_type: Type of filter to use (default: 'madgwick')
        
    Returns:
        Dictionary of windowed data
    """
    try:
        window_data = {}
        
        # Extract window for each modality
        for modality, modality_data in data.items():
            if modality == 'labels':
                continue
                
            if modality_data is None or len(modality_data) == 0:
                continue
                
            try:
                # Handle the modality based on its type
                if modality == 'aligned_timestamps':
                    # Handle 1D array - no second dimension indexing
                    if len(modality_data.shape) == 1:
                        window_data_array = modality_data[start:min(end, len(modality_data))]
                        
                        # Pad if needed
                        if len(window_data_array) < window_size:
                            padded = np.zeros(window_size, dtype=window_data_array.dtype)
                            padded[:len(window_data_array)] = window_data_array
                            window_data_array = padded
                    else:
                        # If somehow it's not 1D, fall back to regular handling
                        window_data_array = modality_data[start:min(end, len(modality_data)), :]
                        if window_data_array.shape[0] < window_size:
                            padded = np.zeros((window_size, window_data_array.shape[1]), dtype=window_data_array.dtype)
                            padded[:window_data_array.shape[0]] = window_data_array
                            window_data_array = padded
                else:
                    # Regular handling for 2D arrays
                    max_idx = min(end, len(modality_data))
                    if start >= max_idx:
                        # Handle invalid indices
                        logger.warning(f"Invalid indices for {modality}: start={start}, end={end}, len={len(modality_data)}")
                        if modality == 'accelerometer':
                            window_data_array = np.zeros((window_size, 3))
                        elif modality == 'gyroscope':
                            window_data_array = np.zeros((window_size, 3))
                        elif modality == 'quaternion':
                            window_data_array = np.zeros((window_size, 4))
                        else:
                            continue
                    else:
                        # Extract window from data
                        window_data_array = modality_data[start:max_idx, :]
                        
                        # Pad if needed
                        if window_data_array.shape[0] < window_size:
                            padded = np.zeros((window_size, window_data_array.shape[1]), dtype=window_data_array.dtype)
                            padded[:window_data_array.shape[0]] = window_data_array
                            window_data_array = padded
                
                # Store in window data dictionary
                window_data[modality] = window_data_array
            except Exception as e:
                logger.error(f"Error extracting {modality} window: {str(e)}")
                # Add empty data with correct shape for critical modalities
                if modality == 'accelerometer':
                    window_data[modality] = np.zeros((window_size, 3))
                elif modality == 'gyroscope':
                    window_data[modality] = np.zeros((window_size, 3))
                elif modality == 'quaternion':
                    window_data[modality] = np.zeros((window_size, 4))
        
        # Apply fusion if requested and we have both accelerometer and gyroscope
        if fuse and 'accelerometer' in window_data and 'gyroscope' in window_data:
            try:
                # Extract the window data
                acc_window = window_data['accelerometer']
                gyro_window = window_data['gyroscope']
                
                # Extract timestamps if available, otherwise create synthetic ones
                timestamps = None
                if 'aligned_timestamps' in window_data:
                    timestamps = window_data['aligned_timestamps']
                    # Convert to 1D array if needed
                    if len(timestamps.shape) > 1:
                        timestamps = timestamps[:, 0] if timestamps.shape[1] > 0 else None
                
                # Process data using Madgwick filter
                fusion_results = process_imu_data(
                    acc_data=acc_window,
                    gyro_data=gyro_window,
                    timestamps=timestamps,
                    filter_type='madgwick',  # Always use Madgwick for now
                    return_features=False
                )
                
                # Add fusion results to window data
                window_data['quaternion'] = fusion_results.get('quaternion', np.zeros((window_size, 4)))
                window_data['linear_acceleration'] = fusion_results.get('linear_acceleration', acc_window)
                    
                logger.debug(f"Added fusion data to window using Madgwick filter")
            except Exception as e:
                logger.error(f"Error in fusion processing: {str(e)}")
                # Add empty quaternion as fallback
                window_data['quaternion'] = np.zeros((window_size, 4))
        else:
            # Always add empty quaternion data as a fallback if not present
            if 'quaternion' not in window_data:
                window_data['quaternion'] = np.zeros((window_size, 4))
        
        # Final validation of quaternion data
        if 'quaternion' not in window_data or window_data['quaternion'] is None:
            window_data['quaternion'] = np.zeros((window_size, 4))
        elif window_data['quaternion'].shape[0] != window_size:
            # Fix quaternion shape if needed
            temp = np.zeros((window_size, 4))
            if window_data['quaternion'].shape[0] < window_size:
                temp[:window_data['quaternion'].shape[0]] = window_data['quaternion']
            else:
                temp = window_data['quaternion'][:window_size]
            window_data['quaternion'] = temp
        
        # IMPORTANT: Remove aligned_timestamps from window data to avoid KeyError in batching
        if 'aligned_timestamps' in window_data:
            del window_data['aligned_timestamps']
            
        return window_data
    except Exception as e:
        logger.error(f"Error in _extract_window: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return minimal valid data
        return {
            'accelerometer': np.zeros((window_size, 3)),
            'quaternion': np.zeros((window_size, 4))
        }

def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int,
                           peaks: Union[List[int], np.ndarray], label: int,
                           fuse: bool, filter_type: str = 'madgwick') -> Dict[str, np.ndarray]:
    """
    Creates windows centered around detected peaks with IMU fusion.
    
    Args:
        data: Dictionary of sensor data arrays
        window_size: Size of each window
        peaks: List of peak indices to center windows on
        label: Label for this activity
        fuse: Whether to apply sensor fusion
        filter_type: Type of fusion filter to use (default: 'madgwick')
        
    Returns:
        Dictionary of windowed data arrays
    """
    start_time = time.time()
    
    # Validate inputs
    num_peaks = len(peaks) if peaks is not None else 0
    logger.info(f"Creating {num_peaks} selective windows with fusion={fuse}, filter={filter_type}")
    
    if num_peaks == 0:
        logger.warning("No peaks provided for windowing")
        # Return empty data dictionary with correct structure
        return {
            'accelerometer': np.zeros((0, window_size, 3)),
            'gyroscope': np.zeros((0, window_size, 3)) if 'gyroscope' in data else None,
            'quaternion': np.zeros((0, window_size, 4)),
            'labels': np.array([])
        }
    
    # Initialize result dictionary
    windowed_data = defaultdict(list)
    
    # Check for required modalities
    has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and len(data['gyroscope']) > 0
    if fuse and not has_gyro:
        logger.warning("Fusion requested but gyroscope data not available, will use accelerometer only")
    
    # Create local thread pool for windows
    max_workers = min(8, len(peaks)) if len(peaks) > 0 else 1
    with ThreadPoolExecutor(max_workers=max_workers) as local_pool:
        # Create windows around peaks with parallel processing
        futures = []
        
        # Submit window extraction tasks
        for peak in peaks:
            # Calculate window boundaries
            half_window = window_size // 2
            start = max(0, peak - half_window)
            end = min(len(data['accelerometer']), start + window_size)
            
            # Skip if window is too small
            if end - start < window_size // 2:
                logger.debug(f"Skipping window at peak {peak}: too small ({end-start} < {window_size//2})")
                continue
                
            # Submit task
            futures.append(local_pool.submit(
                _extract_window,
                data,
                start,
                end,
                window_size,
                fuse,
                filter_type
            ))
            
        # Collect results with progress bar if there are many windows
        windows_created = 0
        use_progress = len(futures) > 10
        
        if use_progress:
            collection_iterator = tqdm(futures, desc="Processing windows")
        else:
            collection_iterator = futures
            
        for future in collection_iterator:
            try:
                window_data = future.result()
                
                # Verify quaternion data exists before adding window
                if 'quaternion' not in window_data or window_data['quaternion'] is None:
                    logger.warning(f"Window missing quaternion data, adding zeros")
                    window_data['quaternion'] = np.zeros((window_size, 4))
                    
                # Add this window's data to the result dictionary
                for modality, modality_window in window_data.items():
                    if modality_window is not None:
                        windowed_data[modality].append(modality_window)
                
                windows_created += 1
            except Exception as e:
                logger.error(f"Error processing window: {str(e)}")
    
    # Convert lists of arrays to arrays
    for modality in list(windowed_data.keys()):
        if modality != 'labels' and len(windowed_data[modality]) > 0:
            try:
                windowed_data[modality] = np.array(windowed_data[modality])
                logger.debug(f"Converted {modality} windows to array with shape {windowed_data[modality].shape}")
            except Exception as e:
                logger.error(f"Error converting {modality} windows to array: {str(e)}")
                # Ensure quaternion data exists in case of error
                if modality == 'quaternion':
                    # Create empty quaternion data of correct shape
                    windowed_data[modality] = np.zeros((windows_created, window_size, 4))
                    
                # Remove problematic modalities
                if modality != 'quaternion' and modality != 'accelerometer':
                    del windowed_data[modality]
    
    # Add labels
    windowed_data['labels'] = np.repeat(label, windows_created)
    
    # CRITICAL: Ensure quaternion data exists
    if 'quaternion' not in windowed_data or len(windowed_data['quaternion']) == 0:
        logger.warning("No quaternion data in final windows, adding zeros")
        if 'accelerometer' in windowed_data and len(windowed_data['accelerometer']) > 0:
            num_windows = len(windowed_data['accelerometer'])
            windowed_data['quaternion'] = np.zeros((num_windows, window_size, 4))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Created {windows_created} windows in {elapsed_time:.2f}s")
    
    return windowed_data

class DatasetBuilder:
    '''
    Builds a dataset from sensor data files for machine learning.

    This class handles loading, pre-processing, alignment, and normalization of
    multi-modal sensor data for human activity recognition and fall detection tasks.

    Args:
        dataset: Dataset object containing matched trials
        mode: Processing mode ('avg_pool' or 'sliding_window')
        max_length: Maximum sequence length
        task: Task type ('fd' for fall detection, 'har' for activity recognition, 'age' for age detection)
        fusion_options: Configuration options for sensor fusion
        **kwargs: Additional arguments
    '''
    def __init__(self, dataset: object, mode: str, max_length: int, task='fd', 
                 fusion_options=None, **kwargs) -> None:
        logger.info(f"Initializing DatasetBuilder with mode={mode}, task={task}")

        if mode not in ['avg_pool', 'sliding_window']:
            logger.error(f"Unsupported processing method: {mode}")
            raise ValueError(f"Unsupported processing method {mode}")

        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.fusion_options = fusion_options or {}

        # Create directory for aligned data
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        os.makedirs(self.aligned_data_dir, exist_ok=True)
        for dir_name in ["accelerometer", "gyroscope", "skeleton", "quaternion"]:
            os.makedirs(os.path.join(self.aligned_data_dir, dir_name), exist_ok=True)

        # Log fusion options if present
        if fusion_options:
            fusion_enabled = fusion_options.get('enabled', False)
            filter_type = fusion_options.get('filter_type', 'madgwick')
            logger.info(f"Fusion options: enabled={fusion_enabled}, filter_type={filter_type}")

    def load_file(self, file_path):
        '''
        Loads sensor data from a file with robust error handling.

        Args:
            file_path: Path to the data file

        Returns:
            Numpy array containing the loaded data
        '''
        logger.debug(f"Loading file: {file_path}")

        try:
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return None
                
            # Import appropriate loader
            loader = self._import_loader(file_path)
            
            # Load data
            data = loader(file_path, **self.kwargs)
            
            # Validate loaded data
            if data is None or len(data) == 0:
                logger.warning(f"Empty data loaded from {file_path}")
                return None
                
            # Check for NaN values
            if np.isnan(data).any():
                logger.warning(f"NaN values found in {file_path}, replacing with zeros")
                data = np.nan_to_num(data, nan=0.0)
                
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _import_loader(self, file_path: str):
        '''
        Determines appropriate loader based on file extension.

        Args:
            file_path: Path to the data file

        Returns:
            Loader function for the file type
        '''
        file_type = file_path.split('.')[-1]

        if file_type not in ['csv', 'mat']:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type {file_type}")

        return LOADER_MAP[file_type]

    def process(self, data, label, fuse=False, filter_type='madgwick', visualize=False):
        '''
        Processes data using either average pooling or peak-based sliding windows.

        Args:
            data: Dictionary of sensor data
            label: Activity label
            fuse: Whether to apply sensor fusion
            filter_type: Type of filter to use
            visualize: Whether to generate visualizations

        Returns:
            Dictionary of processed data
        '''
        logger.info(f"Processing data for label {label} with mode={self.mode}, fusion={fuse}")

        # Basic validation of input data
        if 'accelerometer' not in data or data['accelerometer'] is None or len(data['accelerometer']) == 0:
            logger.error("Missing accelerometer data, cannot process")
            return {}

        if self.mode == 'avg_pool':
            # Use average pooling to create fixed-length data
            logger.debug("Applying average pooling")
            processed_data = {}

            # Process each modality
            for modality, modality_data in data.items():
                if modality != 'labels' and modality_data is not None and len(modality_data) > 0:
                    # Skip aligned_timestamps
                    if modality == 'aligned_timestamps':
                        continue
                        
                    # Pad sequence to fixed length
                    processed_data[modality] = pad_sequence_numpy(
                        sequence=modality_data,
                        max_sequence_length=self.max_length,
                        input_shape=modality_data.shape
                    )

            # Add label
            processed_data['labels'] = np.array([label])

            # Apply fusion if requested
            if fuse and 'accelerometer' in processed_data and 'gyroscope' in processed_data:
                try:
                    logger.debug(f"Applying sensor fusion with Madgwick filter")
                    
                    # Extract timestamps if available
                    timestamps = data.get('aligned_timestamps', None)
                    
                    # Process with IMU fusion
                    fusion_result = process_imu_data(
                        acc_data=processed_data['accelerometer'],
                        gyro_data=processed_data['gyroscope'],
                        timestamps=timestamps,
                        filter_type='madgwick',  # Always use Madgwick
                        return_features=False
                    )
                    
                    # Add fusion results to processed data
                    processed_data.update(fusion_result)
                    
                except Exception as e:
                    logger.error(f"Fusion processing failed: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Add default quaternion data to ensure it exists
                    processed_data['quaternion'] = np.zeros((self.max_length, 4))

            # Ensure quaternion data exists
            if 'quaternion' not in processed_data:
                processed_data['quaternion'] = np.zeros((self.max_length, 4))

            return processed_data
        else:
            # Use peak detection for windowing
            logger.debug("Using peak detection for windowing")

            # Calculate magnitude for peak detection
            sqrt_sum = np.sqrt(np.sum(data['accelerometer']**2, axis=1))

            # Set peak detection parameters based on label
            if label == 1:  # Fall
                logger.debug("Using fall detection peak parameters")
                height = max(10, 0.8 * np.max(sqrt_sum))  # Adaptive threshold
                distance = max(10, min(50, len(sqrt_sum) // 10))  # Adaptive distance
                peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
            else:  # Non-fall
                logger.debug("Using non-fall peak parameters")
                height = max(8, 0.6 * np.max(sqrt_sum))  # Adaptive threshold
                distance = max(20, min(100, len(sqrt_sum) // 5))  # Adaptive distance
                peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)

            # If no peaks found, use simple division of sequence
            if len(peaks) == 0:
                logger.warning("No peaks found, creating evenly spaced windows")
                # Create evenly spaced "peaks"
                num_windows = max(1, len(sqrt_sum) // self.max_length)
                peaks = np.linspace(self.max_length//2, len(sqrt_sum) - self.max_length//2, num_windows).astype(int)

            logger.debug(f"Found {len(peaks)} peaks")

            # Generate visualizations if requested
            if visualize:
                try:
                    # Create directory for visualizations
                    viz_dir = os.path.join(os.getcwd(), "visualizations")
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    # Plot signal with peaks
                    plt.figure(figsize=(12, 6))
                    plt.plot(sqrt_sum)
                    plt.plot(peaks, sqrt_sum[peaks], "x", color='red')
                    plt.title(f"{'Fall' if label == 1 else 'Non-Fall'} Peak Detection")
                    plt.xlabel("Sample")
                    plt.ylabel("Acceleration Magnitude")
                    
                    # Save figure
                    plt.savefig(os.path.join(viz_dir, f"peaks_label{label}_{int(time.time())}.png"))
                    plt.close()
                except Exception as e:
                    logger.error(f"Error creating visualization: {str(e)}")

            # Extract windows around peaks with optional fusion
            processed_data = selective_sliding_window(
                data=data,
                window_size=self.max_length,
                peaks=peaks,
                label=label,
                fuse=fuse,
                filter_type='madgwick'  # Always use Madgwick
            )

            return processed_data

    def _add_trial_data(self, trial_data):
        '''
        Adds processed trial data to the dataset with validation.

        Args:
            trial_data: Dictionary of processed sensor data for a trial
        '''
        logger.debug("Adding trial data to dataset")

        # Skip empty trial data
        if not trial_data or len(trial_data) == 0:
            logger.warning("Empty trial data, skipping")
            return
            
        # Skip trial data without labels
        if 'labels' not in trial_data or trial_data['labels'] is None or len(trial_data['labels']) == 0:
            logger.warning("Trial data missing labels, skipping")
            return

        # Add each modality to dataset
        for modality, modality_data in trial_data.items():
            if modality_data is None or len(modality_data) == 0:
                logger.warning(f"Empty {modality} data, skipping")
                continue
                
            try:
                self.data[modality].append(modality_data)
                logger.debug(f"Added {modality} data with shape {modality_data.shape if hasattr(modality_data, 'shape') else len(modality_data)}")
            except Exception as e:
                logger.error(f"Error adding {modality} data: {str(e)}")

    def _len_check(self, d):
        '''
        Checks if data dictionary has sufficient length in each modality.

        Args:
            d: Dictionary of data arrays

        Returns:
            Boolean indicating if all modalities have sufficient data
        '''
        if not d or len(d) == 0:
            return False
            
        return all(len(v) > 0 for k, v in d.items() if k != 'aligned_timestamps')

    def _process_trial(self, trial, label, fuse, filter_type, visualize, save_aligned=False):
        """
        Process a single trial with robust error handling.

        Args:
            trial: Trial object containing modality file paths
            label: Class label for this trial
            fuse: Whether to use sensor fusion
            filter_type: Type of filter to use
            visualize: Whether to generate visualizations
            save_aligned: Whether to save aligned data to files

        Returns:
            Processed trial data or None if processing failed
        """
        try:
            # Create dictionary to hold trial data
            trial_data = {}
            
            # Load data from each modality
            for modality, file_path in trial.files.items():
                try:
                    unimodal_data = self.load_file(file_path)
                    if unimodal_data is not None and len(unimodal_data) > 0:
                        trial_data[modality] = unimodal_data
                    else:
                        logger.warning(f"Empty or missing data for {modality} from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {modality} from {file_path}: {str(e)}")
            
            # Skip if missing accelerometer data
            if 'accelerometer' not in trial_data or trial_data['accelerometer'] is None or len(trial_data['accelerometer']) == 0:
                logger.warning(f"Missing accelerometer data for trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number}")
                return None
            
            # Align sequence data
            try:
                trial_data = align_sequence(trial_data)
            except Exception as e:
                logger.error(f"Error aligning data: {str(e)}")
            
            # Save aligned data if requested
            if save_aligned:
                try:
                    aligned_acc = trial_data.get('accelerometer')
                    aligned_gyro = trial_data.get('gyroscope')
                    aligned_skl = trial_data.get('skeleton')
                    aligned_timestamps = trial_data.get('aligned_timestamps')
                    
                    if aligned_acc is not None and aligned_gyro is not None:
                        save_aligned_sensor_data(
                            trial.subject_id, 
                            trial.action_id, 
                            trial.sequence_number,
                            aligned_acc,
                            aligned_gyro,
                            aligned_skl,
                            aligned_timestamps if aligned_timestamps is not None else None
                        )
                except Exception as e:
                    logger.error(f"Error saving aligned data: {str(e)}")
            
            # Process the aligned data
            processed_data = self.process(trial_data, label, fuse, filter_type, visualize)
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Trial processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def make_dataset(self, subjects: List[int], fuse: bool, filter_type: str = 'madgwick', 
                    visualize: bool = False, save_aligned: bool = False):
        '''
        Creates a dataset from the sensor data files for the specified subjects.
        
        Args:
            subjects: List of subject IDs to include
            fuse: Whether to apply sensor fusion
            filter_type: Type of fusion filter to use
            visualize: Whether to generate visualizations
            save_aligned: Whether to save aligned data to files
        '''
        # Use madgwick filter type regardless of input
        filter_type = 'madgwick'
        
        logger.info(f"Making dataset for subjects={subjects}, fuse={fuse}, filter_type={filter_type}")

        start_time = time.time()
        self.data = defaultdict(list)
        self.fuse = fuse
        
        # Check if save_aligned is specified in fusion options
        if hasattr(self, 'fusion_options'):
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)

        # Validate subjects list
        if not subjects or len(subjects) == 0:
            logger.error("No subjects specified for dataset creation")
            return
            
        # Validate matched_trials
        if not hasattr(self.dataset, 'matched_trials') or len(self.dataset.matched_trials) == 0:
            logger.error("No matched trials found in dataset")
            return

        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(MAX_WORKER_THREADS, len(self.dataset.matched_trials))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary to track futures for each trial
            future_to_trial = {}
            
            # Submit tasks for processing each trial
            for trial in self.dataset.matched_trials:
                if trial.subject_id not in subjects:
                    continue
                
                # Determine label based on task
                if self.task == 'fd':  # Fall detection
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:  # Activity recognition
                    label = trial.action_id - 1
                
                future = executor.submit(
                    self._process_trial, 
                    trial, 
                    label, 
                    fuse, 
                    filter_type,
                    visualize,
                    save_aligned
                )
                future_to_trial[future] = trial
            
            # Collect results with progress tracking
            count = 0
            processed_count = 0
            skipped_count = 0
            
            # Use tqdm for progress tracking if many trials
            if len(future_to_trial) > 5:
                iterator = tqdm(as_completed(future_to_trial), total=len(future_to_trial), desc="Processing trials")
            else:
                iterator = as_completed(future_to_trial)
            
            for future in iterator:
                trial = future_to_trial[future]
                count += 1
                
                try:
                    trial_data = future.result()
                    if trial_data is not None and self._len_check(trial_data):
                        self._add_trial_data(trial_data)
                        processed_count += 1
                    else:
                        logger.warning(f"Skipping empty or invalid trial data for {trial.subject_id}-{trial.action_id}-{trial.sequence_number}")
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error processing trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number}: {str(e)}")
                    logger.error(traceback.format_exc())
                    skipped_count += 1

        # Concatenate data from all trials
        for key in self.data:
            values = self.data[key]
            if all(isinstance(x, np.ndarray) for x in values):
                try:
                    self.data[key] = np.concatenate(values, axis=0)
                    logger.info(f"Concatenated {key} data with shape {self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error concatenating {key} data: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Try to identify the issue
                    shapes = [x.shape for x in values]
                    logger.error(f"Shapes of arrays for {key}: {shapes}")
                    
                    # Remove problematic key
                    if key != 'accelerometer' and key != 'labels':
                        self.data[key] = None
            else:
                logger.warning(f"Cannot concatenate {key} data - mixed types")
                
                # Remove problematic key
                if key != 'accelerometer' and key != 'labels':
                    self.data[key] = None

        # Ensure quaternion data exists in final dataset
        if 'quaternion' not in self.data and 'accelerometer' in self.data:
            logger.warning("Adding empty quaternion data to final dataset")
            acc_shape = self.data['accelerometer'].shape
            self.data['quaternion'] = np.zeros((acc_shape[0], acc_shape[1], 4))
            
        # Log label distribution
        if 'labels' in self.data and self.data['labels'] is not None and len(self.data['labels']) > 0:
            label_values, label_counts = np.unique(self.data['labels'], return_counts=True)
            label_distribution = {int(l): int(c) for l, c in zip(label_values, label_counts)}
            logger.info(f"Label distribution:")
            for label, count in label_distribution.items():
                percentage = 100.0 * count / len(self.data['labels'])
                logger.info(f"  Class {label}: {count} samples ({percentage:.1f}%)")

        elapsed_time = time.time() - start_time
        logger.info(f"Dataset creation complete: processed {processed_count}/{count} trials, skipped {skipped_count} in {elapsed_time:.2f}s")

    def normalization(self) -> Dict[str, np.ndarray]:
        '''
        Normalizes each modality in the dataset.

        Returns:
            Dictionary with normalized data for each modality
        '''
        logger.info("Normalizing dataset")

        start_time = time.time()
        norm_data = {}

        # Skip if no data
        if not self.data or len(self.data) == 0:
            logger.error("No data to normalize")
            return {}
            
        # Skip normalization if only 'labels' present
        if len(self.data) == 1 and 'labels' in self.data:
            logger.error("Only labels present, no data to normalize")
            return self.data

        # Normalize each modality separately (except labels)
        for key, value in self.data.items():
            # Skip None values
            if value is None:
                continue
                
            # Copy labels directly
            if key == 'labels':
                norm_data[key] = value
                continue
                
            # Skip empty data
            if len(value) == 0:
                logger.warning(f"Empty {key} data, skipping normalization")
                continue
                
            try:
                # Check if this is a feature that needs normalization
                if key in ['accelerometer', 'gyroscope', 'quaternion', 'linear_acceleration'] and len(value.shape) >= 2:
                    # Reshape for standardization
                    num_samples, length = value.shape[:2]
                    orig_shape = value.shape

                    # StandardScaler works on 2D data, so reshape
                    reshaped_data = value.reshape(num_samples * length, -1)

                    # Check for NaN or infinite values
                    if np.isnan(reshaped_data).any() or np.isinf(reshaped_data).any():
                        logger.warning(f"NaN or infinite values found in {key} data, replacing with valid values")
                        reshaped_data = np.nan_to_num(reshaped_data, nan=0.0, posinf=1e6, neginf=-1e6)

                    # Standardize data
                    try:
                        norm_data_array = StandardScaler().fit_transform(reshaped_data)
                    except Exception as e:
                        logger.error(f"StandardScaler failed for {key}: {str(e)}")
                        # Fall back to simple normalization
                        mean = np.mean(reshaped_data, axis=0, keepdims=True)
                        std = np.std(reshaped_data, axis=0, keepdims=True) + 1e-8  # Avoid division by zero
                        norm_data_array = (reshaped_data - mean) / std

                    # Reshape back to original shape
                    norm_data[key] = norm_data_array.reshape(orig_shape)

                    logger.debug(f"Normalized {key} data: shape={norm_data[key].shape}")
                elif key == 'fusion_features' and len(value.shape) == 2:
                    # These are already extracted features, normalize them directly
                    try:
                        norm_data[key] = StandardScaler().fit_transform(value)
                    except Exception as e:
                        logger.error(f"StandardScaler failed for {key}: {str(e)}")
                        # Fall back to simple normalization
                        mean = np.mean(value, axis=0, keepdims=True)
                        std = np.std(value, axis=0, keepdims=True) + 1e-8  # Avoid division by zero
                        norm_data[key] = (value - mean) / std
                        
                    logger.debug(f"Normalized {key} features: shape={norm_data[key].shape}")
                else:
                    # Copy other data without normalization
                    norm_data[key] = value
                    logger.debug(f"Copied {key} without normalization: shape={norm_data[key].shape}")
            except Exception as e:
                logger.error(f"Error normalizing {key} data: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Copy original data as fallback
                norm_data[key] = value

        # CRITICAL: Remove 'aligned_timestamps' to avoid issues with DataLoader
        if 'aligned_timestamps' in norm_data:
            del norm_data['aligned_timestamps']

        # Validate normalized data
        if 'accelerometer' not in norm_data or norm_data['accelerometer'] is None or len(norm_data['accelerometer']) == 0:
            logger.error("Missing accelerometer data after normalization")
            
        if 'quaternion' not in norm_data and 'accelerometer' in norm_data:
            logger.warning("Adding missing quaternion data")
            acc_shape = norm_data['accelerometer'].shape
            norm_data['quaternion'] = np.zeros((acc_shape[0], acc_shape[1], 4))

        elapsed_time = time.time() - start_time
        logger.info(f"Normalization complete in {elapsed_time:.2f}s")

        return norm_data
