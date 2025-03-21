'''
Dataset Builder for SmartFallMM

This module handles loading, preprocessing, and alignment of multi-modal sensor data
for human activity recognition and fall detection. It implements various techniques
for handling variably-sampled sensor data, including hybrid interpolation that
intelligently switches between cubic spline and linear methods.

Key features:
- High-performance parallel processing of multiple files
- Efficient alignment of variable-rate sensor data
- Advanced hybrid interpolation of accelerometer and gyroscope data
- Robust error handling and recovery
'''
import os
from typing import List, Dict, Tuple, Union, Optional, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.io import loadmat
from numpy.linalg import norm
from dtaidistance import dtw
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import time
import logging
from utils.processor.base import Processor
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# Import IMU fusion module for thread pool management
from utils.imu_fusion import (
    process_imu_data, 
    save_aligned_sensor_data,
    ThreadPoolExecutor as IMUThreadPoolExecutor
)

# Configure logging
log_dir = "debug_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "loader.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("loader")

# Also print to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def csvloader(file_path: str, **kwargs) -> np.ndarray:
    '''
    Loads data from CSV files with appropriate handling for different formats.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments

    Returns:
        Numpy array with the loaded data
    '''
    logger.debug(f"Loading CSV file: {file_path}")
    try:
        # Read the file and handle missing values
        file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()

        # Determine number of columns to use based on data type
        if 'skeleton' in file_path:
            cols = 96  # Skeleton data has 32 joints × 3 coordinates
            logger.debug(f"Detected skeleton data with {cols} columns")
        else:
            # Check if this is a meta sensor file or other inertial data
            if file_data.shape[1] > 4:
                # Meta sensor format: epoch, time, elapsed time, x, y, z
                cols = file_data.shape[1] - 3
                file_data = file_data.iloc[:, 3:]  # Skip first 3 columns
            else:
                cols = 3  # Standard inertial data has 3 axes (x, y, z)
            logger.debug(f"Detected inertial sensor data with {cols} columns")

        # Validate data shape
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
            
        logger.debug(f"Extracted data with shape: {activity_data.shape}")

        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise


def matloader(file_path: str, **kwargs) -> np.ndarray:
    '''
    Loads data from MATLAB (.mat) files.

    Args:
        file_path: Path to the MAT file
        **kwargs: Additional arguments including 'key' to specify which data to load

    Returns:
        Numpy array with the loaded data
    '''
    logger.debug(f"Loading MAT file: {file_path}")

    # Check for valid key
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        logger.error(f"Unsupported key for MatLab file: {key}")
        raise ValueError(f"Unsupported {key} for matlab file")

    try:
        # Load data from the specified key
        data = loadmat(file_path)[key]
        logger.debug(f"Loaded MAT data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading MAT {file_path}: {str(e)}")
        raise


# Map file extensions to appropriate loaders
LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}


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

    # Store original shape and reshape for pooling
    shape = sequence.shape if shape is None else shape
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
    result = pooled_np.reshape(-1, *shape[1:])

    elapsed_time = time.time() - start_time
    logger.debug(f"avg_pool complete: input shape {shape} → output shape {result.shape} in {elapsed_time:.4f}s")

    return result


def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int,
                      input_shape: np.ndarray) -> np.ndarray:
    '''
    Pads or truncates a sequence to a specified length.

    Args:
        sequence: Input data sequence
        max_sequence_length: Target sequence length
        input_shape: Shape of the input (used for reshaping)

    Returns:
        Padded/truncated sequence of uniform length
    '''
    logger.debug(f"Padding sequence to length {max_sequence_length}")

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


def sliding_window(data: np.ndarray, clearing_time_index: int, max_time: int,
                  sub_window_size: int, stride_size: int) -> np.ndarray:
    '''
    Extracts sliding windows from a time series.

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

    # Validate parameters
    if clearing_time_index < sub_window_size - 1:
        logger.error(f"Invalid clearing_time_index: {clearing_time_index} < {sub_window_size - 1}")
        raise AssertionError("Clearing value needs to be greater or equal to (window size - 1)")

    # Calculate starting index
    start = clearing_time_index - sub_window_size + 1

    # Adjust max_time if needed to prevent out-of-bounds access
    if max_time >= data.shape[0] - sub_window_size:
        max_time = max_time - sub_window_size + 1
        logger.debug(f"Adjusted max_time to {max_time}")

    # Generate indices for all windows
    sub_windows = (
        start +
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time, step=stride_size), 0).T
    )

    # Extract windows using the indices
    result = data[sub_windows]
    logger.debug(f"Created {result.shape[0]} windows from data with shape {data.shape}")

    return result


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
        rates = np.abs(dy / np.maximum(dx, 1e-10))

        # Smooth the rates to avoid switching too frequently
        if len(rates) >= window_size:
            rates = savgol_filter(rates, window_size, 2)

        # Create mask for rapid changes
        rapid_changes = rates > threshold

        # If no rapid changes detected, use cubic spline for everything
        if not np.any(rapid_changes):
            logger.debug("Using cubic spline interpolation for entire signal")
            try:
                cs = CubicSpline(x, y)
                return cs(x_new)
            except Exception as e:
                logger.warning(f"Cubic spline failed: {e}, falling back to linear")
                linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
                return linear_interp(x_new)

        # If all changes are rapid, use linear for everything
        if np.all(rapid_changes):
            logger.debug("Using linear interpolation for entire signal")
            linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            return linear_interp(x_new)

        # Otherwise, we need a hybrid approach
        logger.debug(f"Using hybrid interpolation: {np.sum(rapid_changes)}/{len(rapid_changes)} points have rapid changes")

        # Create interpolators for both methods
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
        # Fallback to simple linear interpolation
        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        return linear_interp(x_new)


def align_sensors_with_hybrid_interpolation(acc_data: pd.DataFrame, gyro_data: pd.DataFrame,
                                          target_freq: int = 30, acc_threshold: float = 3.0,
                                          gyro_threshold: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns accelerometer and gyroscope data using hybrid interpolation that adapts to signal characteristics.

    Args:
        acc_data: DataFrame with columns [timestamp, x, y, z]
        gyro_data: DataFrame with columns [timestamp, x, y, z]
        target_freq: Target frequency in Hz for the aligned data
        acc_threshold: Rate of change threshold for accelerometer (g/s)
        gyro_threshold: Rate of change threshold for gyroscope (rad/s²)

    Returns:
        Tuple of (aligned_acc, aligned_gyro, aligned_timestamps)
    """
    start_time = time.time()
    logger.info(f"Aligning sensors using hybrid interpolation at {target_freq}Hz")

    try:
        # Extract timestamps and convert to seconds if they're datetime objects
        if isinstance(acc_data.iloc[0, 0], (str, pd.Timestamp)):
            logger.debug("Converting accelerometer timestamps from string to datetime")
            acc_times = pd.to_datetime(acc_data.iloc[:, 0]).astype(np.int64) / 1e9
        else:
            acc_times = acc_data.iloc[:, 0].values

        if isinstance(gyro_data.iloc[0, 0], (str, pd.Timestamp)):
            logger.debug("Converting gyroscope timestamps from string to datetime")
            gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).astype(np.int64) / 1e9
        else:
            gyro_times = gyro_data.iloc[:, 0].values

        # Find common time range
        start_time_point = max(acc_times.min(), gyro_times.min())
        end_time_point = min(acc_times.max(), gyro_times.max())

        logger.debug(f"Common time range: {start_time_point:.3f}s to {end_time_point:.3f}s")

        # Check if there's a valid overlap
        if start_time_point >= end_time_point:
            logger.warning("No temporal overlap between sensors")
            return np.array([]), np.array([]), np.array([])

        # Create common time grid
        duration = end_time_point - start_time_point
        num_samples = int(duration * target_freq)

        if num_samples < 10:  # Minimum viable data
            logger.warning(f"Overlap too short: {duration:.2f}s")
            return np.array([]), np.array([]), np.array([])

        common_times = np.linspace(start_time_point, end_time_point, num_samples)

        # Extract sensor data
        acc_values = acc_data.iloc[:, 1:4].values
        gyro_values = gyro_data.iloc[:, 1:4].values

        # Initialize arrays for interpolated data
        aligned_acc = np.zeros((num_samples, 3))
        aligned_gyro = np.zeros((num_samples, 3))

        # Calculate signal magnitudes for diagnostics
        acc_mag = np.linalg.norm(acc_values, axis=1)
        gyro_mag = np.linalg.norm(gyro_values, axis=1)

        logger.debug(f"Acc magnitude range: {acc_mag.min():.2f}-{acc_mag.max():.2f}g")
        logger.debug(f"Gyro magnitude range: {gyro_mag.min():.2f}-{gyro_mag.max():.2f}rad/s")

        # Process in parallel using thread pool
        thread_pool = ThreadPoolExecutor(max_workers=6)
        futures = []

        # Submit interpolation tasks
        for axis in range(3):
            # Use different thresholds for accelerometer and gyroscope
            futures.append(
                thread_pool.submit(
                    hybrid_interpolate,
                    acc_times,
                    acc_values[:, axis],
                    common_times,
                    threshold=acc_threshold
                )
            )
            
            futures.append(
                thread_pool.submit(
                    hybrid_interpolate,
                    gyro_times,
                    gyro_values[:, axis],
                    common_times,
                    threshold=gyro_threshold
                )
            )

        # Collect results
        for i, future in enumerate(futures):
            result = future.result()
            if i < 3:  # First 3 are accelerometer
                aligned_acc[:, i] = result
            else:  # Last 3 are gyroscope
                aligned_gyro[:, i - 3] = result

        # Clean up thread pool
        thread_pool.shutdown()

        elapsed_time = time.time() - start_time
        logger.info(f"Hybrid interpolation complete: {num_samples} aligned samples in {elapsed_time:.2f}s")

        return aligned_acc, aligned_gyro, common_times

    except Exception as e:
        logger.error(f"Sensor alignment failed: {str(e)}")
        return np.array([]), np.array([]), np.array([])


def visualize_alignment(acc_times, acc_values, gyro_times, gyro_values,
                       common_times, aligned_acc, aligned_gyro):
    """
    Creates visualization of sensor alignment before and after interpolation.

    Args:
        acc_times: Original accelerometer timestamps
        acc_values: Original accelerometer values
        gyro_times: Original gyroscope timestamps
        gyro_values: Original gyroscope values
        common_times: Common timestamps after alignment
        aligned_acc: Aligned accelerometer values
        aligned_gyro: Aligned gyroscope values
    """
    try:
        # Create directory for visualizations
        viz_dir = os.path.join(log_dir, "alignment_viz")
        os.makedirs(viz_dir, exist_ok=True)

        # Create filename with timestamp
        filename = f"alignment_{int(time.time())}.png"

        # Create a new figure to avoid renderer issues
        plt.close('all')
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))

        # Plot accelerometer X axis
        axes[0, 0].plot(acc_times, acc_values[:, 0], 'b.', alpha=0.5, label='Original')
        axes[0, 0].plot(common_times, aligned_acc[:, 0], 'r-', label='Interpolated')
        axes[0, 0].set_title('Accelerometer X-axis')
        axes[0, 0].legend()

        # Plot gyroscope X axis
        axes[0, 1].plot(gyro_times, gyro_values[:, 0], 'b.', alpha=0.5, label='Original')
        axes[0, 1].plot(common_times, aligned_gyro[:, 0], 'r-', label='Interpolated')
        axes[0, 1].set_title('Gyroscope X-axis')
        axes[0, 1].legend()

        # Plot accelerometer Y axis
        axes[1, 0].plot(acc_times, acc_values[:, 1], 'b.', alpha=0.5, label='Original')
        axes[1, 0].plot(common_times, aligned_acc[:, 1], 'r-', label='Interpolated')
        axes[1, 0].set_title('Accelerometer Y-axis')

        # Plot gyroscope Y axis
        axes[1, 1].plot(gyro_times, gyro_values[:, 1], 'b.', alpha=0.5, label='Original')
        axes[1, 1].plot(common_times, aligned_gyro[:, 1], 'r-', label='Interpolated')
        axes[1, 1].set_title('Gyroscope Y-axis')

        # Plot accelerometer Z axis
        axes[2, 0].plot(acc_times, acc_values[:, 2], 'b.', alpha=0.5, label='Original')
        axes[2, 0].plot(common_times, aligned_acc[:, 2], 'r-', label='Interpolated')
        axes[2, 0].set_title('Accelerometer Z-axis')

        # Plot gyroscope Z axis
        axes[2, 1].plot(gyro_times, gyro_values[:, 2], 'b.', alpha=0.5, label='Original')
        axes[2, 1].plot(common_times, aligned_gyro[:, 2], 'r-', label='Interpolated')
        axes[2, 1].set_title('Gyroscope Z-axis')

        plt.tight_layout()
        
        # Ensure the figure is drawn before saving
        fig.canvas.draw()
        plt.savefig(os.path.join(viz_dir, filename))
        plt.close(fig)

        logger.debug(f"Alignment visualization saved to {os.path.join(viz_dir, filename)}")
    except Exception as e:
        logger.warning(f"Failed to create alignment visualization: {e}")
        plt.close('all')  # Clean up any open figures


def visualize_skl_acc_alignment(skeleton_mag, acc_mag, skeleton_indices, inertial_indices):
    """
    Visualizes alignment between skeleton and accelerometer data.

    Args:
        skeleton_mag: Magnitude values from skeleton data
        acc_mag: Magnitude values from accelerometer data
        skeleton_indices: Indices selected from skeleton data after alignment
        inertial_indices: Indices selected from accelerometer data after alignment
    """
    try:
        # Create directory for visualizations
        viz_dir = os.path.join(log_dir, "alignment_viz")
        os.makedirs(viz_dir, exist_ok=True)

        # Create filename with timestamp
        filename = f"skl_acc_alignment_{int(time.time())}.png"

        # Clean up previous plots
        plt.close('all')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot magnitude signals before alignment
        ax1.plot(skeleton_mag, 'b-', label='Skeleton')
        ax1.plot(acc_mag, 'r-', label='Accelerometer')
        ax1.set_title('Signals before alignment')
        ax1.legend()

        # Plot aligned signals
        ax2.plot(skeleton_mag[list(skeleton_indices)], 'b-', label='Aligned Skeleton')
        ax2.plot(acc_mag[list(inertial_indices)], 'r-', label='Aligned Accelerometer')
        ax2.set_title('Signals after alignment')
        ax2.legend()

        # Draw the figure to avoid empty plot issues
        fig.canvas.draw()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, filename))
        plt.close(fig)

        logger.debug(f"Skeleton-Accelerometer alignment visualization saved to {os.path.join(viz_dir, filename)}")
    except Exception as e:
        logger.warning(f"Failed to create skeleton-accelerometer alignment visualization: {e}")
        plt.close('all')


def create_dataframe_with_timestamps(data, start_time=0, sample_rate=None):
    """
    Creates a DataFrame with synthetic timestamps when actual timestamps are not available.

    Args:
        data: Sensor data array
        start_time: Start time in seconds
        sample_rate: Estimated sample rate (if None, assumes evenly spaced samples)

    Returns:
        DataFrame with timestamp column and sensor data
    """
    num_samples = data.shape[0]

    if sample_rate is None:
        # Create evenly spaced timestamps (assuming 30Hz if not specified)
        timestamps = np.linspace(start_time, start_time + num_samples/30, num_samples)
    else:
        # Create timestamps based on sample rate
        timestamps = np.arange(num_samples) / sample_rate + start_time

    # Create a DataFrame with timestamp and data columns
    df = pd.DataFrame()
    df['timestamp'] = timestamps

    # Add the data columns
    for i in range(data.shape[1]):
        df[f'axis_{i}'] = data[:, i]

    return df


def filter_data_by_ids(data: np.ndarray, ids: List[int]) -> np.ndarray:
    '''
    Selects specific indices from a data array.

    Args:
        data: Input data array
        ids: List of indices to select

    Returns:
        Filtered data array containing only the selected indices
    '''
    logger.debug(f"Filtering data with {len(ids)} indices")

    if len(ids) == 0:
        logger.warning("Empty ID list for filtering")
        return np.array([])

    result = data[ids, :]
    logger.debug(f"Filtered data shape: {result.shape}")

    return result


def filter_repeated_ids(path: List[Tuple[int, int]]) -> Tuple[set, set]:
    '''
    Filters DTW path to create one-to-one mapping between indices.

    Args:
        path: List of index pairs from DTW warping path

    Returns:
        Two sets containing unique indices for each sequence
    '''
    logger.debug(f"Filtering repeated IDs from path with {len(path)} points")

    seen_first = set()
    seen_second = set()

    for (first, second) in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)

    logger.debug(f"Filtered to {len(seen_first)} unique first indices and {len(seen_second)} unique second indices")

    return seen_first, seen_second


def align_sequence(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    '''
    Aligns data from multiple sensor modalities using hybrid interpolation and DTW.

    This function first aligns accelerometer and gyroscope data using hybrid interpolation,
    then aligns skeleton data with the synchronized inertial data using DTW.

    Args:
        data: Dictionary containing sensor data for different modalities

    Returns:
        Dictionary with aligned sensor data
    '''
    logger.info("Aligning skeleton and inertial data")

    start_time = time.time()

    # Verify required data exists
    if 'skeleton' not in data:
        logger.warning("Missing skeleton data for alignment")

        # If we still have both accelerometer and gyroscope, align them
        if 'accelerometer' in data and 'gyroscope' in data:
            try:
                # Create DataFrames with timestamps
                acc_df = create_dataframe_with_timestamps(data['accelerometer'])
                gyro_df = create_dataframe_with_timestamps(data['gyroscope'])

                # Align using hybrid interpolation
                aligned_acc, aligned_gyro, aligned_times = align_sensors_with_hybrid_interpolation(
                    acc_df, gyro_df, target_freq=30,
                    acc_threshold=3.0,  # Adjust based on your data characteristics
                    gyro_threshold=1.0
                )

                if len(aligned_acc) > 0:
                    # Update data with interpolated values
                    data['accelerometer'] = aligned_acc
                    data['gyroscope'] = aligned_gyro
                    data['aligned_timestamps'] = aligned_times
                    logger.info(f"Inertial sensors aligned with {len(aligned_acc)} samples")
                else:
                    logger.warning("Sensor alignment failed - insufficient overlap")
            except Exception as e:
                logger.error(f"Sensor alignment error: {str(e)}")

        return data

    # Step 1: Find dynamic modalities (non-skeleton)
    dynamic_keys = sorted([key for key in data.keys() if key != "skeleton"])

    if not dynamic_keys:
        logger.warning("No inertial data found for alignment")
        return data

    # Step 2: Align accelerometer and gyroscope if both exist
    if len(dynamic_keys) > 1:
        logger.debug(f"Found multiple inertial modalities: {dynamic_keys}")
        acc_data = data[dynamic_keys[0]]
        gyro_data = data[dynamic_keys[1]]

        try:
            # Create DataFrames with timestamps
            acc_df = create_dataframe_with_timestamps(acc_data)
            gyro_df = create_dataframe_with_timestamps(gyro_data)

            # Align using hybrid interpolation
            aligned_acc, aligned_gyro, aligned_times = align_sensors_with_hybrid_interpolation(
                acc_df, gyro_df, target_freq=30
            )

            if len(aligned_acc) > 0:
                # Update data with interpolated values
                data[dynamic_keys[0]] = aligned_acc
                data[dynamic_keys[1]] = aligned_gyro
                data['aligned_timestamps'] = aligned_times
                logger.info(f"Inertial sensors aligned: {len(aligned_acc)} samples")
            else:
                logger.warning("Inertial sensor alignment failed - insufficient overlap")
        except Exception as e:
            logger.error(f"Inertial sensor alignment error: {str(e)}")

    # Step 3: Align skeleton with inertial data
    try:
        # Get reference skeleton joint data (typically left wrist)
        joint_id = 9
        skeleton_joint_data = data['skeleton'][:, (joint_id - 1) * 3 : joint_id * 3]
        logger.debug(f"Using joint ID {joint_id} (left wrist) with shape {skeleton_joint_data.shape}")

        # Get primary inertial data (usually accelerometer)
        inertial_data = data[dynamic_keys[0]]
        logger.debug(f"Using {dynamic_keys[0]} with shape {inertial_data.shape} for skeleton alignment")

        # Calculate Frobenius norm for both data types
        skeleton_frob_norm = np.linalg.norm(skeleton_joint_data, axis=1)
        inertial_frob_norm = np.linalg.norm(inertial_data, axis=1)

        # Perform DTW alignment
        path = dtw.warping_path(skeleton_frob_norm, inertial_frob_norm)
        logger.debug(f"DTW path found with {len(path)} points")

        # Filter to unique indices
        skeleton_idx, inertial_ids = filter_repeated_ids(path)

        # Apply filtering
        data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_idx))
        for key in dynamic_keys:
            data[key] = filter_data_by_ids(data[key], list(inertial_ids))
        
        if 'aligned_timestamps' in data:
            data['aligned_timestamps'] = filter_data_by_ids(data['aligned_timestamps'].reshape(-1, 1), list(inertial_ids)).flatten()

        logger.info(f"DTW alignment complete: {len(skeleton_idx)} skeleton frames, {len(inertial_ids)} inertial frames")

        # Visualize alignment if at debug level
        if logger.level <= logging.DEBUG:
            visualize_skl_acc_alignment(skeleton_frob_norm, inertial_frob_norm,
                                      skeleton_idx, inertial_ids)

    except Exception as e:
        logger.error(f"DTW alignment error: {str(e)}")

    elapsed_time = time.time() - start_time
    logger.debug(f"Alignment completed in {elapsed_time:.2f}s")

    return data


def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int,
                            peaks: Union[List[int], np.ndarray], label: int,
                            fuse: bool, filter_type: str = 'madgwick') -> Dict[str, np.ndarray]:
    """
    Creates windows centered around detected peaks with optional fusion processing.

    Args:
        data: Dictionary of sensor data arrays
        window_size: Size of each window
        peaks: List of peak indices to center windows on
        label: Label for this activity
        fuse: Whether to apply sensor fusion
        filter_type: Type of fusion filter to use

    Returns:
        Dictionary of windowed data arrays
    """
    start_time = time.time()
    logger.info(f"Creating selective windows: peaks={len(peaks)}, fusion={fuse}, filter_type={filter_type}")

    windowed_data = defaultdict(list)

    # Check for required modalities
    has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and len(data['gyroscope']) > 0
    if fuse and not has_gyro:
        logger.warning("Fusion requested but gyroscope data not available")
        fuse = False

    # Create local thread pool for windows
    max_workers = min(8, len(peaks)) if len(peaks) > 0 else 1
    with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
        # Create windows around peaks with parallel processing
        futures = []
    
        # Submit window extraction tasks
        for peak_idx, peak in enumerate(peaks):
            # Calculate window boundaries
            start = max(0, peak - window_size // 2)
            end = min(len(data['accelerometer']), start + window_size)
    
            # Skip if window is too small
            if end - start < window_size:
                logger.debug(f"Skipping window at peak {peak}: too small ({end-start} < {window_size})")
                continue
    
            # Submit task
            futures.append((
                peak_idx,
                peak,
                thread_pool.submit(
                    _extract_window,
                    data,
                    start,
                    end,
                    window_size,
                    fuse,
                    filter_type
                )
            ))
    
        # Collect results with progress bar if there are many windows
        windows_created = 0
        collection_iterator = tqdm(futures, desc="Processing windows") if len(futures) > 5 else futures
        
        for peak_idx, peak, future in collection_iterator:
            try:
                window_data = future.result()
                
                # Add this window's data to the result dictionary
                for modality, modality_window in window_data.items():
                    if modality_window is not None:
                        windowed_data[modality].append(modality_window)
                
                windows_created += 1
            except Exception as e:
                logger.error(f"Error processing window at peak {peak}: {str(e)}")

    # Convert lists of arrays to arrays
    for modality in windowed_data:
        if modality != 'labels' and len(windowed_data[modality]) > 0:
            try:
                windowed_data[modality] = np.array(windowed_data[modality])
                logger.debug(f"Converted {modality} windows to array with shape {windowed_data[modality].shape}")
            except Exception as e:
                logger.error(f"Error converting {modality} windows to array: {str(e)}")

    # Add labels
    windowed_data['labels'] = np.repeat(label, windows_created)

    elapsed_time = time.time() - start_time
    logger.info(f"Created {windows_created} windows in {elapsed_time:.2f}s")

    return windowed_data


def _extract_window(data, start, end, window_size, fuse, filter_type):
    """
    Helper function to extract a window from data.
    Designed to be run in a separate thread.
    
    Args:
        data: Dictionary of sensor data arrays
        start: Start index for window
        end: End index for window
        window_size: Target window size
        fuse: Whether to apply fusion
        filter_type: Type of filter to use
        
    Returns:
        Dictionary of windowed data
    """
    window_data = {}
    
    # Extract window for each modality
    for modality, modality_data in data.items():
        if modality != 'labels' and modality_data is not None and len(modality_data) > 0:
            try:
                # Special handling for one-dimensional arrays like aligned_timestamps
                if modality == 'aligned_timestamps':
                    # Handle 1D array - no second dimension indexing
                    if len(modality_data.shape) == 1:
                        window_data_array = modality_data[start:min(end, len(modality_data))]

                        # Pad if needed
                        if len(window_data_array) < window_size:
                            logger.debug(f"Padding {modality} window from {len(window_data_array)} to {window_size}")
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
                    window_data_array = modality_data[start:min(end, len(modality_data)), :]

                    # Pad if needed
                    if window_data_array.shape[0] < window_size:
                        logger.debug(f"Padding {modality} window from {window_data_array.shape[0]} to {window_size}")
                        padded = np.zeros((window_size, window_data_array.shape[1]), dtype=window_data_array.dtype)
                        padded[:window_data_array.shape[0]] = window_data_array
                        window_data_array = padded

                window_data[modality] = window_data_array
            except Exception as e:
                logger.error(f"Error extracting {modality} window: {str(e)}")
                window_data[modality] = None

    # Apply fusion if requested and we have both accelerometer and gyroscope
    if fuse and 'accelerometer' in window_data and 'gyroscope' in window_data:
        try:
            # Extract the window data
            acc_window = window_data['accelerometer']
            gyro_window = window_data['gyroscope']

            # Extract timestamps if available
            timestamps = None
            if 'aligned_timestamps' in window_data:
                # Ensure we handle 1D timestamp array correctly
                if len(window_data['aligned_timestamps'].shape) == 1:
                    timestamps = window_data['aligned_timestamps']
                else:
                    # Handle unexpected 2D timestamps if they occur
                    timestamps = window_data['aligned_timestamps'][:, 0]

            # Process data using IMU fusion
            features = process_imu_data(
                acc_data=acc_window,
                gyro_data=gyro_window,
                timestamps=timestamps,
                filter_type=filter_type,
                return_features=True
            )

            # IMPORTANT: Add fusion results to window data
            # We change the key to 'linear_acceleration' instead of 'accelerometer'
            window_data['linear_acceleration'] = features['linear_acceleration']
            window_data['quaternion'] = features['quaternion']
            if 'fusion_features' in features:
                window_data['fusion_features'] = features['fusion_features']
                
            logger.debug(f"Added fusion data to window")
        except Exception as e:
            logger.error(f"Error in fusion processing: {str(e)}")
    
    return window_data


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
        for dir_name in ["accelerometer", "gyroscope", "skeleton"]:
            os.makedirs(os.path.join(self.aligned_data_dir, dir_name), exist_ok=True)

        # Log fusion options if present
        if fusion_options:
            fusion_enabled = fusion_options.get('enabled', False)
            filter_type = fusion_options.get('filter_type', 'madgwick')
            logger.info(f"Fusion options: enabled={fusion_enabled}, filter_type={filter_type}")

    def load_file(self, file_path):
        '''
        Loads sensor data from a file.

        Args:
            file_path: Path to the data file

        Returns:
            Numpy array containing the loaded data
        '''
        logger.debug(f"Loading file: {file_path}")

        try:
            loader = self._import_loader(file_path)
            data = loader(file_path, **self.kwargs)
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

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

        if self.mode == 'avg_pool':
            # Use average pooling to create fixed-length data
            logger.debug("Applying average pooling")
            processed_data = {}

            for modality, modality_data in data.items():
                if modality != 'labels':
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
                    logger.debug(f"Applying sensor fusion with {filter_type} filter")
                    
                    # Extract timestamps if available
                    timestamps = processed_data.get('aligned_timestamps', None)
                    
                    # Process with IMU fusion
                    fusion_result = process_imu_data(
                        acc_data=processed_data['accelerometer'],
                        gyro_data=processed_data['gyroscope'],
                        timestamps=timestamps,
                        filter_type=filter_type,
                        return_features=True
                    )
                    
                    # Add fusion results to processed data
                    processed_data.update(fusion_result)
                    
                except Exception as e:
                    logger.error(f"Fusion processing failed: {str(e)}")

            return processed_data
        else:
            # Use peak detection for windowing
            logger.debug("Using peak detection for windowing")

            # Calculate magnitude for peak detection
            sqrt_sum = np.sqrt(np.sum(data['accelerometer']**2, axis=1))

            # Set peak detection parameters based on label
            if label == 1:  # Fall
                logger.debug("Using fall detection peak parameters")
                peaks, _ = find_peaks(sqrt_sum, height=12, distance=10)
            else:  # Non-fall
                logger.debug("Using non-fall peak parameters")
                peaks, _ = find_peaks(sqrt_sum, height=10, distance=20)

            logger.debug(f"Found {len(peaks)} peaks")

            # Extract windows around peaks with optional fusion
            processed_data = selective_sliding_window(
                data=data,
                window_size=self.max_length,
                peaks=peaks,
                label=label,
                fuse=fuse,
                filter_type=filter_type
            )

            return processed_data

    def _add_trial_data(self, trial_data):
        '''
        Adds processed trial data to the dataset.

        Args:
            trial_data: Dictionary of processed sensor data for a trial
        '''
        logger.debug("Adding trial data to dataset")

        for modality, modality_data in trial_data.items():
            if len(modality_data) > 0:  # Only add non-empty data
                self.data[modality].append(modality_data)
                logger.debug(f"Added {modality} data with shape {modality_data.shape if hasattr(modality_data, 'shape') else len(modality_data)}")

    def _len_check(self, d):
        '''
        Checks if data dictionary has sufficient length in each modality.

        Args:
            d: Dictionary of data arrays

        Returns:
            Boolean indicating if all modalities have sufficient data
        '''
        return all(len(v) > 1 for v in d.values())

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
                    trial_data[modality] = unimodal_data
                except Exception as e:
                    logger.error(f"Error loading {modality} from {file_path}: {str(e)}")
                    return None
            
            # Align skeleton and inertial data
            trial_data = align_sequence(trial_data)
            
            # Save aligned data if requested
            if save_aligned:
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
                        aligned_timestamps if aligned_timestamps is not None else np.arange(len(aligned_acc))
                    )
            
            # Process the aligned data
            processed_data = self.process(trial_data, label, fuse, filter_type, visualize)
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Trial processing failed: {str(e)}")
            return None

    def make_dataset(self, subjects: List[int], fuse: bool, filter_type: str = 'madgwick', 
                    visualize: bool = False, save_aligned: bool = False):
        '''
        Creates a dataset from the sensor data files for the specified subjects.

        Args:
            subjects: List of subject IDs to include
            fuse: Whether to apply sensor fusion
            filter_type: Type of fusion filter to use ('madgwick', 'comp', 'kalman', 'ekf', 'ukf')
            visualize: Whether to generate visualizations
            save_aligned: Whether to save aligned data to files
        '''
        logger.info(f"Making dataset for subjects={subjects}, fuse={fuse}, filter_type={filter_type}")

        start_time = time.time()
        self.data = defaultdict(list)
        self.fuse = fuse
        
        # Check if save_aligned is specified in fusion options
        if hasattr(self, 'fusion_options'):
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(8, len(self.dataset.matched_trials))) as executor:
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
            
            for future in tqdm(as_completed(future_to_trial), total=len(future_to_trial), desc="Processing trials"):
                trial = future_to_trial[future]
                count += 1
                
                try:
                    trial_data = future.result()
                    if trial_data is not None and self._len_check(trial_data):
                        self._add_trial_data(trial_data)
                        processed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error processing trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number}: {str(e)}")
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
            else:
                logger.warning(f"Cannot concatenate {key} data - mixed types")

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

        # Normalize each modality separately (except labels)
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    # Check if this is a feature that needs normalization
                    if key in ['accelerometer', 'gyroscope', 'quaternion', 'linear_acceleration'] and len(value.shape) >= 2:
                        # Reshape for standardization
                        num_samples, length = value.shape[:2]
                        orig_shape = value.shape

                        # StandardScaler works on 2D data, so reshape
                        reshaped_data = value.reshape(num_samples * length, -1)

                        # Standardize data
                        norm_data = StandardScaler().fit_transform(reshaped_data)

                        # Reshape back to original shape
                        self.data[key] = norm_data.reshape(orig_shape)

                        logger.debug(f"Normalized {key} data: shape={self.data[key].shape}")
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        # These are already extracted features, normalize them directly
                        self.data[key] = StandardScaler().fit_transform(value)
                        logger.debug(f"Normalized {key} features: shape={self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error normalizing {key} data: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(f"Normalization complete in {elapsed_time:.2f}s")

        return self.data
