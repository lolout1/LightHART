'''
IMU Fusion Module for Enhanced Orientation Estimation

This module provides multiple sensor fusion algorithms for converting raw 
accelerometer and gyroscope data into orientation estimates (quaternions) and
derived features. It includes implementations of Madgwick, Complementary,
Kalman, Extended Kalman, and Unscented Kalman filters with multithreading
and GPU acceleration capabilities.

Key features:
- Multiple IMU fusion filters for different accuracy/complexity tradeoffs
- Parallel processing for improved performance
- GPU acceleration for computationally intensive operations
- Progress tracking for long-running operations
- Comprehensive feature extraction for activity recognition
'''

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Union, Optional
import pandas as pd
import time
import traceback
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import logging
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import queue

# Configure logging
log_dir = "debug_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "imu_fusion.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("imu_fusion")

# Also print to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Parallel processing configuration
MAX_TOTAL_THREADS = 48  # Total number of threads to use
MAX_FILES_PARALLEL = 12  # Maximum number of files to process in parallel
THREADS_PER_FILE = 4    # Threads to use per file

# Thread pool managers
_file_thread_pool = None
_processing_thread_pool = {}
_thread_pool_lock = threading.Lock()

def get_file_thread_pool():
    """Get or create the file-level thread pool."""
    global _file_thread_pool
    with _thread_pool_lock:
        if _file_thread_pool is None:
            _file_thread_pool = ThreadPoolExecutor(max_workers=MAX_FILES_PARALLEL)
        return _file_thread_pool

def get_processing_thread_pool(file_id):
    """Get or create a thread pool for a specific file."""
    global _processing_thread_pool
    with _thread_pool_lock:
        if file_id not in _processing_thread_pool:
            _processing_thread_pool[file_id] = ThreadPoolExecutor(max_workers=THREADS_PER_FILE)
        return _processing_thread_pool[file_id]

def cleanup_thread_pools():
    """Shut down all thread pools cleanly."""
    global _file_thread_pool, _processing_thread_pool
    with _thread_pool_lock:
        if _file_thread_pool is not None:
            _file_thread_pool.shutdown(wait=True)
            _file_thread_pool = None
        
        for pool in _processing_thread_pool.values():
            pool.shutdown(wait=True)
        _processing_thread_pool.clear()

def cleanup_resources():
    """Clean up all resources used by the module."""
    cleanup_thread_pools()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# GPU Configuration
def setup_gpu_environment():
    """Configure environment for dual GPU usage"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            logger.info(f"Found {num_gpus} GPUs. Using GPUs 0 and 1 for processing")
            return True, [0, 1]
        elif num_gpus == 1:
            logger.info("Found 1 GPU. Using GPU 0 for processing")
            return True, [0]
        else:
            logger.warning("No GPUs found, falling back to CPU processing")
            return False, []
    else:
        logger.warning("PyTorch CUDA not available, using CPU processing")
        return False, []

# Initialize GPU environment
USE_GPU, GPU_DEVICES = setup_gpu_environment()

def update_thread_configuration(max_files: int, threads_per_file: int):
    """
    Update the thread configuration parameters.

    Args:
        max_files: New value for MAX_FILES_PARALLEL
        threads_per_file: New value for THREADS_PER_FILE
    """
    global MAX_FILES_PARALLEL, THREADS_PER_FILE

    # Clean up existing thread pools first
    cleanup_thread_pools()

    # Update configuration
    MAX_FILES_PARALLEL = max_files
    THREADS_PER_FILE = threads_per_file

    logger.info(f"Thread configuration updated: {MAX_FILES_PARALLEL} files with {THREADS_PER_FILE} threads per file")

def process_multiple_files_parallel(file_data_list, filter_type='madgwick', return_features=True):
    """
    Process multiple IMU data files in parallel.
    
    Args:
        file_data_list: List of tuples (file_id, acc_data, gyro_data, timestamps)
        filter_type: Type of filter to use
        return_features: Whether to return extracted features
        
    Returns:
        List of processing results in the same order as input
    """
    logger.info(f"Processing {len(file_data_list)} files in parallel using {MAX_FILES_PARALLEL} workers")
    
    # Get or create file thread pool
    file_pool = get_file_thread_pool()
    
    # Submit all files for processing
    futures = []
    for idx, (file_id, acc_data, gyro_data, timestamps) in enumerate(file_data_list):
        futures.append(file_pool.submit(
            process_imu_data,
            acc_data,
            gyro_data,
            timestamps,
            filter_type,
            return_features,
            file_id
        ))
    
    # Collect results with progress bar
    results = []
    with tqdm(total=len(futures), desc="Processing files") as pbar:
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                results.append(None)
            finally:
                pbar.update(1)
    
    return results

def align_sensor_data(acc_data: pd.DataFrame, gyro_data: pd.DataFrame,
                     time_tolerance: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align accelerometer and gyroscope data based on timestamps.
    Uses parallel processing for improved performance with large datasets.

    Args:
        acc_data: Accelerometer data with timestamp column
        gyro_data: Gyroscope data with timestamp column
        time_tolerance: Maximum time difference to consider readings aligned

    Returns:
        Tuple of (aligned_acc, aligned_gyro, timestamps)
    """
    start_time = time.time()
    logger.info(f"Starting sensor alignment: acc shape={acc_data.shape}, gyro shape={gyro_data.shape}")

    # Extract timestamps
    if isinstance(acc_data.iloc[0, 0], str):
        logger.debug("Converting accelerometer timestamps from string to datetime")
        acc_times = pd.to_datetime(acc_data.iloc[:, 0]).values
    else:
        acc_times = acc_data.iloc[:, 0].values

    if isinstance(gyro_data.iloc[0, 0], str):
        logger.debug("Converting gyroscope timestamps from string to datetime")
        gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).values
    else:
        gyro_times = gyro_data.iloc[:, 0].values

    # Determine the later start time
    start_time_point = max(acc_times[0], gyro_times[0])
    logger.debug(f"Common start time: {start_time_point}")

    # Filter data to start from the later time
    acc_start_idx = np.searchsorted(acc_times, start_time_point)
    gyro_start_idx = np.searchsorted(gyro_times, start_time_point)

    logger.debug(f"Trimming data: acc from {acc_start_idx}, gyro from {gyro_start_idx}")
    acc_data_filtered = acc_data.iloc[acc_start_idx:].reset_index(drop=True)
    gyro_data_filtered = gyro_data.iloc[gyro_start_idx:].reset_index(drop=True)

    # Extract updated timestamps
    if isinstance(acc_data_filtered.iloc[0, 0], str):
        acc_times = pd.to_datetime(acc_data_filtered.iloc[:, 0]).values
    else:
        acc_times = acc_data_filtered.iloc[:, 0].values

    # Convert timestamps to numeric values for faster computation
    acc_times_np = np.array([t.astype('int64') if hasattr(t, 'astype') else t for t in acc_times])
    gyro_times_np = np.array([t.astype('int64') if hasattr(t, 'astype') else t for t in gyro_times])
    
    # Prepare for parallel processing
    aligned_acc = []
    aligned_gyro = []
    aligned_times = []
    
    # Always use multithreading for alignment regardless of dataset size
    logger.debug(f"Using parallel processing for {len(acc_times)} timestamps")
    
    # Define the function to process a chunk of timestamps
    def process_chunk(start_idx, end_idx):
        local_acc = []
        local_gyro = []
        local_times = []
        
        # Convert tolerance to appropriate units
        if isinstance(acc_times[0], np.datetime64):
            tolerance_ns = np.timedelta64(int(time_tolerance * 1e9), 'ns')
        else:
            tolerance_ns = time_tolerance
            
        for i in range(start_idx, end_idx):
            # Find closest gyro time
            time_diffs = np.abs(gyro_times_np - acc_times_np[i])
            closest_idx = np.argmin(time_diffs)
            
            # If within tolerance, add to matched pairs
            if time_diffs[closest_idx] <= tolerance_ns:
                local_acc.append(acc_data_filtered.iloc[i, 1:4].values)
                local_gyro.append(gyro_data_filtered.iloc[closest_idx, 1:4].values)
                local_times.append(acc_times[i])
                
        return local_acc, local_gyro, local_times
    
    # Split into chunks for parallel processing
    chunk_size = max(1, len(acc_times) // THREADS_PER_FILE)
    futures = []
    
    # Create a thread pool specific to this alignment task
    thread_pool = ThreadPoolExecutor(max_workers=THREADS_PER_FILE)
    
    # Submit tasks to thread pool
    for start_idx in range(0, len(acc_times), chunk_size):
        end_idx = min(start_idx + chunk_size, len(acc_times))
        futures.append(thread_pool.submit(process_chunk, start_idx, end_idx))
    
    # Collect results with progress tracking
    with tqdm(total=len(futures), desc="Aligning sensor data") as pbar:
        for future in as_completed(futures):
            chunk_acc, chunk_gyro, chunk_times = future.result()
            aligned_acc.extend(chunk_acc)
            aligned_gyro.extend(chunk_gyro)
            aligned_times.extend(chunk_times)
            pbar.update(1)
    
    # Clean up thread pool
    thread_pool.shutdown()

    # Convert to numpy arrays
    aligned_acc = np.array(aligned_acc)
    aligned_gyro = np.array(aligned_gyro)
    aligned_times = np.array(aligned_times)

    elapsed_time = time.time() - start_time
    logger.info(f"Alignment complete: {len(aligned_acc)} matched samples in {elapsed_time:.2f}s")
    logger.debug(f"Aligned data shapes: acc={aligned_acc.shape}, gyro={aligned_gyro.shape}")

    # Log data statistics
    if len(aligned_acc) > 0:
        logger.debug(f"Acc min/max/mean: {np.min(aligned_acc):.3f}/{np.max(aligned_acc):.3f}/{np.mean(aligned_acc):.3f}")
        logger.debug(f"Gyro min/max/mean: {np.min(aligned_gyro):.3f}/{np.max(aligned_gyro):.3f}/{np.mean(aligned_gyro):.3f}")

    return aligned_acc, aligned_gyro, aligned_times


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


class OrientationEstimator:
    """Base class for orientation estimation algorithms."""

    def __init__(self, freq: float = 30.0):
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        logger.debug(f"Initialized {self.__class__.__name__} with freq={freq}Hz")

    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: float = None) -> np.ndarray:
        """Update orientation estimate with new sensor readings."""
        # Calculate actual sampling interval if timestamps are provided
        dt = 1.0 / self.freq
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            self.last_time = timestamp
        elif timestamp is not None:
            self.last_time = timestamp

        logger.debug(f"Updating orientation with dt={dt:.6f}s")
        return self._update_impl(acc, gyro, dt)

    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Implementation of the update step to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

    def reset(self):
        """Reset the filter state."""
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        logger.debug(f"Reset {self.__class__.__name__} filter state")


class MadgwickFilter(OrientationEstimator):
    """Optimized Madgwick filter implementation for orientation estimation."""

    def __init__(self, freq: float = 30.0, beta: float = 0.1):
        super().__init__(freq)
        self.beta = beta  # Filter gain
        logger.debug(f"Initialized MadgwickFilter with beta={beta}")

    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Update orientation using Madgwick algorithm."""
        q = self.orientation_q

        # Normalize accelerometer measurement
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            logger.warning("Zero acceleration detected, skipping orientation update")
            return q  # Handle zero acceleration

        acc_norm = acc / acc_norm

        # Convert quaternion to array for calculations
        q0, q1, q2, q3 = q

        # Reference direction of Earth's gravity
        # 2*q1*q3 - 2*q0*q2
        f1 = 2.0 * (q1 * q3 - q0 * q2) - acc_norm[0]
        # 2*q0*q1 + 2*q2*q3
        f2 = 2.0 * (q0 * q1 + q2 * q3) - acc_norm[1]
        # q0^2 - q1^2 - q2^2 + q3^2
        f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - acc_norm[2]

        # Gradient descent algorithm corrective step
        J_t = np.array([
            [-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
            [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
            [0.0, -4.0*q1, -4.0*q2, 0.0]
        ])

        grad = J_t.T @ np.array([f1, f2, f3])
        grad_norm = np.linalg.norm(grad)
        grad = grad / grad_norm if grad_norm > 0 else grad

        # Gyroscope in radians/sec
        qDot = 0.5 * np.array([
            -q1 * gyro[0] - q2 * gyro[1] - q3 * gyro[2],
            q0 * gyro[0] + q2 * gyro[2] - q3 * gyro[1],
            q0 * gyro[1] - q1 * gyro[2] + q3 * gyro[0],
            q0 * gyro[2] + q1 * gyro[1] - q2 * gyro[0]
        ])

        # Apply feedback step
        qDot = qDot - self.beta * grad

        # Integrate to get new quaternion
        q = q + qDot * dt

        # Normalize quaternion
        q = q / np.linalg.norm(q)

        self.orientation_q = q

        logger.debug(f"Updated orientation: q=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
        return q


class CompFilter(OrientationEstimator):
    """Simple complementary filter for IMU fusion."""

    def __init__(self, freq: float = 30.0, alpha: float = 0.98):
        super().__init__(freq)
        self.alpha = alpha  # Weight for gyroscope integration
        logger.debug(f"Initialized ComplementaryFilter with alpha={alpha}")

    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Update orientation using complementary filter."""
        q = self.orientation_q

        # Normalize accelerometer
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            acc_norm = acc / acc_norm

            # Convert accelerometer to orientation (roll and pitch only)
            roll = np.arctan2(acc_norm[1], acc_norm[2])
            pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2))

            # Convert to quaternion (assuming yaw=0)
            acc_q = Rotation.from_euler('xyz', [roll, pitch, 0]).as_quat()
            # Switch from scalar-last (x, y, z, w) to scalar-first (w, x, y, z)
            acc_q = np.array([acc_q[3], acc_q[0], acc_q[1], acc_q[2]])
        else:
            logger.warning("Zero acceleration detected, using previous orientation")
            acc_q = q

        # Integrate gyroscope
        gyro_q = 0.5 * np.array([
            -q[1] * gyro[0] - q[2] * gyro[1] - q[3] * gyro[2],
             q[0] * gyro[0] + q[2] * gyro[2] - q[3] * gyro[1],
             q[0] * gyro[1] - q[1] * gyro[2] + q[3] * gyro[0],
             q[0] * gyro[2] + q[1] * gyro[1] - q[2] * gyro[0]
        ])

        # Integrate rotation
        q_gyro = q + gyro_q * dt

        # Normalize
        q_gyro_norm = np.linalg.norm(q_gyro)
        if q_gyro_norm > 0:
            q_gyro = q_gyro / q_gyro_norm

        # Complementary filter: combine accelerometer and gyroscope information
        result_q = self.alpha * q_gyro + (1.0 - self.alpha) * acc_q

        # Normalize
        result_q_norm = np.linalg.norm(result_q)
        if result_q_norm > 0:
            result_q = result_q / result_q_norm

        self.orientation_q = result_q

        logger.debug(f"Updated orientation: q=[{result_q[0]:.4f}, {result_q[1]:.4f}, {result_q[2]:.4f}, {result_q[3]:.4f}]")
        return result_q


class KalmanFilter(OrientationEstimator):
    """Basic Kalman filter for orientation estimation."""
    
    def __init__(self, freq: float = 30.0):
        super().__init__(freq)
        
        # State vector: quaternion (4), gyro_bias (3)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 1e-4
        self.Q[:4, :4] *= 1e-6  # Lower noise for quaternion
        self.Q[4:, 4:] *= 1e-3  # Higher noise for gyro bias
        
        # Measurement noise covariance
        self.R = np.eye(3) * 0.1  # Accelerometer noise
        
        # Error covariance matrix
        self.P = np.eye(self.state_dim) * 1e-2
        
        logger.debug(f"Initialized KalmanFilter with state_dim={self.state_dim}")
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Update orientation using basic Kalman filter."""
        # Extract current state
        q = self.x[:4]  # Quaternion
        bias = self.x[4:]  # Gyro bias
        
        # Normalize quaternion
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        
        # Correct gyro with estimated bias
        gyro_corrected = gyro - bias
        
        # Prediction step - integrate angular velocity
        omega = np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega)
        
        # State transition
        F = np.eye(self.state_dim)
        F[:4, :4] += 0.5 * dt * self._omega_matrix(gyro_corrected)
        
        # Predict state
        x_pred = self.x.copy()
        x_pred[:4] = q + q_dot * dt
        x_pred[4:] = bias  # Bias assumed constant
        
        # Normalize predicted quaternion
        q_norm = np.linalg.norm(x_pred[:4])
        if q_norm > 0:
            x_pred[:4] = x_pred[:4] / q_norm
        
        # Predict covariance
        P_pred = F @ self.P @ F.T + self.Q
        
        # Measurement update only if we have valid acceleration
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            # Normalize accelerometer
            acc_norm = acc / acc_norm
            
            # Expected gravity direction from orientation
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ np.array([0, 0, 1])  # Gravity in body frame
            
            # Innovation (measurement residual)
            y = acc_norm - g_pred
            
            # Measurement matrix (linearized around current state)
            H = self._compute_H_matrix(x_pred[:4])
            
            # Innovation covariance
            S = H @ P_pred @ H.T + self.R
            
            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.x = x_pred + K @ y
            
            # Update covariance
            self.P = (np.eye(self.state_dim) - K @ H) @ P_pred
        else:
            # No measurement update
            self.x = x_pred
            self.P = P_pred
        
        # Normalize quaternion
        q_norm = np.linalg.norm(self.x[:4])
        if q_norm > 0:
            self.x[:4] = self.x[:4] / q_norm
        
        # Update orientation
        self.orientation_q = self.x[:4]
        
        logger.debug(f"Updated orientation: q=[{self.orientation_q[0]:.4f}, {self.orientation_q[1]:.4f}, "
                    f"{self.orientation_q[2]:.4f}, {self.orientation_q[3]:.4f}]")
        return self.orientation_q
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _omega_matrix(self, gyro):
        """Create omega matrix for quaternion differentiation."""
        wx, wy, wz = gyro
        return np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _compute_H_matrix(self, q):
        """Compute linearized measurement matrix."""
        w, x, y, z = q
        
        # Simplified Jacobian of gravity vector with respect to quaternion
        H_q = np.zeros((3, 4))
        H_q[0, 0] = -2*y
        H_q[0, 1] = 2*z
        H_q[0, 2] = -2*w
        H_q[0, 3] = 2*x
        H_q[1, 0] = 2*x
        H_q[1, 1] = 2*w
        H_q[1, 2] = 2*z
        H_q[1, 3] = 2*y
        H_q[2, 0] = 0
        H_q[2, 1] = -2*y
        H_q[2, 2] = -2*z
        H_q[2, 3] = 0
        
        # Full measurement matrix
        H = np.zeros((3, self.state_dim))
        H[:, :4] = H_q
        
        return H


class ExtendedKalmanFilter(OrientationEstimator):
    """Extended Kalman Filter (EKF) for more accurate orientation estimation."""
    
    def __init__(self, freq: float = 30.0):
        super().__init__(freq)
        
        # State vector: quaternion (4), gyro_bias (3)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 1e-5
        self.Q[:4, :4] *= 1e-6  # Quaternion process noise
        self.Q[4:, 4:] *= 1e-4  # Gyro bias process noise
        
        # Measurement noise covariance for accelerometer
        self.R = np.eye(3) * 0.1
        
        # Error covariance matrix
        self.P = np.eye(self.state_dim) * 1e-2
        
        # Reference vectors
        self.g_ref = np.array([0, 0, 1])  # Gravity reference (normalized)
        
        logger.debug(f"Initialized ExtendedKalmanFilter with state_dim={self.state_dim}")
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Update orientation using EKF."""
        # Extract current state
        q = self.x[:4]  # Quaternion
        bias = self.x[4:]  # Gyro bias
        
        # Normalize quaternion
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        
        # Correct gyro with estimated bias
        gyro_corrected = gyro - bias
        
        # Prediction step - state transition function
        q_dot = 0.5 * self._quaternion_product_matrix(q) @ np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_pred = q + q_dot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)  # Normalize
        
        x_pred = np.zeros_like(self.x)
        x_pred[:4] = q_pred
        x_pred[4:] = bias  # Bias assumed constant
        
        # Jacobian of state transition function
        F = np.eye(self.state_dim)
        F[:4, :4] = self._quaternion_update_jacobian(q, gyro_corrected, dt)
        F[:4, 4:] = -0.5 * dt * self._quaternion_product_matrix(q)[:, 1:]  # Jacobian w.r.t bias
        
        # Predict covariance
        P_pred = F @ self.P @ F.T + self.Q
        
        # Measurement update with accelerometer
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            # Normalize accelerometer
            acc_norm = acc / acc_norm
            
            # Expected gravity direction from orientation
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ self.g_ref
            
            # Create measurement vector and prediction
            z = acc_norm
            h = g_pred
            
            # Measurement Jacobian
            H = self._measurement_jacobian(x_pred[:4])
            
            # Innovation
            y = z - h
            
            # Innovation covariance
            S = H @ P_pred @ H.T + self.R
            
            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.x = x_pred + K @ y
            
            # Update covariance with Joseph form for numerical stability
            I_KH = np.eye(self.state_dim) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            # No measurement update
            self.x = x_pred
            self.P = P_pred
        
        # Ensure quaternion is normalized
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        
        # Set orientation quaternion
        self.orientation_q = self.x[:4]
        
        logger.debug(f"Updated orientation: q=[{self.orientation_q[0]:.4f}, {self.orientation_q[1]:.4f}, "
                     f"{self.orientation_q[2]:.4f}, {self.orientation_q[3]:.4f}]")
        return self.orientation_q
    
    def _quaternion_product_matrix(self, q):
        """Create matrix for quaternion multiplication: p ⊗ q = [q]_L * p."""
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
        ])
    
    def _quaternion_update_jacobian(self, q, gyro, dt):
        """Jacobian of quaternion update with respect to quaternion."""
        wx, wy, wz = gyro
        omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return np.eye(4) + 0.5 * dt * omega
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _measurement_jacobian(self, q):
        """Compute measurement Jacobian for EKF."""
        w, x, y, z = q
        
        # Jacobian for accelerometer measurement (gravity direction)
        H_acc = np.zeros((3, self.state_dim))
        H_acc[:3, :4] = np.array([
            [2*y, 2*z, 2*w, 2*x],
            [-2*z, 2*y, 2*x, -2*w],
            [0, -2*y, -2*z, 0]
        ])
        
        return H_acc


class UnscentedKalmanFilter(OrientationEstimator):
    """Unscented Kalman Filter (UKF) for highly accurate orientation estimation."""
    
    def __init__(self, freq: float = 30.0, alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0):
        super().__init__(freq)
        
        # State vector: quaternion (4), gyro bias (3)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 1e-5
        self.Q[:4, :4] *= 1e-6  # Quaternion process noise
        self.Q[4:, 4:] *= 1e-4  # Gyro bias process noise
        
        # Measurement noise covariance
        self.R = np.eye(3) * 0.1  # Accelerometer noise
        
        # Error covariance matrix
        self.P = np.eye(self.state_dim) * 1e-2
        
        # UKF parameters
        self.alpha = alpha  # Primary scaling parameter
        self.beta = beta    # Secondary scaling parameter
        self.kappa = kappa  # Tertiary scaling parameter
        
        # Derived parameters
        self.lambda_ = self.alpha * self.alpha * (self.state_dim + self.kappa) - self.state_dim
        
        # Calculate weights
        self._calculate_weights()
        
        # Reference gravity vector
        self.g_ref = np.array([0, 0, 1])  # Normalized gravity
        
        logger.debug(f"Initialized UnscentedKalmanFilter with state_dim={self.state_dim}, "
                    f"alpha={alpha}, beta={beta}, kappa={kappa}")
    
    def _calculate_weights(self):
        """Calculate UKF weights for mean and covariance computation."""
        n = self.state_dim
        self.num_sigma_points = 2 * n + 1
        
        # Weights for mean calculation
        self.Wm = np.zeros(self.num_sigma_points)
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wm[1:] = 1.0 / (2 * (n + self.lambda_))
        
        # Weights for covariance calculation
        self.Wc = np.zeros(self.num_sigma_points)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        self.Wc[1:] = self.Wm[1:]
    
    def _generate_sigma_points(self):
        """Generate sigma points for UKF."""
        n = self.state_dim
        
        # Matrix square root of weighted covariance
        U = np.linalg.cholesky((n + self.lambda_) * self.P)
        
        # Sigma points around current estimate
        sigma_points = np.zeros((self.num_sigma_points, n))
        sigma_points[0] = self.x
        
        for i in range(n):
            sigma_points[i+1] = self.x + U[i]
            sigma_points[i+1+n] = self.x - U[i]
        
        return sigma_points
    
    def _quaternion_normalize(self, q):
        """Normalize quaternion."""
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0])  # Default quaternion if normalization fails
    
    def _process_model(self, sigma_point, gyro, dt):
        """Process model for state propagation."""
        # Extract quaternion and bias
        q = sigma_point[:4]
        bias = sigma_point[4:]
        
        # Normalize quaternion
        q = self._quaternion_normalize(q)
        
        # Correct gyro with bias
        gyro_corrected = gyro - bias
        
        # Quaternion integration
        omega = np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega)
        q_pred = q + q_dot * dt
        q_pred = self._quaternion_normalize(q_pred)
        
        # Bias integration (constant model)
        bias_pred = bias
        
        # Predicted state
        x_pred = np.zeros_like(sigma_point)
        x_pred[:4] = q_pred
        x_pred[4:] = bias_pred
        
        return x_pred
    
    def _measurement_model(self, sigma_point):
        """Measurement model for UKF."""
        # Extract quaternion
        q = sigma_point[:4]
        q = self._quaternion_normalize(q)
        
        # Rotation matrix from quaternion
        R = self._quaternion_to_rotation_matrix(q)
        
        # Predicted gravity direction
        g_pred = R @ self.g_ref
        
        return g_pred
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Update orientation using UKF."""
        # Prediction step
        
        # Generate sigma points
        sigma_points = self._generate_sigma_points()
        
        # Propagate sigma points through process model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(self.num_sigma_points):
            sigma_points_pred[i] = self._process_model(sigma_points[i], gyro, dt)
        
        # Calculate predicted mean
        x_pred = np.zeros(self.state_dim)
        for i in range(self.num_sigma_points):
            x_pred += self.Wm[i] * sigma_points_pred[i]
        
        # Ensure quaternion is normalized
        x_pred[:4] = self._quaternion_normalize(x_pred[:4])
        
        # Calculate predicted covariance
        P_pred = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.num_sigma_points):
            diff = sigma_points_pred[i] - x_pred
            # Quaternion difference needs special handling
            diff[:4] = self._quaternion_error(sigma_points_pred[i, :4], x_pred[:4])
            P_pred += self.Wc[i] * np.outer(diff, diff)
        
        # Add process noise
        P_pred += self.Q
        
        # Skip measurement update if acceleration is too small
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            self.x = x_pred
            self.P = P_pred
            self.orientation_q = self.x[:4]
            return self.orientation_q
        
        # Measurement update
        
        # Normalize accelerometer
        acc_norm = acc / acc_norm
        
        # Propagate sigma points through measurement model
        z_pred = np.zeros((self.num_sigma_points, 3))
        for i in range(self.num_sigma_points):
            z_pred[i] = self._measurement_model(sigma_points_pred[i])
        
        # Calculate predicted measurement
        z_mean = np.zeros(3)
        for i in range(self.num_sigma_points):
            z_mean += self.Wm[i] * z_pred[i]
        
        # Innovation covariance
        Pzz = np.zeros((3, 3))
        for i in range(self.num_sigma_points):
            diff = z_pred[i] - z_mean
            Pzz += self.Wc[i] * np.outer(diff, diff)
        
        # Add measurement noise
        Pzz += self.R
        
        # Cross-correlation matrix
        Pxz = np.zeros((self.state_dim, 3))
        for i in range(self.num_sigma_points):
            diff_x = sigma_points_pred[i] - x_pred
            diff_x[:4] = self._quaternion_error(sigma_points_pred[i, :4], x_pred[:4])
            diff_z = z_pred[i] - z_mean
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
        
        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)
        
        # Update state
        innovation = acc_norm - z_mean
        self.x = x_pred + K @ innovation
        
        # Normalize quaternion
        self.x[:4] = self._quaternion_normalize(self.x[:4])
        
        # Update covariance
        self.P = P_pred - K @ Pzz @ K.T
        
        # Set orientation quaternion
        self.orientation_q = self.x[:4]
        
        logger.debug(f"Updated orientation: q=[{self.orientation_q[0]:.4f}, {self.orientation_q[1]:.4f}, "
                    f"{self.orientation_q[2]:.4f}, {self.orientation_q[3]:.4f}]")
        return self.orientation_q
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _quaternion_error(self, q1, q2):
        """Calculate quaternion error (minimal representation)."""
        # Compute quaternion difference
        q_diff = self._quaternion_multiply(q1, self._quaternion_inverse(q2))
        
        # Ensure positive scalar part
        if q_diff[0] < 0:
            q_diff = -q_diff
        
        # For small angles, the vector part is approximately proportional to the rotation angle
        if abs(q_diff[0]) > 0.9999:  # Near identity, avoid numerical issues
            return np.zeros(4)
        
        return q_diff
    
    def _quaternion_inverse(self, q):
        """Calculate quaternion inverse (conjugate for unit quaternions)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])


def apply_adaptive_filter(acc_data: np.ndarray, cutoff_freq: float = 2.0, fs: float = 30.0) -> np.ndarray:
    """
    Apply adaptive Butterworth filter with robust handling for small input sizes.

    Args:
        acc_data: Linear acceleration data
        cutoff_freq: Cutoff frequency for filter
        fs: Sampling frequency

    Returns:
        Filtered acceleration data
    """
    logger.debug(f"Applying adaptive filter with cutoff={cutoff_freq}Hz, fs={fs}Hz")

    # Get data length
    data_length = acc_data.shape[0]

    # For very small inputs, use a simpler filter or return the original data
    if data_length < 15:  # Minimum size needed for default padlen in filtfilt
        logger.warning(f"Input data too small for Butterworth filtering (length={data_length}), using simple smoothing")
        filtered_data = np.zeros_like(acc_data)

        for i in range(acc_data.shape[1]):
            if data_length > 2:
                # Simple moving average for small data
                filtered_data[:, i] = np.convolve(acc_data[:, i], np.ones(3)/3, mode='same')
            else:
                # Just copy the data if too small
                filtered_data[:, i] = acc_data[:, i]

        return filtered_data

    # For normal sized data, use Butterworth filter
    filtered_data = np.zeros_like(acc_data)

    # Calculate appropriate padlen based on data size (must be < data_length)
    padlen = min(data_length - 1, 10)

    try:
        # Apply filter to each axis
        for i in range(acc_data.shape[1]):
            # Design the Butterworth filter
            b, a = butter(4, cutoff_freq / (fs/2), btype='low')

            # Apply the filter with custom padlen
            filtered_data[:, i] = filtfilt(b, a, acc_data[:, i], padlen=padlen)

    except Exception as e:
        logger.error(f"Filtering failed: {str(e)}, returning original data")
        filtered_data = acc_data.copy()

    return filtered_data


def process_imu_data_chunk(chunk_id: int, acc_data: np.ndarray, gyro_data: np.ndarray, 
                         timestamps: Optional[np.ndarray], filter_type: str, 
                         start_quaternion: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Process a chunk of IMU data for parallelization.

    Args:
        chunk_id: Identifier for this chunk
        acc_data: Accelerometer data for this chunk
        gyro_data: Gyroscope data for this chunk
        timestamps: Timestamps for this chunk
        filter_type: Type of filter to use
        start_quaternion: Initial quaternion for continuity between chunks

    Returns:
        Dictionary with processed data
    """
    try:
        # Apply adaptive filtering to linear acceleration
        if acc_data.shape[0] >= 15:  # Only filter if enough samples
            acc_data = apply_adaptive_filter(acc_data)

        # Initialize orientation filter
        if filter_type.lower() == 'madgwick':
            orientation_filter = MadgwickFilter()
        elif filter_type.lower() == 'comp':
            orientation_filter = CompFilter()
        elif filter_type.lower() == 'kalman':
            orientation_filter = KalmanFilter()
        elif filter_type.lower() == 'ekf':
            orientation_filter = ExtendedKalmanFilter()
        elif filter_type.lower() == 'ukf':
            orientation_filter = UnscentedKalmanFilter()
        else:
            logger.warning(f"Unknown filter type: {filter_type}, falling back to Madgwick")
            orientation_filter = MadgwickFilter()

        # Set initial quaternion if provided
        if start_quaternion is not None:
            orientation_filter.orientation_q = start_quaternion

        # Process data
        quaternions = []
        acc_with_gravity = []  # Store accelerometer with added gravity for each frame

        for i in range(len(acc_data)):
            # For orientation estimation, we need to reconstruct raw accelerometer data
            # by adding estimated gravity based on current orientation
            if i > 0:
                # Use current orientation to estimate gravity direction
                q = orientation_filter.orientation_q
                R = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
                gravity = R.apply([0, 0, 9.81])  # Gravity in body frame

                # Add gravity back to get raw-equivalent accelerometer reading for orientation update
                raw_equiv_acc = acc_data[i] + gravity
            else:
                # For first sample, assume gravity is aligned with z-axis
                raw_equiv_acc = acc_data[i] + np.array([0, 0, 9.81])

            # Store for later
            acc_with_gravity.append(raw_equiv_acc)

            # Get sensor readings
            gyro = gyro_data[i]
            timestamp = timestamps[i] if timestamps is not None else None

            # Update orientation using raw-equivalent accelerometer data
            q = orientation_filter.update(raw_equiv_acc, gyro, timestamp)
            quaternions.append(q)

        # Convert to numpy arrays
        quaternions = np.array(quaternions)
        acc_with_gravity = np.array(acc_with_gravity)

        # Return processed data for this chunk
        return {
            'quaternion': quaternions,
            'linear_acceleration': acc_data,
            'accelerometer': acc_with_gravity,  # Added for completeness
            'angular_velocity': gyro_data,
            'final_quaternion': quaternions[-1] if len(quaternions) > 0 else None
        }
    except Exception as e:
        logger.error(f"Error processing IMU data chunk {chunk_id}: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty results with appropriate shapes
        return {
            'quaternion': np.zeros((len(acc_data), 4)) if len(acc_data) > 0 else np.zeros((0, 4)),
            'linear_acceleration': acc_data, 
            'accelerometer': np.zeros_like(acc_data),
            'angular_velocity': gyro_data,
            'final_quaternion': None
        }


def process_imu_data(acc_data: np.ndarray, gyro_data: np.ndarray,
                    timestamps: np.ndarray = None,
                    filter_type: str = 'madgwick',
                    return_features: bool = True,
                    file_id: int = None) -> Dict[str, np.ndarray]:
    """
    Process IMU data to extract orientation and derived features.
    Uses parallel processing for large datasets.

    Important: This function assumes acc_data is already linear acceleration,
    not raw accelerometer data with gravity component.

    Args:
        acc_data: Linear acceleration data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        timestamps: Optional timestamps for variable rate processing
        filter_type: Type of orientation filter ('madgwick', 'comp', 'kalman', 'ekf', 'ukf')
        return_features: Whether to compute derived features
        file_id: Optional identifier for this file (for thread pool management)

    Returns:
        Dictionary with processed data and features
    """
    start_time = time.time()
    logger.info(f"Processing IMU data: acc={acc_data.shape}, gyro={gyro_data.shape}, filter={filter_type}")

    # Input validation
    if acc_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        logger.error("Empty input data")
        return {
            'quaternion': np.zeros((0, 4)),
            'linear_acceleration': np.zeros((0, 3)),  # Pass through the linear acceleration
            'fusion_features': np.zeros(0) if return_features else None
        }

    if acc_data.shape[0] != gyro_data.shape[0]:
        logger.warning(f"Mismatched data lengths: acc={acc_data.shape[0]}, gyro={gyro_data.shape[0]}")
        # Adjust to common length
        min_len = min(acc_data.shape[0], gyro_data.shape[0])
        acc_data = acc_data[:min_len]
        gyro_data = gyro_data[:min_len]
        if timestamps is not None:
            timestamps = timestamps[:min_len]

    try:
        # Process data using parallel chunks for large datasets
        data_length = len(acc_data)
        if data_length > 1000:  # Large dataset
            # Determine optimal chunk size based on data length
            num_chunks = min(THREADS_PER_FILE, max(1, data_length // 500))
            chunk_size = data_length // num_chunks
            
            logger.info(f"Using {num_chunks} parallel chunks for processing {data_length} samples")
            
            # Create a thread pool for this file
            if file_id is None:
                file_id = id(acc_data)  # Use object ID as fallback
            
            # Get or create a thread pool for this file
            thread_pool = get_processing_thread_pool(file_id)
            
            # Create chunks with overlap
            futures = []
            last_quaternion = None
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(data_length, (i + 1) * chunk_size + 5)  # 5-sample overlap
                
                # Extract chunk data
                chunk_acc = acc_data[start_idx:end_idx]
                chunk_gyro = gyro_data[start_idx:end_idx]
                chunk_timestamps = timestamps[start_idx:end_idx] if timestamps is not None else None
                
                # Submit task
                futures.append((
                    i,
                    thread_pool.submit(
                        process_imu_data_chunk,
                        i,
                        chunk_acc,
                        chunk_gyro,
                        chunk_timestamps,
                        filter_type,
                        last_quaternion
                    )
                ))
                
                # Ensure each chunk starts with the final orientation from the previous chunk
                if i < num_chunks - 1:
                    # Process just enough samples to get a good quaternion for the next chunk
                    mini_acc = acc_data[end_idx-10:end_idx+10] if end_idx+10 < data_length else acc_data[end_idx-10:end_idx]
                    mini_gyro = gyro_data[end_idx-10:end_idx+10] if end_idx+10 < data_length else gyro_data[end_idx-10:end_idx]
                    mini_ts = timestamps[end_idx-10:end_idx+10] if timestamps is not None and end_idx+10 < data_length else timestamps[end_idx-10:end_idx] if timestamps is not None else None
                    
                    # Initialize appropriate filter
                    if filter_type.lower() == 'madgwick':
                        mini_filter = MadgwickFilter()
                    elif filter_type.lower() == 'comp':
                        mini_filter = CompFilter()
                    elif filter_type.lower() == 'kalman':
                        mini_filter = KalmanFilter()
                    elif filter_type.lower() == 'ekf':
                        mini_filter = ExtendedKalmanFilter()
                    elif filter_type.lower() == 'ukf':
                        mini_filter = UnscentedKalmanFilter()
                    else:
                        mini_filter = MadgwickFilter()
                    
                    # Start with the last quaternion if available
                    if last_quaternion is not None:
                        mini_filter.orientation_q = last_quaternion
                    
                    # Process the mini segment to get a continuation quaternion
                    for j in range(len(mini_acc)):
                        # Add gravity component based on current orientation
                        q = mini_filter.orientation_q
                        R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                        gravity = R.apply([0, 0, 9.81])
                        acc_with_g = mini_acc[j] + gravity
                        
                        # Update orientation
                        ts = mini_ts[j] if mini_ts is not None else None
                        q = mini_filter.update(acc_with_g, mini_gyro[j], ts)
                    
                    # Use this quaternion for the next chunk
                    last_quaternion = mini_filter.orientation_q
            
            # Collect results
            all_chunks = []
            for chunk_id, future in sorted(futures):
                try:
                    chunk_result = future.result()
                    all_chunks.append(chunk_result)
                except Exception as e:
                    logger.error(f"Error in chunk {chunk_id}: {e}")
                    # Handle failure by inserting empty data
                    start_idx = chunk_id * chunk_size
                    end_idx = min(data_length, (chunk_id + 1) * chunk_size)
                    chunk_length = end_idx - start_idx
                    all_chunks.append({
                        'quaternion': np.zeros((chunk_length, 4)),
                        'linear_acceleration': acc_data[start_idx:end_idx],
                        'accelerometer': np.zeros((chunk_length, 3)),
                        'angular_velocity': gyro_data[start_idx:end_idx],
                        'final_quaternion': None
                    })
            
            # Combine chunks (skipping overlap)
            quaternions = []
            linear_acc = []
            accelerometer = []
            angular_velocity = []
            
            for i, chunk in enumerate(all_chunks):
                # Determine how many samples to take from this chunk
                if i < len(all_chunks) - 1:
                    # Not the last chunk - take all but the overlap
                    end_idx = len(chunk['quaternion']) - 5 if len(chunk['quaternion']) > 5 else len(chunk['quaternion'])
                else:
                    # Last chunk - take everything
                    end_idx = len(chunk['quaternion'])
                
                # Add data to the combined result
                quaternions.append(chunk['quaternion'][:end_idx])
                linear_acc.append(chunk['linear_acceleration'][:end_idx])
                accelerometer.append(chunk['accelerometer'][:end_idx])
                angular_velocity.append(chunk['angular_velocity'][:end_idx])
            
            # Concatenate all chunks
            quaternions = np.concatenate(quaternions, axis=0)
            linear_acc = np.concatenate(linear_acc, axis=0)
            accelerometer = np.concatenate(accelerometer, axis=0)
            angular_velocity = np.concatenate(angular_velocity, axis=0)
            
            # Trim to original length
            quaternions = quaternions[:data_length]
            linear_acc = linear_acc[:data_length]
            accelerometer = accelerometer[:data_length]
            angular_velocity = angular_velocity[:data_length]
            
        else:
            # For smaller datasets, process sequentially
            chunk_result = process_imu_data_chunk(
                0, acc_data, gyro_data, timestamps, filter_type
            )
            quaternions = chunk_result['quaternion']
            linear_acc = chunk_result['linear_acceleration']
            accelerometer = chunk_result['accelerometer']
            angular_velocity = chunk_result['angular_velocity']

        # Create result dictionary
        results = {
            'quaternion': quaternions,
            'linear_acceleration': linear_acc  # Pass through the input linear acceleration
        }

        # Extract features if requested
        if return_features:
            logger.debug("Extracting derived features")
            features = extract_features_from_window({
                'quaternion': quaternions,
                'linear_acceleration': linear_acc,
                'angular_velocity': angular_velocity
            })
            results['fusion_features'] = features

        elapsed_time = time.time() - start_time
        logger.info(f"IMU processing complete in {elapsed_time:.2f}s")

        return results

    except Exception as e:
        logger.error(f"Error in IMU processing: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty results on error
        return {
            'quaternion': np.zeros((len(acc_data), 4)) if len(acc_data) > 0 else np.zeros((0, 4)),
            'linear_acceleration': acc_data,  # Return original accelerometer data
            'fusion_features': np.zeros(43) if return_features else None
        }


def extract_features_from_window(window_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract features from a window of IMU data.

    Args:
        window_data: Dictionary containing quaternion, linear_acceleration, angular_velocity

    Returns:
        Feature vector
    """
    start_time = time.time()
    logger.debug("Extracting features from window")

    try:
        # Extract data
        quaternions = window_data['quaternion']
        acc_data = window_data['linear_acceleration']
        gyro_data = window_data['angular_velocity']

        # Handle empty data
        if len(quaternions) == 0 or len(acc_data) == 0 or len(gyro_data) == 0:
            logger.warning("Empty data in feature extraction, returning zeros")
            return np.zeros(43)

        # Statistical features from linear acceleration
        acc_mean = np.mean(acc_data, axis=0)
        acc_std = np.std(acc_data, axis=0)
        acc_max = np.max(acc_data, axis=0)
        acc_min = np.min(acc_data, axis=0)

        # Magnitude of linear acceleration
        acc_mag = np.linalg.norm(acc_data, axis=1)
        acc_mag_mean = np.mean(acc_mag)
        acc_mag_std = np.std(acc_mag)
        acc_mag_max = np.max(acc_mag)

        # Angular velocity features
        gyro_mean = np.mean(gyro_data, axis=0)
        gyro_std = np.std(gyro_data, axis=0)
        gyro_max = np.max(np.abs(gyro_data), axis=0)

        # Jerk (derivative of acceleration)
        jerk_features = []
        if len(acc_data) > 1:
            # Compute jerk
            jerk = np.diff(acc_data, axis=0, prepend=acc_data[0].reshape(1, -1))
            jerk_mag = np.linalg.norm(jerk, axis=1)
            jerk_mag_mean = np.mean(jerk_mag)
            jerk_mag_max = np.max(jerk_mag)
            jerk_features = [jerk_mag_mean, jerk_mag_max]
        else:
            jerk_features = [0, 0]

        # Convert quaternions to Euler angles for feature extraction
        euler_angles = []
        for q in quaternions:
            # scipy expects x, y, z, w format
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            euler_angles.append(r.as_euler('xyz', degrees=True))

        euler_angles = np.array(euler_angles)

        # Extract orientation features
        euler_mean = np.mean(euler_angles, axis=0)
        euler_std = np.std(euler_angles, axis=0)

        # Calculate orientation change rate
        angle_rate_features = []
        if len(euler_angles) > 1:
            angle_rates = np.diff(euler_angles, axis=0, prepend=euler_angles[0].reshape(1, -1))
            angle_rate_mag = np.linalg.norm(angle_rates, axis=1)
            angle_rate_mean = np.mean(angle_rate_mag)
            angle_rate_max = np.max(angle_rate_mag)
            angle_rate_features = [angle_rate_mean, angle_rate_max]
        else:
            angle_rate_features = [0, 0]

        # Add frequency domain features (FFT-based)
        fft_features = []
        if len(acc_data) >= 8:  # Minimum length for meaningful FFT
            # Use GPU for FFT if available for larger windows
            if USE_GPU and len(acc_data) > 1000:
                try:
                    # Choose GPU device
                    device_id = GPU_DEVICES[0] if GPU_DEVICES else 'cpu'
                    device = torch.device(f'cuda:{device_id}' if device_id != 'cpu' else 'cpu')
                    
                    # Use PyTorch for GPU-accelerated FFT
                    for axis in range(acc_data.shape[1]):
                        # Convert to torch tensor and move to GPU
                        acc_tensor = torch.tensor(acc_data[:, axis], dtype=torch.float32).to(device)
                        
                        # Compute FFT
                        fft_tensor = torch.abs(torch.fft.rfft(acc_tensor))
                        
                        # Extract features
                        if len(fft_tensor) > 3:
                            fft_features.extend([
                                fft_tensor.max().item(),
                                fft_tensor.mean().item(),
                                fft_tensor.var().item()
                            ])
                        else:
                            fft_features.extend([0, 0, 0])
                except Exception as e:
                    logger.warning(f"GPU FFT failed: {str(e)}, falling back to CPU")
                    for axis in range(acc_data.shape[1]):
                        fft = np.abs(np.fft.rfft(acc_data[:, axis]))
                        if len(fft) > 3:
                            fft_features.extend([np.max(fft), np.mean(fft), np.var(fft)])
                        else:
                            fft_features.extend([0, 0, 0])
            else:
                # CPU FFT
                for axis in range(acc_data.shape[1]):
                    fft = np.abs(np.fft.rfft(acc_data[:, axis]))
                    if len(fft) > 3:
                        fft_features.extend([np.max(fft), np.mean(fft), np.var(fft)])
                    else:
                        fft_features.extend([0, 0, 0])
        else:
            fft_features = [0] * 9  # 3 features * 3 axes

        # Compile features into a single vector
        features = np.concatenate([
            acc_mean, acc_std, acc_max, acc_min,
            [acc_mag_mean, acc_mag_std, acc_mag_max],
            gyro_mean, gyro_std, gyro_max,
            jerk_features,
            euler_mean, euler_std,
            angle_rate_features,
            fft_features
        ])

        elapsed_time = time.time() - start_time
        logger.debug(f"Feature extraction complete: {len(features)} features in {elapsed_time:.4f}s")

        return features

    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}\n{traceback.format_exc()}")
        return np.zeros(43)  # Return zeros in case of failure
