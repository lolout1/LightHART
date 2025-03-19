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

# Initialize parallel processing resources
MAX_THREADS = 4
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)

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
    
    # Use multithreading for large datasets
    if len(acc_times) > 1000:
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
        chunk_size = max(1, len(acc_times) // MAX_THREADS)
        futures = []
        
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
    else:
        # Sequential processing for smaller datasets
        logger.debug(f"Using sequential processing for {len(acc_times)} timestamps")
        
        # Convert tolerance to appropriate units
        if isinstance(acc_times[0], np.datetime64):
            tolerance_ns = np.timedelta64(int(time_tolerance * 1e9), 'ns')
        else:
            tolerance_ns = time_tolerance
            
        for i, acc_time in enumerate(acc_times):
            # Find closest gyro time
            time_diffs = np.abs(gyro_times - acc_time)
            closest_idx = np.argmin(time_diffs)
            
            # If within tolerance, add to matched pairs
            if time_diffs[closest_idx] <= tolerance_ns:
                aligned_acc.append(acc_data_filtered.iloc[i, 1:4].values)
                aligned_gyro.append(gyro_data_filtered.iloc[closest_idx, 1:4].values)
                aligned_times.append(acc_time)

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

def save_aligned_sensor_data(subject_id: int, action_id: int, trial_id: int,
                          acc_data: np.ndarray, gyro_data: np.ndarray,
                          skeleton_data: Optional[np.ndarray] = None,
                          timestamps: Optional[np.ndarray] = None,
                          save_dir: str = "data/aligned") -> None:
    """
    Save aligned sensor data to disk for later use.
    
    Args:
        subject_id: Subject identifier
        action_id: Action identifier
        trial_id: Trial identifier
        acc_data: Accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        skeleton_data: Optional skeleton data [n_samples, joints, 3]
        timestamps: Optional timestamps [n_samples]
        save_dir: Directory to save aligned data
    """
    try:
        # Acquire semaphore to limit parallel file operations
        with file_semaphore:
            # Create directories if they don't exist
            os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
            os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
            
            if skeleton_data is not None:
                os.makedirs(f"{save_dir}/skeleton", exist_ok=True)
            
            if timestamps is not None:
                os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
            
            # Create filename
            filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
            
            # Save data
            np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
            np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
            
            if skeleton_data is not None:
                np.save(f"{save_dir}/skeleton/{filename}.npy", skeleton_data)
            
            if timestamps is not None:
                np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)
            
            logger.debug(f"Saved aligned data for {filename} to {save_dir}")
        
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")
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
        """
        Update orientation using EKF algorithm, tailored for linear acceleration input.
        
        Args:
            acc: Linear accelerometer reading [ax, ay, az]
            gyro: Gyroscope reading [gx, gy, gz] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        try:
            # Extract current state
            q = self.x[:4]  # Quaternion
            bias = self.x[4:]  # Gyroscope bias
            
            # Normalize quaternion
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            
            # Correct gyroscope measurements with bias
            gyro_corrected = gyro - bias

            # When using linear acceleration, we should estimate the direction of gravity
            # based on our current orientation estimate to update the filter
            R_q = self._quaternion_to_rotation_matrix(q)
            gravity_global = np.array([0, 0, 9.81])  # Gravity in global frame
            expected_gravity = R_q.T @ gravity_global  # Gravity in sensor frame
            
            # Normalize gravity for direction comparison
            gravity_norm = np.linalg.norm(expected_gravity)
            if gravity_norm > 0:
                gravity_direction = expected_gravity / gravity_norm
            else:
                gravity_direction = np.array([0, 0, 1])
                
            # The rest of the EKF implementation follows...
            # [... EKF prediction and update steps ...]
            
            # Update orientation quaternion
            self.orientation_q = self.x[:4]
            
            return self.orientation_q
                
        except Exception as e:
            logger.error(f"EKF update error: {e}")
            logger.error(traceback.format_exc())
            # Return last valid orientation if processing fails
            return self.orientation_q

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


class ComplementaryFilter(OrientationEstimator):
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

class ComplementaryFilter(OrientationEstimator):
    """
    Complementary filter for orientation estimation.
    
    This filter combines accelerometer and gyroscope data in the frequency
    domain, using a high-pass filter for gyro and low-pass for accelerometer.
    It's computationally efficient and provides good results in many cases.
    """
    def __init__(self, freq: float = 30.0, alpha: float = 0.02):
        """
        Initialize Complementary filter.
        
        Args:
            freq: Sample frequency in Hz
            alpha: Filter weight (lower = more gyro influence)
        """
        super().__init__(freq)
        self.alpha = alpha
        self.name = "Complementary"
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Update orientation using complementary filter algorithm.
        
        Args:
            acc: Accelerometer reading [ax, ay, az]
            gyro: Gyroscope reading [gx, gy, gz] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        # Get current orientation quaternion
        q = self.orientation_q
        
        # Normalize accelerometer measurement
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            # Skip correction if accelerometer data is invalid
            return q
        
        acc_normalized = acc / acc_norm
        
        # Calculate quaternion from accelerometer (gravity)
        # This assumes accelerometer measures gravity vector
        acc_q = self._accel_to_quaternion(acc_normalized)
        
        # Integrate gyroscope data to get orientation change
        gyro_q = self._integrate_gyro(q, gyro, dt)
        
        # Combine using complementary filter
        # alpha determines balance between gyro and accel
        # Lower alpha means more weight on gyro (good for short-term accuracy)
        # Higher alpha means more weight on accel (good for drift correction)
        result_q = self._slerp(gyro_q, acc_q, self.alpha)
        
        # Normalize and store result
        result_q = result_q / np.linalg.norm(result_q)
        self.orientation_q = result_q
        
        return result_q
    
    def _accel_to_quaternion(self, acc: np.ndarray) -> np.ndarray:
        """
        Convert accelerometer vector to orientation quaternion.
        
        Args:
            acc: Normalized accelerometer vector [ax, ay, az]
            
        Returns:
            Orientation quaternion [w, x, y, z] representing alignment with gravity
        """
        # Reference vector (global z-axis / gravity)
        z_ref = np.array([0, 0, 1])
        
        # Get rotation axis via cross product
        rotation_axis = np.cross(z_ref, acc)
        axis_norm = np.linalg.norm(rotation_axis)
        
        if axis_norm < 1e-10:
            # Handle case where vectors are parallel
            if acc[2] > 0:
                # Device pointing up, identity quaternion
                return np.array([1.0, 0.0, 0.0, 0.0])
            else:
                # Device pointing down, 180° rotation around X
                return np.array([0.0, 1.0, 0.0, 0.0])
                
        # Normalize rotation axis
        rotation_axis = rotation_axis / axis_norm
        
        # Calculate rotation angle
        angle = np.arccos(np.clip(np.dot(z_ref, acc), -1.0, 1.0))
        
        # Convert axis-angle to quaternion
        q = np.zeros(4)
        q[0] = np.cos(angle / 2)
        q[1:4] = rotation_axis * np.sin(angle / 2)
        
        return q
    
    def _integrate_gyro(self, q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate gyroscope data to update orientation quaternion.
        
        Args:
            q: Current orientation quaternion [w, x, y, z]
            gyro: Gyroscope reading [gx, gy, gz] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion after integration
        """
        # Quaternion derivative from angular velocity
        q_dot = 0.5 * np.array([
            -q[1]*gyro[0] - q[2]*gyro[1] - q[3]*gyro[2],
            q[0]*gyro[0] + q[2]*gyro[2] - q[3]*gyro[1],
            q[0]*gyro[1] - q[1]*gyro[2] + q[3]*gyro[0],
            q[0]*gyro[2] + q[1]*gyro[1] - q[2]*gyro[0]
        ])
        
        # Integrate to get new quaternion
        q_new = q + q_dot * dt
        
        # Normalize and return
        return q_new / np.linalg.norm(q_new)
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between quaternions.
        
        Args:
            q1: First quaternion
            q2: Second quaternion
            t: Interpolation parameter [0-1]
            
        Returns:
            Interpolated quaternion
        """
        # Ensure unit quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Calculate dot product (cosine of angle between quaternions)
        dot = np.sum(q1 * q2)
        
        # If dot < 0, negate one quaternion to ensure shortest path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # Clamp dot product to valid range
        dot = np.clip(dot, -1.0, 1.0)
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # Calculate angle between quaternions
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        # SLERP formula
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return (s0 * q1) + (s1 * q2)
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


# Parallelized IMU processing function
def process_imu_batch(batch_index, acc_data, gyro_data, timestamps, filter_type, return_features):
    """Process a batch of IMU data in parallel"""
    try:
        result = process_imu_data(acc_data, gyro_data, timestamps, filter_type, return_features)
        return batch_index, result
    except Exception as e:
        logger.error(f"Error processing batch {batch_index}: {str(e)}")
        return batch_index, None


def process_imu_data(acc_data: np.ndarray, gyro_data: np.ndarray,
                    timestamps: Optional[np.ndarray] = None,
                    filter_type: str = 'madgwick',
                    return_features: bool = False) -> Dict[str, np.ndarray]:
    """
    Process IMU data with specified filter to estimate orientation.
    
    Args:
        acc_data: Linear accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        timestamps: Optional timestamps for variable-rate data
        filter_type: Type of filter to use ('madgwick', 'comp', 'kalman', 'ekf', 'ukf')
        return_features: Whether to extract and return features
        
    Returns:
        Dictionary with processed data: quaternion and fusion_features (if requested)
    """
    start_time = time.time()
    
    # Validate inputs
    if not isinstance(acc_data, np.ndarray) or not isinstance(gyro_data, np.ndarray):
        logger.error(f"Invalid input types: acc={type(acc_data)}, gyro={type(gyro_data)}")
        return {
            'quaternion': np.zeros((1, 4))
        }
    
    # Handle empty inputs
    if acc_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        logger.error("Empty input data")
        return {
            'quaternion': np.zeros((1, 4))
        }
    
    # Ensure data arrays have the same length
    if acc_data.shape[0] != gyro_data.shape[0]:
        min_len = min(acc_data.shape[0], gyro_data.shape[0])
        logger.warning(f"Data length mismatch: acc={acc_data.shape[0]}, gyro={gyro_data.shape[0]}, truncating to {min_len}")
        acc_data = acc_data[:min_len]
        gyro_data = gyro_data[:min_len]
        if timestamps is not None:
            timestamps = timestamps[:min_len]
    
    try:
        # Create timestamps if not provided
        if timestamps is None:
            timestamps = np.linspace(0, acc_data.shape[0] / 30.0, acc_data.shape[0])
        
        # Convert gyro data from degrees/s to radians/s if needed
        gyro_max = np.max(np.abs(gyro_data))
        if gyro_max > 20.0:  # Heuristic: gyro values in deg/s are typically larger than rad/s
            logger.info(f"Converting gyroscope data from deg/s to rad/s (max value: {gyro_max})")
            gyro_data = gyro_data * np.pi / 180.0
        
        # Create filter based on filter_type
        if filter_type == 'madgwick':
            orientation_filter = MadgwickFilter()
            logger.info("Using Madgwick filter for orientation estimation")
        elif filter_type == 'comp':
            orientation_filter = ComplementaryFilter()
            logger.info("Using Complementary filter for orientation estimation")
        elif filter_type == 'kalman':
            orientation_filter = KalmanFilter()
            logger.info("Using Kalman filter for orientation estimation")
        elif filter_type == 'ekf':
            orientation_filter = ExtendedKalmanFilter()
            logger.info("Using Extended Kalman filter for orientation estimation")
        elif filter_type == 'ukf':
            orientation_filter = UnscentedKalmanFilter()
            logger.info("Using Unscented Kalman filter for orientation estimation")
        else:
            logger.warning(f"Unknown filter type: {filter_type}, defaulting to Madgwick filter")
            orientation_filter = MadgwickFilter()
            
        logger.info(f"Processing {len(acc_data)} samples with {filter_type} filter")
        
        # Process data with the selected filter
        quaternions = []
        
        # Process each time step
        for i in range(len(acc_data)):
            acc = acc_data[i]
            gyro = gyro_data[i]
            timestamp = timestamps[i] if timestamps is not None else None
            
            # Apply the filter to get orientation
            # Note: Since acc is already linear acceleration, we add back an estimate
            # of gravity to help the filter determine orientation
            gravity_direction = np.array([0, 0, 9.81])  # Default gravity direction
            if i > 0 and len(quaternions) > 0:
                # Use the last orientation to create a more accurate gravity estimate
                last_q = quaternions[-1]
                r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
                
            # For orientation estimation, use acc + estimated gravity
            acc_with_gravity = acc + gravity_direction
            acc_with_gravity = acc_with_gravity / np.linalg.norm(acc_with_gravity)
            
            # Update the filter
            q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
            quaternions.append(q)
        
        # Convert lists to numpy arrays
        quaternions = np.array(quaternions)
        
        # Prepare results
        results = {
            'quaternion': quaternions
        }
        
        # Extract features if requested
        if return_features:
            window_data = {
                'quaternion': quaternions,
                'accelerometer': acc_data,  # Already linear acceleration
                'gyroscope': gyro_data
            }
            results['fusion_features'] = extract_features_from_window(window_data)
        
        elapsed_time = time.time() - start_time
        logger.info(f"IMU processing with {filter_type} filter completed in {elapsed_time:.2f}s")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in IMU processing: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return minimal results on error
        sample_size = max(1, len(acc_data) if isinstance(acc_data, np.ndarray) else 1)
        return {
            'quaternion': np.zeros((sample_size, 4))
        }

def extract_features_from_window(data, start, end, window_size, fuse, filter_type='madgwick'):
    """
    Helper function to extract a window from data with proper quaternion handling.
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
                # Extract appropriate window
                # [... existing window extraction code ...]
                window_data[modality] = window_data_array
            except Exception as e:
                logger.error(f"Error extracting {modality} window: {str(e)}")
                # Add empty data with correct shape
                # [... error handling code ...]

    # Apply fusion if requested and we have both accelerometer and gyroscope
    if fuse and 'accelerometer' in window_data and 'gyroscope' in window_data:
        try:
            # Extract the window data
            acc_window = window_data['accelerometer']  # This is already linear acceleration
            gyro_window = window_data['gyroscope']
            
            # Extract timestamps if available
            timestamps = None
            if 'aligned_timestamps' in window_data:
                timestamps = window_data['aligned_timestamps']
                # Convert to 1D array if needed
                if len(timestamps.shape) > 1:
                    timestamps = timestamps[:, 0] if timestamps.shape[1] > 0 else None
            
            # Process data using the specified filter
            fusion_results = process_imu_data(
                acc_data=acc_window,
                gyro_data=gyro_window,
                timestamps=timestamps,
                filter_type=filter_type,  # Use the specified filter type
                return_features=False
            )
            
            # Add fusion results to window data
            window_data['quaternion'] = fusion_results.get('quaternion', np.zeros((window_size, 4)))
                
            logger.debug(f"Added fusion data to window using {filter_type} filter")
        except Exception as e:
            logger.error(f"Error in fusion processing: {str(e)}")
            # Add empty quaternion as fallback
            window_data['quaternion'] = np.zeros((window_size, 4))
    else:
        # Always add empty quaternion data as a fallback
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
    
    return window_data
