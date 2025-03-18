'''
IMU Fusion Module for Enhanced Orientation Estimation

This module provides multiple sensor fusion algorithms for converting raw 
accelerometer and gyroscope data into orientation estimates (quaternions) and
derived features for fall detection. It implements standard Kalman, Extended Kalman,
and Unscented Kalman filters optimized for wearable sensors.
'''

import numpy as np
from scipy.spatial.transform import Rotation
import math
import pandas as pd
import logging
import os
import time
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
os.makedirs("debug_logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("debug_logs", "imu_fusion.log"),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("imu_fusion")

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Thread pool configuration
MAX_THREADS = 8
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(4)  # Limit concurrent file processing

def update_thread_configuration(max_files: int, threads_per_file: int):
    """
    Update thread pool configuration for parallel processing
    
    Args:
        max_files: Maximum number of files to process in parallel
        threads_per_file: Maximum number of threads to use per file
    """
    global MAX_THREADS, thread_pool, file_semaphore
    
    # Calculate new number of threads
    new_total = max_files * threads_per_file
    
    # Update thread pool if needed
    if new_total != MAX_THREADS:
        # Shutdown existing pool
        thread_pool.shutdown(wait=True)
        
        # Create new pool
        MAX_THREADS = new_total
        thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        file_semaphore = threading.Semaphore(max_files)
        
        logger.info(f"Updated thread pool: {max_files} files × {threads_per_file} threads = {MAX_THREADS} total")
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
def align_sensor_data(acc_data: pd.DataFrame, gyro_data: pd.DataFrame,
                     time_tolerance: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align accelerometer and gyroscope data based on timestamps.

    Args:
        acc_data: Accelerometer data with timestamp column
        gyro_data: Gyroscope data with timestamp column
        time_tolerance: Maximum time difference to consider readings aligned

    Returns:
        Tuple of (aligned_acc, aligned_gyro, timestamps)
    """
    start_time = time.time()
    logger.info(f"Starting sensor alignment: acc shape={acc_data.shape}, gyro shape={gyro_data.shape}")

    try:
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
        end_time_point = min(acc_times[-1], gyro_times[-1])
        logger.debug(f"Common time range: {start_time_point} to {end_time_point}")

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

        if isinstance(gyro_data_filtered.iloc[0, 0], str):
            gyro_times = pd.to_datetime(gyro_data_filtered.iloc[:, 0]).values
        else:
            gyro_times = gyro_data_filtered.iloc[:, 0].values

        # Check if we have enough data after trimming
        if len(acc_times) == 0 or len(gyro_times) == 0:
            logger.warning("No data left after trimming")
            return np.array([]), np.array([]), np.array([])

        # Create a common timeline for both sensors
        # Use 30Hz (typical for smartwatch/phone) or calculate from data
        n_samples = int((end_time_point - start_time_point).total_seconds() * 30)
        if n_samples < 2:
            logger.warning("Time range too short for alignment")
            return np.array([]), np.array([]), np.array([])

        # Create common timeline
        common_times = np.linspace(0, (end_time_point - start_time_point).total_seconds(), n_samples)
        
        # Create numeric versions of timestamps for interpolation
        acc_times_sec = np.array([(t - start_time_point).total_seconds() for t in acc_times])
        gyro_times_sec = np.array([(t - start_time_point).total_seconds() for t in gyro_times])

        # Initialize arrays for aligned data
        aligned_acc = np.zeros((n_samples, 3))
        aligned_gyro = np.zeros((n_samples, 3))

        # Interpolate each axis with hybrid interpolation
        for axis in range(3):
            aligned_acc[:, axis] = hybrid_interpolate(
                acc_times_sec, 
                acc_data_filtered.iloc[:, axis+1].values, 
                common_times
            )
            
            aligned_gyro[:, axis] = hybrid_interpolate(
                gyro_times_sec, 
                gyro_data_filtered.iloc[:, axis+1].values, 
                common_times
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Hybrid interpolation complete: {n_samples} aligned samples in {elapsed_time:.2f}s")

        return aligned_acc, aligned_gyro, common_times

    except Exception as e:
        logger.error(f"Sensor alignment failed: {str(e)}")
        logger.error(traceback.format_exc())
        return np.array([]), np.array([]), np.array([])

def cleanup_resources():
    """Clean up thread pool and other resources"""
    global thread_pool
    
    try:
        # Shutdown thread pool
        thread_pool.shutdown(wait=False)
        
        # Create new default thread pool
        thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        
        logger.info("Cleaned up IMU fusion resources")
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")

###########################################
# Base Orientation Filter
###########################################

class OrientationFilter:
    """
    Base class for orientation estimation algorithms
    
    This class provides common functionality for all orientation filters,
    including initialization from accelerometer data and robust handling
    of variable sampling rates.
    """
    
    def __init__(self, freq: float = 30.0):
        """
        Initialize orientation filter
        
        Args:
            freq: Expected sampling frequency in Hz
        """
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.last_gyro = np.zeros(3)
        self.initialized = False
    
    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Update orientation with new sensor readings
        
        Args:
            acc: Accelerometer reading [x, y, z] in m/s²
            gyro: Gyroscope reading [x, y, z] in rad/s
            timestamp: Optional timestamp for variable sampling rate
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        # Initialize orientation if not already done
        if not self.initialized and np.linalg.norm(acc) > 0.1:
            self._initialize_from_accel(acc)
            self.initialized = True
        
        # Calculate time delta
        dt = 1.0 / self.freq
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            self.last_time = timestamp
        elif timestamp is not None:
            self.last_time = timestamp
            
        # Ensure dt is positive and not too large
        dt = max(0.001, min(dt, 0.1))
        
        # Check for very similar consecutive readings
        # This helps with oversampled or duplicated data
        if np.allclose(gyro, self.last_gyro, atol=1e-7) and dt < 0.01:
            # Skip update for extremely small changes
            return self.orientation_q
            
        # Store gyro for next comparison
        self.last_gyro = np.copy(gyro)
        
        # Call the actual implementation
        return self._update_impl(acc, gyro, dt)
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Implementation to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _initialize_from_accel(self, acc: np.ndarray) -> None:
        """Initialize orientation from accelerometer reading"""
        # Normalize acceleration
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            return
            
        acc_normalized = acc / acc_norm
        
        # Find rotation from [0,0,1] to the normalized acceleration vector
        z_axis = np.array([0, 0, 1])
        
        # Get rotation axis via cross product
        cross = np.cross(z_axis, acc_normalized)
        cross_norm = np.linalg.norm(cross)
        
        if cross_norm < 1e-10:
            # Vectors are parallel, no rotation needed
            if acc_normalized[2] > 0:
                # Pointing up, identity quaternion
                self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                # Pointing down, 180° rotation around x-axis
                self.orientation_q = np.array([0.0, 1.0, 0.0, 0.0])
            return
        
        # Normalize the rotation axis
        axis = cross / cross_norm
        
        # Get rotation angle
        angle = np.arccos(np.dot(z_axis, acc_normalized))
        
        # Convert axis-angle to quaternion
        self.orientation_q = np.array([
            np.cos(angle/2),
            axis[0] * np.sin(angle/2),
            axis[1] * np.sin(angle/2),
            axis[2] * np.sin(angle/2)
        ])
        
        # Normalize quaternion
        self.orientation_q = self.orientation_q / np.linalg.norm(self.orientation_q)
    
    def reset(self) -> None:
        """Reset the filter to initial state"""
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        self.last_gyro = np.zeros(3)
        self.initialized = False

###########################################
# Complementary Filter
###########################################

class ComplementaryFilter(OrientationFilter):
    """
    Complementary filter for orientation estimation
    
    A lightweight filter that combines accelerometer and gyroscope data
    with a simple weighting factor. Good for resource-constrained devices.
    """
    
    def __init__(self, freq: float = 30.0, alpha: float = 0.98):
        """
        Initialize Complementary filter
        
        Args:
            freq: Expected sampling frequency in Hz
            alpha: Weight for gyroscope integration (0-1)
        """
        super().__init__(freq)
        self.alpha = alpha
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Update orientation using complementary filter
        
        Args:
            acc: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        q = self.orientation_q
        
        # Get accelerometer-based orientation (roll/pitch only)
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            # Normalize accelerometer
            acc_norm = acc / acc_norm
            
            # Calculate roll and pitch from accelerometer
            roll = np.arctan2(acc_norm[1], acc_norm[2])
            pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2))
            
            # Convert to quaternion (assuming yaw=0)
            acc_q = Rotation.from_euler('xyz', [roll, pitch, 0]).as_quat()
            # Convert from scipy [x,y,z,w] to our [w,x,y,z] format
            acc_q = np.array([acc_q[3], acc_q[0], acc_q[1], acc_q[2]])
        else:
            # Use current quaternion if accelerometer signal is too weak
            acc_q = q
        
        # Integrate gyroscope
        q0, q1, q2, q3 = q
        qDot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])
        
        # Integrate to get gyro-based quaternion
        q_gyro = q + qDot * dt
        q_gyro = q_gyro / np.linalg.norm(q_gyro)
        
        # Complementary filter fusion
        result_q = self.alpha * q_gyro + (1.0 - self.alpha) * acc_q
        result_q = result_q / np.linalg.norm(result_q)
        
        self.orientation_q = result_q
        return result_q

###########################################
# Madgwick Filter
###########################################

class MadgwickFilter(OrientationFilter):
    """
    Madgwick filter for orientation estimation
    
    A popular filter that uses gradient descent optimization to fuse
    accelerometer and gyroscope data. Good balance of accuracy and performance.
    """
    
    def __init__(self, freq: float = 30.0, beta: float = 0.1):
        """
        Initialize Madgwick filter
        
        Args:
            freq: Expected sampling frequency in Hz
            beta: Filter gain parameter (higher values converge faster)
        """
        super().__init__(freq)
        self.beta = beta
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Update orientation using Madgwick algorithm
        
        Args:
            acc: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        q = self.orientation_q
        
        # Normalize accelerometer measurement
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            # Skip update if acceleration is too small
            return q
        
        acc_norm = acc / acc_norm
        
        # Reference direction of Earth's gravity
        q0, q1, q2, q3 = q
        
        # Gradient descent algorithm corrective step
        f = np.array([
            2.0 * (q1*q3 - q0*q2) - acc_norm[0],
            2.0 * (q0*q1 + q2*q3) - acc_norm[1],
            2.0 * (0.5 - q1*q1 - q2*q2) - acc_norm[2]
        ])
        
        J = np.array([
            [-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
            [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
            [0.0, -4.0*q1, -4.0*q2, 0.0]
        ])
        
        # Compute gradient
        grad = J.T @ f
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
        
        # Gyroscope rate in quaternion form
        qDot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])
        
        # Apply feedback step
        qDot = qDot - self.beta * grad
        
        # Integrate to get new quaternion
        q = q + qDot * dt
        
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        
        self.orientation_q = q
        return q

###########################################
# Standard Kalman Filter
###########################################

class KalmanFilter(OrientationFilter):
    """
    Standard Kalman filter for orientation estimation
    
    Provides optimal estimation for linear systems with Gaussian noise.
    Adapted for orientation tracking with linearization.
    """
    
    def __init__(self, freq: float = 30.0):
        """
        Initialize Kalman filter
        
        Args:
            freq: Expected sampling frequency in Hz
        """
        super().__init__(freq)
        
        # State vector: quaternion (4), gyro_bias (3)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1
        
        # Process noise covariance - tuned for wearable sensors
        self.Q = np.diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])
        
        # Measurement noise covariance for accelerometer
        self.R = np.eye(3) * 0.1
        
        # Error covariance matrix
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Update orientation using Kalman filter
        
        Args:
            acc: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
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
        
        # State transition matrix
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
        if acc_norm > 0.5 and acc_norm < 3.0:  # Only update when near gravity
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
            try:
                K = P_pred @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                K = P_pred @ H.T @ np.linalg.pinv(S)
            
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
        
        return self.orientation_q
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _omega_matrix(self, gyro: np.ndarray) -> np.ndarray:
        """Create omega matrix for quaternion differentiation"""
        wx, wy, wz = gyro
        return np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        
        # Direct computation of rotation matrix elements
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        yy = y * y
        yz = y * z
        yw = y * w
        zz = z * z
        zw = z * w
        
        return np.array([
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
        ])
    
    def _compute_H_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute linearized measurement matrix"""
        w, x, y, z = q
        
        # Jacobian of gravity vector with respect to quaternion
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

###########################################
# Extended Kalman Filter
###########################################

class ExtendedKalmanFilter(OrientationFilter):
    """
    Extended Kalman Filter for more accurate orientation estimation
    
    Handles non-linear systems through linearization, providing better
    accuracy for complex motions than the standard Kalman filter.
    """
    
    def __init__(self, freq: float = 30.0):
        """
        Initialize Extended Kalman filter
        
        Args:
            freq: Expected sampling frequency in Hz
        """
        super().__init__(freq)
        
        # State vector: quaternion (4), gyro_bias (3)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1
        
        # Process noise covariance - tuned for wearable sensors
        self.Q = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5])
        
        # Measurement noise covariance for accelerometer - adaptive
        self.R_base = np.eye(3) * 0.05  # Base noise level
        self.R = self.R_base.copy()
        
        # Error covariance matrix - initialized with moderate uncertainty
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])
        
        # Reference vectors
        self.g_ref = np.array([0, 0, 1])  # Gravity reference (normalized)
        
        # For adaptive noise calculation
        self.acc_history = []
        self.max_history = 10
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Update orientation using EKF
        
        Args:
            acc: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        try:
            # Extract current state
            q = self.x[:4]  # Quaternion
            bias = self.x[4:]  # Gyro bias
            
            # Normalize quaternion
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            
            # Correct gyro with estimated bias
            gyro_corrected = gyro - bias
            
            # Adaptive measurement noise based on acceleration magnitude
            acc_norm = np.linalg.norm(acc)
            self.acc_history.append(acc_norm)
            if len(self.acc_history) > self.max_history:
                self.acc_history.pop(0)
                
            # Calculate variance of recent accelerometer magnitudes
            if len(self.acc_history) >= 3:
                acc_var = np.var(self.acc_history)
                # Increase measurement noise during high dynamics
                dynamic_factor = 1.0 + 10.0 * min(acc_var, 1.0)
                self.R = self.R_base * dynamic_factor
            
            # Prediction step with optimized quaternion integration
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
            if acc_norm > 1e-10:
                # Skip update if acceleration is too small or too large
                if 0.5 < acc_norm < 3.0:  # Near gravity range
                    # Normalize accelerometer
                    acc_normalized = acc / acc_norm
                    
                    # Expected gravity direction from orientation
                    R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
                    g_pred = R_q @ self.g_ref
                    
                    # Innovation
                    y = acc_normalized - g_pred
                    
                    # Measurement Jacobian
                    H = self._measurement_jacobian(x_pred[:4])
                    
                    # Innovation covariance
                    S = H @ P_pred @ H.T + self.R
                    
                    # Kalman gain with robust inversion
                    try:
                        K = P_pred @ H.T @ np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        # Fallback if matrix is singular
                        K = P_pred @ H.T @ np.linalg.pinv(S)
                    
                    # Update state
                    self.x = x_pred + K @ y
                    
                    # Update covariance with Joseph form for numerical stability
                    I_KH = np.eye(self.state_dim) - K @ H
                    self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
                else:
                    # High dynamics - just use prediction
                    self.x = x_pred
                    self.P = P_pred
            else:
                # No measurement update
                self.x = x_pred
                self.P = P_pred
            
            # Ensure quaternion is normalized
            self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
            
            # Set orientation quaternion
            self.orientation_q = self.x[:4]
            
            return self.orientation_q
            
        except Exception as e:
            logger.error(f"EKF update error: {e}")
            logger.error(traceback.format_exc())
            return self.orientation_q
    
    # Optimized helper functions
    def _quaternion_product_matrix(self, q: np.ndarray) -> np.ndarray:
        """Create matrix for quaternion multiplication: p ⊗ q = [q]_L * p"""
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
        ])
    
    def _quaternion_update_jacobian(self, q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Jacobian of quaternion update with respect to quaternion"""
        wx, wy, wz = gyro
        omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return np.eye(4) + 0.5 * dt * omega
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix with optimized implementation"""
        w, x, y, z = q
        
        # Direct computation of rotation matrix elements
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        yy = y * y
        yz = y * z
        yw = y * w
        zz = z * z
        zw = z * w
        
        return np.array([
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
        ])
    
    def _measurement_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute measurement Jacobian for EKF"""
        w, x, y, z = q
        
        # Jacobian of gravity direction with respect to quaternion
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
        H[:3, :4] = H_q
        
        return H

###########################################
# Unscented Kalman Filter
###########################################

class UnscentedKalmanFilter(OrientationFilter):
    """
    Unscented Kalman Filter (UKF) for highly accurate orientation estimation
    
    Handles non-linearities without derivatives through the unscented transform,
    providing the most accurate estimation for complex motions.
    """
    
    def __init__(self, freq: float = 30.0, alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0):
        """
        Initialize UKF
        
        Args:
            freq: Expected sampling frequency in Hz
            alpha: Primary scaling parameter (0 < alpha ≤ 1)
            beta: Secondary scaling parameter (typically 2 for Gaussian priors)
            kappa: Tertiary scaling parameter (typically 0 or 3-n)
        """
        super().__init__(freq)
        
        # State vector: quaternion (4), gyro_bias (3)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1
        
        # Process noise covariance - tuned for wearable sensors
        self.Q = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 5e-5, 5e-5, 5e-5])
        
        # Measurement noise covariance
        self.R = np.eye(3) * 0.1  # Accelerometer noise
        
        # Error covariance matrix - initialized with moderate uncertainty
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])
        
        # UKF parameters
        self.alpha = alpha  # Primary scaling parameter
        self.beta = beta    # Secondary scaling parameter
        self.kappa = kappa  # Tertiary scaling parameter
        
        # Derived parameters
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        
        # Calculate weights
        self._calculate_weights()
        
        # Reference gravity vector
        self.g_ref = np.array([0, 0, 1])  # Normalized gravity
        
    def _calculate_weights(self):
        """Calculate UKF weights for mean and covariance computation"""
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
        """Generate sigma points for UKF"""
        n = self.state_dim
        
        # Matrix square root of weighted covariance
        try:
            U = np.linalg.cholesky((n + self.lambda_) * self.P)
        except np.linalg.LinAlgError:
            # If cholesky decomposition fails, add small positive value to diagonal
            self.P += np.eye(n) * 1e-6
            U = np.linalg.cholesky((n + self.lambda_) * self.P)
        
        # Sigma points around current estimate
        sigma_points = np.zeros((self.num_sigma_points, n))
        sigma_points[0] = self.x
        
        for i in range(n):
            sigma_points[i+1] = self.x + U[i]
            sigma_points[i+1+n] = self.x - U[i]
        
        return sigma_points
    
    def _quaternion_normalize(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion"""
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0])  # Default quaternion if normalization fails
    
    def _process_model(self, sigma_point: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Process model for state propagation"""
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
    
    def _measurement_model(self, sigma_point: np.ndarray) -> np.ndarray:
        """Measurement model for UKF"""
        # Extract quaternion
        q = sigma_point[:4]
        q = self._quaternion_normalize(q)
        
        # Rotation matrix from quaternion
        R = self._quaternion_to_rotation_matrix(q)
        
        # Predicted gravity direction
        g_pred = R @ self.g_ref
        
        return g_pred
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Update orientation using UKF
        
        Args:
            acc: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        try:
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
            
            # Skip measurement update if acceleration is too small or too large
            acc_norm = np.linalg.norm(acc)
            if acc_norm < 0.5 or acc_norm > 3.0:
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
            
            # Kalman gain with robust inversion
            try:
                K = Pxz @ np.linalg.inv(Pzz)
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                K = Pxz @ np.linalg.pinv(Pzz)
            
            # Update state
            innovation = acc_norm - z_mean
            self.x = x_pred + K @ innovation
            
            # Normalize quaternion
            self.x[:4] = self._quaternion_normalize(self.x[:4])
            
            # Update covariance
            self.P = P_pred - K @ Pzz @ K.T
            
            # Set orientation quaternion
            self.orientation_q = self.x[:4]
            
            return self.orientation_q
            
        except Exception as e:
            logger.error(f"UKF update error: {e}")
            logger.error(traceback.format_exc())
            return self.orientation_q
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        
        # Direct computation of rotation matrix elements
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        yy = y * y
        yz = y * z
        yw = y * w
        zz = z * z
        zw = z * w
        
        return np.array([
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
        ])
    
    def _quaternion_error(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Calculate quaternion error (minimal representation)"""
        # Compute quaternion difference
        q_diff = self._quaternion_multiply(q1, self._quaternion_inverse(q2))
        
        # Ensure positive scalar part
        if q_diff[0] < 0:
            q_diff = -q_diff
        
        return q_diff
    
    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Calculate quaternion inverse (conjugate for unit quaternions)"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

###########################################
# Feature Extraction
###########################################

def extract_features_from_window(window_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract features from a window of IMU data.

    Args:
        window_data: Dictionary containing quaternion, linear_acceleration, angular_velocity

    Returns:
        Feature vector of length 43
    """
    try:
        # Extract data
        quaternions = window_data.get('quaternion', np.array([]))
        acc_data = window_data.get('linear_acceleration', window_data.get('accelerometer', np.array([])))
        gyro_data = window_data.get('angular_velocity', window_data.get('gyroscope', np.array([])))

        # Initialize with zeros if any data is missing
        if len(quaternions) == 0 or len(acc_data) == 0 or len(gyro_data) == 0:
            logger.warning("Missing data in feature extraction, returning zeros")
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
            # Compute FFT for each axis
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

        return features

    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        logger.error(traceback.format_exc())
        return np.zeros(43)  # Return zeros in case of failure

###########################################
# Main Processing Functions
###########################################

def process_imu_data(acc_data: np.ndarray, gyro_data: np.ndarray,
                    timestamps: Optional[np.ndarray] = None,
                    filter_type: str = 'ekf',
                    return_features: bool = False) -> Dict[str, np.ndarray]:
    """
    Process IMU data to extract orientation and linear acceleration.
    
    Args:
        acc_data: Accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        timestamps: Optional timestamps for variable rate processing
        filter_type: Type of orientation filter ('madgwick', 'comp', 'kalman', 'ekf', 'ukf')
        return_features: Whether to compute derived features
        
    Returns:
        Dictionary with processed data including quaternion and linear_acceleration
    """
    start_time = time.time()
    logger.info(f"Processing IMU data: acc={acc_data.shape}, gyro={gyro_data.shape}, filter={filter_type}")
    
    # Input validation
    if acc_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        logger.error("Empty input data")
        return {
            'quaternion': np.zeros((0, 4)),
            'linear_acceleration': np.zeros((0, 3)),
            'fusion_features': np.zeros(43) if return_features else None
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
        # Create timestamps if not provided
        if timestamps is None:
            # Create evenly spaced timestamps at 30Hz
            timestamps = np.linspace(0, acc_data.shape[0] / 30.0, acc_data.shape[0])
        
        # Convert gyroscope data from degrees to radians if needed
        # Check if gyro data appears to be in degrees per second
        gyro_max = np.max(np.abs(gyro_data))
        if gyro_max > 20.0:  # Likely in degrees/s
            logger.warning(f"Gyro values appear to be in degrees/s (max={gyro_max:.2f}), converting to rad/s")
            gyro_data = gyro_data * np.pi / 180.0
        
        # Initialize orientation filter based on type
        if filter_type.lower() == 'madgwick':
            orientation_filter = MadgwickFilter()
        elif filter_type.lower() == 'comp':
            orientation_filter = ComplementaryFilter()
        elif filter_type.lower() == 'kalman':
            orientation_filter = KalmanFilter()
        elif filter_type.lower() == 'ekf':
            orientation_filter = ExtendedKalmanFilter()
        elif filter_type.lower() == 'ukf':
            orientation_filter = UnscentedKalmanFilter()
        else:
            logger.warning(f"Unknown filter type: {filter_type}, using Extended Kalman Filter")
            orientation_filter = ExtendedKalmanFilter()
            filter_type = 'ekf'
        
        # Process data
        quaternions = []
        linear_accelerations = []
        
        # Standard processing for all sequences
        for i in range(len(acc_data)):
            # Get sensor readings for this step
            acc = acc_data[i]
            gyro = gyro_data[i]
            timestamp = timestamps[i] if timestamps is not None else None
            
            # Reconstructing raw-equivalent accelerometer
            if i > 0:
                # Estimate gravity direction using current orientation
                q = orientation_filter.orientation_q
                # Convert quaternion to scipy rotation (this is a costly operation, but unavoidable)
                R = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
                gravity = R.apply([0, 0, 9.81])  # Gravity in body frame (assuming Z is down)
                
                # The accelerometer measures acceleration + gravity
                # acc_data is already assumed to be linear acceleration (gravity removed)
                # so we add gravity back to simulate raw accelerometer for orientation filter
                raw_equiv_acc = acc + gravity
            else:
                # For first sample, assume gravity is aligned with z-axis
                raw_equiv_acc = acc + np.array([0, 0, 9.81])
            
            # Update orientation using raw-equivalent accelerometer data
            q = orientation_filter.update(raw_equiv_acc, gyro, timestamp)
            quaternions.append(q)
            
            # The input acc_data is already assumed to be linear acceleration
            # So we store it directly
            linear_accelerations.append(acc)
        
        # Convert to numpy arrays
        quaternions = np.array(quaternions)
        linear_accelerations = np.array(linear_accelerations)
        
        # Create result dictionary with quaternions and linear acceleration
        results = {
            'quaternion': quaternions,
            'linear_acceleration': linear_accelerations
        }
        
        # Extract features if requested
        if return_features:
            features = extract_features_from_window({
                'quaternion': quaternions,
                'linear_acceleration': linear_accelerations,
                'angular_velocity': gyro_data
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

def create_timestamp_array(size: int, freq: float = 30.0, start_time: float = 0.0) -> np.ndarray:
    """
    Create evenly spaced timestamps
    
    Args:
        size: Number of timestamps to generate
        freq: Frequency in Hz
        start_time: Starting time value
        
    Returns:
        Array of evenly spaced timestamps
    """
    dt = 1.0 / freq
    return np.linspace(start_time, start_time + (size-1)*dt, size)

def save_aligned_sensor_data(subject_id: int, action_id: int, trial_id: int,
                            acc_data: np.ndarray, gyro_data: np.ndarray,
                            quaternions: Optional[np.ndarray] = None,
                            timestamps: Optional[np.ndarray] = None,
                            save_dir: str = "data/aligned") -> None:
    """
    Save aligned sensor data to disk for later analysis
    
    Args:
        subject_id: Subject ID
        action_id: Action ID
        trial_id: Trial ID
        acc_data: Accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        quaternions: Orientation quaternions [n_samples, 4]
        timestamps: Timestamps [n_samples]
        save_dir: Base directory for saving data
    """
    try:
        # Create output directories
        os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
        os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
        
        if quaternions is not None:
            os.makedirs(f"{save_dir}/quaternion", exist_ok=True)
        
        if timestamps is not None:
            os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
        
        # Create filename
        filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
        
        # Save files
        np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
        np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
        
        if quaternions is not None:
            np.save(f"{save_dir}/quaternion/{filename}.npy", quaternions)
        
        if timestamps is not None:
            np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)
        
        logger.info(f"Saved aligned data for {filename}")
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")

def compare_filter_performance(acc_data: np.ndarray, gyro_data: np.ndarray,
                              timestamps: Optional[np.ndarray] = None,
                              filter_types: List[str] = None) -> Dict[str, Dict[str, Union[np.ndarray, float]]]:
    """
    Compare performance of different filter types
    
    Args:
        acc_data: Accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        timestamps: Optional timestamps [n_samples]
        filter_types: List of filter types to compare
        
    Returns:
        Dictionary with results for each filter type
    """
    if filter_types is None:
        filter_types = ['madgwick', 'comp', 'kalman', 'ekf', 'ukf']
    
    results = {}
    
    for filter_type in filter_types:
        start_time = time.time()
        
        # Process data with this filter
        filter_results = process_imu_data(
            acc_data=acc_data,
            gyro_data=gyro_data,
            timestamps=timestamps,
            filter_type=filter_type,
            return_features=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Store results
        results[filter_type] = {
            'quaternion': filter_results['quaternion'],
            'linear_acceleration': filter_results['linear_acceleration'],
            'fusion_features': filter_results.get('fusion_features', None),
            'processing_time': elapsed_time,
            'processing_rate': len(acc_data) / elapsed_time  # samples/sec
        }
    
    return results

# Expose key functions and classes
__all__ = [
    'OrientationFilter',
    'MadgwickFilter',
    'ComplementaryFilter',
    'KalmanFilter',
    'ExtendedKalmanFilter',
    'UnscentedKalmanFilter',
    'process_imu_data',
    'extract_features_from_window',
    'compare_filter_performance',
    'save_aligned_sensor_data',
    'update_thread_configuration',
    'cleanup_resources'
]
