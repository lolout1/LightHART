"""
IMU Fusion Module

This module implements various sensor fusion algorithms for IMU data processing,
specifically focusing on accelerometer and gyroscope data to estimate orientation
and linear acceleration. The implemented filters include:

1. Madgwick Filter - Gradient descent-based orientation filter
2. Extended Kalman Filter (EKF) - Nonlinear state estimation for orientation
3. Unscented Kalman Filter (UKF) - Sigma point-based nonlinear estimation
4. Complementary Filter - Simple frequency-domain fusion
5. Kalman Filter - Basic linear Kalman filter for orientation

The module also provides utilities for feature extraction, data alignment,
and visualization of fusion results.
"""

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
import matplotlib.pyplot as plt

# Configure logging
os.makedirs("debug_logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("debug_logs", "imu_fusion.log"),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("imu_fusion")

# Add console handler for more immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Thread pool configuration
MAX_THREADS = 8
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(4)

def update_thread_configuration(max_files: int, threads_per_file: int):
    """
    Update the thread pool configuration for parallel processing.
    
    Args:
        max_files: Maximum number of files to process in parallel
        threads_per_file: Number of threads to dedicate to each file
    """
    global MAX_THREADS, thread_pool, file_semaphore
    new_total = max_files * threads_per_file
    if new_total != MAX_THREADS:
        thread_pool.shutdown(wait=True)
        MAX_THREADS = new_total
        thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        file_semaphore = threading.Semaphore(max_files)
        logger.info(f"Updated thread configuration: {max_files} files × {threads_per_file} threads = {MAX_THREADS} total")

def cleanup_resources():
    """Clean up thread pool resources properly to avoid hanging processes."""
    global thread_pool
    try:
        logger.info("Cleaning up thread pool resources")
        thread_pool.shutdown(wait=False)
        thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")

def align_sensor_data(acc_data: np.ndarray, gyro_data: np.ndarray,
                     time_tolerance: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align accelerometer and gyroscope data to a common time frame.
    
    This function handles both timestamp-based data and raw sensor data
    by creating a common time grid and interpolating sensor values.
    
    Args:
        acc_data: Accelerometer data array or DataFrame with timestamps
        gyro_data: Gyroscope data array or DataFrame with timestamps
        time_tolerance: Time tolerance for alignment (seconds)
        
    Returns:
        Tuple of (aligned_acc, aligned_gyro, common_times)
    """
    start_time = time.time()
    
    try:
        # Handle pandas DataFrame input
        if isinstance(acc_data, pd.DataFrame):
            if isinstance(acc_data.iloc[0, 0], str):
                acc_times = pd.to_datetime(acc_data.iloc[:, 0]).values
            else:
                acc_times = acc_data.iloc[:, 0].values
            acc_values = acc_data.iloc[:, 1:4].values
        else:
            # Handle numpy array input without timestamps
            acc_values = acc_data
            acc_times = np.linspace(0, len(acc_values) / 50.0, len(acc_values))
            
        if isinstance(gyro_data, pd.DataFrame):
            if isinstance(gyro_data.iloc[0, 0], str):
                gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).values
            else:
                gyro_times = gyro_data.iloc[:, 0].values
            gyro_values = gyro_data.iloc[:, 1:4].values
        else:
            # Handle numpy array input without timestamps
            gyro_values = gyro_data
            gyro_times = np.linspace(0, len(gyro_values) / 50.0, len(gyro_values))

        # Find common time range
        if isinstance(acc_times[0], np.datetime64):
            # Convert datetimes to seconds since epoch for easier math
            acc_times_sec = np.array([(t - acc_times[0]).total_seconds() for t in acc_times])
            gyro_times_sec = np.array([(t - acc_times[0]).total_seconds() for t in gyro_times])
            
            start_time_point = max(acc_times_sec[0], gyro_times_sec[0])
            end_time_point = min(acc_times_sec[-1], gyro_times_sec[-1])
        else:
            # Use raw time values
            start_time_point = max(acc_times[0], gyro_times[0])
            end_time_point = min(acc_times[-1], gyro_times[-1])
        
        # Check for valid overlap
        if start_time_point >= end_time_point:
            logger.warning("No time overlap between accelerometer and gyroscope data")
            return np.array([]), np.array([]), np.array([])
        
        # Create common time grid at 50Hz (standard for many IMUs)
        sample_rate = 50.0
        duration = end_time_point - start_time_point
        num_samples = int(duration * sample_rate)
        
        if num_samples < 5:  # Minimum viable data
            logger.warning(f"Overlap too short: {duration:.2f}s")
            return np.array([]), np.array([]), np.array([])
        
        common_times = np.linspace(start_time_point, end_time_point, num_samples)
        
        # Initialize arrays for interpolated data
        aligned_acc = np.zeros((num_samples, 3))
        aligned_gyro = np.zeros((num_samples, 3))
        
        # Convert timestamps if needed
        if isinstance(acc_times[0], np.datetime64):
            acc_times_sec = np.array([(t - acc_times[0]).total_seconds() for t in acc_times])
            gyro_times_sec = np.array([(t - acc_times[0]).total_seconds() for t in gyro_times])
        else:
            acc_times_sec = acc_times
            gyro_times_sec = gyro_times
        
        # Perform linear interpolation for each axis
        for axis in range(3):
            aligned_acc[:, axis] = np.interp(
                common_times, 
                acc_times_sec, 
                acc_values[:, axis]
            )
            
            aligned_gyro[:, axis] = np.interp(
                common_times, 
                gyro_times_sec, 
                gyro_values[:, axis]
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Sensor alignment complete: {num_samples} aligned samples in {elapsed_time:.2f}s")
        
        return aligned_acc, aligned_gyro, common_times
    
    except Exception as e:
        logger.error(f"Sensor alignment failed: {str(e)}")
        logger.error(traceback.format_exc())
        return np.array([]), np.array([]), np.array([])

class OrientationFilter:
    """
    Base class for orientation filters.
    
    This abstract class defines the common interface and functionality for
    all orientation filter implementations (Madgwick, Kalman, etc.).
    """
    def __init__(self, freq: float = 30.0):
        """
        Initialize orientation filter.
        
        Args:
            freq: Sample frequency in Hz (default: 30Hz for SmartFall data)
        """
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.last_gyro = np.zeros(3)
        self.initialized = False
        self.name = "Base OrientationFilter"
    
    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Update orientation estimate with new sensor readings.
        
        Args:
            acc: Accelerometer reading [ax, ay, az]
            gyro: Gyroscope reading [gx, gy, gz]
            timestamp: Optional timestamp for variable rate data
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        # Initialize from accelerometer if not already initialized
        if not self.initialized and np.linalg.norm(acc) > 0.1:
            self._initialize_from_accel(acc)
            self.initialized = True
        
        # Calculate time step
        dt = 1.0 / self.freq
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            self.last_time = timestamp
        elif timestamp is not None:
            self.last_time = timestamp
            
        # Clamp dt to reasonable range
        dt = max(0.001, min(dt, 0.1))
        
        # Skip trivial updates (no change in gyro)
        if np.allclose(gyro, self.last_gyro, atol=1e-7) and dt < 0.01:
            return self.orientation_q
            
        self.last_gyro = np.copy(gyro)
        
        # Call the implementation-specific update method
        return self._update_impl(acc, gyro, dt)
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Implementation-specific update method to be overridden by subclasses.
        
        Args:
            acc: Accelerometer reading [ax, ay, az]
            gyro: Gyroscope reading [gx, gy, gz]
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _initialize_from_accel(self, acc: np.ndarray) -> None:
        """
        Initialize orientation from accelerometer readings.
        
        This assumes the accelerometer is measuring gravity and sets the initial
        orientation to align with the gravity vector.
        
        Args:
            acc: Accelerometer reading [ax, ay, az]
        """
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            return
            
        acc_normalized = acc / acc_norm
        
        # Use global Z-axis as reference for gravity
        z_axis = np.array([0, 0, 1])
        
        # Calculate rotation axis via cross product
        cross = np.cross(z_axis, acc_normalized)
        cross_norm = np.linalg.norm(cross)
        
        if cross_norm < 1e-10:
            # Special case: acceleration aligned with Z axis
            if acc_normalized[2] > 0:
                # Device pointing up, identity quaternion
                self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                # Device pointing down, 180° rotation around X
                self.orientation_q = np.array([0.0, 1.0, 0.0, 0.0])
            return
        
        # Normalize rotation axis
        axis = cross / cross_norm
        
        # Calculate rotation angle
        angle = np.arccos(np.dot(z_axis, acc_normalized))
        
        # Create quaternion from axis-angle representation
        self.orientation_q = np.array([
            np.cos(angle/2),
            axis[0] * np.sin(angle/2),
            axis[1] * np.sin(angle/2),
            axis[2] * np.sin(angle/2)
        ])
        
        # Normalize quaternion
        self.orientation_q = self.orientation_q / np.linalg.norm(self.orientation_q)
    
    def reset(self) -> None:
        """Reset filter state to default values."""
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        self.last_gyro = np.zeros(3)
        self.initialized = False

class MadgwickFilter(OrientationFilter):
    """
    Madgwick orientation filter implementation.
    
    This filter uses gradient descent to correct gyroscope drift using
    accelerometer measurements of gravity. It's efficient and suitable
    for real-time processing on resource-constrained devices.
    """
    def __init__(self, freq: float = 30.0, beta: float = 0.1):
        """
        Initialize Madgwick filter.
        
        Args:
            freq: Sample frequency in Hz
            beta: Filter gain (controls gyro/accel balance)
        """
        super().__init__(freq)
        self.beta = beta
        self.name = "Madgwick"
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Update orientation using Madgwick algorithm.
        
        Args:
            acc: Accelerometer reading [ax, ay, az]
            gyro: Gyroscope reading [gx, gy, gz] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        q = self.orientation_q
        
        # Normalize accelerometer measurement
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            # Skip correction if accelerometer data is invalid
            return q
        
        acc_norm = acc / acc_norm
        
        # Current orientation quaternion components
        q0, q1, q2, q3 = q
        
        # Gradient descent algorithm corrective step
        # Objective function represents the difference between the measured gravity
        # vector and the predicted gravity vector based on current orientation
        f = np.array([
            2.0 * (q1*q3 - q0*q2) - acc_norm[0],
            2.0 * (q0*q1 + q2*q3) - acc_norm[1],
            2.0 * (0.5 - q1*q1 - q2*q2) - acc_norm[2]
        ])
        
        # Jacobian of the objective function
        J = np.array([
            [-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
            [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
            [0.0, -4.0*q1, -4.0*q2, 0.0]
        ])
        
        # Calculate gradient
        grad = J.T @ f
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
        
        # Gyroscope rate of change of quaternion
        qDot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])
        
        # Apply gradient descent correction
        qDot = qDot - self.beta * grad
        
        # Integrate to get new quaternion
        q = q + qDot * dt
        
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        
        # Update internal state
        self.orientation_q = q
        
        return q

class ComplementaryFilter(OrientationFilter):
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

class KalmanFilter(OrientationFilter):
    """
    Basic Kalman filter for orientation estimation.
    
    This filter uses a linear Kalman filter model for orientation tracking.
    It's more suited for systems without strong nonlinearities.
    """
    def __init__(self, freq: float = 30.0):
        """
        Initialize Kalman filter.
        
        Args:
            freq: Sample frequency in Hz
        """
        super().__init__(freq)
        
        # State vector is orientation quaternion and gyro bias
        self.state = np.zeros(7)  # [q0, q1, q2, q3, bias_x, bias_y, bias_z]
        self.state[0] = 1.0  # Initialize with identity quaternion
        
        # Process and measurement noise covariance
        self.P = np.eye(7) * 0.01  # State covariance
        self.Q = np.eye(7) * 0.001  # Process noise
        self.R = np.eye(3) * 0.1  # Measurement noise
        
        self.name = "Kalman"
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Update orientation using Kalman filter algorithm.
        
        Args:
            acc: Accelerometer reading [ax, ay, az]
            gyro: Gyroscope reading [gx, gy, gz] in rad/s
            dt: Time step in seconds
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        # Extract current state
        q = self.state[:4]
        bias = self.state[4:7]
        
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        
        # Correct gyro with bias
        gyro_corrected = gyro - bias
        
        # Prediction step
        # Update quaternion using gyroscope measurements
        q_dot = 0.5 * np.array([
            -q[1]*gyro_corrected[0] - q[2]*gyro_corrected[1] - q[3]*gyro_corrected[2],
            q[0]*gyro_corrected[0] + q[2]*gyro_corrected[2] - q[3]*gyro_corrected[1],
            q[0]*gyro_corrected[1] - q[1]*gyro_corrected[2] + q[3]*gyro_corrected[0],
            q[0]*gyro_corrected[2] + q[1]*gyro_corrected[1] - q[2]*gyro_corrected[0]
        ])
        
        # Predicted quaternion (gyro integration)
        q_pred = q + q_dot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)
        
        # Predicted state
        state_pred = np.zeros(7)
        state_pred[:4] = q_pred
        state_pred[4:7] = bias  # Bias prediction (assumed constant)
        
        # State transition matrix (approximation)
        F = np.eye(7)
        
        # Predict covariance
        P_pred = F @ self.P @ F.T + self.Q
        
        # Check if accelerometer data is valid
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 0.1 and abs(acc_norm - 9.81) < 3.0:
            # Normalize accelerometer measurement
            acc_normalized = acc / acc_norm
            
            # Expected gravity direction based on current orientation
            R_q = self._quaternion_to_rotation_matrix(q_pred)
            expected_gravity = R_q @ np.array([0, 0, 1])
            
            # Measurement residual
            residual = acc_normalized - expected_gravity
            
            # Measurement Jacobian matrix (linearized)
            H = np.zeros((3, 7))
            # Simplified Jacobian - sensitivity of gravity direction to quaternion
            H[:3, :4] = self._gravity_jacobian(q_pred)
            
            # Calculate Kalman gain
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.state = state_pred + K @ residual
            
            # Update covariance
            I_KH = np.eye(7) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            # No valid accelerometer data, use prediction only
            self.state = state_pred
            self.P = P_pred
        
        # Normalize quaternion part of state
        self.state[:4] = self.state[:4] / np.linalg.norm(self.state[:4])
        
        # Update orientation quaternion
        self.orientation_q = self.state[:4]
        
        return self.orientation_q
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        # Extract quaternion components
        w, x, y, z = q
        
        # Calculate common terms
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        yy = y * y
        yz = y * z
        yw = y * w
        zz = z * z
        zw = z * w
        
        # Calculate rotation matrix elements
        R = np.array([
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
        ])
        
        return R
    
    def _gravity_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate Jacobian of gravity direction with respect to quaternion.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x4 Jacobian matrix
        """
        # Extract quaternion components
        w, x, y, z = q
        
        # Jacobian of gravity direction with respect to quaternion
        J = np.zeros((3, 4))
        
        # These are partial derivatives of the expected gravity vector
        # with respect to quaternion components w, x, y, z
        J[0, 0] = -2*y
        J[0, 1] = 2*z
        J[0, 2] = -2*w
        J[0, 3] = 2*x
        
        J[1, 0] = 2*x
        J[1, 1] = 2*w
        J[1, 2] = 2*z
        J[1, 3] = 2*y
        
        J[2, 0] = 0
        J[2, 1] = -2*y
        J[2, 2] = -2*z
        J[2, 3] = 0
        
        return J

def extract_features_from_window(window_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract features from a window of sensor data for model inputs.
    
    This function calculates statistical features from accelerometer, gyroscope,
    and quaternion data that are useful for fall detection.
    
    Args:
        window_data: Dictionary containing sensor data arrays
        
    Returns:
        Feature vector (43 dimensions)
    """
    try:
        # Extract data from window
        quaternions = window_data.get('quaternion', np.array([]))
        acc_data = window_data.get('linear_acceleration', window_data.get('accelerometer', np.array([])))
        gyro_data = window_data.get('angular_velocity', window_data.get('gyroscope', np.array([])))

        # Check if we have sufficient data
        if len(quaternions) == 0 or len(acc_data) == 0 or len(gyro_data) == 0:
            logger.warning("Insufficient data for feature extraction")
            return np.zeros(43)  # Return zero feature vector

        # Basic statistical features from accelerometer
        acc_mean = np.mean(acc_data, axis=0)
        acc_std = np.std(acc_data, axis=0)
        acc_max = np.max(acc_data, axis=0)
        acc_min = np.min(acc_data, axis=0)

        # Magnitude features
        acc_mag = np.linalg.norm(acc_data, axis=1)
        acc_mag_mean = np.mean(acc_mag)
        acc_mag_std = np.std(acc_mag)
        acc_mag_max = np.max(acc_mag)

        # Gyroscope features
        gyro_mean = np.mean(gyro_data, axis=0)
        gyro_std = np.std(gyro_data, axis=0)
        gyro_max = np.max(np.abs(gyro_data), axis=0)

        # Jerk features (derivative of acceleration)
        jerk_features = []
        if len(acc_data) > 1:
            # Calculate jerk (acceleration derivative)
            jerk = np.diff(acc_data, axis=0, prepend=acc_data[0].reshape(1, -1))
            jerk_mag = np.linalg.norm(jerk, axis=1)
            jerk_mag_mean = np.mean(jerk_mag)
            jerk_mag_max = np.max(jerk_mag)
            jerk_features = [jerk_mag_mean, jerk_mag_max]
        else:
            jerk_features = [0, 0]

        # Convert quaternions to Euler angles
        euler_angles = []
        for q in quaternions:
            # Create scipy Rotation object (note: scipy uses [x,y,z,w] order)
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            euler_angles.append(r.as_euler('xyz', degrees=True))

        euler_angles = np.array(euler_angles)

        # Calculate Euler angle statistics
        euler_mean = np.mean(euler_angles, axis=0)
        euler_std = np.std(euler_angles, axis=0)

        # Angular rate features (derivative of orientation)
        angle_rate_features = []
        if len(euler_angles) > 1:
            angle_rates = np.diff(euler_angles, axis=0, prepend=euler_angles[0].reshape(1, -1))
            angle_rate_mag = np.linalg.norm(angle_rates, axis=1)
            angle_rate_mean = np.mean(angle_rate_mag)
            angle_rate_max = np.max(angle_rate_mag)
            angle_rate_features = [angle_rate_mean, angle_rate_max]
        else:
            angle_rate_features = [0, 0]

        # Frequency domain features
        fft_features = []
        if len(acc_data) >= 8:  # Need enough samples for meaningful FFT
            for axis in range(acc_data.shape[1]):
                # Calculate FFT magnitude
                fft = np.abs(np.fft.rfft(acc_data[:, axis]))
                if len(fft) > 3:
                    fft_features.extend([np.max(fft), np.mean(fft), np.var(fft)])
                else:
                    fft_features.extend([0, 0, 0])
        else:
            fft_features = [0] * 9  # 3 features per axis

        # Combine all features into a single vector
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
        return np.zeros(43)  # Return zero feature vector

def process_imu_data(acc_data: np.ndarray, gyro_data: np.ndarray,
                    timestamps: Optional[np.ndarray] = None,
                    filter_type: str = 'madgwick',
                    return_features: bool = False) -> Dict[str, np.ndarray]:
    """
    Process IMU data with Madgwick filter to estimate orientation and linear acceleration.
    
    Args:
        acc_data: Accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        timestamps: Optional timestamps for variable-rate data
        filter_type: Type of filter to use ('madgwick', 'comp', 'kalman', 'ekf', 'ukf')
        return_features: Whether to extract and return features
        
    Returns:
        Dictionary with processed data: quaternion, linear_acceleration, fusion_features (if requested)
    """
    start_time = time.time()
    
    # Validate inputs
    if not isinstance(acc_data, np.ndarray) or not isinstance(gyro_data, np.ndarray):
        logger.error(f"Invalid input types: acc={type(acc_data)}, gyro={type(gyro_data)}")
        # Return empty data with correct shapes
        return {
            'quaternion': np.zeros((max(1, len(acc_data)), 4)),
            'linear_acceleration': np.zeros((max(1, len(acc_data)), 3)) 
        }
    
    # Handle empty inputs
    if acc_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        logger.error("Empty input data")
        return {
            'quaternion': np.zeros((1, 4)),
            'linear_acceleration': np.zeros((1, 3))
        }
    
    # Ensure data arrays have the same length
    if acc_data.shape[0] != gyro_data.shape[0]:
        min_len = min(acc_data.shape[0], gyro_data.shape[0])
        logger.warning(f"Data length mismatch: acc={acc_data.shape[0]}, gyro={gyro_data.shape[0]}, truncating to {min_len}")
        acc_data = acc_data[:min_len]
        gyro_data = gyro_data[:min_len]
        if timestamps is not None and len(timestamps) > min_len:
            timestamps = timestamps[:min_len]
    
    try:
        # Create timestamps if not provided or invalid
        if timestamps is None or len(timestamps) != len(acc_data):
            timestamps = np.linspace(0, acc_data.shape[0] / 30.0, acc_data.shape[0])
            logger.info(f"Created synthetic timestamps at 30Hz for {len(acc_data)} samples")
        
        # Convert timestamp to 1D array if needed
        if hasattr(timestamps, 'shape') and len(timestamps.shape) > 1:
            if timestamps.shape[1] > 0:
                timestamps = timestamps[:, 0]
            else:
                timestamps = np.linspace(0, acc_data.shape[0] / 30.0, acc_data.shape[0])
        
        # Convert gyro data from degrees/s to radians/s if needed
        gyro_max = np.max(np.abs(gyro_data)) if len(gyro_data) > 0 else 0
        if gyro_max > 20.0:  # Heuristic: gyro values in deg/s are typically larger than rad/s
            logger.info(f"Converting gyroscope data from deg/s to rad/s (max value: {gyro_max})")
            gyro_data = gyro_data * np.pi / 180.0
        
        # Create the appropriate filter instance
        orientation_filter = MadgwickFilter(freq=30.0, beta=0.1)  # Always use Madgwick for now
        
        logger.info(f"Applying Madgwick filter to IMU data ({len(acc_data)} samples)")
        
        # Process data with the filter
        quaternions = []
        linear_accelerations = []
        
        # Process each time step
        for i in range(len(acc_data)):
            try:
                acc = acc_data[i]
                gyro = gyro_data[i]
                timestamp = timestamps[i] if i < len(timestamps) else None
                
                # Apply the filter to get orientation
                q = orientation_filter.update(acc, gyro, timestamp)
                quaternions.append(q)
                
                # Calculate linear acceleration by removing gravity
                if i > 0:  # Use current orientation to remove gravity
                    R = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # Convert to scipy format [x,y,z,w]
                    gravity = R.apply([0, 0, 9.81])  # Rotate standard gravity vector
                    lin_acc = acc - gravity  # Remove gravity component
                else:
                    # For the first sample, use a simple gravity removal
                    lin_acc = acc - np.array([0, 0, 9.81])
                
                linear_accelerations.append(lin_acc)
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                # Use previous values or zeros if not available
                prev_q = quaternions[-1] if quaternions else np.array([1.0, 0.0, 0.0, 0.0])
                prev_lin_acc = linear_accelerations[-1] if linear_accelerations else np.zeros(3)
                quaternions.append(prev_q)
                linear_accelerations.append(prev_lin_acc)
        
        # Convert lists to numpy arrays
        quaternions = np.array(quaternions)
        linear_accelerations = np.array(linear_accelerations)
        
        # Prepare results
        results = {
            'quaternion': quaternions,
            'linear_acceleration': linear_accelerations
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"IMU processing with Madgwick filter completed in {elapsed_time:.2f}s")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in IMU processing: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return minimal results on error - ensure consistent structure
        sample_size = max(1, len(acc_data) if isinstance(acc_data, np.ndarray) else 1)
        return {
            'quaternion': np.zeros((sample_size, 4)),
            'linear_acceleration': np.zeros((sample_size, 3)) 
        }
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
        logger.error(traceback.format_exc())
