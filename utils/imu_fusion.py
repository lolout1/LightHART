'''
IMU Fusion Module for Enhanced Orientation Estimation

This module provides multiple sensor fusion algorithms for converting raw 
accelerometer and gyroscope data into orientation estimates (quaternions) and
derived features. It implements Madgwick, Complementary, Kalman, Extended Kalman,
and Unscented Kalman filters with multithreading support.
'''

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
import pandas as pd
import time
import traceback
import logging
import os
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading
from typing import Dict, Tuple, List, Union, Optional, Any

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

# Initialize global resources
MAX_THREADS = 4
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)

# GPU Configuration
def setup_gpu_environment():
    """Configure environment for GPU usage"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 1:
            logger.info(f"Found {num_gpus} GPU{'s' if num_gpus > 1 else ''}. Using GPU 0 for processing")
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
    Update the thread configuration for parallel processing.
    
    Args:
        max_files: Maximum number of files to process in parallel
        threads_per_file: Maximum number of threads to use per file
    """
    global MAX_FILES, MAX_THREADS_PER_FILE, MAX_THREADS, file_semaphore
    
    # Update configuration
    MAX_FILES = max(1, max_files)
    MAX_THREADS_PER_FILE = max(1, threads_per_file)
    MAX_THREADS = MAX_FILES * MAX_THREADS_PER_FILE
    
    # Create new semaphore with updated limit
    file_semaphore = threading.Semaphore(MAX_FILES)
    
    logger.info(f"Updated thread configuration: {MAX_FILES} files × {MAX_THREADS_PER_FILE} threads = {MAX_THREADS} total threads")

def cleanup_resources() -> None:
    """
    Clean up resources used by the module.
    This should be called when the application exits.
    """
    global thread_pool
    
    try:
        # Shutdown thread pool
        thread_pool.shutdown(wait=False)
        
        # Create a new thread pool with default configuration
        thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        
        logger.info("Cleaned up IMU fusion resources")
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")

###########################################
# Data Processing Utility Functions
###########################################
# Base class for all orientation filters
class OrientationFilter:
    """Base class for orientation estimation algorithms"""
    
    def __init__(self, freq: float = 30.0):
        """
        Initialize orientation filter
        
        Args:
            freq: Default sampling frequency in Hz
        """
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    
    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Update orientation with new sensor readings
        
        Args:
            acc: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z]
            timestamp: Optional timestamp for variable sampling rate
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        # Calculate time delta
        dt = 1.0 / self.freq
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            self.last_time = timestamp
        elif timestamp is not None:
            self.last_time = timestamp
            
        # Ensure dt is positive and not too large
        dt = max(0.001, min(dt, 0.1))
        
        # Call the actual implementation
        return self._update_impl(acc, gyro, dt)
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Implementation to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def reset(self) -> None:
        """Reset the filter to initial state"""
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
class KalmanFilter(OrientationFilter):
    """Basic Kalman filter for orientation estimation"""
    
    def __init__(self, freq: float = 30.0):
        """
        Initialize Kalman filter
        
        Args:
            freq: Default sampling frequency in Hz
        """
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
    
    # Helper functions
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
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
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
class ExtendedKalmanFilter(OrientationFilter):
    """Extended Kalman Filter for more accurate orientation estimation"""
    
    def __init__(self, freq: float = 30.0):
        """
        Initialize Extended Kalman filter
        
        Args:
            freq: Default sampling frequency in Hz
        """
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
    
    # Helper functions (similar to KalmanFilter with some additions)
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
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
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
class UnscentedKalmanFilter(OrientationFilter):
    """Unscented Kalman Filter (UKF) for highly accurate orientation estimation"""
    
    def __init__(self, freq: float = 30.0, alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0):
        """
        Initialize UKF
        
        Args:
            freq: Default sampling frequency in Hz
            alpha: Primary scaling parameter
            beta: Secondary scaling parameter
            kappa: Tertiary scaling parameter
        """
        super().__init__(freq)
        
        # State vector: quaternion (4), gyro_bias (3)
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
    
    # Helper functions
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
    
    def _quaternion_normalize(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion"""
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0])  # Default quaternion if normalization fails
    
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
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
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

###########################################
# Orientation Filter Base Class
###########################################

class OrientationFilter:
    """Base class for orientation estimation algorithms"""
    
    def __init__(self, freq: float = 30.0):
        """
        Initialize orientation filter
        
        Args:
            freq: Default sampling frequency in Hz
        """
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    
    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Update orientation with new sensor readings
        
        Args:
            acc: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z]
            timestamp: Optional timestamp for variable sampling rate
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        # Calculate time delta
        dt = 1.0 / self.freq
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            self.last_time = timestamp
        elif timestamp is not None:
            self.last_time = timestamp
            
        # Ensure dt is positive and not too large
        dt = max(0.001, min(dt, 0.1))
        
        # Call the actual implementation
        return self._update_impl(acc, gyro, dt)
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Implementation to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def reset(self) -> None:
        """Reset the filter to initial state"""
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None

###########################################
# Madgwick Filter Implementation
###########################################

class MadgwickFilter(OrientationFilter):
    """Madgwick filter for orientation estimation"""
    
    def __init__(self, freq: float = 30.0, beta: float = 0.1):
        """
        Initialize Madgwick filter
        
        Args:
            freq: Default sampling frequency in Hz
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
# Complementary Filter Implementation
###########################################

class ComplementaryFilter(OrientationFilter):
    """Complementary filter for orientation estimation"""
    
    def __init__(self, freq: float = 30.0, alpha: float = 0.98):
        """
        Initialize Complementary filter
        
        Args:
            freq: Default sampling frequency in Hz
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
        
        # Get accelerometer orientation (roll/pitch only)
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            # Normalize accelerometer
            acc_norm = acc / acc_norm
            
            # Calculate roll and pitch from accelerometer
            roll = np.arctan2(acc_norm[1], acc_norm[2])
            pitch = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2))
            
            # Convert to quaternion (assume yaw=0)
            acc_q = Rotation.from_euler('xyz', [roll, pitch, 0]).as_quat()
            # Convert from [x,y,z,w] to [w,x,y,z]
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
# Kalman Filter Implementation
###########################################

class KalmanFilter(OrientationFilter):
    """Basic Kalman filter for orientation estimation"""
    
    def __init__(self, freq: float = 30.0):
        """
        Initialize Kalman filter
        
        Args:
            freq: Default sampling frequency in Hz
        """
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
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
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
# Extended Kalman Filter Implementation
###########################################

class ExtendedKalmanFilter(OrientationFilter):
    """Extended Kalman Filter for more accurate orientation estimation"""
    
    def __init__(self, freq: float = 30.0):
        """
        Initialize Extended Kalman filter
        
        Args:
            freq: Default sampling frequency in Hz
        """
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
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
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
# Unscented Kalman Filter Implementation
###########################################

class UnscentedKalmanFilter(OrientationFilter):
    """Unscented Kalman Filter (UKF) for highly accurate orientation estimation"""
    
    def __init__(self, freq: float = 30.0, alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0):
        """
        Initialize UKF
        
        Args:
            freq: Default sampling frequency in Hz
            alpha: Primary scaling parameter
            beta: Secondary scaling parameter
            kappa: Tertiary scaling parameter
        """
        super().__init__(freq)
        
        # State vector: quaternion (4), gyro_bias (3)
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
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _quaternion_error(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Calculate quaternion error (minimal representation)"""
        # Compute quaternion difference
        q_diff = self._quaternion_multiply(q1, self._quaternion_inverse(q2))
        
        # Ensure positive scalar part
        if q_diff[0] < 0:
            q_diff = -q_diff
        
        # For small angles, the vector part is approximately proportional to the rotation angle
        if abs(q_diff[0]) > 0.9999:  # Near identity, avoid numerical issues
            return np.zeros(4)
        
        return q_diff
    
    def _quaternion_inverse(self, q: np.ndarray) -> np.ndarray:
        """Calculate quaternion inverse (conjugate for unit quaternions)"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

###########################################
# Feature Extraction Functions
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
        acc_data = window_data.get('linear_acceleration', np.array([]))
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
def process_imu_data(acc_data: np.ndarray, gyro_data: np.ndarray,
                    timestamps: Optional[np.ndarray] = None,
                    filter_type: str = 'madgwick',
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
            timestamps = create_timestamp_array(len(acc_data))
        
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
            logger.error(f"Unknown filter type: {filter_type}, falling back to Madgwick")
            orientation_filter = MadgwickFilter()
        
        # Process data
        quaternions = []
        
        # Process with progress updates for long sequences
        if len(acc_data) > 1000:
            # Create progress bar for long sequences
            for i in tqdm(range(len(acc_data)), desc=f"Processing orientation with {filter_type}"):
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
                
                # Get sensor readings
                gyro = gyro_data[i]
                timestamp = timestamps[i] if timestamps is not None else None
                
                # Update orientation using raw-equivalent accelerometer data
                q = orientation_filter.update(raw_equiv_acc, gyro, timestamp)
                quaternions.append(q)
        else:
            # Standard processing for smaller sequences
            for i in range(len(acc_data)):
                # Process similarly to above but without progress bar
                if i > 0:
                    q = orientation_filter.orientation_q
                    R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                    gravity = R.apply([0, 0, 9.81])
                    raw_equiv_acc = acc_data[i] + gravity
                else:
                    raw_equiv_acc = acc_data[i] + np.array([0, 0, 9.81])
                
                gyro = gyro_data[i]
                timestamp = timestamps[i] if timestamps is not None else None
                q = orientation_filter.update(raw_equiv_acc, gyro, timestamp)
                quaternions.append(q)
        
        # Convert to numpy arrays
        quaternions = np.array(quaternions)
        
        # Create result dictionary with quaternions and linear acceleration
        results = {
            'quaternion': quaternions,
            'linear_acceleration': acc_data  # Pass through the input linear acceleration
        }
        
        # Extract features if requested
        if return_features:
            features = extract_features_from_window({
                'quaternion': quaternions,
                'linear_acceleration': acc_data,
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
def save_aligned_sensor_data(acc_data: np.ndarray, gyro_data: np.ndarray, quaternions: np.ndarray, 
                            subject_id: int, action_id: int, trial_id: int, 
                            save_dir: str = "data/aligned") -> None:
    """
    Save aligned sensor data to disk for later analysis
    
    Args:
        acc_data: Accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        quaternions: Orientation quaternions [n_samples, 4]
        subject_id: Subject ID
        action_id: Action ID
        trial_id: Trial ID
        save_dir: Base directory for saving data
    """
    try:
        # Create output directories
        os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
        os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
        os.makedirs(f"{save_dir}/quaternion", exist_ok=True)
        
        # Create filename
        filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
        
        # Save files
        np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
        np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
        np.save(f"{save_dir}/quaternion/{filename}.npy", quaternions)
        
        logger.info(f"Saved aligned data for {filename}")
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")

def save_model(model, path, name="model"):
    """
    Save model with support for both .pt and .pth extensions
    
    Args:
        model: PyTorch model to save
        path: Directory to save the model
        name: Base name for the model file
    """
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)
    
    # Save with .pt extension (full model)
    pt_path = os.path.join(path, f"{name}.pt")
    torch.save(model, pt_path)
    logger.info(f"Saved full model to {pt_path}")
    
    # Also save weights only with .pth extension
    pth_path = os.path.join(path, f"{name}.pth")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), pth_path)
    else:
        torch.save(model.state_dict(), pth_path)
    logger.info(f"Saved model weights to {pth_path}")

def load_model(model_class, path, name="model", device="cuda:0"):
    """
    Load model with fallback between .pt and .pth extensions
    
    Args:
        model_class: Model class to instantiate if loading .pth
        path: Directory containing the model
        name: Base name of the model file
        device: Device to load the model to
        
    Returns:
        Loaded PyTorch model
    """
    # Try loading .pt file (full model)
    pt_path = os.path.join(path, f"{name}.pt")
    if os.path.exists(pt_path):
        logger.info(f"Loading full model from {pt_path}")
        return torch.load(pt_path, map_location=device)
    
    # If .pt doesn't exist, try loading .pth (weights only)
    pth_path = os.path.join(path, f"{name}.pth")
    if os.path.exists(pth_path):
        logger.info(f"Loading model weights from {pth_path}")
        # Create a new instance of the model
        model = model_class()
        model.load_state_dict(torch.load(pth_path, map_location=device))
        return model
    
    # If neither exists, raise an error
    raise FileNotFoundError(f"Model file not found at {pt_path} or {pth_path}")
