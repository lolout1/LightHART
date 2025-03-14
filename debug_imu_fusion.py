#!/usr/bin/env python
"""
debug_imu_fusion.py - Comprehensive debugging tool for IMU filtering pipeline

This script tests each component of the IMU fusion pipeline:
1. Data loading
2. Filter implementation (Standard, EKF, UKF)
3. Modality alignment
4. Window creation
5. End-to-end performance

It includes detailed logging, visualization, and performance metrics.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import logging
import traceback
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
log_dir = "debug_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"debug_imu_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IMUFusionDebug")

# Performance tracking
timing_stats = {
    "data_loading": [],
    "standard_filter": [],
    "ekf_filter": [],
    "ukf_filter": [],
    "alignment": [],
    "windowing": []
}

# Accuracy metrics
accuracy_metrics = {
    "standard_filter": {"mse": [], "angular_error": []},
    "ekf_filter": {"mse": [], "angular_error": []},
    "ukf_filter": {"mse": [], "angular_error": []}
}

def time_function(category):
    """Decorator to measure execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = None
            error = None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                error = e
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
            
            elapsed = time.time() - start_time
            if category in timing_stats:
                timing_stats[category].append(elapsed)
            
            logger.debug(f"{func.__name__} took {elapsed:.4f} seconds")
            
            if error:
                raise error
            return result
        return wrapper
    return decorator

# =====================================================================
# Data loading functions
# =====================================================================

@time_function("data_loading")
def load_watch_data(file_path, verbose=False):
    """
    Load watch data (accelerometer or gyroscope) from CSV.
    
    Args:
        file_path: Path to CSV file
        verbose: Whether to print detailed info
        
    Returns:
        Array of shape (N, 4) with [time, x, y, z] or empty array if error
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return np.zeros((0, 4), dtype=np.float32)
            
        # Try to read the file with different parsers
        try:
            df = pd.read_csv(file_path, header=None)
        except Exception as e1:
            try:
                # Try with explicit separator
                df = pd.read_csv(file_path, header=None, sep=',')
            except Exception as e2:
                try:
                    # Try with Python engine which is more permissive
                    df = pd.read_csv(file_path, header=None, engine='python')
                except Exception as e3:
                    logger.error(f"Failed to parse {file_path}: {str(e1)}, {str(e2)}, {str(e3)}")
                    return np.zeros((0, 4), dtype=np.float32)
        
        # Clean data
        df = df.dropna(how='all').reset_index(drop=True)
        
        # Check if we have header row
        if isinstance(df.iloc[0, 0], str) and ('time' in df.iloc[0, 0].lower() or 'stamp' in df.iloc[0, 0].lower()):
            df = df.iloc[1:].reset_index(drop=True)
        
        # Check if we have enough columns
        if df.shape[1] < 4:
            logger.error(f"Insufficient columns in {file_path}: found {df.shape[1]}, need at least 4")
            return np.zeros((0, 4), dtype=np.float32)
        
        # Convert to numeric values
        for col in range(min(4, df.shape[1])):
            df.iloc[:, col] = pd.to_numeric(df.iloc[:, col], errors='coerce')
        
        # Convert timestamp
        timestamps = df.iloc[:, 0].values
        
        # If timestamps are very large (epoch milliseconds), convert to seconds
        if np.mean(timestamps) > 1e10:
            timestamps = timestamps / 1000.0
        
        # Make first timestamp zero
        timestamps = timestamps - timestamps[0]
        
        # Get sensor values
        sensor_values = df.iloc[:, 1:4].values.astype(np.float32)
        
        # Combine time and values
        data = np.column_stack([timestamps, sensor_values])
        
        if verbose:
            logger.info(f"Loaded {file_path}: {data.shape[0]} samples, time range: {data[0, 0]:.4f} to {data[-1, 0]:.4f} seconds")
            logger.info(f"Sample rate: {(data.shape[0]-1)/(data[-1, 0] - data[0, 0]):.2f} Hz (avg)")
            logger.info(f"Value ranges: x=[{np.min(data[:, 1]):.2f}, {np.max(data[:, 1]):.2f}], " +
                       f"y=[{np.min(data[:, 2]):.2f}, {np.max(data[:, 2]):.2f}], " +
                       f"z=[{np.min(data[:, 3]):.2f}, {np.max(data[:, 3]):.2f}]")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return np.zeros((0, 4), dtype=np.float32)

@time_function("data_loading")
def load_skeleton_data(file_path, verbose=False):
    """
    Load skeleton data from CSV.
    
    Args:
        file_path: Path to CSV file
        verbose: Whether to print detailed info
        
    Returns:
        Array of skeleton joint positions or empty array if error
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return np.zeros((0, 96), dtype=np.float32)
            
        # Try to read with various options
        try:
            df = pd.read_csv(file_path, header=None)
        except Exception as e1:
            try:
                df = pd.read_csv(file_path, header=None, sep=',')
            except Exception as e2:
                try:
                    df = pd.read_csv(file_path, header=None, engine='python')
                except Exception as e3:
                    logger.error(f"Failed to parse {file_path}: {str(e1)}, {str(e2)}, {str(e3)}")
                    return np.zeros((0, 96), dtype=np.float32)
        
        # Clean data
        df = df.dropna(how='all').reset_index(drop=True)
        df = df.fillna(0)  # Fill NaNs with zeros
        
        # Ensure all values are numeric
        for col in range(df.shape[1]):
            df.iloc[:, col] = pd.to_numeric(df.iloc[:, col], errors='coerce')
        
        # Convert to numpy array
        data = df.values.astype(np.float32)
        
        if verbose:
            logger.info(f"Loaded {file_path}: {data.shape[0]} frames, {data.shape[1]} features")
            # Calculate mean position for each joint to check if data makes sense
            if data.shape[1] >= 96:
                num_joints = data.shape[1] // 3
                joint_means = np.zeros((num_joints, 3))
                for j in range(num_joints):
                    joint_means[j] = np.mean(data[:, j*3:(j+1)*3], axis=0)
                
                logger.info(f"Joint means (first 5 joints): {joint_means[:5]}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return np.zeros((0, 96), dtype=np.float32)

# =====================================================================
# Kalman filter implementations
# =====================================================================

class BaseKalmanFilter:
    """Base class for Kalman filter implementations."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.drift_correction_weight = drift_correction_weight
        self.name = "Base Filter"
        self.initialized = False
        
        # Reference data for drift correction
        self.reference_timestamps = None
        self.reference_orientations = None
        self.use_reference = False
    
    def initialize(self, accel):
        """Initialize the filter with first acceleration."""
        self.initialized = True
    
    def reset(self):
        """Reset filter state."""
        self.initialized = False
    
    def set_reference_data(self, timestamps, orientations):
        """Set reference orientation data for drift correction."""
        if timestamps is None or orientations is None or len(timestamps) == 0:
            self.use_reference = False
            return
            
        self.reference_timestamps = timestamps
        self.reference_orientations = orientations
        self.use_reference = True
        logger.debug(f"Reference data set: {len(timestamps)} points")
    
    def get_reference_orientation(self, timestamp):
        """Get interpolated reference orientation at given timestamp."""
        if not self.use_reference:
            return None
            
        # Check if timestamp is within range
        if (timestamp < self.reference_timestamps[0] or 
            timestamp > self.reference_timestamps[-1]):
            return None
            
        # Interpolate reference orientation
        from scipy.interpolate import interp1d
        
        interp_func = interp1d(
            self.reference_timestamps,
            self.reference_orientations,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        return interp_func(timestamp)
    
    def apply_drift_correction(self, estimated_orientation, timestamp):
        """Apply drift correction using reference orientation."""
        if not self.use_reference:
            return estimated_orientation
            
        reference = self.get_reference_orientation(timestamp)
        if reference is None:
            return estimated_orientation
            
        # Apply weighted correction
        w = self.drift_correction_weight
        corrected = (1 - w) * estimated_orientation + w * reference
        return corrected
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process a single sensor data step."""
        # This should be implemented by subclasses
        pass
    
    def process_sequence(self, accel_data, gyro_data, timestamps=None):
        """Process entire sequence of sensor data."""
        N = accel_data.shape[0]
        
        # If timestamps not provided, generate uniform timestamps
        if timestamps is None:
            timestamps = np.arange(N) * self.dt
        
        # Create output array - 13 features: accel(3), gyro(3), quat(4), euler(3)
        output_features = 13
        output = np.zeros((N, output_features))
        
        # Initialize filter with first acceleration
        if not self.initialized:
            self.initialize(accel_data[0])
        
        # Process each measurement
        for i in range(N):
            accel = accel_data[i]
            gyro = gyro_data[i]
            
            # Compute dt if we have timestamps
            if i > 0:
                dt = timestamps[i] - timestamps[i-1]
                if dt <= 0:  # Handle timestamp errors
                    dt = self.dt
            else:
                dt = self.dt
            
            # Process step
            state, features = self.process_step(accel, gyro, dt, timestamps[i])
            output[i] = features
        
        return output

class StandardKalmanFilter(BaseKalmanFilter):
    """Standard Kalman Filter for IMU fusion."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, drift_correction_weight)
        self.name = "Standard KF"
        
        # State: [orientation (3), gyro_bias (3)]
        self.state_dim = 6
        self.measurement_dim = 3
        
        # Create Kalman filter matrices
        self.x = np.zeros(self.state_dim)  # State
        self.P = np.eye(self.state_dim) * 0.1  # Covariance
        self.F = np.eye(self.state_dim)  # State transition
        self.H = np.zeros((self.measurement_dim, self.state_dim))  # Measurement
        self.H[:3, :3] = np.eye(3)
        self.Q = np.eye(self.state_dim) * self.process_noise  # Process noise
        self.R = np.eye(self.measurement_dim) * self.measurement_noise  # Measurement noise
    
    def initialize(self, accel):
        """Initialize filter using initial acceleration."""
        # Estimate initial orientation from gravity direction
        gravity = -accel / np.linalg.norm(accel)
        
        # Convert to Euler angles (roll, pitch)
        roll = np.arctan2(gravity[1], gravity[2])
        pitch = np.arctan2(-gravity[0], np.sqrt(gravity[1]**2 + gravity[2]**2))
        yaw = 0.0  # Can't determine yaw from gravity alone
        
        # Set initial state
        self.x = np.zeros(self.state_dim)
        self.x[:3] = np.array([roll, pitch, yaw])
        
        # Set initial covariance
        self.P = np.eye(self.state_dim) * 0.1
        self.P[2, 2] = 1.0  # Higher uncertainty for yaw
        
        self.initialized = True
    
    def predict(self, dt, gyro_corrected):
        """Kalman filter prediction step."""
        # Update state transition matrix for current dt
        self.F[:3, 3:] = np.eye(3) * dt
        
        # Update process noise
        self.Q[:3, :3] = np.eye(3) * self.process_noise * dt**2
        self.Q[3:, 3:] = np.eye(3) * self.gyro_bias_noise * dt
        
        # Predict state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Add gyro rotation
        delta_angle = gyro_corrected * dt
        self.x[:3] += delta_angle
    
    def update(self, measurement):
        """Kalman filter update step."""
        # Calculate innovation
        y = measurement - self.H @ self.x
        
        # Calculate innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Calculate Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process one step using standard Kalman filter."""
        if not self.initialized:
            self.initialize(accel)
            
        # Gyro bias correction
        gyro_corrected = gyro - self.x[3:6]
        
        # Prediction step
        self.predict(dt, gyro_corrected)
        
        # Update step using accelerometer for roll and pitch
        gravity = -accel / np.linalg.norm(accel)
        roll_acc = np.arctan2(gravity[1], gravity[2])
        pitch_acc = np.arctan2(-gravity[0], np.sqrt(gravity[1]**2 + gravity[2]**2))
        
        # Only update roll and pitch from accelerometer
        acc_angles = np.array([roll_acc, pitch_acc, self.x[2]])
        
        self.update(acc_angles)
        
        # Apply drift correction if reference data is available
        if timestamp is not None and self.use_reference:
            self.x[:3] = self.apply_drift_correction(self.x[:3], timestamp)
        
        # Convert Euler angles to quaternion
        orientation = self.x[:3]
        quat = R.from_euler('xyz', orientation).as_quat()
        
        # Feature vector: accel, gyro, quaternion, euler
        features = np.concatenate([
            accel,
            gyro,
            quat,
            orientation
        ])
        
        return self.x, features

class ExtendedKalmanFilter(BaseKalmanFilter):
    """Extended Kalman Filter for IMU fusion with quaternion state."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, drift_correction_weight)
        self.name = "EKF"
        
        # State: [quaternion (4), gyro_bias (3)]
        self.state_dim = 7
        self.measurement_dim = 3
        
        # Initialize state and covariance
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1 (identity)
        self.P = np.eye(self.state_dim) * 0.1
    
    def initialize(self, accel):
        """Initialize filter with first acceleration."""
        # Estimate gravity direction
        gravity = -accel / np.linalg.norm(accel)
        
        # Find rotation from [0,0,1] to gravity direction
        v = np.cross([0, 0, 1], gravity)
        s = np.linalg.norm(v)
        
        if s < 1e-10:
            # Vectors are parallel, no rotation needed
            self.x[:4] = np.array([1, 0, 0, 0])  # Identity quaternion
        else:
            c = np.dot([0, 0, 1], gravity)
            
            # Calculate rotation matrix and convert to quaternion
            if hasattr(R, 'from_mrp'):  # More modern SciPy versions
                r = R.from_mrp(v * np.arccos(c) / s)
                quat = r.as_quat()
            else:
                # Manual calculation
                v_skew = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                
                rotation_matrix = np.eye(3) + v_skew + v_skew.dot(v_skew) * (1 - c) / (s**2)
                r = R.from_matrix(rotation_matrix)
                quat = r.as_quat()
            
            # SciPy quaternions are [x, y, z, w], we use [w, x, y, z]
            self.x[:4] = np.array([quat[3], quat[0], quat[1], quat[2]])
        
        # Reset gyro bias
        self.x[4:] = np.zeros(3)
        
        # Initialize covariance
        self.P = np.eye(self.state_dim) * 0.1
        
        self.initialized = True
    
    def quaternion_update(self, q, omega, dt):
        """Update quaternion with angular velocity."""
        omega_norm = np.linalg.norm(omega)
        
        if omega_norm < 1e-10:
            return q
        
        axis = omega / omega_norm
        angle = omega_norm * dt
        
        # Quaternion for this rotation
        quat_delta = np.zeros(4)
        quat_delta[0] = np.cos(angle/2)
        quat_delta[1:] = axis * np.sin(angle/2)
        
        # Quaternion multiplication
        result = np.zeros(4)
        result[0] = q[0]*quat_delta[0] - np.dot(q[1:], quat_delta[1:])
        result[1:] = q[0]*quat_delta[1:] + quat_delta[0]*q[1:] + np.cross(q[1:], quat_delta[1:])
        
        # Normalize
        return result / np.linalg.norm(result)
    
    def state_transition(self, x, dt, gyro):
        """State transition function."""
        # Extract quaternion and bias
        q = x[:4]
        bias = x[4:]
        
        # Correct gyro with bias
        omega_corrected = gyro - bias
        
        # Update quaternion
        q_new = self.quaternion_update(q, omega_corrected, dt)
        
        # State remains the same except for quaternion
        x_new = np.zeros_like(x)
        x_new[:4] = q_new
        x_new[4:] = bias  # Bias model is constant
        
        return x_new
    
    def state_jacobian(self, x, dt, gyro):
        """Jacobian of state transition function."""
        # Approximate Jacobian
        F = np.eye(self.state_dim)
        
        # Effect of gyro bias on quaternion
        omega_norm = np.linalg.norm(gyro)
        if omega_norm > 1e-10:
            # Numerical approximation of Jacobian
            delta = 1e-5
            
            for i in range(3):
                # Original state
                x0 = x.copy()
                
                # Perturbed state
                x1 = x.copy()
                x1[4+i] += delta
                
                # State transition for both
                f0 = self.state_transition(x0, dt, gyro)
                f1 = self.state_transition(x1, dt, gyro)
                
                # Jacobian column for this bias component
                F[:4, 4+i] = (f1[:4] - f0[:4]) / delta
        
        return F
    
    def measurement_function(self, x):
        """Measurement function for EKF."""
        # Extract quaternion
        q = x[:4]
        
        # Convert from [w, x, y, z] to [x, y, z, w]
        quat_scipy = np.array([q[1], q[2], q[3], q[0]])
        
        # Convert quaternion to rotation matrix
        r = R.from_quat(quat_scipy)
        
        # Rotate unit gravity vector [0,0,1]
        gravity_body = r.apply([0, 0, 1])
        
        return gravity_body
    
    def measurement_jacobian(self, x):
        """Jacobian of measurement function."""
        # Numerical approximation
        H = np.zeros((self.measurement_dim, self.state_dim))
        
        # Base measurement
        z = self.measurement_function(x)
        
        # Compute partial derivatives for quaternion elements
        delta = 1e-5
        for i in range(4):
            # Perturb state
            x_plus = x.copy()
            x_plus[i] += delta
            
            # Compute measurement with perturbation
            z_plus = self.measurement_function(x_plus)
            
            # Numerical gradient
            H[:, i] = (z_plus - z) / delta
        
        return H
    
    def predict(self, dt, gyro):
        """EKF prediction step."""
        # Get state transition matrix
        F = self.state_jacobian(self.x, dt, gyro)
        
        # Predict state
        self.x = self.state_transition(self.x, dt, gyro)
        
        # Predict covariance
        # Scale process noise with dt
        Q = np.eye(self.state_dim) * self.process_noise * dt
        Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise * dt
        
        self.P = F @ self.P @ F.T + Q
    
    def update(self, measurement):
        """EKF update step."""
        # Calculate measurement Jacobian
        H = self.measurement_jacobian(self.x)
        
        # Predict measurement
        z_pred = self.measurement_function(self.x)
        
        # Innovation
        y = measurement - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + np.eye(self.measurement_dim) * self.measurement_noise
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
        
        # Normalize quaternion
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process one step using EKF."""
        if not self.initialized:
            self.initialize(accel)
        
        # Prediction step
        self.predict(dt, gyro)
        
        # Update step if not in high dynamics
        accel_norm = np.linalg.norm(accel)
        gravity_norm = 9.81
        
        if abs(accel_norm - gravity_norm) < 3.0:  # Threshold for quasi-static assumption
            # Normalized measured gravity direction
            z = -accel / accel_norm
            
            # Update with measurement
            self.update(z)
        
        # Apply drift correction if reference data is available
        if timestamp is not None and self.use_reference:
            # Convert quaternion to Euler angles
            quat_scipy = np.array([self.x[1], self.x[2], self.x[3], self.x[0]])
            euler = R.from_quat(quat_scipy).as_euler('xyz')
            
            # Correct Euler angles
            corrected_euler = self.apply_drift_correction(euler, timestamp)
            
            # Convert back to quaternion
            corrected_quat = R.from_euler('xyz', corrected_euler).as_quat()
            
            # Back to our format [w, x, y, z]
            self.x[:4] = np.array([corrected_quat[3], corrected_quat[0], corrected_quat[1], corrected_quat[2]])
        
        # Extract orientation for output
        quat_scipy = np.array([self.x[1], self.x[2], self.x[3], self.x[0]])
        euler = R.from_quat(quat_scipy).as_euler('xyz')
        
        # Feature vector: accel, gyro, quaternion, euler
        features = np.concatenate([
            accel,
            gyro,
            quat_scipy,
            euler
        ])
        
        return self.x, features

class UnscentedKalmanFilter(BaseKalmanFilter):
    """Implementation of Unscented Kalman Filter for IMU fusion."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, drift_correction_weight)
        self.name = "UKF"
        
        # State: [quaternion (4), gyro_bias (3)]
        self.state_dim = 7
        self.measurement_dim = 3
        
        # UKF parameters
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        
        # Initialize state and covariance
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1
        self.P = np.eye(self.state_dim) * 0.1
        
        # For saving last gyro
        self.last_gyro = np.zeros(3)
    
    def initialize(self, accel):
        """Initialize filter with first acceleration."""
        # Same as EKF initialization
        gravity = -accel / np.linalg.norm(accel)
        
        v = np.cross([0, 0, 1], gravity)
        s = np.linalg.norm(v)
        
        if s < 1e-10:
            self.x[:4] = np.array([1, 0, 0, 0])
        else:
            c = np.dot([0, 0, 1], gravity)
            
            if hasattr(R, 'from_mrp'):
                r = R.from_mrp(v * np.arccos(c) / s)
                quat = r.as_quat()
            else:
                v_skew = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                
                rotation_matrix = np.eye(3) + v_skew + v_skew.dot(v_skew) * (1 - c) / (s**2)
                r = R.from_matrix(rotation_matrix)
                quat = r.as_quat()
            
            self.x[:4] = np.array([quat[3], quat[0], quat[1], quat[2]])
        
        self.x[4:] = np.zeros(3)
        self.P = np.eye(self.state_dim) * 0.1
        
        self.initialized = True
    
    def quaternion_update(self, q, omega, dt):
        """Update quaternion with angular velocity."""
        omega_norm = np.linalg.norm(omega)
        
        if omega_norm < 1e-10:
            return q
        
        axis = omega / omega_norm
        angle = omega_norm * dt
        
        quat_delta = np.zeros(4)
        quat_delta[0] = np.cos(angle/2)
        quat_delta[1:] = axis * np.sin(angle/2)
        
        result = np.zeros(4)
        result[0] = q[0]*quat_delta[0] - np.dot(q[1:], quat_delta[1:])
        result[1:] = q[0]*quat_delta[1:] + quat_delta[0]*q[1:] + np.cross(q[1:], quat_delta[1:])
        
        return result / np.linalg.norm(result)
    
    def generate_sigma_points(self, x, P):
        """Generate sigma points using Merwe scaled sigma points."""
        n = len(x)
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # Calculate square root of P
        try:
            L = np.linalg.cholesky((n + lambda_) * P)
        except np.linalg.LinAlgError:
            # If cholesky fails, use eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 0)  # Ensure positive eigenvalues
            L = eigvecs @ np.diag(np.sqrt((n + lambda_) * eigvals)) @ eigvecs.T
        
        # Create sigma points
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = x
        
        for i in range(n):
            sigma_points[i + 1] = x + L[i]
            sigma_points[i + 1 + n] = x - L[i]
        
        return sigma_points
    
    def calculate_weights(self):
        """Calculate weights for sigma points."""
        n = self.state_dim
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        wm = np.zeros(2*n + 1)  # Mean weights
        wc = np.zeros(2*n + 1)  # Covariance weights
        
        wm[0] = lambda_ / (n + lambda_)
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2*n + 1):
            wm[i] = 1.0 / (2 * (n + lambda_))
            wc[i] = wm[i]
        
        return wm, wc
    
    def state_transition(self, sigma_points, dt):
        """Apply state transition to sigma points."""
        n_points = sigma_points.shape[0]
        transformed_points = np.zeros_like(sigma_points)
        
        for i in range(n_points):
            # Extract quaternion and bias
            q = sigma_points[i, :4]
            bias = sigma_points[i, 4:]
            
            # Correct gyro with bias
            omega_corrected = self.last_gyro - bias
            
            # Update quaternion
            q_new = self.quaternion_update(q, omega_corrected, dt)
            
            # Update point
            transformed_points[i, :4] = q_new
            transformed_points[i, 4:] = bias  # Bias is constant
        
        return transformed_points
    
    def measurement_function(self, sigma_points):
        """Apply measurement function to sigma points."""
        n_points = sigma_points.shape[0]
        measurements = np.zeros((n_points, self.measurement_dim))
        
        for i in range(n_points):
            # Extract quaternion
            q = sigma_points[i, :4]
            
            # Convert to scipy format
            quat_scipy = np.array([q[1], q[2], q[3], q[0]])
            
            # Rotate unit gravity vector
            r = R.from_quat(quat_scipy)
            gravity_body = r.apply([0, 0, 1])
            
            measurements[i] = gravity_body
        
        return measurements
    
    def predict(self, dt):
        """UKF prediction step."""
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Transform sigma points
        transformed_points = self.state_transition(sigma_points, dt)
        
        # Calculate weights
        wm, wc = self.calculate_weights()
        
        # Calculate predicted mean
        self.x = np.sum(wm.reshape(-1, 1) * transformed_points, axis=0)
        
        # Normalize quaternion part
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        
        # Calculate predicted covariance
        self.P = np.zeros_like(self.P)
        for i in range(len(wc)):
            diff = transformed_points[i] - self.x
            # Special handling for quaternion part
            if np.dot(diff[:4], self.x[:4]) < 0:
                diff[:4] = -diff[:4]  # Flip quaternion if on opposite hemisphere
            
            self.P += wc[i] * np.outer(diff, diff)
        
        # Add process noise
        Q = np.eye(self.state_dim) * self.process_noise * dt
        Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise * dt
        
        self.P += Q
    
    def update(self, measurement):
        """UKF update step."""
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Transform sigma points through measurement function
        measurement_points = self.measurement_function(sigma_points)
        
        # Calculate weights
        wm, wc = self.calculate_weights()
        
        # Calculate predicted measurement
        z_pred = np.sum(wm.reshape(-1, 1) * measurement_points, axis=0)
        
        # Calculate innovation covariance
        S = np.zeros((self.measurement_dim, self.measurement_dim))
        cross_cov = np.zeros((self.state_dim, self.measurement_dim))
        
        for i in range(len(wc)):
            diff_z = measurement_points[i] - z_pred
            diff_x = sigma_points[i] - self.x
            # Fix quaternion sign if needed
            if np.dot(diff_x[:4], self.x[:4]) < 0:
                diff_x[:4] = -diff_x[:4]
                
            S += wc[i] * np.outer(diff_z, diff_z)
            cross_cov += wc[i] * np.outer(diff_x, diff_z)
        
        # Add measurement noise
        S += np.eye(self.measurement_dim) * self.measurement_noise
        
        # Calculate Kalman gain
        K = cross_cov @ np.linalg.inv(S)
        
        # Calculate innovation
        y = measurement - z_pred
        
        # Update state
        self.x = self.x + K @ y
        
        # Normalize quaternion
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        
        # Update covariance
        self.P = self.P - K @ S @ K.T
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process one step using UKF."""
        if not self.initialized:
            self.initialize(accel)
        
        # Save gyro for state transition
        self.last_gyro = gyro
        
        # Prediction step
        self.predict(dt)
        
        # Update step if not in high dynamics
        accel_norm = np.linalg.norm(accel)
        gravity_norm = 9.81
        
        if abs(accel_norm - gravity_norm) < 3.0:
            # Normalize measurement
            z = -accel / accel_norm
            
            # Update with measurement
            self.update(z)
        
        # Apply drift correction
        if timestamp is not None and self.use_reference:
            # Convert to Euler angles
            quat_scipy = np.array([self.x[1], self.x[2], self.x[3], self.x[0]])
            euler = R.from_quat(quat_scipy).as_euler('xyz')
            
            # Apply correction
            corrected_euler = self.apply_drift_correction(euler, timestamp)
            
            # Convert back to quaternion
            corrected_quat = R.from_euler('xyz', corrected_euler).as_quat()
            
            # Back to our format
            self.x[:4] = np.array([corrected_quat[3], corrected_quat[0], corrected_quat[1], corrected_quat[2]])
        
        # Extract orientation
        quat_scipy = np.array([self.x[1], self.x[2], self.x[3], self.x[0]])
        euler = R.from_quat(quat_scipy).as_euler('xyz')
        
        # Feature vector
        features = np.concatenate([
            accel,
            gyro,
            quat_scipy,
            euler
        ])
        
        return self.x, features

# =====================================================================
# Alignment functions
# =====================================================================

@time_function("alignment")
def create_skeleton_timestamps(skel_data, fps=30.0):
    """Create timestamps for skeleton data (30 fps)."""
    n_frames = skel_data.shape[0]
    return np.arange(n_frames) / fps

@time_function("alignment")
def extract_wrist_trajectory(skel_data, wrist_idx=9):
    """Extract wrist joint trajectory from skeleton data."""
    if skel_data.shape[1] < (wrist_idx + 1) * 3:
        logger.error(f"Skeleton data has insufficient joints: {skel_data.shape[1]}")
        return np.zeros((skel_data.shape[0], 3))
    
    # Extract wrist joint (x, y, z)
    start_idx = wrist_idx * 3
    end_idx = start_idx + 3
    return skel_data[:, start_idx:end_idx]

@time_function("alignment")
def align_modalities_dtw(imu_data, skel_data, imu_timestamps=None, skel_fps=30.0, wrist_idx=9):
    """
    Align IMU and skeleton modalities using DTW.
    
    Args:
        imu_data: IMU data array (n_samples, features)
        skel_data: Skeleton data array (n_frames, joints*3)
        imu_timestamps: Optional timestamps for IMU data
        skel_fps: Frame rate of skeleton data
        wrist_idx: Index of wrist joint in skeleton
        
    Returns:
        Tuple of (aligned_imu, aligned_skel, aligned_timestamps)
    """
    if imu_data.shape[0] == 0 or skel_data.shape[0] == 0:
        logger.error("Empty input data for alignment")
        return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
    
    # Generate timestamps if not provided
    if imu_timestamps is None:
        imu_timestamps = np.arange(imu_data.shape[0]) / 50.0  # Assume 50Hz
    
    skel_timestamps = create_skeleton_timestamps(skel_data, skel_fps)
    
    try:
        # Extract features for alignment
        # For IMU: use acceleration magnitude
        imu_mag = np.linalg.norm(imu_data[:, :3], axis=1)
        
        # For skeleton: use wrist velocity
        wrist_traj = extract_wrist_trajectory(skel_data, wrist_idx)
        
        # Calculate velocity (magnitude)
        wrist_vel = np.zeros(skel_data.shape[0])
        for i in range(1, len(wrist_vel)):
            dt = 1.0 / skel_fps
            wrist_vel[i] = np.linalg.norm(wrist_traj[i] - wrist_traj[i-1]) / dt
        
        # Normalize sequences for DTW
        norm_imu = (imu_mag - np.mean(imu_mag)) / (np.std(imu_mag) + 1e-6)
        norm_wrist = (wrist_vel - np.mean(wrist_vel)) / (np.std(wrist_vel) + 1e-6)
        
        # Try to import dtw
        try:
            from dtaidistance import dtw
            
            # Compute warping path
            path = dtw.warping_path(norm_imu, norm_wrist)
            
            # Extract aligned indices
            imu_idx, skel_idx = zip(*path)
            
            # Remove duplicates while preserving order
            imu_unique = []
            skel_unique = []
            seen_imu = set()
            seen_skel = set()
            
            for i, s in zip(imu_idx, skel_idx):
                if i not in seen_imu:
                    imu_unique.append(i)
                    seen_imu.add(i)
                if s not in seen_skel:
                    skel_unique.append(s)
                    seen_skel.add(s)
            
            # Create aligned data
            aligned_imu = imu_data[imu_unique]
            aligned_skel = skel_data[skel_unique]
            aligned_ts = imu_timestamps[imu_unique]
            
            return aligned_imu, aligned_skel, aligned_ts
            
        except ImportError:
            logger.warning("DTW package not available, falling back to interpolation")
    
    except Exception as e:
        logger.error(f"Error in DTW alignment: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Fallback to interpolation
    try:
        # Find common time range
        t_min = max(imu_timestamps[0], skel_timestamps[0])
        t_max = min(imu_timestamps[-1], skel_timestamps[-1])
        
        if t_max <= t_min:
            logger.error("No overlapping time range for alignment")
            return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
        
        # Filter IMU data to common range
        imu_mask = (imu_timestamps >= t_min) & (imu_timestamps <= t_max)
        filtered_imu = imu_data[imu_mask]
        filtered_ts = imu_timestamps[imu_mask]
        
        # Interpolate skeleton to IMU timestamps
        from scipy.interpolate import interp1d
        
        interp_func = interp1d(
            skel_timestamps,
            skel_data,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        interp_skel = interp_func(filtered_ts)
        
        return filtered_imu, interp_skel, filtered_ts
        
    except Exception as e:
        logger.error(f"Error in interpolation alignment: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Last resort: use minimum common length
        min_len = min(imu_data.shape[0], skel_data.shape[0])
        return imu_data[:min_len], skel_data[:min_len], imu_timestamps[:min_len]

# =====================================================================
# Window creation functions
# =====================================================================

@time_function("windowing")
def create_windows(data, window_size_sec=4.0, stride_sec=1.0, max_length=128, is_skeleton=False):
    """
    Create fixed or variable length windows from time-stamped data.
    
    Args:
        data: Array with time in first column
        window_size_sec: Window size in seconds
        stride_sec: Stride in seconds
        max_length: Maximum length for fixed-length windows
        is_skeleton: Whether data is skeleton (variable length) or IMU (fixed length)
        
    Returns:
        List of window arrays
    """
    if data is None or data.shape[0] == 0:
        return []
    
    # Extract timestamps
    timestamps = data[:, 0]
    min_t = timestamps[0]
    max_t = timestamps[-1]
    
    windows = []
    t_start = min_t
    
    while t_start + window_size_sec <= max_t + 1e-9:
        # Extract data in time window
        mask = (timestamps >= t_start) & (timestamps < t_start + window_size_sec)
        window_data = data[mask]
        
        if len(window_data) >= 5:  # Minimum points threshold
            if is_skeleton:
                # For skeleton, keep variable length
                windows.append(window_data)
            else:
                # For IMU, resample to fixed length if needed
                if window_data.shape[0] != max_length:
                    # Linear interpolation to fixed length
                    indices = np.linspace(0, window_data.shape[0] - 1, max_length).astype(int)
                    resampled = window_data[indices]
                    windows.append(resampled)
                else:
                    windows.append(window_data)
        
        t_start += stride_sec
    
    return windows

# =====================================================================
# Main debug functions
# =====================================================================

def extract_orientation_from_skeleton(skel_data, wrist_idx=9):
    """
    Extract orientation from skeleton joint positions.
    
    Args:
        skel_data: Skeleton joint positions
        wrist_idx: Index of wrist joint
        
    Returns:
        Euler angles orientation
    """
    n_frames = skel_data.shape[0]
    orientations = np.zeros((n_frames, 3))
    
    if skel_data.shape[1] < 96:  # Need enough joints
        return orientations
    
    try:
        # Define key joint indices (adapt to your skeleton format)
        # These are example indices - adjust based on your actual skeleton model
        NECK = 2
        SPINE = 1
        RIGHT_SHOULDER = 8
        LEFT_SHOULDER = 4
        RIGHT_HIP = 12
        LEFT_HIP = 16
        
        # Process each frame
        for i in range(n_frames):
            # Reshape to get joint positions
            joints = skel_data[i].reshape(-1, 3)
            
            if joints.shape[0] > max(NECK, SPINE, RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_HIP, LEFT_HIP):
                # Calculate body orientation
                # Forward vector (spine)
                forward = joints[NECK] - joints[SPINE]
                forward = forward / (np.linalg.norm(forward) + 1e-10)
                
                # Right vector (shoulders)
                right = joints[RIGHT_SHOULDER] - joints[LEFT_SHOULDER]
                right = right / (np.linalg.norm(right) + 1e-10)
                
                # Up vector (cross product)
                up = np.cross(right, forward)
                up = up / (np.linalg.norm(up) + 1e-10)
                
                # Recalculate right to ensure orthogonality
                right = np.cross(forward, up)
                
                # Create rotation matrix
                rotation_matrix = np.column_stack([right, up, forward])
                
                # Convert to Euler angles
                try:
                    r = R.from_matrix(rotation_matrix)
                    euler = r.as_euler('xyz')
                    orientations[i] = euler
                except:
                    if i > 0:
                        orientations[i] = orientations[i-1]
    except Exception as e:
        logger.error(f"Error extracting skeleton orientation: {str(e)}")
    
    return orientations

def compare_filters(accel_data, gyro_data, skel_data=None, timestamps=None, 
                    filter_params=None, wrist_idx=9):
    """
    Compare different Kalman filter implementations.
    
    Args:
        accel_data: Accelerometer data (n_samples, 3)
        gyro_data: Gyroscope data (n_samples, 3)
        skel_data: Optional skeleton data for reference
        timestamps: Optional timestamps
        filter_params: Filter parameters
        wrist_idx: Wrist joint index
        
    Returns:
        Dictionary with filter outputs and metrics
    """
    if filter_params is None:
        filter_params = {
            'process_noise': 0.01,
            'measurement_noise': 0.1,
            'gyro_bias_noise': 0.01,
            'drift_correction_weight': 0.3
        }
    
    # Create filters
    filters = {
        'standard': StandardKalmanFilter(**filter_params),
        'ekf': ExtendedKalmanFilter(**filter_params),
        'ukf': UnscentedKalmanFilter(**filter_params)
    }
    
    # Extract reference orientation from skeleton if available
    reference_orientations = None
    if skel_data is not None and skel_data.shape[0] > 0:
        reference_orientations = extract_orientation_from_skeleton(skel_data, wrist_idx)
        
        # Set reference data for drift correction
        for f in filters.values():
            if timestamps is not None and reference_orientations is not None:
                f.set_reference_data(timestamps, reference_orientations)
    
    # Process data with each filter
    results = {}
    
    for name, filter_obj in filters.items():
        start_time = time.time()
        
        # Process data
        output = filter_obj.process_sequence(accel_data, gyro_data, timestamps)
        
        elapsed = time.time() - start_time
        timing_stats[f"{name}_filter"].append(elapsed)
        
        results[name] = {
            'output': output,
            'processing_time': elapsed
        }
        
        # Calculate error metrics if reference available
        if reference_orientations is not None and len(reference_orientations) == len(output):
            # Extract Euler angles from output (last 3 columns)
            estimated_orientations = output[:, -3:]
            
            # Calculate MSE
            mse = np.mean(np.sum((estimated_orientations - reference_orientations)**2, axis=1))
            
            # Calculate average angular error in degrees
            angular_errors = []
            for i in range(len(estimated_orientations)):
                # Create rotation matrices
                r1 = R.from_euler('xyz', estimated_orientations[i])
                r2 = R.from_euler('xyz', reference_orientations[i])
                
                # Calculate relative rotation
                r_diff = r1 * r2.inv()
                
                # Convert to angle
                angle = np.degrees(np.linalg.norm(r_diff.as_rotvec()))
                angular_errors.append(angle)
            
            avg_angular_error = np.mean(angular_errors)
            
            results[name]['mse'] = mse
            results[name]['angular_error'] = avg_angular_error
            
            # Update global metrics
            accuracy_metrics[f"{name}_filter"]["mse"].append(mse)
            accuracy_metrics[f"{name}_filter"]["angular_error"].append(avg_angular_error)
    
    return results

def plot_filter_comparison(filter_results, timestamps=None, title="Filter Comparison"):
    """
    Plot comparison of different filter outputs.
    
    Args:
        filter_results: Dictionary with filter outputs
        timestamps: Optional timestamps for x-axis
        title: Plot title
    """
    if not filter_results:
        return
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Sample filter output to determine shape
    first_filter = next(iter(filter_results.values()))
    output = first_filter['output']
    
    if timestamps is None:
        timestamps = np.arange(len(output))
    
    # Plot quaternion components
    plt.subplot(3, 1, 1)
    for name, result in filter_results.items():
        output = result['output']
        for i in range(4):
            component = ['w', 'x', 'y', 'z'][i]
            plt.plot(timestamps, output[:, 3+i], label=f"{name} {component}")
    
    plt.title("Quaternion Components")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # Plot Euler angles
    plt.subplot(3, 1, 2)
    for name, result in filter_results.items():
        output = result['output']
        for i in range(3):
            angle = ['Roll', 'Pitch', 'Yaw'][i]
            plt.plot(timestamps, np.degrees(output[:, 10+i]), label=f"{name} {angle}")
    
    plt.title("Euler Angles")
    plt.xlabel("Time (s)")
    plt.ylabel("Degrees")
    plt.legend()
    plt.grid(True)
    
    # Plot metrics
    plt.subplot(3, 1, 3)
    bar_width = 0.35
    index = np.arange(len(filter_results))
    
    metrics = []
    names = []
    process_times = []
    angular_errors = []
    
    for name, result in filter_results.items():
        names.append(name)
        process_times.append(result['processing_time'])
        if 'angular_error' in result:
            angular_errors.append(result['angular_error'])
        else:
            angular_errors.append(0)
    
    # Plot bars
    plt.bar(index, process_times, bar_width, label='Processing Time (s)')
    
    if angular_errors and any(angular_errors):
        # Normalize to make comparable with time
        max_time = max(process_times)
        norm_errors = [e * max_time / max(angular_errors) for e in angular_errors]
        plt.bar(index + bar_width, norm_errors, bar_width, label='Angular Error (normalized)')
        
        # Add text for actual errors
        for i, v in enumerate(angular_errors):
            plt.text(i + bar_width, norm_errors[i] + 0.01, f"{v:.2f}°", ha='center')
    
    plt.xlabel('Filter')
    plt.ylabel('Time (s) / Normalized Error')
    plt.title('Performance Metrics')
    plt.xticks(index + bar_width/2, names)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle(title)
    
    # Save plot
    plt.savefig(f"filter_comparison_{title.replace(' ', '_')}.png", dpi=150)
    plt.close()

def debug_trial(trial_info, output_dir="debug_output", window_size_sec=4.0, 
               stride_sec=1.0, max_length=128, wrist_idx=9):
    """
    Run comprehensive debug for a specific trial.
    
    Args:
        trial_info: Dictionary with file paths and trial info
        output_dir: Directory to save debug output
        window_size_sec: Window size in seconds
        stride_sec: Stride in seconds
        max_length: Max length for fixed windows
        wrist_idx: Wrist joint index
    """
    os.makedirs(output_dir, exist_ok=True)
    
    subject_id = trial_info.get('subject_id', 'unknown')
    action_id = trial_info.get('action_id', 'unknown')
    trial_id = trial_info.get('trial_id', 'unknown')
    
    trial_name = f"S{subject_id}A{action_id}T{trial_id}"
    logger.info(f"===== Debugging trial {trial_name} =====")
    
    # Create debug report file
    report_file = os.path.join(output_dir, f"{trial_name}_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"Debug Report for Trial {trial_name}\n")
        f.write("="*80 + "\n\n")
        
        # 1. Load data
        f.write("1. Data Loading\n")
        f.write("-"*80 + "\n")
        
        accel_path = trial_info.get('accelerometer', '')
        gyro_path = trial_info.get('gyroscope', '')
        skel_path = trial_info.get('skeleton', '')
        
        f.write(f"Accelerometer: {accel_path}\n")
        f.write(f"Gyroscope: {gyro_path}\n")
        f.write(f"Skeleton: {skel_path}\n\n")
        
        accel_data = load_watch_data(accel_path, verbose=True)
        f.write(f"Loaded accelerometer data: {accel_data.shape}\n")
        
        if gyro_path:
            gyro_data = load_watch_data(gyro_path, verbose=True)
            f.write(f"Loaded gyroscope data: {gyro_data.shape}\n")
        else:
            gyro_data = np.zeros((accel_data.shape[0], 4))
            gyro_data[:, 0] = accel_data[:, 0]  # Copy timestamps
            f.write("No gyroscope data - using zeros\n")
        
        if skel_path:
            skel_data = load_skeleton_data(skel_path, verbose=True)
            f.write(f"Loaded skeleton data: {skel_data.shape}\n")
        else:
            skel_data = None
            f.write("No skeleton data\n")
        
        f.write("\n")
        
        # Check for empty data
        if accel_data.shape[0] == 0:
            f.write("ERROR: Empty accelerometer data - aborting debug\n")
            logger.error(f"Empty accelerometer data for trial {trial_name}")
            return
        
        if gyro_data.shape[0] == 0:
            f.write("WARNING: Empty gyroscope data - using zeros\n")
            gyro_data = np.zeros((accel_data.shape[0], 4))
            gyro_data[:, 0] = accel_data[:, 0]  # Copy timestamps
        
        # 2. Align modalities
        f.write("2. Modality Alignment\n")
        f.write("-"*80 + "\n")
        
        accel_values = accel_data[:, 1:4]
        gyro_values = gyro_data[:, 1:4]
        timestamps = accel_data[:, 0]
        
        if skel_data is not None and skel_data.shape[0] > 0:
            f.write("Aligning skeleton and IMU data using DTW...\n")
            
            aligned_imu, aligned_skel, aligned_ts = align_modalities_dtw(
                accel_data[:, 1:4],  # Skip timestamp column
                skel_data,
                accel_data[:, 0],
                wrist_idx=wrist_idx
            )
            
            f.write(f"Alignment results:\n")
            f.write(f"  Original IMU samples: {accel_data.shape[0]}\n")
            f.write(f"  Original skeleton frames: {skel_data.shape[0]}\n")
            f.write(f"  Aligned IMU samples: {aligned_imu.shape[0]}\n")
            f.write(f"  Aligned skeleton frames: {aligned_skel.shape[0]}\n")
            
            if aligned_imu.shape[0] > 0:
                # Update data for filtering
                accel_values = aligned_imu
                timestamps = aligned_ts
                
                # Need to interpolate gyro to aligned timestamps
                from scipy.interpolate import interp1d
                
                gyro_interp = interp1d(
                    gyro_data[:, 0],
                    gyro_data[:, 1:4],
                    axis=0,
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                
                gyro_values = gyro_interp(aligned_ts)
                
                f.write("Using aligned data for filtering\n")
            else:
                f.write("WARNING: Alignment failed - using original data\n")
        else:
            f.write("No skeleton data for alignment - using original data\n")
            aligned_skel = None
        
        f.write("\n")
        
        # 3. Filter comparison
        f.write("3. Filter Comparison\n")
        f.write("-"*80 + "\n")
        
        filter_results = compare_filters(
            accel_values, 
            gyro_values, 
            aligned_skel, 
            timestamps,
            wrist_idx=wrist_idx
        )
        
        f.write("Filter processing times:\n")
        for name, result in filter_results.items():
            f.write(f"  {name.upper()}: {result['processing_time']:.4f} seconds\n")
        
        f.write("\nFilter accuracy metrics (if reference available):\n")
        for name, result in filter_results.items():
            if 'mse' in result:
                f.write(f"  {name.upper()}: MSE={result['mse']:.4f}, Angular Error={result['angular_error']:.2f}°\n")
        
        f.write("\n")
        
        # Create comparison plot
        plot_filter_comparison(filter_results, timestamps, title=trial_name)
        f.write(f"Filter comparison plot saved to filter_comparison_{trial_name}.png\n\n")
        
        # 4. Window creation
        f.write("4. Window Creation\n")
        f.write("-"*80 + "\n")
        
        # Get the best filter output
        best_filter = 'ekf'  # Default to EKF
        if filter_results:
            # Find filter with lowest angular error or processing time
            if all('angular_error' in r for _, r in filter_results.items()):
                best_filter = min(filter_results.items(), 
                                  key=lambda x: x[1]['angular_error'])[0]
            else:
                best_filter = min(filter_results.items(),
                                  key=lambda x: x[1]['processing_time'])[0]
        
        f.write(f"Using {best_filter.upper()} filter for window creation\n")
        
        # Reconstruct full data with timestamps
        fused_data = np.column_stack([timestamps, filter_results[best_filter]['output']])
        
        # Create windows
        imu_windows = create_windows(
            fused_data, 
            window_size_sec=window_size_sec,
            stride_sec=stride_sec,
            max_length=max_length,
            is_skeleton=False
        )
        
        f.write(f"Created {len(imu_windows)} IMU windows\n")
        
        if aligned_skel is not None and aligned_skel.shape[0] > 0:
            # Create skeleton windows
            skel_with_time = np.column_stack([timestamps, aligned_skel])
            
            skel_windows = create_windows(
                skel_with_time,
                window_size_sec=window_size_sec,
                stride_sec=stride_sec,
                is_skeleton=True
            )
            
            f.write(f"Created {len(skel_windows)} skeleton windows\n")
            
            # Match window counts
            min_windows = min(len(imu_windows), len(skel_windows))
            f.write(f"Using {min_windows} matched windows\n")
        else:
            f.write("No skeleton windows\n")
        
        f.write("\n")
        
        # 5. Visualize windows
        if imu_windows:
            f.write("5. Window Visualization\n")
            f.write("-"*80 + "\n")
            
            # Plot first window
            first_window = imu_windows[0]
            
            plt.figure(figsize=(15, 10))
            
            # Plot acceleration
            plt.subplot(3, 1, 1)
            plt.plot(first_window[:, 1], label='X')
            plt.plot(first_window[:, 2], label='Y')
            plt.plot(first_window[:, 3], label='Z')
            plt.title("Acceleration")
            plt.legend()
            plt.grid(True)
            
            # Plot quaternion
            plt.subplot(3, 1, 2)
            plt.plot(first_window[:, 7], label='W')
            plt.plot(first_window[:, 8], label='X')
            plt.plot(first_window[:, 9], label='Y')
            plt.plot(first_window[:, 10], label='Z')
            plt.title("Quaternion")
            plt.legend()
            plt.grid(True)
            
            # Plot Euler angles
            plt.subplot(3, 1, 3)
            plt.plot(np.degrees(first_window[:, 11]), label='Roll')
            plt.plot(np.degrees(first_window[:, 12]), label='Pitch')
            plt.plot(np.degrees(first_window[:, 13]), label='Yaw')
            plt.title("Euler Angles")
            plt.legend()
            plt.grid(True)
            
            plt.suptitle(f"First Window - Trial {trial_name}")
            plt.tight_layout()
            
            # Save plot
            window_plot_path = os.path.join(output_dir, f"{trial_name}_window.png")
            plt.savefig(window_plot_path, dpi=150)
            plt.close()
            
            f.write(f"First window visualization saved to {window_plot_path}\n\n")
        
        # 6. Summary
        f.write("6. Summary\n")
        f.write("-"*80 + "\n")
        
        f.write(f"Trial {trial_name} processing completed.\n")
        f.write(f"Recommended filter: {best_filter.upper()}\n")
        f.write(f"Number of windows: {len(imu_windows)}\n")
        
        if filter_results and all('angular_error' in r for _, r in filter_results.items()):
            best_accuracy = min(filter_results.items(), key=lambda x: x[1]['angular_error'])
            f.write(f"Best accuracy filter: {best_accuracy[0].upper()} "
                   f"with angular error {best_accuracy[1]['angular_error']:.2f}°\n")
        
        f.write("\nTiming statistics:\n")
        for category, times in timing_stats.items():
            if times:
                f.write(f"  {category}: avg={np.mean(times):.4f}s, total={np.sum(times):.4f}s\n")
    
    logger.info(f"Debug report saved to {report_file}")
    return filter_results

def get_trial_list(base_dir, subjects=None, max_trials=None):
    """
    Get list of available trials.
    
    Args:
        base_dir: Base directory containing data
        subjects: Optional list of subject IDs to include
        max_trials: Maximum number of trials to process
        
    Returns:
        List of trial info dictionaries
    """
    trials = []
    
    # Get young/old directories
    young_dir = os.path.join(base_dir, 'young')
    old_dir = os.path.join(base_dir, 'old')
    
    age_groups = []
    if os.path.exists(young_dir):
        age_groups.append(('young', young_dir))
    if os.path.exists(old_dir):
        age_groups.append(('old', old_dir))
    
    for age_group, group_dir in age_groups:
        # Find accelerometer/watch directory
        accel_dir = os.path.join(group_dir, 'accelerometer', 'watch')
        gyro_dir = os.path.join(group_dir, 'gyroscope', 'watch')
        skel_dir = os.path.join(group_dir, 'skeleton')
        
        if not os.path.exists(accel_dir):
            logger.warning(f"Accelerometer directory not found: {accel_dir}")
            continue
        
        # List accelerometer files (required)
        for file in os.listdir(accel_dir):
            if not file.endswith('.csv'):
                continue
                
            # Parse SxxAxxTxx
            match = re.match(r'S(\d+)A(\d+)T(\d+)\.csv', file)
            if not match:
                continue
                
            subject_id = int(match.group(1))
            action_id = int(match.group(2))
            trial_id = int(match.group(3))
            
            # Filter by subject if provided
            if subjects and subject_id not in subjects:
                continue
            
            # Create trial info
            trial_info = {
                'subject_id': subject_id,
                'action_id': action_id,
                'trial_id': trial_id,
                'age_group': age_group,
                'accelerometer': os.path.join(accel_dir, file)
            }
            
            # Add gyroscope if exists
            gyro_file = os.path.join(gyro_dir, file)
            if os.path.exists(gyro_file):
                trial_info['gyroscope'] = gyro_file
            
            # Add skeleton if exists
            skel_file = os.path.join(skel_dir, file)
            if os.path.exists(skel_file):
                trial_info['skeleton'] = skel_file
            
            trials.append(trial_info)
            
            # Check max trials
            if max_trials and len(trials) >= max_trials:
                break
    
    logger.info(f"Found {len(trials)} trials")
    return trials

# =====================================================================
# Main function
# =====================================================================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Debug IMU fusion and alignment")
    parser.add_argument('--data_dir', type=str, default='data/smartfallmm',
                        help='Data directory containing young/old subdirectories')
    parser.add_argument('--subject', type=int, default=None,
                        help='Process specific subject ID')
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma-separated list of subject IDs')
    parser.add_argument('--actions', type=str, default=None,
                        help='Comma-separated list of action IDs')
    parser.add_argument('--max_trials', type=int, default=5,
                        help='Maximum number of trials to process')
    parser.add_argument('--window_size', type=float, default=4.0,
                        help='Window size in seconds')
    parser.add_argument('--stride', type=float, default=1.0,
                        help='Stride size in seconds')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for fixed windows')
    parser.add_argument('--wrist_idx', type=int, default=9,
                        help='Index of wrist joint in skeleton data')
    parser.add_argument('--output_dir', type=str, default='debug_output',
                        help='Output directory for debug files')
    parser.add_argument('--filters', type=str, default='standard,ekf,ukf',
                        help='Comma-separated list of filters to test')
    
    args = parser.parse_args()
    
    # Parse subjects
    if args.subjects:
        subjects = [int(s) for s in args.subjects.split(',')]
    elif args.subject:
        subjects = [args.subject]
    else:
        subjects = None
    
    # Parse actions
    if args.actions:
        actions = [int(a) for a in args.actions.split(',')]
    else:
        actions = None
    
    # Parse filters
    filters = [f.strip() for f in args.filters.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info(f"Debug configuration:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Subjects: {subjects}")
    logger.info(f"  Actions: {actions}")
    logger.info(f"  Max trials: {args.max_trials}")
    logger.info(f"  Window size: {args.window_size} seconds")
    logger.info(f"  Stride: {args.stride} seconds")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Wrist index: {args.wrist_idx}")
    logger.info(f"  Filters: {filters}")
    
    # Get trial list
    trials = get_trial_list(args.data_dir, subjects, args.max_trials)
    
    # Filter by actions if provided
    if actions:
        trials = [t for t in trials if t['action_id'] in actions]
        logger.info(f"Filtered to {len(trials)} trials matching actions {actions}")
    
    # Process trials
    all_results = {}
    
    for trial in trials:
        trial_name = f"S{trial['subject_id']}A{trial['action_id']}T{trial['trial_id']}"
        logger.info(f"Processing trial {trial_name}")
        
        results = debug_trial(
            trial,
            output_dir=args.output_dir,
            window_size_sec=args.window_size,
            stride_sec=args.stride,
            max_length=args.max_length,
            wrist_idx=args.wrist_idx
        )
        
        all_results[trial_name] = results
    
    # Generate summary
    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("IMU Fusion Debug Summary\n")
        f.write("="*80 + "\n\n")
        
        # Overall timing stats
        f.write("Timing Statistics\n")
        f.write("-"*80 + "\n")
        
        for category, times in timing_stats.items():
            if times:
                f.write(f"{category}: avg={np.mean(times):.4f}s, total={np.sum(times):.4f}s, count={len(times)}\n")
        
        f.write("\n")
        
        # Filter comparison
        f.write("Filter Performance Comparison\n")
        f.write("-"*80 + "\n")
        
        for filter_name in filters:
            if f"{filter_name}_filter" in timing_stats:
                times = timing_stats[f"{filter_name}_filter"]
                f.write(f"{filter_name.upper()}: avg_time={np.mean(times):.4f}s, count={len(times)}\n")
        
        f.write("\n")
        
        # Accuracy metrics
        f.write("Filter Accuracy Metrics\n")
        f.write("-"*80 + "\n")
        
        for filter_name in filters:
            key = f"{filter_name}_filter"
            if key in accuracy_metrics:
                metrics = accuracy_metrics[key]
                if metrics["angular_error"]:
                    f.write(f"{filter_name.upper()}: ")
                    f.write(f"avg_angular_error={np.mean(metrics['angular_error']):.2f}°, ")
                    f.write(f"avg_mse={np.mean(metrics['mse']):.4f}, ")
                    f.write(f"count={len(metrics['angular_error'])}\n")
        
        f.write("\n")
        
        # Best filter recommendation
        best_filter = None
        best_error = float('inf')
        
        for filter_name in filters:
            key = f"{filter_name}_filter"
            if key in accuracy_metrics and accuracy_metrics[key]["angular_error"]:
                avg_error = np.mean(accuracy_metrics[key]["angular_error"])
                if avg_error < best_error:
                    best_error = avg_error
                    best_filter = filter_name
        
        if best_filter:
            f.write(f"Recommended filter: {best_filter.upper()} with average angular error {best_error:.2f}°\n")
        else:
            # If no accuracy metrics, recommend based on processing time
            for filter_name in filters:
                key = f"{filter_name}_filter"
                if key in timing_stats and timing_stats[key]:
                    avg_time = np.mean(timing_stats[key])
                    if best_filter is None or avg_time < best_error:
                        best_error = avg_time
                        best_filter = filter_name
            
            if best_filter:
                f.write(f"Recommended filter (based on processing time): {best_filter.upper()} with average time {best_error:.4f}s\n")
    
    logger.info(f"Summary saved to {summary_file}")
    
    # Create performance plot
    plt.figure(figsize=(12, 8))
    
    # Plot timing comparison
    plt.subplot(2, 1, 1)
    filter_names = []
    avg_times = []
    
    for filter_name in filters:
        key = f"{filter_name}_filter"
        if key in timing_stats and timing_stats[key]:
            filter_names.append(filter_name.upper())
            avg_times.append(np.mean(timing_stats[key]))
    
    if filter_names:
        plt.bar(filter_names, avg_times)
        plt.title("Average Processing Time")
        plt.ylabel("Time (seconds)")
        plt.grid(True)
    
    # Plot accuracy comparison
    plt.subplot(2, 1, 2)
    filter_names = []
    avg_errors = []
    
    for filter_name in filters:
        key = f"{filter_name}_filter"
        if key in accuracy_metrics and accuracy_metrics[key]["angular_error"]:
            filter_names.append(filter_name.upper())
            avg_errors.append(np.mean(accuracy_metrics[key]["angular_error"]))
    
    if filter_names:
        plt.bar(filter_names, avg_errors)
        plt.title("Average Angular Error")
        plt.ylabel("Error (degrees)")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "performance_comparison.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
