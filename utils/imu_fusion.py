"""
IMU fusion module with multiple Kalman filter implementations for orientation estimation.
Supports Standard, Extended, and Unscented Kalman Filters with quaternion representation.

Features:
- Drift correction using skeleton reference data
- Quaternion-based orientation representation
- Fall-specific feature extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time
import logging
from typing import Optional, Tuple, List, Dict, Union

# Configure logging
logger = logging.getLogger("IMUFusion")

class BaseKalmanFilter:
    """Base class for IMU fusion with drift correction capabilities."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
        """
        Initialize the base Kalman filter.
        
        Args:
            dt: Default time step in seconds
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            gyro_bias_noise: Gyroscope bias noise variance
            drift_correction_weight: Weight for skeleton-based drift correction
        """
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
        
        # Fall detection specific parameters
        self.fall_detection_features = True  # Enable fall-specific features
        
    def initialize(self, accel):
        """Initialize the filter - implemented by subclasses."""
        raise NotImplementedError
        
    def set_reference_data(self, timestamps, orientations):
        """
        Set reference orientation data from skeleton.
        
        Args:
            timestamps: Array of reference timestamps
            orientations: Array of reference orientations (euler angles)
        """
        if timestamps is None or orientations is None:
            self.use_reference = False
            return
        
        if len(timestamps) == 0 or len(orientations) == 0:
            self.use_reference = False
            return
        
        if len(timestamps) != len(orientations):
            logger.warning(f"Reference mismatch: {len(timestamps)} timestamps vs {len(orientations)} orientations. Disabling reference.")
            self.use_reference = False
            return
            
        self.reference_timestamps = np.array(timestamps).flatten()
        self.reference_orientations = np.array(orientations)
        self.use_reference = True
        logger.info(f"Reference data set: {len(timestamps)} points")
        
    def get_reference_orientation(self, timestamp):
        """
        Get reference orientation at a specific timestamp.
        
        Args:
            timestamp: Time point to get reference orientation
            
        Returns:
            Reference orientation or None if not available
        """
        if not self.use_reference:
            return None
            
        # Check if we have valid reference data
        if self.reference_timestamps is None or self.reference_orientations is None:
            return None
            
        if len(self.reference_timestamps) != len(self.reference_orientations):
            return None
            
        # Check if timestamp is within range
        if (timestamp < self.reference_timestamps[0] or 
            timestamp > self.reference_timestamps[-1]):
            return None
        
        try:
            # Interpolate reference orientation
            interp_func = interp1d(
                self.reference_timestamps,
                self.reference_orientations,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            return interp_func(timestamp)
        except Exception as e:
            logger.warning(f"Error interpolating reference orientation: {e}")
            return None
        
    def apply_drift_correction(self, estimated_orientation, timestamp):
        """
        Apply drift correction using reference orientation.
        
        Args:
            estimated_orientation: Orientation from Kalman filter
            timestamp: Current timestamp
            
        Returns:
            Corrected orientation
        """
        if not self.use_reference:
            return estimated_orientation
            
        reference = self.get_reference_orientation(timestamp)
        if reference is None:
            return estimated_orientation
            
        # Apply weighted correction
        w = self.drift_correction_weight
        corrected = (1 - w) * estimated_orientation + w * reference
        return corrected
    
    def extract_fall_features(self, accel, gyro, orientation, prev_accel=None):
        """
        Extract fall-specific features.
        
        Args:
            accel: Current acceleration (3,)
            gyro: Current angular velocity (3,)
            orientation: Current orientation (3,) (euler angles)
            prev_accel: Previous acceleration or None
            
        Returns:
            Dictionary of fall-specific features
        """
        if not self.fall_detection_features:
            return {}
            
        features = {}
        
        # 1. Vertical acceleration component (using orientation)
        r = R.from_euler('xyz', orientation)
        world_accel = r.apply(accel)
        features['vert_accel'] = world_accel[2]  # Vertical component
        
        # 2. Orientation change rate - useful for detecting sudden posture changes
        features['pitch'] = orientation[1]  # Pitch angle
        
        # 3. Impact detection - sudden changes in acceleration magnitude
        accel_mag = np.linalg.norm(accel)
        features['accel_magnitude'] = accel_mag
        
        # 4. Jerk - rate of acceleration change (first derivative of acceleration)
        if prev_accel is not None:
            jerk = (accel - prev_accel) / self.dt
            features['jerk_magnitude'] = np.linalg.norm(jerk)
        
        return features
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """
        Process a single step of sensor data.
        
        Args:
            accel: Acceleration measurement (3,)
            gyro: Angular velocity measurement (3,)
            dt: Time step
            timestamp: Current timestamp (for drift correction)
            
        Returns:
            Updated state estimate and features
        """
        # Base implementation - to be overridden
        pass
    
    def process_sequence(self, accel_data, gyro_data, timestamps=None):
        """
        Process a sequence of sensor data.
        
        Args:
            accel_data: Accelerometer data (N, 3)
            gyro_data: Gyroscope data (N, 3)
            timestamps: Optional timestamps (N,)
            
        Returns:
            Fused data with orientation (N, features)
        """
        N = accel_data.shape[0]
        
        # If timestamps not provided, generate uniform timestamps
        if timestamps is None:
            timestamps = np.arange(N) * self.dt
        
        # Create output array with appropriate size for chosen filter
        # Output: [accel(3), gyro(3), quat(4), euler(3)]
        output_features = 13
        output = np.zeros((N, output_features))
        
        # Initialize filter with first acceleration
        if not self.initialized:
            self.initialize(accel_data[0])
        
        # Process each measurement
        prev_accel = None
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
            try:
                state, features = self.process_step(accel, gyro, dt, timestamps[i])
                output[i] = features
                prev_accel = accel
            except Exception as e:
                logger.warning(f"Error processing step {i}: {e}")
                if i > 0:
                    # Use previous output if available
                    output[i] = output[i-1]
        
        return output

class StandardKalmanIMU(BaseKalmanFilter):
    """
    Standard Kalman Filter for IMU fusion.
    
    This uses a simplified linear model that works well for small movements
    but has limitations for complex rotations.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, gyro_bias_noise=0.01,
                drift_correction_weight=0.3):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, drift_correction_weight)
        
        # State: [orientation (3), gyro_bias (3)]
        self.state_dim = 6
        self.measurement_dim = 3
        
        # Initialize state and covariance
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 0.1
        
        # State transition matrix (F)
        self.F = np.eye(self.state_dim)
        
        # Measurement matrix (H)
        # We only directly measure orientation from accelerometer
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[:3, :3] = np.eye(3)
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * self.measurement_noise
        
        self.name = "Standard Kalman"
    
    def initialize(self, accel):
        """Initialize filter state using initial acceleration."""
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
    
    def predict(self, dt, gyro):
        """Kalman filter prediction step."""
        # Update state transition matrix for current dt
        self.F[:3, 3:] = np.eye(3) * dt
        
        # Update process noise
        Q = np.zeros((self.state_dim, self.state_dim))
        Q[:3, :3] = np.eye(3) * self.process_noise * dt**2
        Q[3:, 3:] = np.eye(3) * self.gyro_bias_noise * dt
        
        # Predict state
        gyro_corrected = gyro - self.x[3:6]
        delta_angle = gyro_corrected * dt
        
        # State transition: x_new = F*x + delta_angle
        self.x = self.F @ self.x
        self.x[:3] += delta_angle  # Add rotation
        
        # Update covariance
        self.P = self.F @ self.P @ self.F.T + Q
    
    def update(self, measurement):
        """Kalman filter update step."""
        # Innovation: y = z - Hx
        y = measurement - self.H @ self.x
        
        # Innovation covariance: S = HPH' + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = PH'S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + Ky
        self.x = self.x + K @ y
        
        # Update covariance: P = (I-KH)P
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process one step using standard Kalman filter."""
        if not self.initialized:
            self.initialize(accel)
        
        # Prediction step
        self.predict(dt, gyro)
        
        # Update step using accelerometer for roll and pitch
        # (Accelerometer can't directly measure yaw)
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:  # Avoid division by zero
            gravity = -accel / accel_norm
            roll_acc = np.arctan2(gravity[1], gravity[2])
            pitch_acc = np.arctan2(-gravity[0], np.sqrt(gravity[1]**2 + gravity[2]**2))
            
            # Only update roll and pitch from accelerometer
            acc_angles = np.array([roll_acc, pitch_acc, self.x[2]])
            
            # Skip update if acceleration is far from gravity (high dynamics)
            if abs(accel_norm - 9.81) < 3.0:  # Within ~3m/s² of gravity
                self.update(acc_angles)
        
        # Apply drift correction if reference data is available
        if timestamp is not None and self.use_reference:
            try:
                self.x[:3] = self.apply_drift_correction(self.x[:3], timestamp)
            except Exception as e:
                logger.warning(f"Drift correction failed: {e}")
        
        # Convert Euler angles to quaternion
        orientation = self.x[:3]
        
        # Create quaternion
        try:
            quat = R.from_euler('xyz', orientation).as_quat()
        except Exception as e:
            # Handle gimbal lock or other conversion issues
            logger.warning(f"Quaternion conversion error: {e}")
            quat = np.array([1.0, 0.0, 0.0, 0.0])  # Default to identity quaternion
        
        # Feature vector: accel, gyro, quaternion, euler
        features = np.concatenate([
            accel,
            gyro,
            quat,
            orientation
        ])
        
        return self.x, features

class ExtendedKalmanIMU(BaseKalmanFilter):
    """
    Extended Kalman Filter for IMU fusion.
    
    Uses quaternion representation for better handling of orientation
    and nonlinear state transitions.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, gyro_bias_noise=0.01,
                drift_correction_weight=0.3):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, drift_correction_weight)
        
        # State: [quaternion (4), gyro_bias (3)]
        self.state_dim = 7
        self.measurement_dim = 3
        
        # Initialize state
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1 (identity)
        
        # Initialize covariance
        self.P = np.eye(self.state_dim) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * self.process_noise
        self.Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise  # Bias noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * self.measurement_noise
        
        self.name = "Extended Kalman"
    
    def initialize(self, accel):
        """Initialize filter with first acceleration measurement."""
        # Estimate gravity direction
        gravity = -accel / np.linalg.norm(accel)
        
        # Find rotation from [0,0,1] to gravity direction
        v = np.cross([0, 0, 1], gravity)
        s = np.linalg.norm(v)
        
        if s < 1e-10:
            # Vectors are parallel, no rotation needed
            quat = np.array([1, 0, 0, 0])  # Identity quaternion
        else:
            c = np.dot([0, 0, 1], gravity)
            v_skew = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            
            rotation_matrix = np.eye(3) + v_skew + v_skew.dot(v_skew) * (1 - c) / (s**2)
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()
            
            # Convert from scipy [x,y,z,w] to [w,x,y,z]
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        
        # Initialize state [quat, gyro_bias]
        self.x = np.zeros(self.state_dim)
        self.x[:4] = quat
        
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
        # Convention: q = [w, x, y, z]
        result = np.zeros(4)
        result[0] = q[0]*quat_delta[0] - np.dot(q[1:], quat_delta[1:])
        result[1:] = q[0]*quat_delta[1:] + quat_delta[0]*q[1:] + np.cross(q[1:], quat_delta[1:])
        
        # Normalize
        return result / np.linalg.norm(result)
    
    def state_transition_function(self, x, dt, omega):
        """
        State transition function for EKF.
        
        Args:
            x: Current state (quaternion, gyro_bias)
            dt: Time step
            omega: Angular velocity
            
        Returns:
            New state
        """
        # Extract current quaternion and bias
        q = x[:4]
        bias = x[4:]
        
        # Correct gyro with bias
        omega_corrected = omega - bias
        
        # Update quaternion
        q_new = self.quaternion_update(q, omega_corrected, dt)
        
        # State remains the same except for quaternion
        x_new = np.zeros_like(x)
        x_new[:4] = q_new
        x_new[4:] = bias  # Bias model is constant
        
        return x_new
    
    def state_transition_jacobian(self, x, dt, omega):
        """Jacobian of state transition function."""
        # This is a complex calculation for quaternion dynamics
        F = np.eye(self.state_dim)
        
        # Effect of gyro bias on quaternion
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-10:
            # Approximate partial derivatives
            delta = 0.0001
            
            for i in range(3):
                # Small perturbation to measure Jacobian
                delta_vec = np.zeros(3)
                delta_vec[i] = delta
                
                # Forward difference approximation
                x_plus = np.copy(x)
                x_plus[4+i] += delta
                
                # Calculate difference
                f_plus = self.state_transition_function(x_plus, dt, omega)
                f = self.state_transition_function(x, dt, omega)
                
                # Partial derivative
                F[:4, 4+i] = (f_plus[:4] - f[:4]) / delta
        
        return F
    
    def measurement_function(self, x):
        """
        Measurement function for EKF.
        
        Converts quaternion to expected accelerometer measurement
        (assuming only gravity when stationary).
        """
        # Extract quaternion
        q = x[:4]
        
        # Convert our quaternion [w,x,y,z] to scipy [x,y,z,w]
        q_scipy = np.array([q[1], q[2], q[3], q[0]])
        
        # Convert quaternion to rotation matrix
        r = R.from_quat(q_scipy)
        
        # Rotate unit gravity vector [0,0,1]
        gravity_body = r.apply([0, 0, 1])
        
        return gravity_body
    
    def measurement_jacobian(self, x):
        """Jacobian of measurement function."""
        # Numerical approximation
        H = np.zeros((self.measurement_dim, self.state_dim))
        
        # Base measurement
        z = self.measurement_function(x)
        
        # Compute partials for quaternion elements
        delta = 0.0001
        
        for i in range(4):
            # Small perturbation
            x_plus = np.copy(x)
            x_plus[i] += delta
            
            # Numerical partial derivative
            z_plus = self.measurement_function(x_plus)
            H[:, i] = (z_plus - z) / delta
        
        # Partials for gyro bias are zero (no direct impact on measurement)
        
        return H
    
    def predict(self, dt, gyro):
        """EKF prediction step."""
        # State transition Jacobian
        F = self.state_transition_jacobian(self.x, dt, gyro)
        
        # Predict state
        self.x = self.state_transition_function(self.x, dt, gyro)
        
        # Predict covariance
        # Scale process noise with dt
        Q = self.Q.copy() * dt
        
        self.P = F @ self.P @ F.T + Q
    
    def update(self, z):
        """EKF update step."""
        # Measurement Jacobian
        H = self.measurement_jacobian(self.x)
        
        # Predicted measurement
        z_pred = self.measurement_function(self.x)
        
        # Innovation
        y = z - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
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
        
        # Update step if not in high dynamics (acceleration close to g)
        accel_norm = np.linalg.norm(accel)
        gravity_norm = 9.81
        
        if abs(accel_norm - gravity_norm) < 3.0:  # Threshold for quasi-static assumption
            # Normalize measurement
            z = -accel / accel_norm  # Measured gravity direction
            
            # Update with measurement
            self.update(z)
        
        # Apply drift correction if reference data is available
        if timestamp is not None and self.use_reference:
            try:
                # Extract Euler angles from quaternion
                # Convert quaternion [w,x,y,z] to scipy [x,y,z,w]
                q_scipy = np.array([self.x[1], self.x[2], self.x[3], self.x[0]])
                euler = R.from_quat(q_scipy).as_euler('xyz')
                
                # Correct Euler angles
                corrected_euler = self.apply_drift_correction(euler, timestamp)
                
                # Convert back to quaternion
                q_corrected = R.from_euler('xyz', corrected_euler).as_quat()
                
                # Convert scipy [x,y,z,w] back to [w,x,y,z]
                self.x[:4] = [q_corrected[3], q_corrected[0], q_corrected[1], q_corrected[2]]
            except Exception as e:
                logger.warning(f"Drift correction error: {e}")
        
        # Extract orientation for output
        quat = self.x[:4]
        
        # Convert quaternion [w,x,y,z] to scipy [x,y,z,w]
        q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        
        try:
            euler = R.from_quat(q_scipy).as_euler('xyz')
        except Exception as e:
            logger.warning(f"Euler conversion error: {e}")
            euler = np.zeros(3)
        
        # Feature vector: accel, gyro, quaternion, euler
        features = np.concatenate([
            accel,
            gyro,
            quat,
            euler
        ])
        
        return self.x, features

class UnscentedKalmanIMU(BaseKalmanFilter):
    """
    Unscented Kalman Filter for IMU fusion.
    
    Handles highly nonlinear systems better than EKF by using sigma points
    to capture the probability distribution.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, gyro_bias_noise=0.01,
                drift_correction_weight=0.3):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, drift_correction_weight)
        
        # State: [quaternion (4), gyro_bias (3)]
        self.state_dim = 7
        self.measurement_dim = 3
        
        # UKF parameters
        self.alpha = 1e-3  # Spread of sigma points
        self.beta = 2.0    # Prior knowledge of distribution (2 is optimal for Gaussian)
        self.kappa = 0.0   # Secondary scaling parameter
        
        # Initialize state
        self.x = np.zeros(self.state_dim)  # State
        self.x[0] = 1.0  # Initial quaternion w=1
        self.P = np.eye(self.state_dim) * 0.1  # Covariance
        
        # Process noise
        self.Q = np.eye(self.state_dim) * self.process_noise
        self.Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * self.measurement_noise
        
        # Save last gyro for state transition
        self.last_gyro = np.zeros(3)
        
        self.name = "Unscented Kalman"
    
    def initialize(self, accel):
        """Initialize filter with first acceleration."""
        # Similar to EKF initialization
        gravity = -accel / np.linalg.norm(accel)
        
        v = np.cross([0, 0, 1], gravity)
        s = np.linalg.norm(v)
        
        if s < 1e-10:
            quat = np.array([1, 0, 0, 0])  # Identity quaternion
        else:
            c = np.dot([0, 0, 1], gravity)
            v_skew = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            
            rotation_matrix = np.eye(3) + v_skew + v_skew.dot(v_skew) * (1 - c) / (s**2)
            r = R.from_matrix(rotation_matrix)
            quat_scipy = r.as_quat()
            
            # Convert scipy [x,y,z,w] to [w,x,y,z]
            quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        
        self.x[:4] = quat
        self.x[4:] = np.zeros(3)  # Initial bias
        
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
    
    def generate_sigma_points(self):
        """Generate sigma points using Merwe scaled sigma points."""
        n = self.state_dim
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # Calculate square root of P
        try:
            # Try Cholesky decomposition first (more efficient)
            L = np.linalg.cholesky((n + lambda_) * self.P)
        except np.linalg.LinAlgError:
            # Fall back to eigenvalue decomposition if Cholesky fails
            eigvals, eigvecs = np.linalg.eigh(self.P)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 0)
            L = eigvecs @ np.diag(np.sqrt((n + lambda_) * eigvals)) @ eigvecs.T
        
        # Create 2n+1 sigma points
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = self.x  # Center point
        
        for i in range(n):
            sigma_points[i + 1] = self.x + L[i]
            sigma_points[i + 1 + n] = self.x - L[i]
            
            # Normalize quaternion part of each sigma point
            for j in range(2*n + 1):
                q_norm = np.linalg.norm(sigma_points[j, :4])
                if q_norm > 0:
                    sigma_points[j, :4] /= q_norm
        
        return sigma_points, lambda_, n
    
    def calculate_weights(self, lambda_, n):
        """Calculate weights for sigma points."""
        # Weights for mean
        wm = np.zeros(2*n + 1)
        # Weights for covariance
        wc = np.zeros(2*n + 1)
        
        wm[0] = lambda_ / (n + lambda_)
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2*n + 1):
            wm[i] = 1.0 / (2 * (n + lambda_))
            wc[i] = wm[i]
        
        return wm, wc
    
    def state_transition(self, sigma_points, dt):
        """Apply state transition to sigma points."""
        n_points = sigma_points.shape[0]
        n_dim = sigma_points.shape[1]
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
            
            # Convert [w,x,y,z] to scipy [x,y,z,w]
            q_scipy = np.array([q[1], q[2], q[3], q[0]])
            
            # Rotate unit gravity vector
            try:
                r = R.from_quat(q_scipy)
                gravity_body = r.apply([0, 0, 1])
                measurements[i] = gravity_body
            except Exception as e:
                logger.warning(f"Measurement function error: {e}")
                measurements[i] = np.array([0, 0, 1])
        
        return measurements
    
    def predict(self, dt):
        """UKF prediction step."""
        # Generate sigma points
        sigma_points, lambda_, n = self.generate_sigma_points()
        
        # Transform sigma points
        transformed_points = self.state_transition(sigma_points, dt)
        
        # Calculate weights
        wm, wc = self.calculate_weights(lambda_, n)
        
        # Calculate predicted mean
        self.x = np.zeros(self.state_dim)
        for i in range(len(wm)):
            self.x += wm[i] * transformed_points[i]
        
        # Normalize quaternion part
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        
        # Calculate predicted covariance
        self.P = np.zeros_like(self.P)
        for i in range(len(wc)):
            diff = transformed_points[i] - self.x
            
            # Handle quaternion differences properly
            if np.dot(diff[:4], self.x[:4]) < 0:
                diff[:4] = -diff[:4]  # Flip quaternion if on opposite hemisphere
            
            self.P += wc[i] * np.outer(diff, diff)
        
        # Add process noise
        self.P += self.Q * dt
    
    def update(self, measurement):
        """UKF update step."""
        # Generate sigma points
        sigma_points, lambda_, n = self.generate_sigma_points()
        
        # Transform sigma points through measurement function
        measurement_points = self.measurement_function(sigma_points)
        
        # Calculate weights
        wm, wc = self.calculate_weights(lambda_, n)
        
        # Calculate predicted measurement
        z_pred = np.zeros(self.measurement_dim)
        for i in range(len(wm)):
            z_pred += wm[i] * measurement_points[i]
        
        # Calculate innovation covariance and cross-correlation
        Pzz = np.zeros((self.measurement_dim, self.measurement_dim))
        Pxz = np.zeros((self.state_dim, self.measurement_dim))
        
        for i in range(len(wc)):
            z_diff = measurement_points[i] - z_pred
            x_diff = sigma_points[i] - self.x
            
            # Handle quaternion differences properly
            if np.dot(x_diff[:4], self.x[:4]) < 0:
                x_diff[:4] = -x_diff[:4]
            
            Pzz += wc[i] * np.outer(z_diff, z_diff)
            Pxz += wc[i] * np.outer(x_diff, z_diff)
        
        # Add measurement noise
        Pzz += self.R
        
        # Calculate Kalman gain
        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError as e:
            logger.warning(f"Matrix inversion error in UKF update: {e}")
            return
        
        # Update state
        self.x = self.x + K @ (measurement - z_pred)
        
        # Normalize quaternion
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        
        # Update covariance
        self.P = self.P - K @ Pzz @ K.T
    
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
            try:
                # Convert to Euler angles
                q_scipy = np.array([self.x[1], self.x[2], self.x[3], self.x[0]])
                euler = R.from_quat(q_scipy).as_euler('xyz')
                
                # Apply correction
                corrected_euler = self.apply_drift_correction(euler, timestamp)
                
                # Convert back to quaternion
                corrected_quat_scipy = R.from_euler('xyz', corrected_euler).as_quat()
                
                # Convert back to our format [w,x,y,z]
                self.x[:4] = [
                    corrected_quat_scipy[3],
                    corrected_quat_scipy[0], 
                    corrected_quat_scipy[1], 
                    corrected_quat_scipy[2]
                ]
            except Exception as e:
                logger.warning(f"UKF drift correction error: {e}")
        
        # Extract orientation
        quat = self.x[:4]
        
        # Convert to scipy format for euler angles
        q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        
        try:
            euler = R.from_quat(q_scipy).as_euler('xyz')
        except Exception as e:
            logger.warning(f"UKF euler conversion error: {e}")
            euler = np.zeros(3)
        
        # Feature vector: accel, gyro, quaternion, euler
        features = np.concatenate([
            accel,
            gyro,
            quat,
            euler
        ])
        
        return self.x, features

def extract_orientation_from_skeleton(skeleton_data, num_joints=32, joint_dim=3, wrist_idx=9):
    """
    Extract orientation information from skeleton joint positions.
    
    Args:
        skeleton_data: Skeleton joint positions (n_frames, num_joints*joint_dim)
        num_joints: Number of joints in skeleton
        joint_dim: Dimension of each joint (typically 3 for XYZ)
        wrist_idx: Index of wrist joint for special handling
    
    Returns:
        Orientation as Euler angles (n_frames, 3)
    """
    n_frames = skeleton_data.shape[0]
    orientations = np.zeros((n_frames, 3))
    
    # Reshape if needed
    if skeleton_data.shape[1] == num_joints * joint_dim:
        skeleton_reshaped = skeleton_data.reshape(n_frames, num_joints, joint_dim)
    else:
        # Can't determine orientation reliably if joint structure is unknown
        logger.warning("Unable to extract orientation from skeleton - unexpected dimensions")
        return orientations
    
    for i in range(n_frames):
        joints = skeleton_reshaped[i]
        
        # Define key joints (modify indices based on actual skeleton structure)
        NECK = 2
        SPINE = 1
        RIGHT_SHOULDER = 8
        LEFT_SHOULDER = 4
        RIGHT_HIP = 12
        LEFT_HIP = 16
        
        try:
            if num_joints > max(NECK, SPINE, RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_HIP, LEFT_HIP, wrist_idx):
                # Calculate forward vector (spine)
                spine_vec = joints[NECK] - joints[SPINE]
                spine_vec = spine_vec / (np.linalg.norm(spine_vec) + 1e-6)
                
                # Calculate right vector (shoulders)
                shoulder_vec = joints[RIGHT_SHOULDER] - joints[LEFT_SHOULDER]
                shoulder_vec = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-6)
                
                # Calculate up vector (cross product)
                up_vec = np.cross(shoulder_vec, spine_vec)
                up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-6)
                
                # Recalculate right to ensure orthogonality
                right_vec = np.cross(spine_vec, up_vec)
                right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-6)
                
                # Create rotation matrix from these vectors
                rot_matrix = np.column_stack([right_vec, up_vec, spine_vec])
                
                # Convert to Euler angles
                try:
                    r = R.from_matrix(rot_matrix)
                    orientations[i] = r.as_euler('xyz')
                except Exception as e:
                    # If conversion fails, use previous orientation or zeros
                    if i > 0:
                        orientations[i] = orientations[i-1]
                    logger.warning(f"Error converting rotation matrix to Euler angles: {e}")
        except Exception as e:
            logger.warning(f"Error extracting orientation from skeleton frame {i}: {e}")
            if i > 0:
                orientations[i] = orientations[i-1]
    
    return orientations

def calibrate_filter(accel_data, gyro_data, skeleton_data, filter_type='ekf', timestamps=None, wrist_idx=9):
    """
    Calibrate filter parameters using skeleton as ground truth.
    
    Args:
        accel_data: Accelerometer data (n_samples, 3)
        gyro_data: Gyroscope data (n_samples, 3) or None
        skeleton_data: Skeleton data (n_samples, num_joints*3)
        filter_type: Type of filter to calibrate ('standard', 'ekf', 'ukf')
        timestamps: Optional timestamps for data
        wrist_idx: Wrist joint index
        
    Returns:
        Tuple of (calibrated_filter, optimal_parameters)
    """
    from scipy.optimize import minimize
    
    # Extract orientation from skeleton for reference
    reference_orient = extract_orientation_from_skeleton(skeleton_data, wrist_idx=wrist_idx)
    
    # If no gyro data provided, use zeros
    if gyro_data is None:
        gyro_data = np.zeros_like(accel_data)
    
    # Define error function for optimization
    def error_function(params):
        # Unpack parameters
        process_noise, measurement_noise, gyro_bias_noise = params
        
        # Create filter with these parameters
        if filter_type == 'standard':
            test_filter = StandardKalmanIMU(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                gyro_bias_noise=gyro_bias_noise
            )
        elif filter_type == 'ekf':
            test_filter = ExtendedKalmanIMU(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                gyro_bias_noise=gyro_bias_noise
            )
        else:  # 'ukf'
            test_filter = UnscentedKalmanIMU(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                gyro_bias_noise=gyro_bias_noise
            )
        
        # Process a subset of data for efficiency
        sample_rate = 5  # Use every 5th point
        accel_subset = accel_data[::sample_rate]
        gyro_subset = gyro_data[::sample_rate]
        ts_subset = None if timestamps is None else timestamps[::sample_rate]
        ref_subset = reference_orient[::sample_rate]
        
        # Process data with filter
        output = test_filter.process_sequence(accel_subset, gyro_subset, ts_subset)
        
        # Extract Euler angles (last 3 columns)
        estimated_orient = output[:, -3:]
        
        # Calculate error (mean squared error)
        mse = np.mean(np.sum((estimated_orient - ref_subset)**2, axis=1))
        return mse
    
    # Initial parameter guess
    initial_params = np.array([0.01, 0.1, 0.01])
    
    # Parameter bounds
    bounds = [(0.001, 0.1), (0.01, 1.0), (0.001, 0.1)]
    
    # Minimize error
    try:
        result = minimize(
            error_function,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Create final calibrated filter
        process_noise, measurement_noise, gyro_bias_noise = result.x
        
        # Create calibrated filter
        if filter_type == 'standard':
            calibrated_filter = StandardKalmanIMU(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                gyro_bias_noise=gyro_bias_noise
            )
        elif filter_type == 'ekf':
            calibrated_filter = ExtendedKalmanIMU(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                gyro_bias_noise=gyro_bias_noise
            )
        else:  # 'ukf'
            calibrated_filter = UnscentedKalmanIMU(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                gyro_bias_noise=gyro_bias_noise
            )
        
        logger.info(f"Calibrated {filter_type} filter parameters: process_noise={process_noise:.6f}, "
                f"measurement_noise={measurement_noise:.6f}, gyro_bias_noise={gyro_bias_noise:.6f}")
        
        return calibrated_filter, result.x
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        # Return default parameters
        return None, np.array([0.01, 0.1, 0.01])
