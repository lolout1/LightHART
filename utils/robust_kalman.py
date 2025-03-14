"""
robust_kalman.py

Robust implementations of Kalman filters for IMU sensor fusion:
- StandardKalmanIMU: Linear Kalman filter for simple orientation estimation
- ExtendedKalmanIMU: EKF for nonlinear quaternion-based orientation
- UnscentedKalmanIMU: UKF for highly nonlinear dynamics with better uncertainty representation

Features:
- Quaternion normalization safeguards
- Covariance regularization to ensure positive definiteness
- Drift correction using skeleton reference data
- Fall-specific feature extraction
- Graceful error handling and fallbacks
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logger = logging.getLogger("RobustKalman")

class BaseRobustKalmanIMU:
    """Base class for robust IMU fusion with shared functionality."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
        """
        Initialize base Kalman filter.
        
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
        self.name = "Base Robust Filter"
        self.initialized = False
        
        # Reference data for drift correction
        self.reference_timestamps = None
        self.reference_orientations = None
        self.use_reference = False
    
    def _ensure_positive_definite(self, matrix, epsilon=1e-8):
        """
        Ensure matrix is positive definite for numerical stability.
        
        Args:
            matrix: Input matrix to stabilize
            epsilon: Small positive value for eigenvalue floor
            
        Returns:
            Positive definite matrix
        """
        # Use symmetric eigenvalue decomposition
        S = (matrix + matrix.T) / 2  # Ensure symmetry
        eigvals, eigvecs = np.linalg.eigh(S)
        
        # Replace negative or very small eigenvalues with small positive values
        eigvals = np.maximum(eigvals, epsilon)
        
        # Reconstruct matrix
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
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
            from scipy.interpolate import interp1d
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
        features = {}
        
        try:
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
                
            # 5. Angular velocity magnitude
            gyro_mag = np.linalg.norm(gyro)
            features['gyro_magnitude'] = gyro_mag
            
        except Exception as e:
            logger.warning(f"Error extracting fall features: {e}")
        
        return features
    
    def initialize(self, accel):
        """Initialize filter state using initial acceleration."""
        raise NotImplementedError("Subclasses must implement initialize()")
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process a single step of sensor data."""
        raise NotImplementedError("Subclasses must implement process_step()")
    
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
            try:
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
                prev_accel = accel
                
            except Exception as e:
                logger.warning(f"Error processing step {i}: {e}")
                if i > 0:
                    # Use previous output if available
                    output[i] = output[i-1]
        
        return output

class RobustStandardKalmanIMU(BaseRobustKalmanIMU):
    """Enhanced Standard Kalman Filter with improved numerical stability."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
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
        
        self.name = "Robust Standard Kalman"
    
    def initialize(self, accel):
        """Initialize filter state using initial acceleration."""
        try:
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
            
        except Exception as e:
            logger.warning(f"Error initializing SKF: {e}")
            # Default initialization
            self.x = np.zeros(self.state_dim)
            self.P = np.eye(self.state_dim) * 0.1
            self.initialized = True
    
    def predict(self, dt, gyro):
        """Kalman filter prediction step."""
        try:
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
            
            # Ensure covariance is positive definite
            self.P = self._ensure_positive_definite(self.P)
            
        except Exception as e:
            logger.warning(f"Standard KF prediction error: {e}. Using identity update.")
            # Fallback to identity update with small noise
            self.P += np.eye(self.state_dim) * self.process_noise * dt
    
    def update(self, measurement):
        """Kalman filter update step."""
        try:
            # Innovation: y = z - Hx
            y = measurement - self.H @ self.x
            
            # Innovation covariance: S = HPH' + R
            S = self.H @ self.P @ self.H.T + self.R
            
            # Ensure S is positive definite for inversion
            S = self._ensure_positive_definite(S)
            
            # Kalman gain: K = PH'S^-1
            K = self.P @ self.H.T @ np.linalg.inv(S)
            
            # Update state: x = x + Ky
            self.x = self.x + K @ y
            
            # Joseph form for covariance update (more stable)
            I = np.eye(self.state_dim)
            self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
            
            # Ensure covariance remains positive definite
            self.P = self._ensure_positive_definite(self.P)
            
        except Exception as e:
            logger.warning(f"Standard KF update error: {e}. Skipping update.")
    
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
            # Convert scipy [x,y,z,w] to [w,x,y,z]
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
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

class RobustExtendedKalmanIMU(BaseRobustKalmanIMU):
    """Enhanced Extended Kalman Filter with robust quaternion handling."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
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
        
        self.name = "Robust Extended Kalman"
    
    def initialize(self, accel):
        """Initialize filter with first acceleration measurement."""
        try:
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
                quat_scipy = r.as_quat()
                
                # Convert from scipy [x,y,z,w] to [w,x,y,z]
                quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
            
            # Initialize state [quat, gyro_bias]
            self.x = np.zeros(self.state_dim)
            self.x[:4] = quat
            
            # Initialize covariance
            self.P = np.eye(self.state_dim) * 0.1
            
            self.initialized = True
            
        except Exception as e:
            logger.warning(f"Error initializing EKF: {e}")
            # Default initialization
            self.x = np.zeros(self.state_dim)
            self.x[0] = 1.0  # Identity quaternion
            self.P = np.eye(self.state_dim) * 0.1
            self.initialized = True
    
    def quaternion_update(self, q, omega, dt):
        """Update quaternion with angular velocity."""
        try:
            # Check for near-zero quaternion
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-10:
                logger.warning("Near-zero quaternion in EKF. Resetting to identity.")
                return np.array([1.0, 0.0, 0.0, 0.0])
                
            # Normalize quaternion
            q = q / q_norm
                
            # Original quaternion update code with safeguards
            omega_norm = np.linalg.norm(omega)
            
            if omega_norm < 1e-10:
                # For very small rotations, return normalized input quaternion
                return q
            
            # Standard quaternion update with rotation
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
            
            # Robust normalization
            result_norm = np.linalg.norm(result)
            if result_norm < 1e-10:
                logger.warning("EKF quaternion update resulted in near-zero norm. Using identity.")
                return np.array([1.0, 0.0, 0.0, 0.0])
                
            return result / result_norm
            
        except Exception as e:
            logger.warning(f"Quaternion update error: {e}")
            return np.array([1.0, 0.0, 0.0, 0.0])  # Return identity quaternion
    
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
        try:
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
            
        except Exception as e:
            logger.warning(f"State transition error: {e}")
            # Return copy of input state as fallback
            return x.copy()
    
    def state_transition_jacobian(self, x, dt, omega):
        """Jacobian of state transition function."""
        try:
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
            
        except Exception as e:
            logger.warning(f"Jacobian calculation error: {e}")
            # Return identity matrix as fallback
            return np.eye(self.state_dim)
    
    def measurement_function(self, x):
        """
        Measurement function for EKF.
        
        Converts quaternion to expected accelerometer measurement
        (assuming only gravity when stationary).
        """
        try:
            # Extract quaternion
            q = x[:4]
            
            # Check quaternion norm
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-10:
                logger.warning("Measurement function: zero norm quaternion. Using identity.")
                q = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                q = q / q_norm
            
            # Convert our quaternion [w,x,y,z] to scipy [x,y,z,w]
            q_scipy = np.array([q[1], q[2], q[3], q[0]])
            
            # Convert quaternion to rotation matrix
            r = R.from_quat(q_scipy)
            
            # Rotate unit gravity vector [0,0,1]
            gravity_body = r.apply([0, 0, 1])
            
            return gravity_body
            
        except Exception as e:
            logger.warning(f"EKF measurement function error: {e}")
            return np.array([0.0, 0.0, 1.0])  # Default to gravity along z-axis
    
    def measurement_jacobian(self, x):
        """Jacobian of measurement function."""
        try:
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
            
        except Exception as e:
            logger.warning(f"Measurement Jacobian error: {e}")
            # Return zeros as fallback
            H = np.zeros((self.measurement_dim, self.state_dim))
            H[:3, :3] = np.eye(3)  # Simple approximation
            return H
    
    def predict(self, dt, gyro):
        """EKF prediction step."""
        try:
            # State transition Jacobian
            F = self.state_transition_jacobian(self.x, dt, gyro)
            
            # Predict state
            self.x = self.state_transition_function(self.x, dt, gyro)
            
            # Predict covariance
            # Scale process noise with dt
            Q = self.Q.copy() * dt
            
            # Robust covariance update
            self.P = F @ self.P @ F.T + Q
            
            # Ensure covariance stays positive definite
            self.P = self._ensure_positive_definite(self.P)
            
            # Normalize quaternion part
            norm = np.linalg.norm(self.x[:4])
            if norm > 1e-10:
                self.x[:4] /= norm
            else:
                logger.warning("EKF state quaternion has near-zero norm. Resetting to identity.")
                self.x[:4] = np.array([1.0, 0.0, 0.0, 0.0])
                
        except Exception as e:
            logger.warning(f"EKF prediction error: {e}. Using simplified update.")
            # Simplified fallback
            if np.linalg.norm(self.x[:4]) < 1e-10:
                self.x[:4] = np.array([1.0, 0.0, 0.0, 0.0])
            self.P += np.eye(self.state_dim) * self.process_noise * dt
    
    def update(self, z):
        """EKF update step."""
        try:
            # Original update code
            H = self.measurement_jacobian(self.x)
            z_pred = self.measurement_function(self.x)
            y = z - z_pred
            
            S = H @ self.P @ H.T + self.R
            
            # Ensure innovation covariance is positive definite
            S = self._ensure_positive_definite(S)
            
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.x = self.x + K @ y
            
            # Joseph form for covariance update (more numerically stable)
            I = np.eye(self.state_dim)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
            
            # Normalize quaternion part
            norm = np.linalg.norm(self.x[:4])
            if norm > 1e-10:
                self.x[:4] /= norm
            else:
                logger.warning("EKF state quaternion has near-zero norm after update. Resetting.")
                self.x[:4] = np.array([1.0, 0.0, 0.0, 0.0])
                
        except Exception as e:
            logger.warning(f"EKF update error: {e}. Skipping update.")
    
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
                
                # Check quaternion norm
                if np.linalg.norm(q_scipy) < 1e-10:
                    logger.warning("Zero norm quaternion during Euler conversion")
                    euler = np.zeros(3)
                else:
                    # Normalize quaternion
                    q_scipy = q_scipy / np.linalg.norm(q_scipy)
                    euler = R.from_quat(q_scipy).as_euler('xyz')
                
                # Correct Euler angles
                corrected_euler = self.apply_drift_correction(euler, timestamp)
                
                # Convert back to quaternion
                q_corrected = R.from_euler('xyz', corrected_euler).as_quat()
                
                # Convert scipy [x,y,z,w] back to [w,x,y,z]
                self.x[:4] = [q_corrected[3], q_corrected[0], q_corrected[1], q_corrected[2]]
                
                # Ensure quaternion is normalized
                self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
                
            except Exception as e:
                logger.warning(f"Drift correction error: {e}")
        
        # Extract orientation for output
        quat = self.x[:4].copy()
        
        # Convert quaternion [w,x,y,z] to scipy [x,y,z,w]
        q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        
        try:
            # Check quaternion norm
            q_norm = np.linalg.norm(q_scipy)
            if q_norm < 1e-10:
                logger.warning("Zero norm quaternion during Euler conversion in output")
                euler = np.zeros(3)
            else:
                # Normalize quaternion
                q_scipy = q_scipy / q_norm
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

class RobustUnscentedKalmanIMU(BaseRobustKalmanIMU):
    """Enhanced Unscented Kalman Filter with robust matrix operations."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, drift_correction_weight)
        
        # State: [quaternion (4), gyro_bias (3)]
        self.state_dim = 7
        self.measurement_dim = 3
        
        # UKF parameters
        self.alpha = 1e-2  # Larger alpha for more stable sigma points
        self.beta = 2.0    # Optimal for Gaussian
        self.kappa = 0.0   # Secondary scaling parameter
        
        # Initialize state
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initial quaternion w=1 (identity)
        self.P = np.eye(self.state_dim) * 0.1
        
        # Process noise
        self.Q = np.eye(self.state_dim) * self.process_noise
        self.Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * self.measurement_noise
        
        # Save last gyro for state transition
        self.last_gyro = np.zeros(3)
        
        self.name = "Robust Unscented Kalman"
    
    def initialize(self, accel):
        """Initialize filter with first acceleration."""
        try:
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
            
        except Exception as e:
            logger.warning(f"Error initializing UKF: {e}")
            # Default initialization
            self.x = np.zeros(self.state_dim)
            self.x[0] = 1.0  # Identity quaternion
            self.P = np.eye(self.state_dim) * 0.1
            self.initialized = True
    
    def quaternion_update(self, q, omega, dt):
        """Update quaternion with angular velocity."""
        try:
            # Check quaternion norm
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-10:
                return np.array([1.0, 0.0, 0.0, 0.0])
            
            # Normalize quaternion
            q = q / q_norm
            
            omega_norm = np.linalg.norm(omega)
            
            if omega_norm < 1e-10:
                return q
            
            # Quaternion for this rotation
            axis = omega / omega_norm
            angle = omega_norm * dt
            
            quat_delta = np.zeros(4)
            quat_delta[0] = np.cos(angle/2)
            quat_delta[1:] = axis * np.sin(angle/2)
            
            # Quaternion multiplication
            result = np.zeros(4)
            result[0] = q[0]*quat_delta[0] - np.dot(q[1:], quat_delta[1:])
            result[1:] = q[0]*quat_delta[1:] + quat_delta[0]*q[1:] + np.cross(q[1:], quat_delta[1:])
            
            # Normalize
            result_norm = np.linalg.norm(result)
            if result_norm < 1e-10:
                return np.array([1.0, 0.0, 0.0, 0.0])
                
            return result / result_norm
            
        except Exception as e:
            logger.warning(f"Quaternion update error: {e}")
            return np.array([1.0, 0.0, 0.0, 0.0])  # Return identity quaternion
    
    def generate_sigma_points(self):
        """Generate sigma points with robust matrix handling."""
        try:
            n = self.state_dim
            lambda_ = self.alpha**2 * (n + self.kappa) - n
            
            # Ensure covariance is positive definite
            P_regularized = self._ensure_positive_definite(self.P)
            
            # Calculate square root of P
            try:
                # Try Cholesky decomposition first
                L = np.linalg.cholesky((n + lambda_) * P_regularized)
            except np.linalg.LinAlgError:
                # Fall back to eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(P_regularized)
                
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-8)
                
                # Compute square root via eigendecomposition
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
                    if q_norm > 1e-10:
                        sigma_points[j, :4] /= q_norm
                    else:
                        sigma_points[j, :4] = np.array([1.0, 0.0, 0.0, 0.0])
            
            return sigma_points, lambda_, n
            
        except Exception as e:
            logger.warning(f"Error generating sigma points: {e}")
            # Return minimal set of points
            sigma_points = np.zeros((3, self.state_dim))
            sigma_points[0] = self.x
            sigma_points[1] = self.x.copy()
            sigma_points[1][:4] = np.array([1.0, 0.0, 0.0, 0.0])
            sigma_points[2] = self.x.copy() 
            sigma_points[2][4:] = np.zeros(3)
            return sigma_points, 0, self.state_dim
    
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
        try:
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
            
        except Exception as e:
            logger.warning(f"Error in state transition: {e}")
            # Return input points as fallback
            return sigma_points.copy()
    
    def measurement_function(self, sigma_points):
        """Apply measurement function to sigma points."""
        try:
            n_points = sigma_points.shape[0]
            measurements = np.zeros((n_points, self.measurement_dim))
            
            for i in range(n_points):
                # Extract quaternion
                q = sigma_points[i, :4]
                
                # Check quaternion norm
                q_norm = np.linalg.norm(q)
                if q_norm < 1e-10:
                    logger.debug("Zero norm quaternion in measurement function")
                    q = np.array([1.0, 0.0, 0.0, 0.0])
                else:
                    q = q / q_norm
                
                # Convert [w,x,y,z] to scipy [x,y,z,w]
                q_scipy = np.array([q[1], q[2], q[3], q[0]])
                
                # Rotate unit gravity vector
                r = R.from_quat(q_scipy)
                gravity_body = r.apply([0, 0, 1])
                measurements[i] = gravity_body
            
            return measurements
            
        except Exception as e:
            logger.warning(f"Error in measurement function: {e}")
            # Return gravity along z as fallback
            measurements = np.zeros((sigma_points.shape[0], self.measurement_dim))
            measurements[:, 2] = 1.0  # z-direction gravity
            return measurements
    
    def predict(self, dt):
        """UKF prediction step."""
        try:
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
            q_norm = np.linalg.norm(self.x[:4])
            if q_norm > 1e-10:
                self.x[:4] /= q_norm
            else:
                logger.warning("UKF state quaternion has near-zero norm. Resetting to identity.")
                self.x[:4] = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Calculate predicted covariance with robust handling
            self.P = np.zeros_like(self.P)
            for i in range(len(wc)):
                diff = transformed_points[i] - self.x
                
                # Handle quaternion differences properly
                if np.dot(diff[:4], self.x[:4]) < 0:
                    diff[:4] = -diff[:4]  # Flip quaternion if on opposite hemisphere
                
                self.P += wc[i] * np.outer(diff, diff)
            
            # Add process noise
            self.P += self.Q * dt
            
            # Ensure covariance remains positive definite
            self.P = self._ensure_positive_definite(self.P)
            
        except Exception as e:
            logger.warning(f"UKF prediction error: {e}, using simplified update")
            # Fallback to simple prediction
            self.P += np.eye(self.state_dim) * self.process_noise * dt * 10  # Extra noise for stability
    
    def update(self, measurement):
        """UKF update step."""
        try:
            # Generate sigma points
            sigma_points, lambda_, n = self.generate_sigma_points()
            
            # Transform sigma points through measurement function
            measurement_points = self.measurement_function(sigma_points)
            
            # Calculate weights
            wm, wc = self.calculate_weights(lambda_, n)
            
            # Calculate predicted measurement with proper handling of failed points
            z_pred = np.zeros(self.measurement_dim)
            valid_points = 0
            
            for i in range(len(wm)):
                if not np.any(np.isnan(measurement_points[i])):
                    z_pred += wm[i] * measurement_points[i]
                    valid_points += 1
            
            if valid_points == 0:
                logger.warning("UKF update: All measurement points invalid. Skipping update.")
                return
                
            # Normalize if needed
            if valid_points < len(wm):
                z_pred /= np.sum(wm[:valid_points])
            
            # Calculate innovation covariance and cross-correlation with robust handling
            Pzz = np.zeros((self.measurement_dim, self.measurement_dim))
            Pxz = np.zeros((self.state_dim, self.measurement_dim))
            
            valid_points = 0
            for i in range(len(wc)):
                if not np.any(np.isnan(measurement_points[i])):
                    z_diff = measurement_points[i] - z_pred
                    x_diff = sigma_points[i] - self.x
                    
                    # Handle quaternion differences properly
                    if np.dot(x_diff[:4], self.x[:4]) < 0:
                        x_diff[:4] = -x_diff[:4]
                    
                    Pzz += wc[i] * np.outer(z_diff, z_diff)
                    Pxz += wc[i] * np.outer(x_diff, z_diff)
                    valid_points += 1
            
            if valid_points == 0:
                logger.warning("UKF update: No valid measurements for covariance. Skipping update.")
                return
                
            # Add measurement noise
            Pzz += self.R
            
            # Ensure innovation covariance is positive definite
            Pzz = self._ensure_positive_definite(Pzz)
            
            # Calculate Kalman gain
            try:
                K = Pxz @ np.linalg.inv(Pzz)
            except np.linalg.LinAlgError as e:
                logger.warning(f"UKF update: Matrix inversion failed: {e}. Skipping update.")
                return
            
            # Update state
            innovation = measurement - z_pred
            self.x = self.x + K @ innovation
            
            # Normalize quaternion
            q_norm = np.linalg.norm(self.x[:4])
            if q_norm > 1e-10:
                self.x[:4] /= q_norm
            else:
                logger.warning("UKF state quaternion has near-zero norm after update. Resetting.")
                self.x[:4] = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Update covariance
            self.P = self.P - K @ Pzz @ K.T
            
            # Ensure positive definiteness
            self.P = self._ensure_positive_definite(self.P)
            
        except Exception as e:
            logger.warning(f"UKF update error: {e}. Skipping update.")
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process one step with UKF."""
        if not self.initialized:
            self.initialize(accel)
        
        # Save gyro for state transition
        self.last_gyro = gyro.copy()  # Create a copy to avoid reference issues
        
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
                
                # Check quaternion norm
                q_norm = np.linalg.norm(q_scipy)
                if q_norm < 1e-10:
                    logger.warning("Zero norm quaternion during Euler conversion. Using zeros.")
                    euler = np.zeros(3)
                else:
                    # Normalize quaternion
                    q_scipy = q_scipy / q_norm
                    euler = R.from_euler('xyz', q_scipy).as_euler('xyz')
                
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
                
                # Ensure quaternion is normalized
                q_norm = np.linalg.norm(self.x[:4])
                if q_norm > 1e-10:
                    self.x[:4] /= q_norm
                else:
                    logger.warning("Zero norm quaternion after drift correction. Using identity.")
                    self.x[:4] = np.array([1.0, 0.0, 0.0, 0.0])
                    
            except Exception as e:
                logger.warning(f"UKF drift correction error: {e}")
        
        # Extract orientation
        quat = self.x[:4].copy()
        
        # Convert to scipy format for euler angles
        q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        
        try:
            # Check for valid quaternion
            q_norm = np.linalg.norm(q_scipy)
            if q_norm < 1e-10:
                logger.warning("UKF zero norm quaternion during Euler conversion. Using zeros.")
                euler = np.zeros(3)
            else:
                # Normalize and convert
                q_scipy = q_scipy / q_norm
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
