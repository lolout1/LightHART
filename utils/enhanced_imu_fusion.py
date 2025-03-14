# utils/enhanced_imu_fusion.py

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time
import os
import logging
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedIMUFusion")

class EnhancedIMUFusionBase:
    """Base class for enhanced IMU fusion with optimized filtering."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3,
                 fall_specific_features=True):
        """
        Initialize IMU fusion base class with configurable parameters.
        
        Args:
            dt: Default time step in seconds (default: 1/30.0)
            process_noise: Process noise variance (default: 0.01)
            measurement_noise: Measurement noise variance (default: 0.1)
            gyro_bias_noise: Gyroscope bias noise variance (default: 0.01)
            drift_correction_weight: Weight for skeleton-based drift correction (default: 0.3)
            fall_specific_features: Enable extraction of fall-specific features (default: True)
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.drift_correction_weight = drift_correction_weight
        self.name = "Enhanced Base"
        self.calibrated = False
        
        # Reference data for drift correction
        self.reference_timestamps = None
        self.reference_orientations = None
        self.use_reference = False
        
        # Fall detection specific parameters
        self.fall_detection_features = fall_specific_features
        
        # Cache for computed features
        self.cache = {}
        
    def set_reference_data(self, timestamps: np.ndarray, orientations: np.ndarray):
        """
        Set reference orientation data from skeleton for drift correction.
        
        Args:
            timestamps: Array of reference timestamps
            orientations: Array of reference orientations (Euler angles)
        """
        if timestamps is None or orientations is None or len(timestamps) == 0:
            self.use_reference = False
            return
            
        if len(timestamps) != len(orientations):
            logger.warning(f"Reference data length mismatch: {len(timestamps)} timestamps vs {len(orientations)} orientations")
            self.use_reference = False
            return
            
        self.reference_timestamps = timestamps
        self.reference_orientations = orientations
        self.use_reference = True
        logger.info(f"Reference data set: {len(timestamps)} points")
        
    def get_reference_orientation(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get reference orientation at a specific timestamp using interpolation.
        
        Args:
            timestamp: Time point to get reference orientation
            
        Returns:
            Reference orientation or None if not available
        """
        if not self.use_reference:
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
            logger.warning(f"Reference interpolation error: {e}")
            return None
        
    def apply_drift_correction(self, estimated_orientation: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Apply drift correction using reference orientation when available.
        
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
    
    def extract_fall_features(self, accel: np.ndarray, gyro: np.ndarray, 
                             orientation: np.ndarray, timestamp: float = None,
                             prev_accel: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract fall-specific features for better fall detection.
        
        Args:
            accel: Current acceleration (3,)
            gyro: Current angular velocity (3,)
            orientation: Current orientation (3,) (Euler angles)
            timestamp: Current timestamp (optional)
            prev_accel: Previous acceleration for derivative features
            
        Returns:
            Dictionary of fall-specific features
        """
        if not self.fall_detection_features:
            return {}
            
        features = {}
        
        # 1. Vertical acceleration component (using orientation)
        try:
            r = R.from_euler('xyz', orientation)
            world_accel = r.apply(accel)
            features['vert_accel'] = world_accel[2]  # Vertical component
        except Exception as e:
            features['vert_accel'] = accel[2]  # Fallback to raw Z-axis
            
        # 2. Orientation features
        features['pitch'] = orientation[1]  # Pitch angle is critical for falls
        features['roll'] = orientation[0]   # Roll also important
        
        # 3. Impact detection features
        accel_mag = np.linalg.norm(accel)
        features['accel_magnitude'] = accel_mag
        
        # Threshold-based impact detection (tuned for falls)
        features['potential_impact'] = float(accel_mag > 19.6)  # >2g threshold
        
        # 4. Angular velocity features
        gyro_mag = np.linalg.norm(gyro)
        features['angular_velocity_mag'] = gyro_mag
        
        # 5. Jerk - rate of acceleration change (first derivative of acceleration)
        if prev_accel is not None:
            jerk = (accel - prev_accel) / self.dt
            jerk_mag = np.linalg.norm(jerk)
            features['jerk_magnitude'] = jerk_mag
            
            # High jerk indicates sudden movement (common in falls)
            features['high_jerk'] = float(jerk_mag > 100.0)  # Threshold tuned for falls
        
        # 6. Combined fall indicators
        if 'jerk_magnitude' in features:
            # Combine vertical acceleration and jerk for stronger fall indicator
            features['fall_indicator'] = (
                (features['vert_accel'] < -4.9) and  # Strong downward acceleration
                (features['jerk_magnitude'] > 100.0) and  # High jerk
                (gyro_mag > 1.5)  # Significant angular velocity
            )
        
        return features

    def calibrate_parameters(self, accel_data: np.ndarray, gyro_data: np.ndarray, 
                            reference_orientations: np.ndarray, timestamps: np.ndarray = None) -> Dict[str, float]:
        """
        Calibrate filter parameters using reference data (usually from skeleton).
        
        Args:
            accel_data: Accelerometer data (N, 3)
            gyro_data: Gyroscope data (N, 3)
            reference_orientations: Reference orientations (N, 3)
            timestamps: Optional timestamps (N,)
            
        Returns:
            Dictionary of optimized parameters
        """
        # Default implementation - override in specific filter classes
        logger.warning("Using default calibration method in base class")
        return {
            'process_noise': self.process_noise,
            'measurement_noise': self.measurement_noise,
            'gyro_bias_noise': self.gyro_bias_noise
        }
    
    def fuse_inertial_modalities(self, data_dict: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        """
        Fuse different inertial sensor data (watch, phone, etc.) into a single stream.
        
        Args:
            data_dict: Dictionary with various sensor data
            
        Returns:
            Updated dictionary with fused data
        """
        # This method should be called externally and not by individual filter classes
        result = data_dict.copy()
        
        # Check if we have both accelerometer and gyroscope data
        if 'accelerometer' not in data_dict or len(data_dict['accelerometer']) == 0:
            logger.warning("No accelerometer data available for fusion")
            return result
            
        if 'gyroscope' not in data_dict or len(data_dict['gyroscope']) == 0:
            logger.warning("No gyroscope data available for fusion")
            # Can still proceed with just accelerometer data
        
        # Get data windows
        accel_windows = data_dict['accelerometer']
        gyro_windows = data_dict.get('gyroscope', [])
        
        # Create fused windows
        fused_windows = []
        
        # Skip time column (assumed to be first column)
        for i in range(len(accel_windows)):
            accel_window = accel_windows[i]
            
            # Get corresponding gyro window if available
            gyro_window = None
            if i < len(gyro_windows):
                gyro_window = gyro_windows[i]
            
            # Process each window
            fused_window = self._fuse_window(accel_window, gyro_window)
            fused_windows.append(fused_window)
        
        # Add fused windows to result
        result['fused_imu'] = fused_windows
        
        return result
    
    def _fuse_window(self, accel_window: np.ndarray, gyro_window: Optional[np.ndarray]) -> np.ndarray:
        """
        Fuse accelerometer and gyroscope data for a single window.
        
        Args:
            accel_window: Accelerometer window with time in first column
            gyro_window: Gyroscope window with time in first column (or None)
            
        Returns:
            Fused data window
        """
        # Extract timestamps from accelerometer
        timestamps = accel_window[:, 0]
        accel_values = accel_window[:, 1:4]  # Assume XYZ in columns 1-3
        
        # Prepare gyro values (zeros if not available)
        if gyro_window is not None and gyro_window.shape[0] > 0:
            gyro_timestamps = gyro_window[:, 0]
            gyro_values = gyro_window[:, 1:4]
            
            # Check if timestamps match
            if gyro_timestamps.shape != timestamps.shape or not np.array_equal(gyro_timestamps, timestamps):
                # Interpolate gyro to match accel timestamps
                try:
                    gyro_interp = interp1d(
                        gyro_timestamps, 
                        gyro_values,
                        axis=0, 
                        bounds_error=False, 
                        fill_value="extrapolate"
                    )
                    gyro_values = gyro_interp(timestamps)
                except Exception as e:
                    logger.warning(f"Gyro interpolation failed: {e}")
                    # Fall back to zeros
                    gyro_values = np.zeros_like(accel_values)
        else:
            # No gyro data, use zeros
            gyro_values = np.zeros_like(accel_values)
        
        # Initialize filter if not already done
        if not hasattr(self, 'initialized') or not self.initialized:
            self.initialize(accel_values[0])
            self.initialized = True
        
        # Process through the filter
        fused_data = self.process_sequence(accel_values, gyro_values, timestamps)
        
        # Combine with original timestamps
        return np.column_stack([timestamps, fused_data])

class EnhancedStandardKalman(EnhancedIMUFusionBase):
    """
    Enhanced Standard Kalman Filter specialized for fall detection.
    Optimized for computational efficiency and better handling of linear motion.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.005, measurement_noise=0.08, 
                 gyro_bias_noise=0.008, drift_correction_weight=0.3,
                 fall_specific_features=True):
        """
        Initialize enhanced standard Kalman filter.
        
        Process noise, measurement noise, and gyro_bias_noise are tuned
        specifically for fall detection scenarios based on validation data.
        """
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, 
                        drift_correction_weight, fall_specific_features)
        
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
        
        self.name = "Enhanced Standard Kalman"
        self.prev_accel = None  # For computing jerk features
        
    def initialize(self, accel):
        """Initialize filter state using initial acceleration measurement."""
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
        self.prev_accel = accel
    
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
    
    def update(self, measurement, acceleration_magnitude):
        """
        Kalman filter update step with adaptive measurement noise.
        
        Args:
            measurement: Measured orientation
            acceleration_magnitude: Magnitude of acceleration vector
        """
        # Adaptive measurement noise based on acceleration
        # If acceleration is far from gravity, reduce its weight
        g_error = abs(acceleration_magnitude - 9.81)
        
        # Scale measurement noise based on difference from gravity
        adaptive_R = self.R.copy()
        
        # Increase uncertainty during high dynamics
        if g_error > 2.0:
            # Higher measurement noise during high acceleration
            adaptive_factor = min(10.0, 1.0 + g_error)
            adaptive_R = self.R * adaptive_factor
        
        # Innovation: y = z - Hx
        y = measurement - self.H @ self.x
        
        # Innovation covariance: S = HPH' + R
        S = self.H @ self.P @ self.H.T + adaptive_R
        
        # Kalman gain: K = PH'S^-1
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
            
            # Update state: x = x + Ky
            self.x = self.x + K @ y
            
            # Update covariance: P = (I-KH)P
            I = np.eye(self.state_dim)
            self.P = (I - K @ self.H) @ self.P
        except np.linalg.LinAlgError:
            # If matrix inversion fails, skip this update
            logger.warning("Matrix inversion failed in Kalman update step")
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """
        Process one step of IMU data.
        
        Args:
            accel: Acceleration measurement (3,)
            gyro: Angular velocity measurement (3,)
            dt: Time step
            timestamp: Current timestamp (for drift correction)
            
        Returns:
            Updated state estimate and features
        """
        if not self.initialized:
            self.initialize(accel)
        
        # Prediction step
        self.predict(dt, gyro)
        
        # Update step using accelerometer for roll and pitch
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:  # Avoid division by zero
            gravity = -accel / accel_norm
            roll_acc = np.arctan2(gravity[1], gravity[2])
            pitch_acc = np.arctan2(-gravity[0], np.sqrt(gravity[1]**2 + gravity[2]**2))
            
            # Only update roll and pitch from accelerometer
            acc_angles = np.array([roll_acc, pitch_acc, self.x[2]])
            
            # Adaptive update based on acceleration magnitude
            if abs(accel_norm - 9.81) < 3.0:  # Only trust accelerometer when close to gravity
                self.update(acc_angles, accel_norm)
        
        # Apply drift correction if reference data is available
        if timestamp is not None and self.use_reference:
            try:
                self.x[:3] = self.apply_drift_correction(self.x[:3], timestamp)
            except Exception as e:
                logger.warning(f"Drift correction failed: {e}")
        
        # Extract orientation
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
        
        # Extract fall-specific features
        fall_features = self.extract_fall_features(accel, gyro, orientation, timestamp, self.prev_accel)
        
        # Update previous acceleration for jerk calculation
        self.prev_accel = accel
        
        # Create combined feature vector
        # [accel(3), gyro(3), quaternion(4), euler(3)]
        feature_vector = np.concatenate([
            accel,                  # Original acceleration
            gyro,                   # Original gyro
            quat,                   # Quaternion orientation
            orientation             # Euler angles
        ])
        
        # Add fall-specific features if any
        if fall_features:
            # Extract numerical features in consistent order
            fall_feature_values = np.array([
                fall_features.get('vert_accel', 0),
                fall_features.get('accel_magnitude', 0),
                fall_features.get('jerk_magnitude', 0),
                fall_features.get('angular_velocity_mag', 0),
                fall_features.get('potential_impact', 0)
            ])
            
            # Append to feature vector
            feature_vector = np.concatenate([feature_vector, fall_feature_values])
        
        return self.x, feature_vector
    
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
        
        # Determine output feature dimension
        feature_dim = 13  # Standard: [accel(3), gyro(3), quat(4), euler(3)]
        if self.fall_detection_features:
            feature_dim += 5  # Additional fall-specific features
        
        # Create output array
        output = np.zeros((N, feature_dim))
        
        # Initialize filter if needed
        if not self.initialized:
            self.initialize(accel_data[0])
        
        # Process each measurement
        for i in range(N):
            accel = accel_data[i]
            gyro = gyro_data[i]
            
            # Compute dt if we have timestamps
            if i > 0:
                dt = timestamps[i] - timestamps[i-1]
                if dt <= 0 or dt > 0.5:  # Handle timestamp errors or large gaps
                    dt = self.dt
            else:
                dt = self.dt
            
            # Process step
            try:
                _, features = self.process_step(accel, gyro, dt, timestamps[i])
                output[i, :len(features)] = features
            except Exception as e:
                logger.warning(f"Error processing step {i}: {e}")
                if i > 0:
                    # Use previous output if available
                    output[i] = output[i-1]
        
        return output
    
    def calibrate_parameters(self, accel_data, gyro_data, reference_orientations, timestamps=None):
        """
        Calibrate filter parameters using reference data.
        
        Args:
            accel_data: Accelerometer data (N, 3)
            gyro_data: Gyroscope data (N, 3)
            reference_orientations: Reference orientations (N, 3)
            timestamps: Optional timestamps (N,)
            
        Returns:
            Dictionary of optimized parameters
        """
        def objective_function(params):
            # Unpack parameters
            p_noise, m_noise, g_noise = params
            
            # Create test filter
            test_filter = EnhancedStandardKalman(
                process_noise=p_noise,
                measurement_noise=m_noise,
                gyro_bias_noise=g_noise
            )
            
            # Process sequence with these parameters
            output = test_filter.process_sequence(accel_data, gyro_data, timestamps)
            
            # Extract Euler angles (last 3 columns of standard output)
            euler_angles = output[:, -3:]
            
            # Calculate error (MSE)
            min_len = min(euler_angles.shape[0], reference_orientations.shape[0])
            mse = np.mean(np.sum((euler_angles[:min_len] - reference_orientations[:min_len])**2, axis=1))
            
            return mse
        
        # Initial parameters
        initial_params = [self.process_noise, self.measurement_noise, self.gyro_bias_noise]
        
        # Parameter bounds
        bounds = [
            (0.001, 0.05),    # Process noise
            (0.01, 0.5),      # Measurement noise
            (0.001, 0.05)     # Gyro bias noise
        ]
        
        try:
            # Optimize parameters
            result = minimize(
                objective_function,
                initial_params,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Get optimized parameters
            optimized_params = {
                'process_noise': float(result.x[0]),
                'measurement_noise': float(result.x[1]),
                'gyro_bias_noise': float(result.x[2])
            }
            
            # Update filter parameters
            self.process_noise = optimized_params['process_noise']
            self.measurement_noise = optimized_params['measurement_noise']
            self.gyro_bias_noise = optimized_params['gyro_bias_noise']
            
            # Update measurement noise matrix
            self.R = np.eye(self.measurement_dim) * self.measurement_noise
            
            self.calibrated = True
            logger.info(f"Calibrated parameters: {optimized_params}")
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return {
                'process_noise': self.process_noise,
                'measurement_noise': self.measurement_noise,
                'gyro_bias_noise': self.gyro_bias_noise
            }

class EnhancedExtendedKalman(EnhancedIMUFusionBase):
    """
    Enhanced Extended Kalman Filter using quaternion representation.
    Better handles nonlinear motion and prevents gimbal lock.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.007, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3,
                 fall_specific_features=True):
        """
        Initialize enhanced EKF with parameters tuned for fall detection.
        """
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, 
                        drift_correction_weight, fall_specific_features)
        
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
        
        self.name = "Enhanced Extended Kalman"
        self.prev_accel = None  # For computing jerk features
    
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
            quat_scipy = r.as_quat()
            
            # Convert from scipy [x,y,z,w] to [w,x,y,z]
            quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        
        # Initialize state [quat, gyro_bias]
        self.x = np.zeros(self.state_dim)
        self.x[:4] = quat
        
        # Reset covariance with lower uncertainty for quaternion
        self.P = np.eye(self.state_dim) * 0.1
        self.P[:4, :4] *= 0.01  # Lower initial uncertainty for quaternion
        
        self.initialized = True
        self.prev_accel = accel
    
    def quaternion_update(self, q, omega, dt):
        """
        Update quaternion with angular velocity.
        
        Args:
            q: Current quaternion [w,x,y,z]
            omega: Angular velocity [wx,wy,wz]
            dt: Time step
            
        Returns:
            Updated quaternion
        """
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
        """
        Jacobian of state transition function.
        
        Args:
            x: Current state
            dt: Time step
            omega: Angular velocity
            
        Returns:
            Jacobian matrix
        """
        # This is a complex calculation for quaternion dynamics
        F = np.eye(self.state_dim)
        
        # Effect of gyro bias on quaternion
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-10:
            # Use numerical differentiation for Jacobian
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
        """
        Jacobian of measurement function.
        
        Args:
            x: Current state
            
        Returns:
            Jacobian matrix
        """
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
        """
        EKF prediction step.
        
        Args:
            dt: Time step
            gyro: Angular velocity
        """
        # State transition Jacobian
        F = self.state_transition_jacobian(self.x, dt, gyro)
        
        # Predict state
        self.x = self.state_transition_function(self.x, dt, gyro)
        
        # Predict covariance
        # Scale process noise with dt
        Q = self.Q.copy() * dt
        
        self.P = F @ self.P @ F.T + Q
        
        # Ensure quaternion is normalized
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
    
    def update(self, z, confidence_factor=1.0):
        """
        EKF update step with adaptive noise.
        
        Args:
            z: Measurement
            confidence_factor: Factor to scale measurement noise (higher = less confidence)
        """
        # Measurement Jacobian
        H = self.measurement_jacobian(self.x)
        
        # Predicted measurement
        z_pred = self.measurement_function(self.x)
        
        # Innovation
        y = z - z_pred
        
        # Adaptive measurement noise
        R_adaptive = self.R * confidence_factor
        
        # Innovation covariance
        S = H @ self.P @ H.T + R_adaptive
        
        try:
            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.x = self.x + K @ y
            
            # Update covariance
            I = np.eye(self.state_dim)
            self.P = (I - K @ H) @ self.P
            
            # Normalize quaternion
            self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        except np.linalg.LinAlgError:
            # Skip update if matrix inversion fails
            logger.warning("Matrix inversion failed in EKF update")
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """
        Process one step of IMU data.
        
        Args:
            accel: Acceleration measurement (3,)
            gyro: Angular velocity measurement (3,)
            dt: Time step
            timestamp: Current timestamp (for drift correction)
            
        Returns:
            Updated state estimate and features
        """
        if not self.initialized:
            self.initialize(accel)
        
        # Prediction step
        self.predict(dt, gyro)
        
        # Update step using accelerometer
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:  # Avoid division by zero
            gravity_norm = 9.81
            confidence_factor = 1.0
            
            # Determine confidence based on difference from gravity
            g_error = abs(accel_norm - gravity_norm)
            if g_error > 1.0:
                # Increase uncertainty during high dynamics
                confidence_factor = 1.0 + min(9.0, g_error)
            
            # Normalize acceleration for measurement
            z = -accel / accel_norm
            
            # Only update if acceleration is somewhat reasonable
            if g_error < 5.0:  # Wider threshold than standard KF
                self.update(z, confidence_factor)
        
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
                
                # Normalize
                self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
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
        
        # Extract fall-specific features
        fall_features = self.extract_fall_features(accel, gyro, euler, timestamp, self.prev_accel)
        
        # Update previous acceleration for jerk calculation
        self.prev_accel = accel
        
        # Create combined feature vector
        # [accel(3), gyro(3), quaternion(4), euler(3)]
        feature_vector = np.concatenate([
            accel,
            gyro,
            quat,
            euler
        ])
        
        # Add fall-specific features if any
        if fall_features:
            # Extract numerical features in consistent order
            fall_feature_values = np.array([
                fall_features.get('vert_accel', 0),
                fall_features.get('accel_magnitude', 0),
                fall_features.get('jerk_magnitude', 0),
                fall_features.get('angular_velocity_mag', 0),
                fall_features.get('potential_impact', 0)
            ])
            
            # Append to feature vector
            feature_vector = np.concatenate([feature_vector, fall_feature_values])
        
        return self.x, feature_vector
    
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
        
        # Determine output feature dimension
        feature_dim = 13  # Standard: [accel(3), gyro(3), quat(4), euler(3)]
        if self.fall_detection_features:
            feature_dim += 5  # Additional fall-specific features
        
        # Create output array
        output = np.zeros((N, feature_dim))
        
        # Initialize filter if needed
        if not self.initialized:
            self.initialize(accel_data[0])
        
        # Process each measurement
        for i in range(N):
            accel = accel_data[i]
            gyro = gyro_data[i]
            
            # Compute dt if we have timestamps
            if i > 0:
                dt = timestamps[i] - timestamps[i-1]
                if dt <= 0 or dt > 0.5:  # Handle timestamp errors or large gaps
                    dt = self.dt
            else:
                dt = self.dt
            
            # Process step
            try:
                _, features = self.process_step(accel, gyro, dt, timestamps[i])
                output[i, :len(features)] = features
            except Exception as e:
                logger.warning(f"Error processing step {i}: {e}")
                if i > 0:
                    # Use previous output if available
                    output[i] = output[i-1]
        
        return output
    
    def calibrate_parameters(self, accel_data, gyro_data, reference_orientations, timestamps=None):
        """
        Calibrate filter parameters using reference data.
        
        Args:
            accel_data: Accelerometer data (N, 3)
            gyro_data: Gyroscope data (N, 3)
            reference_orientations: Reference orientations (N, 3)
            timestamps: Optional timestamps (N,)
            
        Returns:
            Dictionary of optimized parameters
        """
        def objective_function(params):
            # Unpack parameters
            p_noise, m_noise, g_noise = params
            
            # Create test filter
            test_filter = EnhancedExtendedKalman(
                process_noise=p_noise,
                measurement_noise=m_noise,
                gyro_bias_noise=g_noise
            )
            
            # Process sequence with these parameters
            output = test_filter.process_sequence(accel_data, gyro_data, timestamps)
            
            # Extract Euler angles (columns -3, -2, -1)
            euler_angles = output[:, -3:]
            
            # Calculate error (MSE)
            min_len = min(euler_angles.shape[0], reference_orientations.shape[0])
            mse = np.mean(np.sum((euler_angles[:min_len] - reference_orientations[:min_len])**2, axis=1))
            
            return mse
        
        # Initial parameters
        initial_params = [self.process_noise, self.measurement_noise, self.gyro_bias_noise]
        
        # Parameter bounds
        bounds = [
            (0.001, 0.05),    # Process noise
            (0.01, 0.5),      # Measurement noise
            (0.001, 0.05)     # Gyro bias noise
        ]
        
        try:
            # Optimize parameters
            result = minimize(
                objective_function,
                initial_params,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Get optimized parameters
            optimized_params = {
                'process_noise': float(result.x[0]),
                'measurement_noise': float(result.x[1]),
                'gyro_bias_noise': float(result.x[2])
            }
            
            # Update filter parameters
            self.process_noise = optimized_params['process_noise']
            self.measurement_noise = optimized_params['measurement_noise']
            self.gyro_bias_noise = optimized_params['gyro_bias_noise']
            
            # Update matrices
            self.Q = np.eye(self.state_dim) * self.process_noise
            self.Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise
            self.R = np.eye(self.measurement_dim) * self.measurement_noise
            
            self.calibrated = True
            logger.info(f"Calibrated parameters: {optimized_params}")
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return {
                'process_noise': self.process_noise,
                'measurement_noise': self.measurement_noise,
                'gyro_bias_noise': self.gyro_bias_noise
            }

class EnhancedUnscentedKalman(EnhancedIMUFusionBase):
    """
    Enhanced Unscented Kalman Filter for IMU fusion.
    Most accurate but computationally expensive.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.008, measurement_noise=0.12, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3,
                 fall_specific_features=True):
        """
        Initialize enhanced UKF with parameters tuned for fall detection.
        """
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, 
                        drift_correction_weight, fall_specific_features)
        
        # State: [quaternion (4), gyro_bias (3)]
        self.state_dim = 7
        self.measurement_dim = 3
        
        # UKF parameters
        self.alpha = 1e-3  # Spread of sigma points
        self.beta = 2.0    # Prior knowledge of distribution (2 is optimal for Gaussian)
        self.kappa = 0.0   # Secondary scaling parameter
        
        # Initialize state
        self.x = np.zeros(self.state_dim)  # State
        self.x[0] = 1.0    # Initial quaternion w=1
        self.P = np.eye(self.state_dim) * 0.1  # Covariance
        
        # Process noise
        self.Q = np.eye(self.state_dim) * self.process_noise
        self.Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * self.measurement_noise
        
        # Save last gyro for state transition
        self.last_gyro = np.zeros(3)
        
        self.name = "Enhanced Unscented Kalman"
        self.prev_accel = None  # For computing jerk features
        
        # Performance optimization: cache sigma points
        self.cache_valid = False
        self.cached_sigma_points = None
        self.cached_weights = None
    
    def initialize(self, accel):
        """Initialize filter with first acceleration measurement."""
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
        
        # Reset covariance with lower uncertainty for quaternion
        self.P = np.eye(self.state_dim) * 0.1
        self.P[:4, :4] *= 0.01  # Lower initial uncertainty for quaternion
        
        self.initialized = True
        self.prev_accel = accel
        self.cache_valid = False
    
    def quaternion_update(self, q, omega, dt):
        """
        Update quaternion with angular velocity.
        
        Args:
            q: Current quaternion [w,x,y,z]
            omega: Angular velocity [wx,wy,wz] 
            dt: Time step
            
        Returns:
            Updated quaternion
        """
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
        """
        Generate sigma points using scaled unscented transform.
        
        Returns:
            Tuple of (sigma_points, lambda, n)
        """
        if self.cache_valid and self.cached_sigma_points is not None:
            return self.cached_sigma_points, self.cached_lambda, self.cached_n, self.cached_weights
            
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
        
        # Calculate weights
        wm = np.zeros(2*n + 1)  # Weights for mean
        wc = np.zeros(2*n + 1)  # Weights for covariance
        
        wm[0] = lambda_ / (n + lambda_)
        wc[0] = wm[0] + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2*n + 1):
            wm[i] = 1.0 / (2 * (n + lambda_))
            wc[i] = wm[i]
        
        # Cache results to avoid recomputing sigma points when state hasn't changed
        self.cached_sigma_points = sigma_points
        self.cached_lambda = lambda_
        self.cached_n = n
        self.cached_weights = (wm, wc)
        self.cache_valid = True
        
        return sigma_points, lambda_, n, (wm, wc)
    
    def state_transition(self, sigma_points, dt):
        """
        Apply state transition to sigma points.
        
        Args:
            sigma_points: Sigma points
            dt: Time step
            
        Returns:
            Transformed sigma points
        """
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
        """
        Apply measurement function to sigma points.
        
        Args:
            sigma_points: Sigma points
            
        Returns:
            Predicted measurements
        """
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
        """
        UKF prediction step.
        
        Args:
            dt: Time step
        """
        # Generate sigma points and weights
        sigma_points, lambda_, n, (wm, wc) = self.generate_sigma_points()
        
        # Transform sigma points
        transformed_points = self.state_transition(sigma_points, dt)
        
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
        
        # Cache is no longer valid
        self.cache_valid = False
    
    def update(self, measurement, confidence_factor=1.0):
        """
        UKF update step with adaptive trust.
        
        Args:
            measurement: Measurement
            confidence_factor: Factor to scale measurement noise (higher = less confidence)
        """
        # Generate sigma points
        sigma_points, lambda_, n, (wm, wc) = self.generate_sigma_points()
        
        # Transform sigma points through measurement function
        measurement_points = self.measurement_function(sigma_points)
        
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
        
        # Add adaptive measurement noise
        R_adaptive = self.R * confidence_factor
        Pzz += R_adaptive
        
        try:
            # Calculate Kalman gain
            K = Pxz @ np.linalg.inv(Pzz)
            
            # Update state
            innovation = measurement - z_pred
            self.x = self.x + K @ innovation
            
            # Normalize quaternion
            self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
            
            # Update covariance
            self.P = self.P - K @ Pzz @ K.T
            
            # Cache is no longer valid
            self.cache_valid = False
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed in UKF update")
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """
        Process one step of IMU data.
        
        Args:
            accel: Acceleration measurement (3,)
            gyro: Angular velocity measurement (3,)
            dt: Time step
            timestamp: Current timestamp (for drift correction)
            
        Returns:
            Updated state estimate and features
        """
        if not self.initialized:
            self.initialize(accel)
        
        # Save gyro for state transition
        self.last_gyro = gyro
        
        # Prediction step
        self.predict(dt)
        
        # Update step using accelerometer
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:  # Avoid division by zero
            gravity_norm = 9.81
            confidence_factor = 1.0
            
            # Determine confidence based on difference from gravity
            g_error = abs(accel_norm - gravity_norm)
            if g_error > 1.0:
                # Increase uncertainty during high dynamics
                confidence_factor = 1.0 + min(9.0, g_error)
            
            # Only update if not in extreme acceleration
            if g_error < 6.0:  # Even wider threshold than EKF
                # Normalize measurement
                z = -accel / accel_norm
                
                # Update with measurement
                self.update(z, confidence_factor)
        
        # Apply drift correction if reference data is available
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
                
                # Normalize
                self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
                
                # Cache is no longer valid
                self.cache_valid = False
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
        
        # Extract fall-specific features
        fall_features = self.extract_fall_features(accel, gyro, euler, timestamp, self.prev_accel)
        
        # Update previous acceleration for jerk calculation
        self.prev_accel = accel
        
        # Create combined feature vector
        # [accel(3), gyro(3), quaternion(4), euler(3)]
        feature_vector = np.concatenate([
            accel,
            gyro,
            quat,
            euler
        ])
        
        # Add fall-specific features if any
        if fall_features:
            # Extract numerical features in consistent order
            fall_feature_values = np.array([
                fall_features.get('vert_accel', 0),
                fall_features.get('accel_magnitude', 0),
                fall_features.get('jerk_magnitude', 0),
                fall_features.get('angular_velocity_mag', 0),
                fall_features.get('potential_impact', 0)
            ])
            
            # Append to feature vector
            feature_vector = np.concatenate([feature_vector, fall_feature_values])
        
        return self.x, feature_vector
    
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
        
        # Determine output feature dimension
        feature_dim = 13  # Standard: [accel(3), gyro(3), quat(4), euler(3)]
        if self.fall_detection_features:
            feature_dim += 5  # Additional fall-specific features
        
        # Create output array
        output = np.zeros((N, feature_dim))
        
        # Initialize filter if needed
        if not self.initialized:
            self.initialize(accel_data[0])
        
        # Process each measurement
        for i in range(N):
            accel = accel_data[i]
            gyro = gyro_data[i]
            
            # Compute dt if we have timestamps
            if i > 0:
                dt = timestamps[i] - timestamps[i-1]
                if dt <= 0 or dt > 0.5:  # Handle timestamp errors or large gaps
                    dt = self.dt
            else:
                dt = self.dt
            
            # Process step
            try:
                _, features = self.process_step(accel, gyro, dt, timestamps[i])
                output[i, :len(features)] = features
            except Exception as e:
                logger.warning(f"Error processing step {i}: {e}")
                if i > 0:
                    # Use previous output if available
                    output[i] = output[i-1]
        
        return output
    
    def calibrate_parameters(self, accel_data, gyro_data, reference_orientations, timestamps=None):
        """
        Calibrate filter parameters using reference data.
        
        Args:
            accel_data: Accelerometer data (N, 3)
            gyro_data: Gyroscope data (N, 3)
            reference_orientations: Reference orientations (N, 3)
            timestamps: Optional timestamps (N,)
            
        Returns:
            Dictionary of optimized parameters
        """
        # For UKF, we'll use a simpler grid search rather than full optimization
        # due to computational constraints
        process_noise_values = [0.005, 0.008, 0.01, 0.015]
        measurement_noise_values = [0.05, 0.1, 0.15, 0.2]
        gyro_bias_noise_values = [0.005, 0.01, 0.015, 0.02]
        
        best_mse = float('inf')
        best_params = None
        
        # Sample data to speed up calibration (use every 10th point)
        sample_rate = 10
        accel_sample = accel_data[::sample_rate]
        gyro_sample = gyro_data[::sample_rate]
        reference_sample = reference_orientations[::sample_rate]
        
        if timestamps is not None:
            time_sample = timestamps[::sample_rate]
        else:
            time_sample = None
        
        # Grid search
        for p_noise in process_noise_values:
            for m_noise in measurement_noise_values:
                for g_noise in gyro_bias_noise_values:
                    # Create test filter
                    test_filter = EnhancedUnscentedKalman(
                        process_noise=p_noise,
                        measurement_noise=m_noise,
                        gyro_bias_noise=g_noise
                    )
                    
                    # Process sequence with these parameters
                    try:
                        output = test_filter.process_sequence(accel_sample, gyro_sample, time_sample)
                        
                        # Extract Euler angles
                        euler_angles = output[:, -3:]
                        
                        # Calculate error (MSE)
                        min_len = min(euler_angles.shape[0], reference_sample.shape[0])
                        mse = np.mean(np.sum((euler_angles[:min_len] - reference_sample[:min_len])**2, axis=1))
                        
                        if mse < best_mse:
                            best_mse = mse
                            best_params = {
                                'process_noise': p_noise,
                                'measurement_noise': m_noise,
                                'gyro_bias_noise': g_noise
                            }
                    except Exception as e:
                        logger.warning(f"Error during UKF parameter testing: {e}")
        
        if best_params is not None:
            # Update filter parameters
            self.process_noise = best_params['process_noise']
            self.measurement_noise = best_params['measurement_noise']
            self.gyro_bias_noise = best_params['gyro_bias_noise']
            
            # Update matrices
            self.Q = np.eye(self.state_dim) * self.process_noise
            self.Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise
            self.R = np.eye(self.measurement_dim) * self.measurement_noise
            
            self.calibrated = True
            logger.info(f"Calibrated UKF parameters: {best_params}, MSE: {best_mse:.6f}")
            
            # Cache is no longer valid
            self.cache_valid = False
            
            return best_params
        else:
            logger.error("UKF calibration failed to find optimal parameters")
            return {
                'process_noise': self.process_noise,
                'measurement_noise': self.measurement_noise,
                'gyro_bias_noise': self.gyro_bias_noise
            }

def select_optimal_filter(accel_data, gyro_data, reference_orientations=None, timestamps=None):
    """
    Automatically select the optimal filter based on data characteristics.
    
    Args:
        accel_data: Accelerometer data (N, 3) 
        gyro_data: Gyroscope data (N, 3)
        reference_orientations: Optional reference orientations (N, 3)
        timestamps: Optional timestamps (N,)
        
    Returns:
        The most appropriate filter instance
    """
    # Calculate data characteristics
    if accel_data.size == 0:
        logger.warning("Empty accelerometer data, defaulting to Enhanced Standard Kalman")
        return EnhancedStandardKalman()
    
    # Check for gyro data
    has_gyro = gyro_data.size > 0
    
    # Check data dynamics
    accel_magnitude = np.linalg.norm(accel_data, axis=1)
    accel_variance = np.var(accel_magnitude)
    
    # Check if we have timestamps - variable sampling rate
    variable_sampling = False
    if timestamps is not None and len(timestamps) > 1:
        dt = np.diff(timestamps)
        dt_variance = np.var(dt)
        variable_sampling = dt_variance > 1e-6
    
    # Decision logic
    if not has_gyro:
        # Without gyro, standard Kalman is most reliable
        logger.info("No gyroscope data available, using Enhanced Standard Kalman")
        return EnhancedStandardKalman()
    
    if accel_variance > 15.0:
        # High dynamics - EKF or UKF are better
        if variable_sampling:
            # Variable sampling with high dynamics - UKF handles this best
            logger.info("High dynamics with variable sampling, using Enhanced Unscented Kalman")
            return EnhancedUnscentedKalman()
        else:
            # Constant sampling with high dynamics - EKF is good balance
            logger.info("High dynamics with constant sampling, using Enhanced Extended Kalman")
            return EnhancedExtendedKalman()
    else:
        # Low dynamics - Standard Kalman is efficient and accurate
        logger.info("Low dynamics detected, using Enhanced Standard Kalman")
        return EnhancedStandardKalman()

def fuse_inertial_modalities(data_dict, filter_type='auto', calibrated_params=None):
    """
    Fuse accelerometer and gyroscope data using the specified filter type.
    
    Args:
        data_dict: Dictionary with accelerometer and gyroscope data
        filter_type: 'standard', 'ekf', 'ukf', or 'auto'
        calibrated_params: Optional dictionary of calibrated parameters
        
    Returns:
        Updated dictionary with fused data
    """
    result = data_dict.copy()
    
    # Check if we have required data
    if 'accelerometer' not in data_dict or len(data_dict['accelerometer']) == 0:
        logger.warning("No accelerometer data available for fusion")
        return result
    
    has_gyro = 'gyroscope' in data_dict and len(data_dict['gyroscope']) > 0
    
    # Get data windows
    accel_windows = data_dict['accelerometer']
    gyro_windows = data_dict.get('gyroscope', [])
    
    # Choose filter type
    if filter_type == 'auto':
        # Sample a few windows to determine characteristics
        sample_accel = accel_windows[0][:, 1:4] if len(accel_windows) > 0 else np.zeros((0, 3))
        sample_gyro = gyro_windows[0][:, 1:4] if has_gyro and len(gyro_windows) > 0 else np.zeros((0, 3))
        sample_time = accel_windows[0][:, 0] if len(accel_windows) > 0 else None
        
        filter_obj = select_optimal_filter(sample_accel, sample_gyro, timestamps=sample_time)
    elif filter_type == 'standard':
        filter_obj = EnhancedStandardKalman()
    elif filter_type == 'ekf':
        filter_obj = EnhancedExtendedKalman()
    elif filter_type == 'ukf':
        filter_obj = EnhancedUnscentedKalman()
    else:
        logger.warning(f"Unknown filter type: {filter_type}, using standard Kalman")
        filter_obj = EnhancedStandardKalman()
    
    # Apply calibrated parameters if provided
    if calibrated_params is not None:
        filter_obj.process_noise = calibrated_params.get('process_noise', filter_obj.process_noise)
        filter_obj.measurement_noise = calibrated_params.get('measurement_noise', filter_obj.measurement_noise)
        filter_obj.gyro_bias_noise = calibrated_params.get('gyro_bias_noise', filter_obj.gyro_bias_noise)
        
        # Update filter matrices
        if hasattr(filter_obj, 'R'):
            filter_obj.R = np.eye(filter_obj.measurement_dim) * filter_obj.measurement_noise
        
        if hasattr(filter_obj, 'Q'):
            if isinstance(filter_obj, EnhancedStandardKalman):
                # Standard Kalman updates Q during predict()
                pass
            else:
                # EKF and UKF have persistent Q
                filter_obj.Q = np.eye(filter_obj.state_dim) * filter_obj.process_noise
                if hasattr(filter_obj, 'Q') and filter_obj.Q.shape[0] > 4:
                    filter_obj.Q[4:, 4:] = np.eye(3) * filter_obj.gyro_bias_noise
    
    # Create fused windows
    fused_windows = []
    
    # Process each window
    for i in range(len(accel_windows)):
        accel_window = accel_windows[i]
        
        # Get corresponding gyro window if available
        gyro_window = None
        if i < len(gyro_windows) and has_gyro:
            gyro_window = gyro_windows[i]
        
        # Extract timestamps and sensor values
        timestamps = accel_window[:, 0]
        accel_values = accel_window[:, 1:4]
        
        # Get gyro values (zeros if not available)
        if gyro_window is not None and gyro_window.shape[0] > 0:
            gyro_timestamps = gyro_window[:, 0]
            gyro_values = gyro_window[:, 1:4]
            
            # Check if timestamps match
            if not np.array_equal(timestamps, gyro_timestamps):
                # Interpolate gyro to match accel timestamps
                from scipy.interpolate import interp1d
                try:
                    gyro_interp = interp1d(
                        gyro_timestamps, 
                        gyro_values,
                        axis=0, 
                        bounds_error=False, 
                        fill_value="extrapolate"
                    )
                    gyro_values = gyro_interp(timestamps)
                except Exception as e:
                    logger.warning(f"Gyro interpolation failed: {e}")
                    gyro_values = np.zeros_like(accel_values)
        else:
            gyro_values = np.zeros_like(accel_values)
        
        # Process through filter
        try:
            # We need to reinitialize for each window since they're independent
            filter_obj.initialized = False
            fused_features = filter_obj.process_sequence(accel_values, gyro_values, timestamps)
            
            # Combine with original timestamps
            fused_window = np.column_stack([timestamps, fused_features])
            fused_windows.append(fused_window)
        except Exception as e:
            logger.error(f"Error processing window {i}: {e}")
            if len(fused_windows) > 0:
                # Use previous window as fallback
                fused_windows.append(fused_windows[-1])
            else:
                # Create minimal fallback window
                fallback = np.column_stack([
                    timestamps,
                    accel_values,
                    np.zeros_like(accel_values),  # Zero gyro
                    np.zeros((accel_values.shape[0], 4)),  # Identity quaternion [1,0,0,0]
                    np.zeros((accel_values.shape[0], 3))   # Zero Euler angles
                ])
                fused_windows.append(fallback)
    
    # Add fused windows to result
    result['fused_imu'] = fused_windows
    
    return result
