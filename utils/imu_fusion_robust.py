# utils/imu_fusion_robust.py

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import logging
import math

logger = logging.getLogger("IMUFusion")

class RobustIMUFilterBase:
    """Base class for robust IMU filters with quaternion stabilization."""
    
    def __init__(self, dt=1/50.0, process_noise=0.01, measurement_noise=0.1, 
                gyro_bias_noise=0.01, drift_correction_weight=0.05):
        """
        Initialize the robust IMU filter.
        
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
        
        # Initialize state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # Unit quaternion [w, x, y, z]
        self.gyro_bias = np.zeros(3)
        
        # Reference data for drift correction
        self.reference_orientations = None
        self.reference_timestamps = None
        self.use_reference = False
        
        # Initialize derived filter parameters
        self._initialize_filter()
    
    def _initialize_filter(self):
        """Initialize filter-specific parameters. To be implemented by subclasses."""
        raise NotImplementedError
    
    def set_reference_data(self, timestamps, orientations):
        """
        Set reference orientation data from skeleton.
        
        Args:
            timestamps: Array of reference timestamps
            orientations: Array of reference orientations (euler angles)
        """
        if timestamps is None or orientations is None or len(timestamps) == 0 or len(orientations) == 0:
            self.use_reference = False
            return
        
        # Ensure equal lengths
        min_len = min(len(timestamps), len(orientations))
        self.reference_timestamps = timestamps[:min_len]
        self.reference_orientations = orientations[:min_len]
        self.use_reference = True
        logger.info(f"Reference data set with {min_len} points")
    
    def normalize_quaternion(self, quat):
        """
        Safely normalize a quaternion, handling near-zero cases.
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            Normalized quaternion
        """
        norm = np.linalg.norm(quat)
        
        if norm < 1e-10:
            # Reinitialize to identity quaternion if normalization would fail
            logger.warning("Near-zero quaternion detected, resetting to identity")
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        return quat / norm
    
    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions.
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            
        Returns:
            Product quaternion
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([w, x, y, z])
    
    def quaternion_from_euler(self, euler_angles):
        """
        Convert Euler angles to quaternion.
        
        Args:
            euler_angles: Euler angles [roll, pitch, yaw] in radians
            
        Returns:
            Quaternion [w, x, y, z]
        """
        try:
            r = R.from_euler('xyz', euler_angles)
            q = r.as_quat()  # [x, y, z, w] format
            return np.array([q[3], q[0], q[1], q[2]])  # Convert to [w, x, y, z]
        except Exception as e:
            logger.warning(f"Euler to quaternion conversion failed: {e}")
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    def quaternion_to_euler(self, quat):
        """
        Convert quaternion to Euler angles with safe handling.
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        try:
            # Ensure normalized
            quat = self.normalize_quaternion(quat)
            
            # Convert to scipy format [x, y, z, w]
            q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
            
            # Create rotation object
            r = R.from_quat(q_scipy)
            
            # Get Euler angles
            euler = r.as_euler('xyz')
            return euler
        except Exception as e:
            logger.warning(f"Quaternion to Euler conversion failed: {e}")
            return np.zeros(3)  # Default to zero angles
    
    def get_reference_orientation(self, timestamp):
        """
        Get reference orientation for a specific timestamp.
        
        Args:
            timestamp: Time point to get reference orientation
            
        Returns:
            Reference Euler angles or None if not available
        """
        if not self.use_reference:
            return None
            
        # Check if timestamp is within range
        if (timestamp < self.reference_timestamps[0] or 
            timestamp > self.reference_timestamps[-1]):
            return None
            
        # Find closest timestamp
        idx = np.argmin(np.abs(self.reference_timestamps - timestamp))
        return self.reference_orientations[idx]
    
    def apply_drift_correction(self, quat, timestamp):
        """
        Apply drift correction using reference orientation when available.
        
        Args:
            quat: Current quaternion orientation
            timestamp: Current timestamp
            
        Returns:
            Corrected quaternion
        """
        if not self.use_reference:
            return quat
            
        # Get reference orientation
        ref_euler = self.get_reference_orientation(timestamp)
        if ref_euler is None:
            return quat
            
        # Convert reference Euler to quaternion
        ref_quat = self.quaternion_from_euler(ref_euler)
        
        # Apply weighted correction (SLERP)
        alpha = self.drift_correction_weight
        
        # Calculate dot product between quaternions
        dot = np.clip(np.sum(quat * ref_quat), -1.0, 1.0)
        
        # If quaternions are nearly opposite, negate one
        if dot < 0:
            ref_quat = -ref_quat
            dot = -dot
        
        # If quaternions are very close, just do linear interpolation
        if dot > 0.9995:
            result = quat + alpha * (ref_quat - quat)
            return self.normalize_quaternion(result)
        
        # Otherwise do proper spherical interpolation
        theta_0 = np.arccos(dot)
        theta = theta_0 * alpha
        
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        
        return self.normalize_quaternion(s1 * quat + s2 * ref_quat)
    
    def estimate_gravity(self, quat):
        """
        Estimate gravity direction in sensor frame.
        
        Args:
            quat: Orientation quaternion [w, x, y, z]
            
        Returns:
            Gravity vector in sensor frame [x, y, z]
        """
        try:
            # Ensure normalized quaternion
            quat = self.normalize_quaternion(quat)
            
            # Gravity in world frame (assuming +z is up)
            g_world = np.array([0, 0, 1])
            
            # Convert quaternion to scipy format [x, y, z, w]
            q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
            
            # Create rotation object
            r = R.from_quat(q_scipy)
            
            # Rotate gravity to sensor frame (inverse rotation)
            g_sensor = r.inv().apply(g_world)
            
            return g_sensor
        except Exception as e:
            logger.warning(f"Gravity estimation error: {e}")
            return np.array([0, 0, 1])  # Default to +z
    
    def process_step(self, accel, gyro, timestamp):
        """
        Process a single timestep of sensor data.
        
        Args:
            accel: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z]
            timestamp: Current timestamp
            
        Returns:
            Dictionary with current state
        """
        raise NotImplementedError
    
    def process_sequence(self, accel_data, gyro_data, timestamps=None):
        """
        Process a sequence of accelerometer and gyroscope readings.
        
        Args:
            accel_data: Accelerometer readings (N, 3)
            gyro_data: Gyroscope readings (N, 3)
            timestamps: Optional timestamps (N,)
            
        Returns:
            Array of shape (N, 16) containing:
                - Linear acceleration (3)
                - Angular velocity (3)
                - Quaternion orientation (4) 
                - Euler angles (3)
                - Gravity direction (3)
        """
        n_samples = accel_data.shape[0]
        
        # Create output array
        # [accel (3), gyro (3), quaternion (4), euler (3), gravity (3)]
        output = np.zeros((n_samples, 16))
        
        # Reset state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.gyro_bias = np.zeros(3)
        
        # If timestamps not provided, create evenly spaced ones
        if timestamps is None:
            timestamps = np.arange(n_samples) * self.dt
        
        # Check for length mismatches and fix
        gyro_len = len(gyro_data)
        accel_len = len(accel_data)
        timestamp_len = len(timestamps)
        
        if gyro_len != accel_len or gyro_len != timestamp_len:
            logger.warning(f"Length mismatch: accel={accel_len}, gyro={gyro_len}, timestamps={timestamp_len}")
            
            # Use the shortest length
            min_len = min(accel_len, gyro_len, timestamp_len)
            accel_data = accel_data[:min_len]
            gyro_data = gyro_data[:min_len]
            timestamps = timestamps[:min_len]
            
            # Update n_samples
            n_samples = min_len
            
            # Resize output array
            output = np.zeros((n_samples, 16))
        
        # Process each sample
        previous_quat = None
        
        for i in range(n_samples):
            # Get current sensor readings
            accel = accel_data[i].copy()
            
            # Safely handle gyro data which might be zeros
            if i < gyro_len:
                gyro = gyro_data[i].copy()
            else:
                gyro = np.zeros(3)
            
            timestamp = timestamps[i]
            
            # Store previous quaternion for recovery
            previous_quat = self.quat.copy() if previous_quat is None else self.quat.copy()
            
            # Process the step
            try:
                state = self.process_step(accel, gyro, timestamp)
                
                # Store results in output array
                output[i, 0:3] = accel  # Accelerometer
                output[i, 3:6] = gyro   # Gyroscope
                output[i, 6:10] = self.quat  # Quaternion
                
                # Get Euler angles
                euler = self.quaternion_to_euler(self.quat)
                output[i, 10:13] = euler
                
                # Estimate gravity
                gravity = self.estimate_gravity(self.quat)
                output[i, 13:16] = gravity
                
            except Exception as e:
                logger.warning(f"Error processing step {i}: {e}")
                
                # Recover using previous quaternion
                self.quat = previous_quat
                
                # Still fill output with best available data
                output[i, 0:3] = accel
                output[i, 3:6] = gyro
                output[i, 6:10] = self.quat
                
                # Use previously computed values for orientation
                if i > 0:
                    output[i, 10:16] = output[i-1, 10:16]
                else:
                    # First step failed, use defaults
                    output[i, 10:13] = np.zeros(3)  # Zero Euler angles
                    output[i, 13:16] = np.array([0, 0, 1])  # Default gravity
        
        return output


class RobustStandardKalmanIMU(RobustIMUFilterBase):
    """Standard Kalman filter with robust quaternion handling."""
    
    def _initialize_filter(self):
        """Initialize filter-specific parameters."""
        # For a standard Kalman filter, we don't need additional parameters
        # State is already maintained in base class
        pass
    
    def process_step(self, accel, gyro, timestamp):
        """
        Process a single IMU reading with a standard approach.
        
        Args:
            accel: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z]
            timestamp: Current timestamp
            
        Returns:
            Dictionary with current state
        """
        # 1. Update gyro bias estimate (simple model for bias drift)
        self.gyro_bias += np.random.normal(0, self.gyro_bias_noise, 3) * self.dt
        
        # 2. Correct gyro reading for bias
        gyro_corrected = gyro - self.gyro_bias
        
        # 3. Convert angular velocity to quaternion derivative
        # First convert gyro to quaternion form [0, wx, wy, wz]
        gyro_quat = np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        
        # Compute quaternion derivative (q_dot = 0.5 * q * gyro_quat)
        q_dot = 0.5 * self.quaternion_multiply(self.quat, gyro_quat)
        
        # 4. Update quaternion using first-order integration
        new_quat = self.quat + q_dot * self.dt
        
        # 5. Normalize quaternion to prevent numerical drift
        self.quat = self.normalize_quaternion(new_quat)
        
        # 6. Estimate gravity direction in sensor frame
        gravity = self.estimate_gravity(self.quat)
        
        # 7. If accelerometer reading is reliable, use it to correct orientation
        accel_mag = np.linalg.norm(accel)
        
        # Only use accelerometer if magnitude is close to gravity (not during linear acceleration)
        if 8.5 < accel_mag < 11.0:  # ~9.8 m/s² ±10%
            # Normalize accelerometer reading
            accel_norm = accel / accel_mag
            
            # Calculate error between measured and expected gravity
            error = np.cross(accel_norm, gravity)
            
            # Apply correction to quaternion
            error_quat = np.array([1.0, error[0]*0.01, error[1]*0.01, error[2]*0.01])
            error_quat = self.normalize_quaternion(error_quat)
            
            # Apply correction
            self.quat = self.quaternion_multiply(self.quat, error_quat)
            self.quat = self.normalize_quaternion(self.quat)
        
        # 8. Apply drift correction if reference data is available
        self.quat = self.apply_drift_correction(self.quat, timestamp)
        
        # Return current state
        return {
            'quaternion': self.quat,
            'gyro_bias': self.gyro_bias,
            'gravity': gravity
        }


class RobustExtendedKalmanIMU(RobustIMUFilterBase):
    """Extended Kalman Filter with robust quaternion handling."""
    
    def _initialize_filter(self):
        """Initialize EKF parameters."""
        # State: [quaternion (4), gyro_bias (3)]
        self.state = np.zeros(7)
        self.state[0] = 1.0  # w component of quaternion
        
        # State covariance matrix
        self.P = np.eye(7) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(7)
        self.Q[:4, :4] *= self.process_noise
        self.Q[4:, 4:] *= self.gyro_bias_noise
        
        # Measurement noise covariance
        self.R = np.eye(3) * self.measurement_noise
    
    def quaternion_derivative(self, q, omega):
        """
        Compute quaternion derivative from angular velocity.
        
        Args:
            q: Quaternion [w, x, y, z]
            omega: Angular velocity [wx, wy, wz]
            
        Returns:
            Quaternion derivative
        """
        # Convert to quaternion
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        
        # Compute derivative
        q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)
        return q_dot
    
    def state_transition(self, state, gyro, dt):
        """
        EKF state transition function.
        
        Args:
            state: Current state [quat, bias]
            gyro: Gyroscope reading [wx, wy, wz]
            dt: Time step
            
        Returns:
            New state
        """
        q = state[:4]
        bias = state[4:]
        
        # Correct gyro with estimated bias
        gyro_corrected = gyro - bias
        
        # Compute quaternion derivative
        q_dot = self.quaternion_derivative(q, gyro_corrected)
        
        # Update quaternion using first-order integration
        new_q = q + q_dot * dt
        
        # Normalize quaternion
        new_q = self.normalize_quaternion(new_q)
        
        # Bias stays relatively constant
        new_bias = bias.copy()
        
        # Combine into new state
        new_state = np.concatenate([new_q, new_bias])
        return new_state
    
    def measurement_function(self, state):
        """
        EKF measurement function - predict accelerometer reading.
        
        Args:
            state: Current state [quat, bias]
            
        Returns:
            Predicted normalized gravity in sensor frame
        """
        q = state[:4]
        
        # Estimate gravity direction in sensor frame
        gravity = self.estimate_gravity(q)
        
        return gravity
    
    def compute_jacobian_F(self, state, gyro, dt):
        """
        Compute Jacobian of state transition function.
        
        Args:
            state: Current state
            gyro: Gyroscope reading
            dt: Time step
            
        Returns:
            Jacobian matrix F
        """
        # Simplified implementation - linearized model
        F = np.eye(7)
        
        # Quaternion part is approximated
        F[:4, :4] += np.eye(4) * dt * 0.5
        
        # Coupling between quaternion and gyro bias
        F[:4, 4:] = -np.eye(4, 3) * dt * 0.5
        
        return F
    
    def compute_jacobian_H(self, state):
        """
        Compute Jacobian of measurement function.
        
        Args:
            state: Current state
            
        Returns:
            Jacobian matrix H
        """
        # Simplified implementation - linearized measurement model
        H = np.zeros((3, 7))
        
        # Relationship between quaternion and gravity direction
        # This is a simplification - ideally would compute actual derivatives
        H[:3, :4] = np.eye(3, 4) * 2.0
        
        return H
    
    def process_step(self, accel, gyro, timestamp):
        """
        Process a single IMU reading with EKF.
        
        Args:
            accel: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z]
            timestamp: Current timestamp
            
        Returns:
            Dictionary with current state
        """
        try:
            # Extract current state
            q = self.state[:4]
            bias = self.state[4:]
            
            # 1. Predict step
            # State transition
            self.state = self.state_transition(self.state, gyro, self.dt)
            
            # Compute state transition Jacobian
            F = self.compute_jacobian_F(self.state, gyro, self.dt)
            
            # Update covariance
            self.P = F @ self.P @ F.T + self.Q * self.dt
            
            # 2. Update step
            # Only update if accelerometer is reliable (not during high linear acceleration)
            accel_mag = np.linalg.norm(accel)
            
            if 8.5 < accel_mag < 11.0:  # ~9.8 m/s² ±10%
                # Normalize accelerometer
                accel_norm = accel / accel_mag
                
                # Get predicted measurement
                predicted_gravity = self.measurement_function(self.state)
                
                # Compute measurement Jacobian
                H = self.compute_jacobian_H(self.state)
                
                # Innovation
                y = accel_norm - predicted_gravity
                
                # Innovation covariance
                S = H @ self.P @ H.T + self.R
                
                # Kalman gain
                K = self.P @ H.T @ np.linalg.inv(S)
                
                # Update state
                self.state = self.state + K @ y
                
                # Update covariance
                self.P = (np.eye(7) - K @ H) @ self.P
            
            # Normalize quaternion part of state
            self.state[:4] = self.normalize_quaternion(self.state[:4])
            
            # Apply drift correction if reference data is available
            self.state[:4] = self.apply_drift_correction(self.state[:4], timestamp)
            
            # Update instance quaternion and bias for compatibility
            self.quat = self.state[:4]
            self.gyro_bias = self.state[4:]
            
            # Return current state
            return {
                'quaternion': self.quat,
                'gyro_bias': self.gyro_bias,
                'gravity': self.estimate_gravity(self.quat)
            }
            
        except Exception as e:
            logger.warning(f"EKF process step error: {e}")
            
            # Keep previous state if error occurs
            return {
                'quaternion': self.quat,
                'gyro_bias': self.gyro_bias,
                'gravity': self.estimate_gravity(self.quat)
            }


class RobustUnscentedKalmanIMU(RobustIMUFilterBase):
    """Unscented Kalman Filter with robust quaternion handling."""
    
    def _initialize_filter(self):
        """Initialize UKF specific parameters."""
        self.n_dim = 7  # [quaternion (4), gyro_bias (3)]
        self.n_meas = 3  # Accelerometer measurements (3)
        
        # State vector
        self.x = np.zeros(self.n_dim)
        self.x[0] = 1.0  # Initial quaternion w component
        
        # State covariance
        self.P = np.eye(self.n_dim) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(self.n_dim)
        self.Q[:4, :4] *= self.process_noise
        self.Q[4:, 4:] *= self.gyro_bias_noise
        
        # Measurement noise covariance
        self.R = np.eye(self.n_meas) * self.measurement_noise
        
        # UKF parameters
        self.alpha = 0.1  # Small alpha for minimal spread
        self.beta = 2.0   # Optimal for Gaussian distributions
        self.kappa = 0.0  # Default
        
        # Derived parameters
        self.n_sigma = 2 * self.n_dim + 1  # Number of sigma points
        self.lambda_param = self.alpha**2 * (self.n_dim + self.kappa) - self.n_dim
        
        # Weights
        self.Wm = np.zeros(self.n_sigma)  # Weights for mean
        self.Wc = np.zeros(self.n_sigma)  # Weights for covariance
        
        self.Wm[0] = self.lambda_param / (self.n_dim + self.lambda_param)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1.0 / (2.0 * (self.n_dim + self.lambda_param))
            self.Wc[i] = self.Wm[i]
        
        # Current gyro for state transition
        self.current_gyro = np.zeros(3)
    
    def generate_sigma_points(self):
        """
        Generate sigma points using the unscented transform.
        
        Returns:
            Array of sigma points
        """
        # Compute matrix square root of (n+lambda)*P
        scaling = self.n_dim + self.lambda_param
        L = np.linalg.cholesky(scaling * self.P)
        
        # Initialize sigma points
        sigma_points = np.zeros((self.n_sigma, self.n_dim))
        
        # First sigma point is the current state
        sigma_points[0] = self.x
        
        # Generate remaining sigma points
        for i in range(self.n_dim):
            sigma_points[i+1] = self.x + L[i]
            sigma_points[i+1+self.n_dim] = self.x - L[i]
            
            # Normalize quaternion part of each sigma point
            sigma_points[i+1, :4] = self.normalize_quaternion(sigma_points[i+1, :4])
            sigma_points[i+1+self.n_dim, :4] = self.normalize_quaternion(sigma_points[i+1+self.n_dim, :4])
        
        return sigma_points
    
    def state_transition_fn(self, sigma_point):
        """
        State transition function for UKF.
        
        Args:
            sigma_point: Sigma point state vector
            
        Returns:
            Propagated sigma point
        """
        q = sigma_point[:4]
        bias = sigma_point[4:]
        
        # Use stored gyro reading
        gyro_corrected = self.current_gyro - bias
        
        # Convert to quaternion form
        omega_quat = np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        
        # Compute quaternion derivative
        q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)
        
        # Update quaternion
        new_q = q + q_dot * self.dt
        
        # Normalize quaternion
        new_q = self.normalize_quaternion(new_q)
        
        # Bias transition (assume constant with small noise)
        new_bias = bias.copy()
        
        # Return full state
        return np.concatenate([new_q, new_bias])
    
    def measurement_fn(self, sigma_point):
        """
        Measurement function for UKF.
        
        Args:
            sigma_point: Sigma point state vector
            
        Returns:
            Predicted gravity direction in sensor frame
        """
        try:
            q = sigma_point[:4]
            
            # Normalize quaternion
            q = self.normalize_quaternion(q)
            
            # Estimate gravity
            gravity = self.estimate_gravity(q)
            
            return gravity
        except Exception as e:
            logger.warning(f"Measurement function error: {e}")
            return np.array([0, 0, 1])  # Default +z direction
    
    def mean_quaternion(self, quaternions, weights):
        """
        Calculate weighted mean of quaternions.
        
        Args:
            quaternions: Array of quaternions
            weights: Array of weights
            
        Returns:
            Mean quaternion
        """
        # Initialize with zeros
        mean = np.zeros(4)
        
        # Simple weighted average for initial guess
        for i in range(len(quaternions)):
            mean += weights[i] * quaternions[i]
        
        # Normalize
        return self.normalize_quaternion(mean)
    
    def process_step(self, accel, gyro, timestamp):
        """
        Process a single IMU reading with UKF.
        
        Args:
            accel: Accelerometer reading [x, y, z]
            gyro: Gyroscope reading [x, y, z]
            timestamp: Current timestamp
            
        Returns:
            Dictionary with current state
        """
        try:
            # Store gyro for state transition
            self.current_gyro = gyro
            
            # 1. Generate sigma points
            sigma_points = self.generate_sigma_points()
            
            # 2. Propagate sigma points through state transition function
            transformed_sigmas = np.zeros_like(sigma_points)
            for i in range(self.n_sigma):
                transformed_sigmas[i] = self.state_transition_fn(sigma_points[i])
            
            # 3. Calculate predicted mean and covariance
            # For the quaternion part, use special quaternion averaging
            quat_mean = self.mean_quaternion(transformed_sigmas[:, :4], self.Wm)
            bias_mean = np.zeros(3)
            
            for i in range(self.n_sigma):
                bias_mean += self.Wm[i] * transformed_sigmas[i, 4:]
            
            # Combined mean
            x_pred = np.concatenate([quat_mean, bias_mean])
            
            # Predicted covariance
            P_pred = np.zeros_like(self.P)
            
            for i in range(self.n_sigma):
                # For quaternion part, compute difference in tangent space
                state_diff = transformed_sigmas[i] - x_pred
                P_pred += self.Wc[i] * np.outer(state_diff, state_diff)
            
            # Add process noise
            P_pred += self.Q * self.dt
            
            # 4. Measurement update
            # Only update if accelerometer is reliable
            accel_mag = np.linalg.norm(accel)
            
            if 8.5 < accel_mag < 11.0:  # ~9.8 m/s² ±10%
                # Normalize accelerometer
                accel_norm = accel / accel_mag
                
                # Propagate sigma points through measurement function
                meas_sigmas = np.zeros((self.n_sigma, self.n_meas))
                for i in range(self.n_sigma):
                    try:
                        meas_sigmas[i] = self.measurement_fn(transformed_sigmas[i])
                    except Exception:
                        # If measurement function fails, use default
                        meas_sigmas[i] = np.array([0, 0, 1])
                
                # Predicted measurement mean
                z_pred = np.zeros(self.n_meas)
                for i in range(self.n_sigma):
                    z_pred += self.Wm[i] * meas_sigmas[i]
                
                # Measurement covariance
                Pzz = np.zeros((self.n_meas, self.n_meas))
                for i in range(self.n_sigma):
                    meas_diff = meas_sigmas[i] - z_pred
                    Pzz += self.Wc[i] * np.outer(meas_diff, meas_diff)
                
                # Add measurement noise
                Pzz += self.R
                
                # Cross-correlation
                Pxz = np.zeros((self.n_dim, self.n_meas))
                for i in range(self.n_sigma):
                    state_diff = transformed_sigmas[i] - x_pred
                    meas_diff = meas_sigmas[i] - z_pred
                    Pxz += self.Wc[i] * np.outer(state_diff, meas_diff)
                
                # Kalman gain
                K = Pxz @ np.linalg.inv(Pzz)
                
                # Update state
                innovation = accel_norm - z_pred
                self.x = x_pred + K @ innovation
                
                # Update covariance
                self.P = P_pred - K @ Pzz @ K.T
            else:
                # No measurement update, just use prediction
                self.x = x_pred
                self.P = P_pred
            
            # Normalize quaternion part
            self.x[:4] = self.normalize_quaternion(self.x[:4])
            
            # Apply drift correction if reference data is available
            self.x[:4] = self.apply_drift_correction(self.x[:4], timestamp)
            
            # Update instance quaternion and bias for compatibility
            self.quat = self.x[:4]
            self.gyro_bias = self.x[4:]
            
            # Ensure no NaNs or Infs
            if not np.all(np.isfinite(self.quat)) or not np.all(np.isfinite(self.gyro_bias)):
                logger.warning("Non-finite values detected, resetting state")
                self.quat = np.array([1.0, 0.0, 0.0, 0.0])
                self.gyro_bias = np.zeros(3)
                self.x[:4] = self.quat
                self.x[4:] = self.gyro_bias
            
            # Return current state
            return {
                'quaternion': self.quat,
                'gyro_bias': self.gyro_bias,
                'gravity': self.estimate_gravity(self.quat)
            }
            
        except Exception as e:
            logger.warning(f"UKF process step error: {e}")
            
            # Keep previous state if error occurs
            return {
                'quaternion': self.quat,
                'gyro_bias': self.gyro_bias,
                'gravity': self.estimate_gravity(self.quat)
            }


def extract_orientation_from_skeleton(skeleton_data, wrist_idx=9):
    """
    Extract orientation from skeleton data at the wrist joint.
    
    Args:
        skeleton_data: Skeleton joint positions of shape (n_frames, n_joints*3)
        wrist_idx: Index of wrist joint
        
    Returns:
        Array of Euler angles [roll, pitch, yaw] for each frame
    """
    n_frames = skeleton_data.shape[0]
    orientations = np.zeros((n_frames, 3))
    
    try:
        # Calculate joint dimensions
        joint_dim = 3  # x, y, z
        
        # Check dimensions
        if skeleton_data.shape[1] < (wrist_idx + 1) * joint_dim:
            logger.warning(f"Skeleton data doesn't have enough dimensions for wrist_idx={wrist_idx}")
            return orientations
        
        # Extract wrist and elbow positions
        wrist_idx_start = wrist_idx * joint_dim
        elbow_idx = max(0, wrist_idx - 1)  # Elbow usually 1 joint before wrist
        elbow_idx_start = elbow_idx * joint_dim
        
        for i in range(n_frames):
            # Extract joint positions
            wrist_pos = skeleton_data[i, wrist_idx_start:wrist_idx_start+joint_dim]
            elbow_pos = skeleton_data[i, elbow_idx_start:elbow_idx_start+joint_dim]
            
            # Calculate forearm direction (from elbow to wrist)
            forearm_dir = wrist_pos - elbow_pos
            forearm_norm = np.linalg.norm(forearm_dir)
            
            if forearm_norm > 1e-6:
                # Normalize direction
                forearm_dir = forearm_dir / forearm_norm
                
                # Use forearm direction as forward axis (z)
                z_axis = forearm_dir
                
                # Use global up direction as reference
                global_up = np.array([0, 1, 0])
                
                # Calculate right axis (x) as perpendicular to forearm and global up
                x_axis = np.cross(global_up, z_axis)
                x_norm = np.linalg.norm(x_axis)
                
                if x_norm > 1e-6:
                    x_axis = x_axis / x_norm
                    
                    # Calculate up axis (y) to complete right-handed system
                    y_axis = np.cross(z_axis, x_axis)
                    
                    # Form rotation matrix
                    rot_mat = np.column_stack((x_axis, y_axis, z_axis))
                    
                    # Convert to Euler angles
                    try:
                        r = R.from_matrix(rot_mat)
                        orientations[i] = r.as_euler('xyz')
                    except Exception as e:
                        logger.warning(f"Error converting to Euler angles: {e}")
                else:
                    # Handle degenerate case (aligned with global up)
                    orientations[i] = np.zeros(3)
            else:
                # Handle zero-length forearm
                orientations[i] = np.zeros(3)
                
    except Exception as e:
        logger.warning(f"Error extracting orientation from skeleton: {e}")
    
    return orientations


def calibrate_filter(accel_data, gyro_data, skel_data=None, filter_type='ekf', 
                   timestamps=None, wrist_idx=9):
    """
    Calibrate filter parameters using skeleton data as ground truth.
    
    Args:
        accel_data: Accelerometer data (n_samples, 3)
        gyro_data: Gyroscope data (n_samples, 3)
        skel_data: Optional skeleton data for ground truth (n_samples, n_joints*3)
        filter_type: 'standard', 'ekf', or 'ukf'
        timestamps: Optional timestamps
        wrist_idx: Index of wrist joint
        
    Returns:
        Tuple of (calibrated_filter, optimal_params)
    """
    # Default parameters (starting point)
    default_params = np.array([0.01, 0.1, 0.01])  # process_noise, measurement_noise, gyro_bias_noise
    
    # Extract reference orientations from skeleton if available
    ref_orientations = None
    if skel_data is not None:
        try:
            ref_orientations = extract_orientation_from_skeleton(skel_data, wrist_idx)
        except Exception as e:
            logger.warning(f"Failed to extract reference orientations: {e}")
    
    # If we don't have ref_orientations, just return default
    if ref_orientations is None or len(ref_orientations) == 0:
        logger.warning("No reference orientations available for calibration, using defaults")
        
        # Create filter with default parameters
        if filter_type == 'standard':
            filter_obj = RobustStandardKalmanIMU()
        elif filter_type == 'ekf':
            filter_obj = RobustExtendedKalmanIMU()
        else:  # ukf
            filter_obj = RobustUnscentedKalmanIMU()
            
        return filter_obj, default_params
    
    # Otherwise, find optimal parameters
    # Define objective function for optimization
    def objective_fn(params):
        # Create filter with these parameters
        if filter_type == 'standard':
            filt = RobustStandardKalmanIMU(
                process_noise=params[0],
                measurement_noise=params[1],
                gyro_bias_noise=params[2]
            )
        elif filter_type == 'ekf':
            filt = RobustExtendedKalmanIMU(
                process_noise=params[0],
                measurement_noise=params[1],
                gyro_bias_noise=params[2]
            )
        else:  # ukf
            filt = RobustUnscentedKalmanIMU(
                process_noise=params[0],
                measurement_noise=params[1],
                gyro_bias_noise=params[2]
            )
        
        # Set reference data
        if timestamps is not None:
            filt.set_reference_data(timestamps, ref_orientations)
        
        # Process data
        try:
            output = filt.process_sequence(accel_data, gyro_data, timestamps)
            
            # Extract Euler angles
            euler_estimated = output[:, 10:13]
            
            # Compute error
            min_len = min(len(euler_estimated), len(ref_orientations))
            mse = np.mean(np.sum((euler_estimated[:min_len] - ref_orientations[:min_len])**2, axis=1))
            return mse
            
        except Exception as e:
            logger.warning(f"Error in calibration: {e}")
            return 1e6  # Large penalty
    
    # Simple grid search for optimization
    from scipy.optimize import minimize
    
    try:
        # Bounds for parameters
        bounds = [(1e-4, 1.0), (1e-4, 1.0), (1e-4, 1.0)]
        
        result = minimize(
            objective_fn,
            default_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20}
        )
        
        optimal_params = result.x
        
        # Create calibrated filter
        if filter_type == 'standard':
            filter_obj = RobustStandardKalmanIMU(
                process_noise=optimal_params[0],
                measurement_noise=optimal_params[1],
                gyro_bias_noise=optimal_params[2]
            )
        elif filter_type == 'ekf':
            filter_obj = RobustExtendedKalmanIMU(
                process_noise=optimal_params[0],
                measurement_noise=optimal_params[1],
                gyro_bias_noise=optimal_params[2]
            )
        else:  # ukf
            filter_obj = RobustUnscentedKalmanIMU(
                process_noise=optimal_params[0],
                measurement_noise=optimal_params[1],
                gyro_bias_noise=optimal_params[2]
            )
        
        return filter_obj, optimal_params
        
    except Exception as e:
        logger.warning(f"Optimization failed: {e}")
        
        # Return filter with default parameters
        if filter_type == 'standard':
            filter_obj = RobustStandardKalmanIMU()
        elif filter_type == 'ekf':
            filter_obj = RobustExtendedKalmanIMU()
        else:  # ukf
            filter_obj = RobustUnscentedKalmanIMU()
            
        return filter_obj, default_params
