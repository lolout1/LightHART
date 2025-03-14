# utils/imu_fusion.py

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import filterpy.kalman as kf
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints

class BaseKalmanFilter:
    """Base class for all Kalman filter implementations."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.3):
        self.dt = dt  # Default time step
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.drift_correction_weight = drift_correction_weight
        self.initialized = False
        
        # For drift correction using reference data (skeleton)
        self.reference_timestamps = None
        self.reference_orientations = None
        self.use_reference = False
        
        # State dimensions will depend on filter type
        self.state_dim = 0
        self.measurement_dim = 0
        
    def initialize(self, accel_data):
        """Initialize filter using initial acceleration data."""
        # Base implementation - to be overridden
        self.initialized = True
    
    def reset(self):
        """Reset filter state."""
        self.initialized = False
    
    def set_reference_data(self, timestamps, orientations):
        """
        Set reference orientation data from skeleton for drift correction.
        
        Args:
            timestamps: Array of timestamps for reference orientations
            orientations: Array of reference orientations (Euler angles or quaternions)
        """
        if timestamps is None or orientations is None or len(timestamps) == 0:
            self.use_reference = False
            return
            
        self.reference_timestamps = timestamps
        self.reference_orientations = orientations
        self.use_reference = True
        print(f"Reference data set for drift correction: {len(timestamps)} points")
    
    def get_reference_orientation(self, timestamp):
        """
        Get reference orientation at a specific timestamp through interpolation.
        
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
            
        # Interpolate reference orientation
        interp_func = interp1d(
            self.reference_timestamps,
            self.reference_orientations,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        return interp_func(timestamp)
    
    def apply_drift_correction(self, estimated_orientation, timestamp, quaternion=False):
        """
        Apply drift correction using reference orientation from skeleton.
        
        Args:
            estimated_orientation: Current estimated orientation
            timestamp: Current timestamp
            quaternion: Whether orientation is in quaternion format
            
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
        
        if quaternion:
            # For quaternions, we need to handle special SLERP interpolation
            est_quat = R.from_quat(estimated_orientation).as_quat()
            ref_quat = R.from_euler('xyz', reference).as_quat()
            
            # Ensure quaternions have same sign (shortest path interpolation)
            if np.dot(est_quat, ref_quat) < 0:
                ref_quat = -ref_quat
                
            # Spherical linear interpolation
            corrected_quat = (1 - w) * est_quat + w * ref_quat
            corrected_quat = corrected_quat / np.linalg.norm(corrected_quat)
            
            # Convert back to original format
            corrected = R.from_quat(corrected_quat).as_euler('xyz')
        else:
            # Simple linear interpolation for Euler angles
            corrected = (1 - w) * estimated_orientation + w * reference
            
        return corrected
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """
        Process a single step of sensor data.
        
        Args:
            accel: Acceleration measurement (3,)
            gyro: Angular velocity measurement (3,)
            dt: Time step
            timestamp: Current timestamp (for drift correction)
            
        Returns:
            Updated state estimate and covariance
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
        
        # Create output array - vary size depending on filter output features
        output_features = 13  # Default: 3 accel + 3 gyro + 4 quaternion + 3 euler
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
        
        # Create Kalman filter
        self.kf = kf.KalmanFilter(dim_x=self.state_dim, dim_z=self.measurement_dim)
        
        # Initialize state transition matrix
        self.F = np.eye(self.state_dim)
        
        # Initialize measurement matrix (H)
        # We only directly measure orientation from accelerometer
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[:3, :3] = np.eye(3)
        
        # Set measurement noise
        self.R = np.eye(self.measurement_dim) * self.measurement_noise
    
    def initialize(self, accel):
        """Initialize filter state using initial acceleration."""
        # Estimate initial orientation from gravity direction
        gravity = -accel / np.linalg.norm(accel)
        
        # Convert to Euler angles (roll, pitch)
        roll = np.arctan2(gravity[1], gravity[2])
        pitch = np.arctan2(-gravity[0], np.sqrt(gravity[1]**2 + gravity[2]**2))
        yaw = 0.0  # Can't determine yaw from gravity alone
        
        # Set initial state
        self.kf.x = np.zeros(self.state_dim)
        self.kf.x[:3] = np.array([roll, pitch, yaw])
        
        # Set initial covariance
        self.kf.P = np.eye(self.state_dim) * 0.1
        self.kf.P[2, 2] = 1.0  # Higher uncertainty for yaw
        
        # Set fixed matrices
        self.kf.F = self.F
        self.kf.H = self.H
        self.kf.R = self.R
        
        self.initialized = True
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process one step using standard Kalman filter."""
        if not self.initialized:
            self.initialize(accel)
            
        # Update state transition matrix for current dt
        self.F[:3, 3:] = np.eye(3) * dt
        self.kf.F = self.F
        
        # Set process noise
        q = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise, block_size=3)
        self.kf.Q = q
        
        # Gyro bias noise
        self.kf.Q[3:, 3:] += np.eye(3) * self.gyro_bias_noise * dt
        
        # Prediction step using gyroscope
        gyro_corrected = gyro - self.kf.x[3:6]  # Apply bias correction
        
        # Apply gyro for prediction
        self.kf.predict()
        
        # Add gyro rotation to state (integrated angular velocity)
        delta_angle = gyro_corrected * dt
        self.kf.x[:3] += delta_angle
        
        # Update step using accelerometer for roll and pitch
        # (Accelerometer can't directly measure yaw)
        gravity = -accel / np.linalg.norm(accel)
        roll_acc = np.arctan2(gravity[1], gravity[2])
        pitch_acc = np.arctan2(-gravity[0], np.sqrt(gravity[1]**2 + gravity[2]**2))
        
        # Only update roll and pitch from accelerometer
        acc_angles = np.array([roll_acc, pitch_acc, self.kf.x[2]])
        
        self.kf.update(acc_angles)
        
        # Apply drift correction if reference data is available
        if timestamp is not None and self.use_reference:
            self.kf.x[:3] = self.apply_drift_correction(self.kf.x[:3], timestamp)
        
        # Convert Euler angles to quaternion
        orientation = self.kf.x[:3]
        quat = R.from_euler('xyz', orientation).as_quat()
        
        # Feature vector: accel, gyro, quaternion, euler
        features = np.concatenate([
            accel,
            gyro,
            quat,
            orientation
        ])
        
        return self.kf.x, features

class ExtendedKalmanIMU(BaseKalmanFilter):
    """
    Extended Kalman Filter for IMU fusion.
    
    This handles nonlinear state transitions for better orientation tracking.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, gyro_bias_noise=0.01,
                drift_correction_weight=0.3):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, drift_correction_weight)
        
        # State: [quaternion (4), gyro_bias (3)]
        self.state_dim = 7
        self.measurement_dim = 3
        
        # Create EKF
        self.ekf = kf.ExtendedKalmanFilter(dim_x=self.state_dim, dim_z=self.measurement_dim)
        
        # Process noise
        self.Q = np.eye(self.state_dim) * self.process_noise
        self.Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise  # Bias noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * self.measurement_noise
    
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
        
        # Initialize state [quat, gyro_bias]
        self.ekf.x = np.zeros(self.state_dim)
        self.ekf.x[:4] = quat
        
        # Initialize covariance
        self.ekf.P = np.eye(self.state_dim) * 0.1
        
        # Set measurement noise
        self.ekf.R = self.R
        
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
        # Simplified version that works reasonably well
        F = np.eye(self.state_dim)
        
        # Effect of gyro bias on quaternion
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-10:
            q = x[:4]
            # Approximate partial derivatives
            for i in range(3):
                # Small perturbation to measure Jacobian
                delta = np.zeros(3)
                delta[i] = 0.001
                
                # Forward difference approximation
                x_plus = np.copy(x)
                x_plus[4+i] += delta[i]
                
                f_plus = self.state_transition_function(x_plus, dt, omega)
                f = self.state_transition_function(x, dt, omega)
                
                # Compute partial derivative
                F[:4, 4+i] = (f_plus[:4] - f[:4]) / delta[i]
        
        return F
    
    def measurement_function(self, x):
        """
        Measurement function for EKF.
        
        Converts quaternion to expected accelerometer measurement
        (assuming only gravity when stationary).
        """
        # Extract quaternion
        q = x[:4]
        
        # Convert quaternion to rotation matrix
        r = R.from_quat(q)
        
        # Rotate unit gravity vector [0,0,1]
        gravity_body = r.apply([0, 0, 1])
        
        return gravity_body
    
    def measurement_jacobian(self, x):
        """Jacobian of measurement function."""
        # Simplified version - numerical approximation
        H = np.zeros((self.measurement_dim, self.state_dim))
        
        # Base measurement
        z = self.measurement_function(x)
        
        # Compute partials for quaternion elements
        for i in range(4):
            # Small perturbation
            delta = 0.001
            x_plus = np.copy(x)
            x_plus[i] += delta
            
            # Numerical partial derivative
            z_plus = self.measurement_function(x_plus)
            H[:, i] = (z_plus - z) / delta
        
        # Partials for gyro bias are zero (no direct impact on measurement)
        
        return H
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process one step using EKF."""
        if not self.initialized:
            self.initialize(accel)
        
        # Set process noise (scales with dt)
        self.ekf.Q = self.Q * dt
        
        # Prediction step
        self.ekf.predict_x = lambda x, u: self.state_transition_function(x, dt, gyro)
        self.ekf.F = self.state_transition_jacobian(self.ekf.x, dt, gyro)
        
        self.ekf.predict()
        
        # Update step if not in high dynamics (acceleration close to g)
        accel_norm = np.linalg.norm(accel)
        gravity_norm = 9.81
        
        if abs(accel_norm - gravity_norm) < 3.0:  # Threshold for quasi-static assumption
            # Normalize measurement
            z = -accel / accel_norm  # Measured gravity direction (negative of acceleration)
            
            # Update with measurement
            self.ekf.update(z, HJacobian=self.measurement_jacobian, 
                           Hx=self.measurement_function)
        
        # Normalize quaternion
        self.ekf.x[:4] = self.ekf.x[:4] / np.linalg.norm(self.ekf.x[:4])
        
        # Apply drift correction if reference data is available
        if timestamp is not None and self.use_reference:
            quat = self.ekf.x[:4]
            euler = R.from_quat(quat).as_euler('xyz')
            
            # Correct Euler angles
            corrected_euler = self.apply_drift_correction(euler, timestamp)
            
            # Convert back to quaternion
            corrected_quat = R.from_euler('xyz', corrected_euler).as_quat()
            
            self.ekf.x[:4] = corrected_quat
        
        # Extract orientation for output
        quat = self.ekf.x[:4]
        euler = R.from_euler('xyz', R.from_quat(quat).as_euler('xyz')).as_euler('xyz')
        
        # Feature vector: accel, gyro, quaternion, euler
        features = np.concatenate([
            accel,
            gyro,
            quat,
            euler
        ])
        
        return self.ekf.x, features

class UnscentedKalmanIMU(BaseKalmanFilter):
    """
    Unscented Kalman Filter for IMU fusion.
    
    This handles highly nonlinear systems better than EKF.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, gyro_bias_noise=0.01,
                drift_correction_weight=0.3):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise, drift_correction_weight)
        
        # State: [quaternion (4), gyro_bias (3)]
        self.state_dim = 7
        self.measurement_dim = 3
        
        # Create UKF with Merwe scaled sigma points
        points = MerweScaledSigmaPoints(
            n=self.state_dim,
            alpha=1e-3,
            beta=2.0,
            kappa=0.0
        )
        
        self.ukf = kf.UnscentedKalmanFilter(
            dim_x=self.state_dim,
            dim_z=self.measurement_dim,
            dt=dt,
            hx=self.measurement_function,
            fx=self.state_transition_function,
            points=points
        )
        
        # Process noise
        self.Q = np.eye(self.state_dim) * self.process_noise
        self.Q[4:, 4:] = np.eye(3) * self.gyro_bias_noise  # Bias noise
        
        # Measurement noise
        self.R = np.eye(self.measurement_dim) * self.measurement_noise
        
        # Save last gyro for state transition
        self.last_gyro = np.zeros(3)
    
    def initialize(self, accel):
        """Initialize filter with first acceleration measurement."""
        # Similar to EKF initialization
        gravity = -accel / np.linalg.norm(accel)
        
        # Find rotation from [0,0,1] to gravity direction
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
            quat = r.as_quat()
        
        # Initialize state [quat, gyro_bias]
        self.ukf.x = np.zeros(self.state_dim)
        self.ukf.x[:4] = quat
        
        # Initialize covariance
        self.ukf.P = np.eye(self.state_dim) * 0.1
        
        # Set noise matrices
        self.ukf.Q = self.Q * self.dt
        self.ukf.R = self.R
        
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
    
    def state_transition_function(self, x, dt):
        """
        State transition function for UKF.
        
        Args:
            x: Current state (quaternion, gyro_bias)
            dt: Time step
            
        Returns:
            New state
        """
        # Extract current quaternion and bias
        q = x[:4]
        bias = x[4:]
        
        # Use saved gyro - UKF passes dt but not additional parameters
        omega = self.last_gyro
        
        # Correct gyro with bias
        omega_corrected = omega - bias
        
        # Update quaternion
        q_new = self.quaternion_update(q, omega_corrected, dt)
        
        # State remains the same except for quaternion
        x_new = np.zeros_like(x)
        x_new[:4] = q_new
        x_new[4:] = bias  # Bias model is constant
        
        return x_new
    
    def measurement_function(self, x):
        """
        Measurement function for UKF.
        
        Converts quaternion to expected accelerometer measurement.
        """
        # Extract quaternion
        q = x[:4]
        
        # Convert quaternion to rotation matrix
        r = R.from_quat(q)
        
        # Rotate unit gravity vector [0,0,1]
        gravity_body = r.apply([0, 0, 1])
        
        return gravity_body
    
    def process_step(self, accel, gyro, dt, timestamp=None):
        """Process one step using UKF."""
        if not self.initialized:
            self.initialize(accel)
        
        # Set process noise (scales with dt)
        self.ukf.Q = self.Q * dt
        
        # Save gyro for state transition function
        self.last_gyro = gyro
        
        # Predict step
        self.ukf.predict(dt=dt)
        
        # Update step if not in high dynamics (acceleration close to g)
        accel_norm = np.linalg.norm(accel)
        gravity_norm = 9.81
        
        if abs(accel_norm - gravity_norm) < 3.0:  # Threshold for quasi-static assumption
            # Normalize measurement
            z = -accel / accel_norm  # Measured gravity direction (negative of acceleration)
            
            # Update with measurement
            self.ukf.update(z)
        
        # Normalize quaternion
        self.ukf.x[:4] = self.ukf.x[:4] / np.linalg.norm(self.ukf.x[:4])
        
        # Apply drift correction if reference data is available
        if timestamp is not None and self.use_reference:
            quat = self.ukf.x[:4]
            euler = R.from_quat(quat).as_euler('xyz')
            
            # Correct Euler angles
            corrected_euler = self.apply_drift_correction(euler, timestamp)
            
            # Convert back to quaternion
            corrected_quat = R.from_euler('xyz', corrected_euler).as_quat()
            
            self.ukf.x[:4] = corrected_quat
        
        # Extract orientation for output
        quat = self.ukf.x[:4]
        euler = R.from_quat(quat).as_euler('xyz')
        
        # Feature vector: accel, gyro, quaternion, euler
        features = np.concatenate([
            accel,
            gyro,
            quat,
            euler
        ])
        
        return self.ukf.x, features

def extract_orientation_from_skeleton(skeleton_data, num_joints=32, joint_dim=3):
    """
    Extract orientation information from skeleton joint positions.
    
    Args:
        skeleton_data: Skeleton joint positions (n_frames, num_joints*joint_dim)
        num_joints: Number of joints in skeleton
        joint_dim: Dimension of each joint (typically 3 for XYZ)
    
    Returns:
        Orientation as Euler angles (n_frames, 3)
    """
    n_frames = skeleton_data.shape[0]
    orientations = np.zeros((n_frames, 3))
    
    # Indices for key joints (adapt to your skeleton structure)
    # These are example indices - adjust based on your actual skeleton format
    NECK = 2
    SPINE = 1
    RIGHT_SHOULDER = 8
    LEFT_SHOULDER = 4
    RIGHT_HIP = 12
    LEFT_HIP = 16
    
    for i in range(n_frames):
        # Get joint positions
        joints = skeleton_data[i].reshape(num_joints, joint_dim)
        
        # Use spine and shoulders to compute orientation
        if num_joints > max(NECK, SPINE, RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_HIP, LEFT_HIP):
            # Calculate body plane normal vector (cross product of shoulders vector and spine vector)
            shoulder_vector = joints[RIGHT_SHOULDER] - joints[LEFT_SHOULDER]
            spine_vector = joints[NECK] - joints[SPINE]
            
            # Calculate roll (around forward axis)
            # Roll is the angle between shoulders and horizonal plane
            roll = np.arctan2(shoulder_vector[1], shoulder_vector[0])
            
            # Calculate pitch (around side axis)
            # Pitch is the angle between spine and vertical axis
            pitch = np.arctan2(spine_vector[0], spine_vector[2])
            
            # Calculate yaw (around vertical axis)
            # Use hip vector or shoulders vector
            yaw = np.arctan2(shoulder_vector[2], shoulder_vector[0])
            
            orientations[i] = [roll, pitch, yaw]
        
    return orientations

def calibrate_filter(accel, gyro, skeleton, filter_type='ekf', timestamps=None):
    """
    Calibrate filter parameters using skeleton ground truth.
    
    Args:
        accel: Accelerometer data (N, 3)
        gyro: Gyroscope data (N, 3) or None
        skeleton: Skeleton data (N, num_joints*3)
        filter_type: Type of filter to calibrate ('standard', 'ekf', 'ukf')
        timestamps: Optional timestamps (N,)
    
    Returns:
        Tuple of (calibrated_filter, optimal_parameters)
    """
    from scipy.optimize import minimize
    
    # Extract orientation from skeleton
    skeleton_orientation = extract_orientation_from_skeleton(skeleton)
    
    # Define error function to minimize
    def error_function(params):
        # Create filter with trial parameters
        process_noise, measurement_noise, gyro_bias_noise = params
        
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
        
        # Process data with trial filter
        if gyro is None:
            # If no gyro, use zeros
            gyro_data = np.zeros_like(accel)
        else:
            gyro_data = gyro
            
        # Process a subset for efficiency
        test_size = min(len(accel), 1000)
        sample_indices = np.linspace(0, len(accel)-1, test_size).astype(int)
        
        accel_sample = accel[sample_indices]
        gyro_sample = gyro_data[sample_indices]
        timestamps_sample = timestamps[sample_indices] if timestamps is not None else None
        skeleton_sample = skeleton_orientation[sample_indices]
        
        output = test_filter.process_sequence(accel_sample, gyro_sample, timestamps_sample)
        
        # Extract estimated orientation
        estimated_orientation = output[:, 10:]  # Last 3 values are Euler angles
        
        # Compute error with skeleton orientation
        error = np.mean(np.sum((estimated_orientation - skeleton_sample)**2, axis=1))
        
        return error
    
    # Initial parameter guess
    initial_params = np.array([0.01, 0.1, 0.01])
    
    # Parameter bounds
    bounds = [(0.001, 0.1), (0.01, 1.0), (0.001, 0.1)]
    
    # Minimize error
    result = minimize(
        error_function,
        initial_params,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    # Create final filter with optimal parameters
    process_noise, measurement_noise, gyro_bias_noise = result.x
    
    if filter_type == 'standard':
        optimal_filter = StandardKalmanIMU(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            gyro_bias_noise=gyro_bias_noise
        )
    elif filter_type == 'ekf':
        optimal_filter = ExtendedKalmanIMU(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            gyro_bias_noise=gyro_bias_noise
        )
    else:  # 'ukf'
        optimal_filter = UnscentedKalmanIMU(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            gyro_bias_noise=gyro_bias_noise
        )
    
    print(f"Calibrated {filter_type} parameters: process_noise={process_noise:.6f}, "
          f"measurement_noise={measurement_noise:.6f}, gyro_bias_noise={gyro_bias_noise:.6f}")
    
    return optimal_filter, result.x

def robust_align_modalities(imu_data, skel_data, timestamps, method='dtw', wrist_idx=9):
    """
    Align IMU and skeleton data using various methods.
    
    Args:
        imu_data: IMU data [accel/gyro values, no timestamps] (N, features)
        skel_data: Skeleton data (M, features)
        timestamps: IMU timestamps (N,)
        method: Alignment method ('dtw', 'interpolation', 'crop')
        wrist_idx: Index of wrist joint (to align with watch data)
        
    Returns:
        Tuple of (aligned_imu, aligned_skel, aligned_timestamps)
    """
    if imu_data.shape[0] == 0 or skel_data.shape[0] == 0:
        # Return empty arrays if either input is empty
        return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
    
    if method == 'dtw':
        try:
            from dtaidistance import dtw
            
            # Extract wrist movement from skeleton for alignment
            if skel_data.shape[1] >= 96:  # Full skeleton
                # Extract wrist joint (assuming XYZ format for each joint)
                wrist_start_idx = wrist_idx * 3
                wrist_xyz = skel_data[:, wrist_start_idx:wrist_start_idx+3]
                
                # Calculate wrist velocity (magnitude)
                wrist_vel = np.zeros(skel_data.shape[0])
                for i in range(1, skel_data.shape[0]):
                    dt = 1/30.0  # Assuming 30fps for skeleton
                    wrist_vel[i] = np.linalg.norm(wrist_xyz[i] - wrist_xyz[i-1]) / dt
                
                # Calculate IMU total acceleration magnitude for comparison
                if imu_data.shape[1] >= 3:  # At least XYZ acceleration
                    imu_mag = np.linalg.norm(imu_data[:, :3], axis=1)
                else:
                    imu_mag = imu_data[:, 0]  # Use first column if fewer dimensions
                
                # Normalize sequences for comparison
                norm_imu = (imu_mag - np.mean(imu_mag)) / (np.std(imu_mag) + 1e-6)
                norm_wrist = (wrist_vel - np.mean(wrist_vel)) / (np.std(wrist_vel) + 1e-6)
                
                # Compute DTW path
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
                
                # Create aligned arrays
                aligned_imu = imu_data[imu_unique]
                aligned_skel = skel_data[skel_unique]
                aligned_ts = timestamps[imu_unique]
                
                return aligned_imu, aligned_skel, aligned_ts
            else:
                # Fall back to interpolation if skeleton format unexpected
                method = 'interpolation'
        except ImportError:
            print("DTW package not found, falling back to interpolation")
            method = 'interpolation'
        except Exception as e:
            print(f"DTW alignment failed: {e}, falling back to interpolation")
            method = 'interpolation'
    
    if method == 'interpolation':
        try:
            # Generate timestamps for skeleton (assuming 30fps)
            skel_timestamps = np.arange(skel_data.shape[0]) / 30.0
            
            # Find common time range
            t_min = max(timestamps[0], skel_timestamps[0])
            t_max = min(timestamps[-1], skel_timestamps[-1])
            
            # Ensure sufficient overlap
            if t_max <= t_min:
                print("No time overlap between modalities")
                return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
            
            # Filter IMU data to common range
            imu_mask = (timestamps >= t_min) & (timestamps <= t_max)
            if np.sum(imu_mask) < 10:  # Need at least 10 points
                print("Insufficient IMU data in common time range")
                return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
            
            filtered_imu = imu_data[imu_mask]
            filtered_ts = timestamps[imu_mask]
            
            # Interpolate skeleton data to IMU timestamps
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
            print(f"Interpolation failed: {e}, falling back to crop")
            method = 'crop'
    
    if method == 'crop':
        # Simple approach - use common length
        min_len = min(imu_data.shape[0], skel_data.shape[0])
        
        if min_len < 10:  # Need at least 10 points
            print("Insufficient common data length")
            return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
        
        return imu_data[:min_len], skel_data[:min_len], timestamps[:min_len]
    
    # If we reach here, all methods failed
    print(f"All alignment methods failed for method {method}")
    return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
