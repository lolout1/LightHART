"""
Enhanced IMU fusion implementations with quaternion-based orientation.

Provides three Kalman filter variants:
1. Standard Kalman Filter: Linear approximation for sensor fusion
2. Extended Kalman Filter: Nonlinear model with first-order linearization
3. Unscented Kalman Filter: Nonlinear model using sigma points for better approximation

Each filter is optimized for wearable sensor fusion with quaternion representation 
to avoid gimbal lock and ensure smooth orientation estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from dtaidistance import dtw

class IMUFusionBase:
    """
    Base class for IMU fusion implementations with parameter calibration.
    Specifically designed for pre-processed linear acceleration data.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01):
        """
        Initialize the IMU fusion base class.
        
        Args:
            dt: Time step in seconds (default: 1/30 s)
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            gyro_bias_noise: Gyroscope bias noise variance
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.name = "Base"
        self.calibrated = False
        
    def initialize(self):
        """Initialize the filter - implemented by subclasses."""
        raise NotImplementedError
        
    def update(self, accel, gyro, dt=None):
        """Update filter with new measurements - implemented by subclasses."""
        raise NotImplementedError
    
    def process_sequence(self, accel_seq, gyro_seq, timestamps=None):
        """Process a sequence of measurements."""
        raise NotImplementedError
        
    def reset(self):
        """Reset filter state."""
        self.initialize()
        
    def visualize_orientation(self, orientations, timestamps=None, title=None):
        """
        Visualize orientation estimates.
        
        Args:
            orientations: Array of orientation estimates (N, 3)
            timestamps: Optional array of timestamps (N,)
            title: Optional plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        if timestamps is None:
            timestamps = np.arange(len(orientations))
            
        labels = ['Roll', 'Pitch', 'Yaw']
        for i in range(3):
            axes[i].plot(timestamps, orientations[:, i], '-')
            axes[i].set_ylabel(f'{labels[i]} (rad)')
            axes[i].grid(True)
            
        axes[2].set_xlabel('Time (s)')
        if title:
            plt.suptitle(f'Orientation Estimates: {title}')
        plt.tight_layout()
        return fig
    
    def set_parameters(self, process_noise, measurement_noise, gyro_bias_noise):
        """
        Update filter parameters.
        
        Args:
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            gyro_bias_noise: Gyroscope bias noise variance
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.initialize()  # Reinitialize filter with new parameters


class StandardKalmanIMU(IMUFusionBase):
    """
    Standard Kalman Filter for IMU processing.
    Specifically designed for pre-processed linear acceleration data.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1,
                 gyro_bias_noise=0.01):
        """
        Initialize Standard Kalman Filter.
        
        Args:
            dt: Time step in seconds (default: 1/30 s)
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            gyro_bias_noise: Gyroscope bias noise variance
        """
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise)
        self.name = "Standard KF"
        # State vector: [ax, ay, az, gx, gy, gz, roll, pitch, yaw, bias_gx, bias_gy, bias_gz]
        # Measurement vector: [ax, ay, az, gx, gy, gz]
        self.dim_x = 12
        self.dim_z = 6
        self.filter = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        self.initialize()
        
    def initialize(self):
        """Initialize filter matrices."""
        dt = self.dt
        
        # State transition matrix (F)
        self.filter.F = np.eye(self.dim_x)
        # Orientation updated by gyro rates minus biases
        self.filter.F[6:9, 3:6] = np.eye(3) * dt
        self.filter.F[6:9, 9:12] = -np.eye(3) * dt
        
        # Measurement matrix (H) - direct measurements of linear accel and gyro
        self.filter.H = np.zeros((self.dim_z, self.dim_x))
        self.filter.H[0:3, 0:3] = np.eye(3)  # linear accel measurements
        self.filter.H[3:6, 3:6] = np.eye(3)  # gyro measurements
        
        # Process noise covariance (Q)
        q_accel = Q_discrete_white_noise(dim=3, dt=dt, var=self.measurement_noise*5)
        q_gyro = Q_discrete_white_noise(dim=3, dt=dt, var=self.process_noise)
        q_orientation = Q_discrete_white_noise(dim=3, dt=dt, var=self.process_noise)
        q_bias = Q_discrete_white_noise(dim=3, dt=dt, var=self.gyro_bias_noise)
        self.filter.Q = block_diag(q_accel, q_gyro, q_orientation, q_bias)
        
        # Measurement noise covariance (R)
        self.filter.R = np.eye(self.dim_z) * self.measurement_noise
        self.filter.R[0:3, 0:3] *= 5  # Higher noise for accelerometer
        
        # Initial state
        self.filter.x = np.zeros(self.dim_x)
        
        # Initial state covariance (P)
        self.filter.P = np.eye(self.dim_x)
        self.filter.P[0:3, 0:3] *= 5  # Higher uncertainty for accel
        self.filter.P[3:6, 3:6] *= 1  # Gyro initial uncertainty
        self.filter.P[6:9, 6:9] *= 10  # High uncertainty in initial orientation
        self.filter.P[9:12, 9:12] *= 0.1  # Low uncertainty in initial bias (assume zero)
        
    def update(self, accel, gyro, dt=None):
        """
        Update filter with new measurements.
        
        Args:
            accel: Accelerometer measurement (3,)
            gyro: Gyroscope measurement (3,)
            dt: Time step (optional)
            
        Returns:
            Tuple of (orientation, filtered_accel)
        """
        if dt is not None and dt > 0:
            # Update state transition matrix for variable time steps
            self.filter.F[6:9, 3:6] = np.eye(3) * dt
            self.filter.F[6:9, 9:12] = -np.eye(3) * dt
            
            # Update process noise for variable time steps
            q_accel = Q_discrete_white_noise(dim=3, dt=dt, var=self.measurement_noise*5)
            q_gyro = Q_discrete_white_noise(dim=3, dt=dt, var=self.process_noise)
            q_orientation = Q_discrete_white_noise(dim=3, dt=dt, var=self.process_noise)
            q_bias = Q_discrete_white_noise(dim=3, dt=dt, var=self.gyro_bias_noise)
            self.filter.Q = block_diag(q_accel, q_gyro, q_orientation, q_bias)
        
        # Concatenate measurements
        z = np.hstack((accel, gyro))
        
        # Predict and update
        self.filter.predict()
        self.filter.update(z)
        
        # Extract orientation (roll, pitch, yaw)
        orientation = self.filter.x[6:9]
        
        # Return orientation and filtered linear acceleration
        # Note: We're NOT removing gravity again since the input is already linear acceleration
        filtered_accel = self.filter.x[0:3]
        
        return orientation, filtered_accel
    
    def process_sequence(self, accel_seq, gyro_seq, timestamps=None):
        """
        Process an entire sequence of IMU data.
        
        Args:
            accel_seq: Accelerometer data (N, 3)
            gyro_seq: Gyroscope data (N, 3)
            timestamps: Optional timestamps (N,)
            
        Returns:
            Augmented data with orientation and derived features
        """
        n_samples = len(accel_seq)
        orientation_seq = np.zeros((n_samples, 3))
        filtered_accel_seq = np.zeros((n_samples, 3))
        
        self.reset()  # Make sure we start fresh
        
        # Handle variable time steps
        if timestamps is not None:
            dt_values = np.diff(timestamps, prepend=timestamps[0])
            dt_values[0] = self.dt  # Use default for first sample
        else:
            dt_values = np.ones(n_samples) * self.dt
            
        for i in range(n_samples):
            dt = dt_values[i]
            orientation, filtered_accel = self.update(accel_seq[i], gyro_seq[i], dt)
            orientation_seq[i] = orientation
            filtered_accel_seq[i] = filtered_accel
            
        # Calculate derived features
        accel_mag = np.linalg.norm(accel_seq, axis=1).reshape(-1, 1)  # Use original linear accel
        gyro_mag = np.linalg.norm(gyro_seq, axis=1).reshape(-1, 1)
        
        # Calculate jerk (derivative of acceleration)
        jerk = np.zeros_like(accel_seq)
        if n_samples > 1:
            for i in range(1, n_samples):
                dt = dt_values[i] if timestamps is not None else self.dt
                if dt > 0:
                    jerk[i] = (accel_seq[i] - accel_seq[i-1]) / dt
        jerk_mag = np.linalg.norm(jerk, axis=1).reshape(-1, 1)
        
        # Convert Euler orientation to quaternions for better representation
        quat_orientation = np.zeros((n_samples, 4))
        for i in range(n_samples):
            r = R.from_euler('xyz', orientation_seq[i])
            quat_orientation[i] = r.as_quat()  # [x, y, z, w] format
        
        # Combine all features into a single array
        augmented_data = np.hstack((
            accel_seq,           # Original linear acceleration (3)
            gyro_seq,            # Original gyroscope (3)
            filtered_accel_seq,  # Filtered linear acceleration (3)
            orientation_seq,     # Orientation angles (3)
            quat_orientation,    # Quaternion orientation (4)
            accel_mag,           # Magnitude of linear acceleration (1)
            gyro_mag,            # Magnitude of angular velocity (1)
            jerk_mag             # Magnitude of jerk (1)
        ))
        
        return augmented_data


class ExtendedKalmanIMU(IMUFusionBase):
    """
    Extended Kalman Filter for IMU processing.
    Designed specifically for pre-processed linear acceleration data.
    Uses nonlinear state transition model for quaternion-based orientation.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1,
                 gyro_bias_noise=0.01):
        """
        Initialize Extended Kalman Filter.
        
        Args:
            dt: Time step in seconds (default: 1/30 s)
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            gyro_bias_noise: Gyroscope bias noise variance
        """
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise)
        self.name = "Extended KF"
        # State: [q0,q1,q2,q3,wx,wy,wz,bx,by,bz,ax,ay,az] 
        # (quaternion, angular velocity, gyro bias, linear acceleration)
        self.state = np.zeros(13)
        self.state[0] = 1.0  # Initial quaternion is identity rotation
        
        # Covariance matrix
        self.P = np.eye(13) * 0.01
        self.P[0:4, 0:4] *= 0.1    # Quaternion covariance
        self.P[4:7, 4:7] *= 1.0    # Angular velocity covariance
        self.P[7:10, 7:10] *= 0.1  # Bias covariance
        self.P[10:13, 10:13] *= 5.0  # Linear acceleration covariance
        
        # Process noise covariance
        self.Q = np.eye(13) * 0.01
        self.Q[0:4, 0:4] *= 0.01              # Quaternion process noise
        self.Q[4:7, 4:7] *= self.process_noise  # Angular velocity process noise
        self.Q[7:10, 7:10] *= self.gyro_bias_noise  # Bias process noise
        self.Q[10:13, 10:13] *= self.measurement_noise * 5  # Linear accel process noise
        
        # Measurement noise covariance
        self.R = np.eye(6) * self.measurement_noise
        self.R[0:3, 0:3] *= 5.0  # Linear accelerometer noise
        self.R[3:6, 3:6] *= 1.0  # Gyroscope noise
        
    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Euler angles [roll, pitch, yaw]
        """
        r = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w] format
        return r.as_euler('xyz')
    
    def normalize_quaternion(self, q):
        """
        Normalize quaternion to maintain unit length.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Normalized quaternion
        """
        return q / np.linalg.norm(q)
    
    def state_transition(self, x, dt):
        """
        Nonlinear state transition function.
        
        Args:
            x: State vector
            dt: Time step
            
        Returns:
            Updated state vector
        """
        # Extract states
        q = x[0:4]  # quaternion [w, x, y, z]
        w = x[4:7]  # angular velocity
        b = x[7:10]  # gyro bias
        a = x[10:13]  # linear acceleration
        
        # Corrected angular velocity
        w_corrected = w - b
        
        # Quaternion derivative
        q_dot = 0.5 * np.array([
            -q[1]*w_corrected[0] - q[2]*w_corrected[1] - q[3]*w_corrected[2],
            q[0]*w_corrected[0] + q[2]*w_corrected[2] - q[3]*w_corrected[1],
            q[0]*w_corrected[1] + q[3]*w_corrected[0] - q[1]*w_corrected[2],
            q[0]*w_corrected[2] + q[1]*w_corrected[1] - q[2]*w_corrected[0]
        ])
        
        # Update quaternion
        q_new = q + q_dot * dt
        q_new = self.normalize_quaternion(q_new)
        
        # Bias is modeled as constant (slow random walk)
        b_new = b
        
        # Angular velocity is also modeled as constant (with noise)
        w_new = w
        
        # Linear acceleration is modeled as constant (with noise)
        a_new = a
        
        # Combine updated states
        x_new = np.concatenate([q_new, w_new, b_new, a_new])
        return x_new
    
    def state_transition_jacobian(self, x, dt):
        """
        Jacobian of state transition function.
        
        Args:
            x: State vector
            dt: Time step
            
        Returns:
            Jacobian matrix of state transition
        """
        # Extract states
        q = x[0:4]  # quaternion [w, x, y, z]
        w = x[4:7]  # angular velocity
        b = x[7:10]  # gyro bias
        
        # Corrected angular velocity
        w_corrected = w - b
        
        # Initialize Jacobian
        F = np.eye(13)
        
        # Jacobian of quaternion update with respect to quaternion
        F[0:4, 0:4] = np.array([
            [1, -0.5*w_corrected[0]*dt, -0.5*w_corrected[1]*dt, -0.5*w_corrected[2]*dt],
            [0.5*w_corrected[0]*dt, 1, 0.5*w_corrected[2]*dt, -0.5*w_corrected[1]*dt],
            [0.5*w_corrected[1]*dt, -0.5*w_corrected[2]*dt, 1, 0.5*w_corrected[0]*dt],
            [0.5*w_corrected[2]*dt, 0.5*w_corrected[1]*dt, -0.5*w_corrected[0]*dt, 1]
        ])
        
        # Jacobian of quaternion update with respect to angular velocity
        F[0:4, 4:7] = np.array([
            [-0.5*q[1]*dt, -0.5*q[2]*dt, -0.5*q[3]*dt],
            [0.5*q[0]*dt, -0.5*q[3]*dt, 0.5*q[2]*dt],
            [0.5*q[3]*dt, 0.5*q[0]*dt, -0.5*q[1]*dt],
            [-0.5*q[2]*dt, 0.5*q[1]*dt, 0.5*q[0]*dt]
        ])
        
        # Jacobian of quaternion update with respect to bias
        F[0:4, 7:10] = -F[0:4, 4:7]
        
        return F
    
    def measurement_function(self, x):
        """
        Nonlinear measurement function - adapted for linear acceleration
        
        Args:
            x: State vector
            
        Returns:
            Expected measurement vector
        """
        # Extract states
        q = x[0:4]      # quaternion [w, x, y, z]
        w = x[4:7]      # angular velocity
        a = x[10:13]    # linear acceleration in sensor frame
        
        # Expected accelerometer reading (linear acceleration)
        expected_accel = a
        
        # Expected gyroscope reading is just angular velocity
        expected_gyro = w
        
        # Combine expected measurements
        z_expected = np.concatenate([expected_accel, expected_gyro])
        return z_expected
    
    def measurement_jacobian(self, x):
        """
        Jacobian of measurement function.
        
        Args:
            x: State vector
            
        Returns:
            Jacobian matrix of measurement function
        """
        # Initialize measurement Jacobian
        H = np.zeros((6, 13))
        
        # Linear acceleration directly measures accel state
        H[0:3, 10:13] = np.eye(3)
        
        # Gyroscope directly measures angular velocity
        H[3:6, 4:7] = np.eye(3)
        
        return H
    
    def update(self, accel, gyro, dt=None):
        """
        Update filter with new measurements.
        
        Args:
            accel: Accelerometer measurement (3,)
            gyro: Gyroscope measurement (3,)
            dt: Time step (optional)
            
        Returns:
            Tuple of (orientation, filtered_accel)
        """
        if dt is None:
            dt = self.dt
            
        # Measurement vector
        z = np.concatenate([accel, gyro])
        
        # Predict step
        x_pred = self.state_transition(self.state, dt)
        F = self.state_transition_jacobian(self.state, dt)
        self.P = F @ self.P @ F.T + self.Q
        
        # Update step
        z_pred = self.measurement_function(x_pred)
        H = self.measurement_jacobian(x_pred)
        
        y = z - z_pred  # Innovation
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = x_pred + K @ y
        self.state[0:4] = self.normalize_quaternion(self.state[0:4])
        
        # Joseph form for covariance update (more numerically stable)
        I = np.eye(len(self.state))
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        
        # Extract orientation as Euler angles
        orientation = self.quaternion_to_euler(self.state[0:4])
        
        # Return orientation and filtered linear acceleration
        filtered_accel = self.state[10:13]
        
        return orientation, filtered_accel
    
    def process_sequence(self, accel_seq, gyro_seq, timestamps=None):
        """
        Process an entire sequence of IMU data.
        
        Args:
            accel_seq: Accelerometer data (N, 3)
            gyro_seq: Gyroscope data (N, 3)
            timestamps: Optional timestamps (N,)
            
        Returns:
            Augmented data with orientation and derived features
        """
        n_samples = len(accel_seq)
        orientation_seq = np.zeros((n_samples, 3))
        filtered_accel_seq = np.zeros((n_samples, 3))
        quat_orientation = np.zeros((n_samples, 4))
        
        # Reset filter state
        self.state = np.zeros(13)
        self.state[0] = 1.0  # Initial quaternion is identity rotation
        self.P = np.eye(13) * 0.01
        self.P[0:4, 0:4] *= 0.1    # Quaternion covariance
        self.P[4:7, 4:7] *= 1.0    # Angular velocity covariance
        self.P[7:10, 7:10] *= 0.1  # Bias covariance
        self.P[10:13, 10:13] *= 5.0  # Linear acceleration covariance
        
        # Handle variable time steps
        if timestamps is not None:
            dt_values = np.diff(timestamps, prepend=timestamps[0])
            dt_values[0] = self.dt  # Use default for first sample
        else:
            dt_values = np.ones(n_samples) * self.dt
            
        for i in range(n_samples):
            dt = dt_values[i]
            orientation, filtered_accel = self.update(accel_seq[i], gyro_seq[i], dt)
            orientation_seq[i] = orientation
            filtered_accel_seq[i] = filtered_accel
            quat_orientation[i] = self.state[0:4]  # Store quaternion orientation
        
        # Calculate derived features
        accel_mag = np.linalg.norm(accel_seq, axis=1).reshape(-1, 1)  # Original linear accel magnitude
        gyro_mag = np.linalg.norm(gyro_seq, axis=1).reshape(-1, 1)
        
        # Calculate jerk (derivative of acceleration)
        jerk = np.zeros_like(accel_seq)
        if n_samples > 1:
            for i in range(1, n_samples):
                dt = dt_values[i] if timestamps is not None else self.dt
                if dt > 0:
                    jerk[i] = (accel_seq[i] - accel_seq[i-1]) / dt
        jerk_mag = np.linalg.norm(jerk, axis=1).reshape(-1, 1)
        
        # Combine all features into a single array
        augmented_data = np.hstack((
            accel_seq,           # Original linear acceleration (3)
            gyro_seq,            # Original gyroscope (3)
            filtered_accel_seq,  # Filtered linear acceleration (3)
            orientation_seq,     # Orientation angles (3)
            quat_orientation,    # Quaternion orientation (4)
            accel_mag,           # Magnitude of linear acceleration (1)
            gyro_mag,            # Magnitude of angular velocity (1)
            jerk_mag             # Magnitude of jerk (1)
        ))
        
        return augmented_data


class UnscentedKalmanIMU(IMUFusionBase):
    """
    Unscented Kalman Filter implementation for orientation estimation.
    Adapted to work with pre-processed linear acceleration data.
    Uses sigma points to better handle nonlinearities in orientation.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, gyro_bias_noise=0.01,
                 alpha=0.1, beta=2.0, kappa=0.0):
        """
        Initialize Unscented Kalman Filter.
        
        Args:
            dt: Time step in seconds (default: 1/30 s)
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            gyro_bias_noise: Gyroscope bias noise variance
            alpha: UKF tuning parameter (default: 0.1)
            beta: UKF tuning parameter (default: 2.0)
            kappa: UKF tuning parameter (default: 0.0)
        """
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise)
        self.name = "Unscented KF"
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # State dimension: [q0,q1,q2,q3,wx,wy,wz,bx,by,bz,ax,ay,az]
        # (quaternion, angular velocity, gyro bias, linear acceleration)
        self.dim_x = 13
        self.dim_z = 6  # Measurement dimension: [ax,ay,az,gx,gy,gz]
        
        # Define sigma point calculation parameters
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=alpha, beta=beta, kappa=kappa)
        
        # Create UKF
        self.filter = UnscentedKalmanFilter(dim_x=self.dim_x, 
                                           dim_z=self.dim_z,
                                           dt=self.dt,
                                           fx=self.state_transition_fn,
                                           hx=self.measurement_fn,
                                           points=points)
        
        # Initialize state and covariance
        self.initialize()
        
    def initialize(self):
        """Initialize UKF state and covariance matrices."""
        # Initial state: identity quaternion, zero angular velocity, zero bias, zero linear accel
        self.filter.x = np.zeros(self.dim_x)
        self.filter.x[0] = 1.0  # w component of quaternion
        
        # Initial covariance
        self.filter.P = np.eye(self.dim_x) * 0.01
        self.filter.P[0:4, 0:4] *= 0.1      # Quaternion covariance
        self.filter.P[4:7, 4:7] *= 1.0      # Angular velocity covariance
        self.filter.P[7:10, 7:10] *= 0.1    # Bias covariance
        self.filter.P[10:13, 10:13] *= 5.0  # Linear acceleration covariance
        
        # Process noise covariance
        self.filter.Q = np.eye(self.dim_x) * 0.01
        self.filter.Q[0:4, 0:4] *= 0.01     # Quaternion process noise
        self.filter.Q[4:7, 4:7] *= self.process_noise  # Angular velocity process noise
        self.filter.Q[7:10, 7:10] *= self.gyro_bias_noise  # Bias process noise
        self.filter.Q[10:13, 10:13] *= self.measurement_noise * 5  # Linear accel process noise
        
        # Measurement noise covariance
        self.filter.R = np.eye(self.dim_z) * self.measurement_noise
        self.filter.R[0:3, 0:3] *= 5.0  # Accelerometer noise
        self.filter.R[3:6, 3:6] *= 1.0  # Gyroscope noise
        
    def normalize_quaternion(self, q):
        """
        Normalize quaternion to maintain unit length.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Normalized quaternion
        """
        return q / np.linalg.norm(q)
    
    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Euler angles [roll, pitch, yaw]
        """
        r = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w] format
        return r.as_euler('xyz')
    
    def state_transition_fn(self, x, dt):
        """
        State transition function for UKF.
        
        Args:
            x: State vector
            dt: Time step
            
        Returns:
            Updated state vector
        """
        # Extract states
        q = x[0:4]    # quaternion [w, x, y, z]
        w = x[4:7]    # angular velocity
        b = x[7:10]   # gyro bias
        a = x[10:13]  # linear acceleration
        
        # Corrected angular velocity
        w_corrected = w - b
        
        # Quaternion derivative
        q_dot = 0.5 * np.array([
            -q[1]*w_corrected[0] - q[2]*w_corrected[1] - q[3]*w_corrected[2],
            q[0]*w_corrected[0] + q[2]*w_corrected[2] - q[3]*w_corrected[1],
            q[0]*w_corrected[1] + q[3]*w_corrected[0] - q[1]*w_corrected[2],
            q[0]*w_corrected[2] + q[1]*w_corrected[1] - q[2]*w_corrected[0]
        ])
        
        # Update quaternion
        q_new = q + q_dot * dt
        q_new = self.normalize_quaternion(q_new)
        
        # Bias is modeled as constant (slow random walk)
        b_new = b
        
        # Angular velocity is also modeled as constant (with noise)
        w_new = w
        
        # Linear acceleration is modeled as constant (with noise)
        a_new = a
        
        # Combine updated states
        x_new = np.concatenate([q_new, w_new, b_new, a_new])
        return x_new
    
    def measurement_fn(self, x):
        """
        Measurement function for UKF - adapted for linear acceleration
        
        Args:
            x: State vector
            
        Returns:
            Expected measurement vector
        """
        # Extract states
        w = x[4:7]      # angular velocity
        a = x[10:13]    # linear acceleration in sensor frame
        
        # Expected accelerometer reading is the linear acceleration state
        expected_accel = a
        
        # Expected gyroscope reading is just angular velocity
        expected_gyro = w
        
        # Combine expected measurements
        z_expected = np.concatenate([expected_accel, expected_gyro])
        return z_expected
    
    def update(self, accel, gyro, dt=None):
        """
        Update filter with new measurements.
        
        Args:
            accel: Accelerometer measurement (3,)
            gyro: Gyroscope measurement (3,)
            dt: Time step (optional)
            
        Returns:
            Tuple of (orientation, filtered_accel)
        """
        if dt is not None:
            self.filter.dt = dt
            
        # Measurement vector - using linear acceleration directly
        z = np.concatenate([accel, gyro])
        
        # Predict and update
        self.filter.predict()
        self.filter.update(z)
        
        # Normalize quaternion part of state
        self.filter.x[0:4] = self.normalize_quaternion(self.filter.x[0:4])
        
        # Extract orientation
        orientation = self.quaternion_to_euler(self.filter.x[0:4])
        
        # Return filtered linear acceleration directly
        filtered_accel = self.filter.x[10:13]
        
        return orientation, filtered_accel
    
    def process_sequence(self, accel_seq, gyro_seq, timestamps=None):
        """
        Process an entire sequence of IMU data with UKF.
        
        Args:
            accel_seq: Accelerometer data (N, 3)
            gyro_seq: Gyroscope data (N, 3)
            timestamps: Optional timestamps (N,)
            
        Returns:
            Augmented data with orientation and derived features
        """
        n_samples = len(accel_seq)
        orientation_seq = np.zeros((n_samples, 3))
        filtered_accel_seq = np.zeros((n_samples, 3))
        quat_orientation = np.zeros((n_samples, 4))
        
        # Reset filter state
        self.initialize()
        
        # Handle variable time steps
        if timestamps is not None:
            dt_values = np.diff(timestamps, prepend=timestamps[0])
            dt_values[0] = self.dt  # Use default for first sample
        else:
            dt_values = np.ones(n_samples) * self.dt
            
        for i in range(n_samples):
            dt = dt_values[i]
            orientation, filtered_accel = self.update(accel_seq[i], gyro_seq[i], dt)
            orientation_seq[i] = orientation
            filtered_accel_seq[i] = filtered_accel
            quat_orientation[i] = self.filter.x[0:4]  # Store quaternion orientation
        
        # Calculate derived features
        accel_mag = np.linalg.norm(accel_seq, axis=1).reshape(-1, 1)  # Original linear accel magnitude
        gyro_mag = np.linalg.norm(gyro_seq, axis=1).reshape(-1, 1)
        
        # Calculate jerk (derivative of acceleration)
        jerk = np.zeros_like(accel_seq)
        if n_samples > 1:
            for i in range(1, n_samples):
                dt = dt_values[i] if timestamps is not None else self.dt
                if dt > 0:
                    jerk[i] = (accel_seq[i] - accel_seq[i-1]) / dt
        jerk_mag = np.linalg.norm(jerk, axis=1).reshape(-1, 1)
        
        # Calculate angular acceleration
        angular_accel = np.zeros_like(gyro_seq)
        if n_samples > 1:
            for i in range(1, n_samples):
                dt = dt_values[i] if timestamps is not None else self.dt
                if dt > 0:
                    angular_accel[i] = (gyro_seq[i] - gyro_seq[i-1]) / dt
        angular_accel_mag = np.linalg.norm(angular_accel, axis=1).reshape(-1, 1)
        
        # Combine all features into a single array
        augmented_data = np.hstack((
            accel_seq,           # Original linear acceleration (3)
            gyro_seq,            # Original gyroscope (3)
            filtered_accel_seq,  # Filtered linear acceleration (3)
            orientation_seq,     # Orientation angles (3)
            quat_orientation,    # Quaternion orientation (4)
            accel_mag,           # Magnitude of linear acceleration (1)
            gyro_mag,            # Magnitude of angular velocity (1)
            jerk_mag,            # Magnitude of jerk (1)
            angular_accel_mag    # Magnitude of angular acceleration (1)
        ))
        
        return augmented_data


def calibrate_filter(accel, gyro, skeleton, filter_type='ekf', timestamps=None, wrist_idx=9, num_joints=32):
    """
    Calibrate filter parameters by optimizing against skeleton ground truth.
    
    Args:
        accel: Accelerometer data (N, 3)
        gyro: Gyroscope data (N, 3)
        skeleton: Skeleton data (N, 3*num_joints)
        filter_type: Filter type ('standard', 'ekf', 'ukf')
        timestamps: Optional timestamps (N,)
        wrist_idx: Index of wrist joint in skeleton (default: 9)
        num_joints: Number of joints in skeleton (default: 32)
        
    Returns:
        Tuple of (calibrated_filter, [process_noise, measurement_noise, gyro_bias_noise])
    """
    # Extract wrist orientation from skeleton
    ref_orientation = extract_orientation_from_skeleton(skeleton, num_joints, wrist_idx)
    
    # Define optimization objective function
    def objective(params):
        # Extract parameters
        process_noise, measurement_noise, gyro_bias_noise = params
        
        # Create filter
        if filter_type == 'standard':
            filt = StandardKalmanIMU(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                gyro_bias_noise=gyro_bias_noise
            )
        elif filter_type == 'ekf':
            filt = ExtendedKalmanIMU(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                gyro_bias_noise=gyro_bias_noise
            )
        else:  # ukf
            filt = UnscentedKalmanIMU(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
                gyro_bias_noise=gyro_bias_noise
            )
        
        # Process sequence
        n_samples = min(len(accel), len(gyro), len(ref_orientation))
        orientations = np.zeros((n_samples, 3))
        
        # Handle timestamps
        if timestamps is not None and len(timestamps) >= n_samples:
            dt_values = np.diff(timestamps[:n_samples], prepend=timestamps[0])
        else:
            dt_values = np.ones(n_samples) * 1/30.0
        
        # Run filter
        filt.reset()
        for i in range(n_samples):
            orientation, _ = filt.update(accel[i], gyro[i], dt_values[i])
            orientations[i] = orientation
        
        # Calculate error (MSE)
        error = np.mean(np.sum((orientations - ref_orientation[:n_samples])**2, axis=1))
        return error
    
    # Initial parameter guess
    initial_guess = [0.01, 0.1, 0.01]
    
    # Parameter bounds
    bounds = [(0.001, 0.1), (0.01, 1.0), (0.001, 0.1)]
    
    # Optimize
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    # Get optimal parameters
    optimal_params = result.x
    
    # Create calibrated filter
    if filter_type == 'standard':
        calibrated_filter = StandardKalmanIMU(
            process_noise=optimal_params[0],
            measurement_noise=optimal_params[1],
            gyro_bias_noise=optimal_params[2]
        )
    elif filter_type == 'ekf':
        calibrated_filter = ExtendedKalmanIMU(
            process_noise=optimal_params[0],
            measurement_noise=optimal_params[1],
            gyro_bias_noise=optimal_params[2]
        )
    else:  # ukf
        calibrated_filter = UnscentedKalmanIMU(
            process_noise=optimal_params[0],
            measurement_noise=optimal_params[1],
            gyro_bias_noise=optimal_params[2]
        )
    
    return calibrated_filter, optimal_params


def extract_orientation_from_skeleton(skeleton_data, num_joints=32, wrist_idx=9):
    """
    Extract orientation from skeleton data.
    
    Args:
        skeleton_data: Skeleton data (N, 3*num_joints)
        num_joints: Number of joints in skeleton (default: 32)
        wrist_idx: Index of wrist joint (default: 9)
        
    Returns:
        Orientation angles (N, 3)
    """
    n_samples = skeleton_data.shape[0]
    
    # Reshape to (N, num_joints, 3) for easier access
    if skeleton_data.shape[1] == 3 * num_joints:
        skel_reshaped = skeleton_data.reshape(n_samples, num_joints, 3)
    else:
        raise ValueError(f"Unexpected skeleton data shape: {skeleton_data.shape}")
    
    # Get wrist position
    wrist_pos = skel_reshaped[:, wrist_idx, :]
    
    # Calculate orientations using wrist motion
    orientations = np.zeros((n_samples, 3))
    
    # Simple approach: use consecutive positions to estimate direction
    if n_samples > 1:
        for i in range(1, n_samples):
            delta = wrist_pos[i] - wrist_pos[i-1]
            
            # Calculate roll, pitch, yaw based on movement direction
            if np.linalg.norm(delta) > 1e-6:
                # Normalize
                delta = delta / np.linalg.norm(delta)
                
                # Calculate angles (simplified)
                pitch = np.arcsin(-delta[1])  # Elevation angle
                yaw = np.arctan2(delta[0], delta[2])  # Azimuth angle
                
                # Roll is harder to estimate from position only
                # Here's a simplified approach based on wrist position relative to body
                # In a real app, you'd use more joints for a better estimate
                roll = 0.0  # Default
                
                orientations[i] = [roll, pitch, yaw]
    
    return orientations


def robust_align_modalities(imu_data, skel_data, imu_ts, skel_fps=30.0, method='dtw', wrist_idx=9):
    """
    Align IMU (watch) and skeleton data using DTW or other methods.
    
    Args:
        imu_data: Array of shape (n_imu, 4+) with IMU data (time at index 0)
        skel_data: Array of shape (n_skel, 1+96) with skeleton data (time at index 0)
        imu_ts: Array of shape (n_imu,) with IMU timestamps
        skel_fps: Frame rate of skeleton data (default: 30.0 Hz)
        method: Alignment method ('dtw', 'simple')
        wrist_idx: Index of wrist joint to align with watch data (default: 9)
        
    Returns:
        Tuple of (aligned_imu, aligned_skel, aligned_timestamps)
    """
    if imu_data.shape[0] == 0 or skel_data.shape[0] == 0:
        return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
    
    # Extract skeleton wrist position for alignment
    skel_cols = skel_data.shape[1]
    joint_cols = 3  # x, y, z per joint
    
    if skel_cols >= 96 + 1:  # Time column + 96 joint coordinates (32 joints * 3)
        wrist_start_col = 1 + wrist_idx * joint_cols  # +1 for time column
        wrist_end_col = wrist_start_col + joint_cols
        
        # Extract wrist position from skeleton data
        wrist_pos = skel_data[:, wrist_start_col:wrist_end_col]
        
        # Calculate wrist velocity (magnitude)
        wrist_vel = np.zeros(skel_data.shape[0])
        for i in range(1, skel_data.shape[0]):
            dt = (skel_data[i, 0] - skel_data[i-1, 0]) if skel_data.shape[1] > 96 else 1.0/skel_fps
            if dt == 0:
                dt = 1.0/skel_fps
            wrist_vel[i] = np.linalg.norm(wrist_pos[i] - wrist_pos[i-1]) / dt
    else:
        # Fallback if skeleton data doesn't match expected format
        print("Warning: Skeleton data doesn't have expected number of columns for wrist extraction")
        wrist_vel = np.ones(skel_data.shape[0])
    
    # Calculate IMU total acceleration magnitude
    imu_mag = np.linalg.norm(imu_data[:, 1:4], axis=1)
    
    if method == 'dtw':
        try:
            # Normalize sequences
            norm_imu_mag = (imu_mag - np.mean(imu_mag)) / (np.std(imu_mag) + 1e-6)
            norm_wrist_vel = (wrist_vel - np.mean(wrist_vel)) / (np.std(wrist_vel) + 1e-6)
            
            # Calculate DTW path
            path = dtw.warping_path(norm_imu_mag, norm_wrist_vel)
            
            # Extract aligned indices
            imu_idx, skel_idx = zip(*path)
            
            # Remove duplicates while preserving order
            imu_unique_idx = []
            skel_unique_idx = []
            seen_imu = set()
            seen_skel = set()
            
            for i, s in zip(imu_idx, skel_idx):
                if i not in seen_imu:
                    imu_unique_idx.append(i)
                    seen_imu.add(i)
                if s not in seen_skel:
                    skel_unique_idx.append(s)
                    seen_skel.add(s)
            
            # Create aligned arrays
            aligned_imu = imu_data[imu_unique_idx]
            aligned_skel = skel_data[skel_unique_idx]
            aligned_ts = imu_ts[imu_unique_idx]
            
            return aligned_imu, aligned_skel, aligned_ts
            
        except Exception as e:
            print(f"DTW alignment failed: {e}, falling back to simple alignment")
            method = 'simple'
    
    if method == 'simple':
        # Simple approach: use the smaller length
        length = min(imu_data.shape[0], skel_data.shape[0])
        if length < 10:
            return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0)
        
        aligned_imu = imu_data[:length]
        aligned_skel = skel_data[:length]
        aligned_ts = imu_ts[:length]
        
        return aligned_imu, aligned_skel, aligned_ts


def compare_kalman_filters(accel_seq, gyro_seq, timestamps=None, window_size_sec=4.0, stride_sec=1.0):
    """
    Compare Standard, Extended, and Unscented Kalman filters on the same data.
    
    Args:
        accel_seq: Accelerometer data (N, 3)
        gyro_seq: Gyroscope data (N, 3)
        timestamps: Optional timestamps (N,)
        window_size_sec: Window size in seconds for segmentation
        stride_sec: Stride size in seconds for segmentation
        
    Returns:
        Dictionary with filter results
    """
    # Create filters
    standard_kf = StandardKalmanIMU()
    ekf = ExtendedKalmanIMU()
    ukf = UnscentedKalmanIMU()
    
    # Process sequences
    standard_result = standard_kf.process_sequence(accel_seq, gyro_seq, timestamps)
    ekf_result = ekf.process_sequence(accel_seq, gyro_seq, timestamps)
    ukf_result = ukf.process_sequence(accel_seq, gyro_seq, timestamps)
    
    # Extract orientations
    if timestamps is None:
        timestamps = np.arange(len(accel_seq)) / 30.0  # Assume 30 Hz
    
    standard_orientation = standard_result[:, 3:6]  # Orientation at indices 3-5
    ekf_orientation = ekf_result[:, 3:6]
    ukf_orientation = ukf_result[:, 3:6]
    
    # Extract quaternions
    standard_quat = standard_result[:, 6:10]  # Quaternion at indices 6-9
    ekf_quat = ekf_result[:, 6:10]
    ukf_quat = ukf_result[:, 6:10]
    
    # Create windows for analysis
    def create_windows(data, ts, window_size, stride):
        windows = []
        window_ts = []
        
        if len(ts) == 0:
            return windows, window_ts
        
        start_time = ts[0]
        end_time = ts[-1]
        
        t = start_time
        while t + window_size <= end_time:
            # Find indices within window
            indices = (ts >= t) & (ts < t + window_size)
            
            if np.sum(indices) > 0:
                windows.append(data[indices])
                window_ts.append(ts[indices])
            
            t += stride
        
        return windows, window_ts
    
    # Create windows for each result
    standard_windows, window_ts = create_windows(standard_result, timestamps, window_size_sec, stride_sec)
    ekf_windows, _ = create_windows(ekf_result, timestamps, window_size_sec, stride_sec)
    ukf_windows, _ = create_windows(ukf_result, timestamps, window_size_sec, stride_sec)
    
    # Calculate smoothness metric (average angular jerk)
    def calc_smoothness(orientation_windows, ts_windows):
        smoothness = []
        
        for i, (window, ts) in enumerate(zip(orientation_windows, ts_windows)):
            # Get orientation part
            orientation = window[:, 3:6]
            
            # Calculate angular jerk (derivative of angular acceleration)
            jerk = np.zeros_like(orientation)
            if len(orientation) > 2:
                for j in range(2, len(orientation)):
                    dt1 = ts[j] - ts[j-1]
                    dt2 = ts[j-1] - ts[j-2]
                    if dt1 > 0 and dt2 > 0:
                        acc1 = (orientation[j] - orientation[j-1]) / dt1
                        acc2 = (orientation[j-1] - orientation[j-2]) / dt2
                        jerk[j] = (acc1 - acc2) / ((dt1 + dt2) / 2)
            
            # RMS jerk magnitude
            jerk_mag = np.linalg.norm(jerk, axis=1)
            rms_jerk = np.sqrt(np.mean(jerk_mag**2))
            
            smoothness.append(rms_jerk)
        
        return np.mean(smoothness) if smoothness else 0
    
    # Calculate feature statistics
    def calc_statistics(windows):
        # If no windows, return empty stats
        if not windows:
            return {
                'mean': None,
                'std': None,
                'range': None,
                'features': None
            }
        
        # Concatenate all windows
        all_data = np.vstack(windows)
        
        # Calculate statistics for each feature
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        min_val = np.min(all_data, axis=0)
        max_val = np.max(all_data, axis=0)
        feature_range = max_val - min_val
        
        return {
            'mean': mean,
            'std': std,
            'range': feature_range,
            'features': all_data
        }
    
    # Calculate smoothness
    standard_smoothness = calc_smoothness(standard_windows, window_ts)
    ekf_smoothness = calc_smoothness(ekf_windows, window_ts)
    ukf_smoothness = calc_smoothness(ukf_windows, window_ts)
    
    # Calculate statistics
    standard_stats = calc_statistics(standard_windows)
    ekf_stats = calc_statistics(ekf_windows)
    ukf_stats = calc_statistics(ukf_windows)
    
    # Compare computation time (on a small batch for fairness)
    num_samples = min(1000, len(accel_seq))
    
    start_time = time.time()
    standard_kf.process_sequence(accel_seq[:num_samples], gyro_seq[:num_samples], 
                              None if timestamps is None else timestamps[:num_samples])
    standard_time = time.time() - start_time
    
    start_time = time.time()
    ekf.process_sequence(accel_seq[:num_samples], gyro_seq[:num_samples], 
                      None if timestamps is None else timestamps[:num_samples])
    ekf_time = time.time() - start_time
    
    start_time = time.time()
    ukf.process_sequence(accel_seq[:num_samples], gyro_seq[:num_samples], 
                      None if timestamps is None else timestamps[:num_samples])
    ukf_time = time.time() - start_time
    
    # Return comparison results
    return {
        'standard_kf': {
            'orientation': standard_orientation,
            'quaternion': standard_quat,
            'windows': standard_windows,
            'stats': standard_stats,
            'smoothness': standard_smoothness,
            'time': standard_time,
            'size': standard_result.shape[1]
        },
        'extended_kf': {
            'orientation': ekf_orientation,
            'quaternion': ekf_quat,
            'windows': ekf_windows,
            'stats': ekf_stats,
            'smoothness': ekf_smoothness,
            'time': ekf_time,
            'size': ekf_result.shape[1]
        },
        'unscented_kf': {
            'orientation': ukf_orientation,
            'quaternion': ukf_quat,
            'windows': ukf_windows,
            'stats': ukf_stats,
            'smoothness': ukf_smoothness,
            'time': ukf_time,
            'size': ukf_result.shape[1]
        }
    }
