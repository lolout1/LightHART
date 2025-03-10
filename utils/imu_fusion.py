# utils/imu_fusion.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from dtaidistance import dtw

class IMUFusionBase:
    """
    Base class for IMU fusion implementations with parameter calibration.
    Specifically designed for pre-processed linear acceleration data.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01):
        self.dt = dt  # Base time step (will be adjusted with timestamps)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.name = "Base"
        self.calibrated = False
        
    def initialize(self):
        """Initialize the filter - implemented by subclasses"""
        raise NotImplementedError
        
    def update(self, accel, gyro, dt=None):
        """Update filter with new measurements - implemented by subclasses"""
        raise NotImplementedError
    
    def process_sequence(self, accel_seq, gyro_seq, timestamps=None):
        """Process a sequence of measurements"""
        raise NotImplementedError
        
    def reset(self):
        """Reset filter state"""
        self.initialize()
        
    def visualize_orientation(self, orientations, timestamps=None, title=None):
        """Visualize orientation estimates"""
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
        """Update filter parameters"""
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
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise)
        self.name = "Standard KF"
        # State vector: [ax, ay, az, gx, gy, gz, roll, pitch, yaw, bias_gx, bias_gy, bias_gz]
        # Measurement vector: [ax, ay, az, gx, gy, gz]
        self.dim_x = 12
        self.dim_z = 6
        self.filter = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        self.initialize()
        
    def initialize(self):
        """Initialize filter matrices"""
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
        """Update filter with new measurements"""
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
        """Process an entire sequence of IMU data"""
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
        
        # Combine all features into a single array
        augmented_data = np.hstack((
            accel_seq,           # Original linear acceleration (3)
            gyro_seq,            # Original gyroscope (3)
            filtered_accel_seq,  # Filtered linear acceleration (3)
            orientation_seq,     # Orientation angles (3)
            accel_mag,           # Magnitude of linear acceleration (1)
            gyro_mag,            # Magnitude of angular velocity (1)
            jerk_mag             # Magnitude of jerk (1)
        ))
        
        return augmented_data


class ExtendedKalmanIMU(IMUFusionBase):
    """
    Extended Kalman Filter for IMU processing.
    Designed specifically for pre-processed linear acceleration data.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1,
                 gyro_bias_noise=0.01):
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
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        r = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w] format
        return r.as_euler('xyz')
    
    def normalize_quaternion(self, q):
        """Normalize quaternion to maintain unit length"""
        return q / np.linalg.norm(q)
    
    def state_transition(self, x, dt):
        """Nonlinear state transition function"""
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
        """Jacobian of state transition function"""
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
        
        For linear acceleration data, we directly measure acceleration in sensor frame,
        so we use the linear acceleration state directly.
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
        """Jacobian of measurement function"""
        # Initialize measurement Jacobian
        H = np.zeros((6, 13))
        
        # Linear acceleration directly measures accel state
        H[0:3, 10:13] = np.eye(3)
        
        # Gyroscope directly measures angular velocity
        H[3:6, 4:7] = np.eye(3)
        
        return H
    
    def update(self, accel, gyro, dt=None):
        """Update filter with new measurements"""
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
        """Process an entire sequence of IMU data"""
        n_samples = len(accel_seq)
        orientation_seq = np.zeros((n_samples, 3))
        filtered_accel_seq = np.zeros((n_samples, 3))
        
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
            accel_mag,           # Magnitude of linear acceleration (1)
            gyro_mag,            # Magnitude of angular velocity (1)
            jerk_mag             # Magnitude of jerk (1)
        ))
        
        return augmented_data


class UnscentedKalmanIMU(IMUFusionBase):
    """
    Unscented Kalman Filter implementation for orientation estimation.
    Adapted to work with pre-processed linear acceleration data.
    """
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, gyro_bias_noise=0.01):
        super().__init__(dt, process_noise, measurement_noise, gyro_bias_noise)
        self.name = "Unscented KF"
        
        from filterpy.kalman import UnscentedKalmanFilter
        from filterpy.kalman import MerweScaledSigmaPoints
        
        # State dimension: [q0,q1,q2,q3,wx,wy,wz,bx,by,bz,ax,ay,az]
        # (quaternion, angular velocity, gyro bias, linear acceleration)
        self.dim_x = 13
        self.dim_z = 6  # Measurement dimension: [ax,ay,az,gx,gy,gz]
        
        # Define sigma point calculation parameters
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2., kappa=0.)
        
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
        """Initialize UKF state and covariance matrices"""
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
        """Normalize quaternion to maintain unit length"""
        return q / np.linalg.norm(q)
    
    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        r = R.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w] format
        return r.as_euler('xyz')
    
    def state_transition_fn(self, x, dt):
        """State transition function for UKF"""
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
        
        With pre-processed linear acceleration data, measurements directly correspond
        to the linear acceleration state variables and gyroscope readings.
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
        """Update filter with new measurements"""
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
        """Process an entire sequence of IMU data with UKF"""
        n_samples = len(accel_seq)
        orientation_seq = np.zeros((n_samples, 3))
        filtered_accel_seq = np.zeros((n_samples, 3))
        
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
            accel_mag,           # Magnitude of linear acceleration (1)
            gyro_mag,            # Magnitude of angular velocity (1)
            jerk_mag             # Magnitude of jerk (1)
        ))
        
        return augmented_data
