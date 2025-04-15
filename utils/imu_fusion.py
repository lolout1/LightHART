import numpy as np
from scipy.spatial.transform import Rotation
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("imu_fusion")
thread_pool = ThreadPoolExecutor(max_workers=4)

VISUALIZATION_DIR = 'visualization_output'
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def hybrid_interpolate(time1, data1, time2, data2, target_time=None, method='linear'):
    if target_time is None:
        target_time = time1
    try:
        f1 = interp1d(time1, data1, kind=method, axis=0, bounds_error=False, fill_value="extrapolate")
        interp_data1 = f1(target_time)
    except:
        f1 = interp1d(time1, data1, kind='linear', axis=0, bounds_error=False, fill_value="extrapolate")
        interp_data1 = f1(target_time)
    try:
        f2 = interp1d(time2, data2, kind=method, axis=0, bounds_error=False, fill_value="extrapolate")
        interp_data2 = f2(target_time)
    except:
        f2 = interp1d(time2, data2, kind='linear', axis=0, bounds_error=False, fill_value="extrapolate")
        interp_data2 = f2(target_time)
    return interp_data1, interp_data2
def save_aligned_sensor_data(subject_id, action_id, trial_id, acc_data, gyro_data, quaternions=None, timestamps=None, save_dir="data/aligned"):
    try:
        os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
        os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
        if quaternions is not None: os.makedirs(f"{save_dir}/quaternion", exist_ok=True)
        filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
        np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
        np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
        if quaternions is not None: np.save(f"{save_dir}/quaternion/{filename}.npy", quaternions)
        if timestamps is not None:
            os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
            np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)
        logger.info(f"Saved aligned data for {filename}")
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")
def bandpass_filter(data, lowcut=0.5, highcut=15.0, fs=30.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(0.001, min(0.999, low))
    high = max(low + 0.001, min(0.999, high))
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    return filtered

def align_sensor_data(acc_df, gyro_df, target_freq=30.0, visualize=False, trial_id=None):
    logger.info(f"Starting sensor alignment with target frequency {target_freq}Hz")
    start_time = time.time()
    try:
        acc_timestamps = acc_df['timestamp'].values
        gyro_timestamps = gyro_df['timestamp'].values
        acc_data = acc_df.iloc[:, 1:4].values
        gyro_data = gyro_df.iloc[:, 1:4].values
        if len(acc_timestamps) < 3 or len(gyro_timestamps) < 3:
            logger.warning("Insufficient data points for alignment")
            return None, None, None
        start_time_point = max(acc_timestamps[0], gyro_timestamps[0])
        end_time_point = min(acc_timestamps[-1], gyro_timestamps[-1])
        if start_time_point >= end_time_point:
            logger.warning("No temporal overlap between sensors")
            return None, None, None
        duration_sec = (end_time_point - start_time_point)
        if duration_sec <= 0:
            logger.warning(f"Invalid duration: {duration_sec} seconds")
            return None, None, None
        n_samples = max(10, int(duration_sec * target_freq))
        common_timestamps = np.linspace(start_time_point, end_time_point, n_samples)
        aligned_acc = np.zeros((n_samples, 3))
        aligned_gyro = np.zeros((n_samples, 3))
        for axis in range(3):
            acc_interp = interp1d(acc_timestamps, acc_data[:, axis], bounds_error=False, fill_value="extrapolate")
            aligned_acc[:, axis] = acc_interp(common_timestamps)
            gyro_interp = interp1d(gyro_timestamps, gyro_data[:, axis], bounds_error=False, fill_value="extrapolate")
            aligned_gyro[:, axis] = gyro_interp(common_timestamps)
        safe_freq = max(5.0, min(1000.0, target_freq))
        aligned_acc = bandpass_filter(aligned_acc, lowcut=0.1, highcut=min(safe_freq/2.1, 15.0), fs=safe_freq)
        aligned_gyro = bandpass_filter(aligned_gyro, lowcut=0.1, highcut=min(safe_freq/2.1, 12.0), fs=safe_freq)
        if visualize and trial_id:
            visualize_alignment(acc_data, gyro_data, aligned_acc, aligned_gyro,
                              acc_timestamps, gyro_timestamps, common_timestamps, trial_id)
        elapsed = time.time() - start_time
        logger.info(f"Sensor alignment complete: {n_samples} aligned samples in {elapsed:.2f}s")
        return aligned_acc, aligned_gyro, common_timestamps
    except Exception as e:
        logger.error(f"Error during sensor alignment: {str(e)}")
        return None, None, None

def add_gravity(linear_acc, quaternion):
    """
    Add gravity component back to linear acceleration based on orientation quaternion.
    
    Args:
        linear_acc: Linear acceleration vector [ax, ay, az] in m/s^2
        quaternion: Orientation quaternion [qw, qx, qy, qz]
        
    Returns:
        Raw acceleration with gravity component
    """
    try:
        # Convert to scipy's rotation quaternion format [qx, qy, qz, qw]
        rot = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        
        # Compute gravity vector in sensor frame
        gravity = rot.apply([0, 0, 9.81], inverse=True)
        
        # Add gravity to linear acceleration to get raw acceleration
        return linear_acc + gravity
    except Exception as e:
        logger.error(f"Error adding gravity: {e}")
        return linear_acc

def cleanup_resources():
    global thread_pool
    if 'thread_pool' in globals() and thread_pool is not None:
        thread_pool.shutdown(wait=True)

def update_thread_configuration(max_workers=None):
    global thread_pool
    if 'thread_pool' in globals() and thread_pool is not None:
        thread_pool.shutdown(wait=True)
    if max_workers is None:
        import os
        max_workers = min(os.cpu_count(), 8)
    thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"Thread pool reconfigured with {max_workers} workers")

class MadgwickFilter:
    def __init__(self, beta=0.1, sample_rate=30.0):
        self.beta = beta
        self.sample_rate = sample_rate
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [qw, qx, qy, qz]
        
    def update(self, acc, gyro, dt=None, is_linear_acc=True):
        """
        Update orientation estimate using accelerometer and gyroscope measurements.
        
        Args:
            acc: Acceleration vector [ax, ay, az] in m/s^2
            gyro: Angular velocity vector [wx, wy, wz] in rad/s
            dt: Time step in seconds
            is_linear_acc: Whether acceleration is linear (gravity removed)
            
        Returns:
            Updated quaternion [qw, qx, qy, qz]
        """
        if dt is None: 
            dt = 1.0 / self.sample_rate
            
        q = self.quaternion
        
        # If dealing with linear acceleration, we need to handle it specially
        if is_linear_acc:
            # First iteration or significant change in direction - use simple approach
            if np.array_equal(q, [1.0, 0.0, 0.0, 0.0]) or np.linalg.norm(acc) > 12.0:
                # For initialization, assume gravity is in the -z direction
                acc_with_gravity = np.array([acc[0], acc[1], acc[2] - 9.81])
            else:
                # Use current orientation estimate to add gravity component
                acc_with_gravity = add_gravity(acc, q)
        else:
            # Already raw acceleration with gravity component
            acc_with_gravity = acc
        
        # Normalize acceleration if magnitude is non-zero
        acc_norm = np.linalg.norm(acc_with_gravity)
        if acc_norm < 1e-10:
            acc_norm_vector = np.array([0, 0, 1])
        else:
            acc_norm_vector = acc_with_gravity / acc_norm
        
        # Extract quaternion components
        q0, q1, q2, q3 = q
        
        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q1*q3 - q0*q2) - acc_norm_vector[0],
            2*(q0*q1 + q2*q3) - acc_norm_vector[1],
            2*(0.5 - q1**2 - q2**2) - acc_norm_vector[2]
        ])
        
        J = np.array([
            [-2*q2, 2*q3, -2*q0, 2*q1],
            [2*q1, 2*q0, 2*q3, 2*q2],
            [0, -4*q1, -4*q2, 0]
        ])
        
        gradient = J.T @ f
        
        # Normalize gradient if magnitude is non-zero
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 0:
            gradient = gradient / gradient_norm
        
        # Gyroscope quaternion rate
        qDot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])
        
        # Apply feedback step
        qDot = qDot - self.beta * gradient
        
        # Integrate to get new quaternion
        q = q + qDot * dt
        
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        
        self.quaternion = q
        return q
    
    def reset(self):
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

class KalmanFilter:
    def __init__(self, sample_rate=30.0):
        self.sample_rate = sample_rate
        # State: [qw, qx, qy, qz, bias_x, bias_y, bias_z]
        self.state = np.zeros(7)
        self.state[0] = 1.0  # Initial quaternion [1, 0, 0, 0]
        self.P = np.diag([1e-2]*4 + [1e-3]*3)  # State covariance
        self.Q = np.diag([1e-5]*4 + [1e-4]*3)  # Process noise
        self.R = np.eye(3) * 0.1  # Measurement noise
        
    def update(self, acc, gyro, dt=None, is_linear_acc=True):
        """
        Update orientation estimate using Kalman filter.
        
        Args:
            acc: Acceleration vector [ax, ay, az] in m/s^2
            gyro: Angular velocity vector [wx, wy, wz] in rad/s
            dt: Time step in seconds
            is_linear_acc: Whether acceleration is linear (gravity removed)
            
        Returns:
            Updated quaternion [qw, qx, qy, qz]
        """
        if dt is None: 
            dt = 1.0 / self.sample_rate
            
        q = self.state[:4]
        bias = self.state[4:]
        
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        
        # Handle linear acceleration
        if is_linear_acc:
            # First iteration or high acceleration - use simple approach
            if np.array_equal(q, [1.0, 0.0, 0.0, 0.0]) or np.linalg.norm(acc) > 12.0:
                # For initialization, assume gravity is in the -z direction
                acc_with_gravity = np.array([acc[0], acc[1], acc[2] - 9.81])
            else:
                # Use current orientation estimate to add gravity component
                acc_with_gravity = add_gravity(acc, q)
        else:
            # Already raw acceleration with gravity component
            acc_with_gravity = acc
        
        # Apply bias correction to gyro
        gyro_corrected = gyro - bias
        
        # Quaternion derivative from angular velocity
        q_dot = 0.5 * self._quaternion_multiply(q, np.array([0, *gyro_corrected]))
        
        # State transition matrix
        F = np.eye(7)
        F[:4, :4] += dt * 0.5 * self._omega_matrix(gyro_corrected)
        
        # Predict state
        x_pred = np.zeros(7)
        x_pred[:4] = q + q_dot * dt
        x_pred[4:] = bias
        x_pred[:4] = x_pred[:4] / np.linalg.norm(x_pred[:4])
        
        # Predict covariance
        P_pred = F @ self.P @ F.T + self.Q
        
        # Update step - only if acceleration magnitude is in reasonable range
        acc_norm = np.linalg.norm(acc_with_gravity)
        if 0.5 < acc_norm < 1.5 * 9.81:  # Reasonable gravity range
            # Compute expected gravity direction from orientation
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ np.array([0, 0, 1])  # Normalized gravity vector
            
            # Normalize measured acceleration
            acc_unit = acc_with_gravity / acc_norm
            
            # Innovation
            y = acc_unit - g_pred
            
            # Measurement Jacobian
            H = self._compute_H_matrix(x_pred[:4])
            
            # Innovation covariance
            S = H @ P_pred @ H.T + self.R
            
            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.state = x_pred + K @ y
            
            # Update covariance
            self.P = (np.eye(7) - K @ H) @ P_pred
        else:
            # Skip update if acceleration is not reliable
            self.state = x_pred
            self.P = P_pred
        
        # Normalize quaternion part of state
        self.state[:4] = self.state[:4] / np.linalg.norm(self.state[:4])
        
        return self.state[:4]
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _omega_matrix(self, gyro):
        """Skew-symmetric matrix from angular velocity"""
        wx, wy, wz = gyro
        return np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
    
    def _compute_H_matrix(self, q):
        """Compute measurement Jacobian matrix"""
        w, x, y, z = q
        H_q = np.zeros((3, 4))
        H_q[0, :] = [-2*y, 2*z, -2*w, 2*x]
        H_q[1, :] = [2*x, 2*w, 2*z, 2*y]
        H_q[2, :] = [0, -2*y, -2*z, 0]
        H = np.zeros((3, 7))
        H[:, :4] = H_q
        return H
    
    def reset(self):
        self.state = np.zeros(7)
        self.state[0] = 1.0
        self.P = np.diag([1e-2]*4 + [1e-3]*3)

class ExtendedKalmanFilter:
    def __init__(self, sample_rate=30.0):
        self.sample_rate = sample_rate
        # State: [qw, qx, qy, qz, bias_x, bias_y, bias_z]
        self.state = np.zeros(7)
        self.state[0] = 1.0  # Initial quaternion [1, 0, 0, 0]
        self.P = np.diag([1e-2]*4 + [1e-4]*3)  # State covariance
        self.Q = np.diag([1e-6]*4 + [1e-5]*3)  # Process noise
        self.R_base = np.eye(3) * 0.05  # Base measurement noise
        self.R = self.R_base.copy()  # Current measurement noise
        self.g_ref = np.array([0, 0, 1])  # Reference gravity vector (normalized)
        self.acc_history = []  # History of acceleration magnitudes
        self.max_history = 10  # Maximum history length
        
    def update(self, acc, gyro, dt=None, is_linear_acc=True):
        """
        Update orientation estimate using Extended Kalman filter.
        
        Args:
            acc: Acceleration vector [ax, ay, az] in m/s^2
            gyro: Angular velocity vector [wx, wy, wz] in rad/s
            dt: Time step in seconds
            is_linear_acc: Whether acceleration is linear (gravity removed)
            
        Returns:
            Updated quaternion [qw, qx, qy, qz]
        """
        if dt is None: 
            dt = 1.0 / self.sample_rate
            
        q = self.state[:4]
        bias = self.state[4:]
        
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        
        # Handle linear acceleration
        if is_linear_acc:
            # First iteration or high acceleration - use simple approach
            if np.array_equal(q, [1.0, 0.0, 0.0, 0.0]) or np.linalg.norm(acc) > 12.0:
                # For initialization, assume gravity is in the -z direction
                acc_with_gravity = np.array([acc[0], acc[1], acc[2] - 9.81])
            else:
                # Use current orientation estimate to add gravity component
                acc_with_gravity = add_gravity(acc, q)
        else:
            # Already raw acceleration with gravity component
            acc_with_gravity = acc
        
        # Track acceleration magnitude for adaptive filtering
        acc_norm = np.linalg.norm(acc_with_gravity)
        self.acc_history.append(acc_norm)
        if len(self.acc_history) > self.max_history:
            self.acc_history.pop(0)
        
        # Update measurement noise based on acceleration variance
        if len(self.acc_history) >= 3:
            acc_var = np.var(self.acc_history)
            dynamic_factor = 1.0 + 10.0 * min(acc_var, 1.0)
            self.R = self.R_base * dynamic_factor
        
        # Apply bias correction to gyro
        gyro_corrected = gyro - bias
        
        # Quaternion derivative from angular velocity
        q_dot = 0.5 * self._quaternion_product_matrix(q) @ np.array([0, *gyro_corrected])
        
        # Predict quaternion
        q_pred = q + q_dot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)
        
        # Predict state
        x_pred = np.zeros(7)
        x_pred[:4] = q_pred
        x_pred[4:] = bias
        
        # State transition matrix
        F = np.eye(7)
        F[:4, :4] = self._quaternion_update_jacobian(q, gyro_corrected, dt)
        F[:4, 4:] = -0.5 * dt * self._quaternion_product_matrix(q)[:, 1:]
        
        # Predict covariance
        P_pred = F @ self.P @ F.T + self.Q
        
        # Update step - only if acceleration magnitude is in reasonable range
        if 0.5 * 9.81 < acc_norm < 1.5 * 9.81:  # Reasonable gravity range
            # Normalize measured acceleration
            acc_normalized = acc_with_gravity / acc_norm
            
            # Compute expected gravity direction from orientation
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ self.g_ref
            
            # Innovation
            y = acc_normalized - g_pred
            
            # Measurement Jacobian
            H = self._measurement_jacobian(x_pred[:4])
            
            # Innovation covariance
            S = H @ P_pred @ H.T + self.R
            
            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.state = x_pred + K @ y
            
            # Joseph form for covariance update (more numerically stable)
            I_KH = np.eye(7) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            # Skip update if acceleration is not reliable
            self.state = x_pred
            self.P = P_pred
        
        # Normalize quaternion part of state
        self.state[:4] = self.state[:4] / np.linalg.norm(self.state[:4])
        
        return self.state[:4]
    
    def _quaternion_product_matrix(self, q):
        """Matrix form of quaternion multiplication"""
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
        ])
    
    def _quaternion_update_jacobian(self, q, gyro, dt):
        """Jacobian of quaternion update equation"""
        wx, wy, wz = gyro
        omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return np.eye(4) + 0.5 * dt * omega
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        xx, xy, xz, xw = x*x, x*y, x*z, x*w
        yy, yz, yw = y*y, y*z, y*w
        zz, zw = z*z, z*w
        return np.array([
            [1-2*(yy+zz), 2*(xy-zw), 2*(xz+yw)],
            [2*(xy+zw), 1-2*(xx+zz), 2*(yz-xw)],
            [2*(xz-yw), 2*(yz+xw), 1-2*(xx+yy)]
        ])
    
    def _measurement_jacobian(self, q):
        """Compute measurement Jacobian matrix"""
        w, x, y, z = q
        H_q = np.zeros((3, 4))
        H_q[0, :] = [-2*y, 2*z, -2*w, 2*x]
        H_q[1, :] = [2*x, 2*w, 2*z, 2*y]
        H_q[2, :] = [0, -2*y, -2*z, 0]
        H = np.zeros((3, 7))
        H[:, :4] = H_q
        return H
    
    def reset(self):
        self.state = np.zeros(7)
        self.state[0] = 1.0
        self.P = np.diag([1e-2]*4 + [1e-4]*3)
        self.acc_history = []

def process_imu_data(acc_data, gyro_data, timestamps=None, filter_type='ekf', return_features=False, is_linear_acc=True):
    """
    Process IMU data with orientation filter to estimate quaternions and linear acceleration.
    
    Args:
        acc_data: Accelerometer data (N, 3)
        gyro_data: Gyroscope data (N, 3)
        timestamps: Optional timestamps in milliseconds
        filter_type: Type of orientation filter ('madgwick', 'kalman', or 'ekf')
        return_features: Whether to extract features from the processed data
        is_linear_acc: Whether input acceleration is linear (gravity removed)
        
    Returns:
        Dictionary with processed data
    """
    logger.info(f"Processing IMU data: filter={filter_type}, is_linear_acc={is_linear_acc}")
    
    if len(acc_data) == 0 or len(gyro_data) == 0:
        logger.error("Empty input data")
        return {'quaternion': np.zeros((0, 4)),
                'linear_acceleration': np.zeros((0, 3)),
                'fusion_features': np.zeros(43) if return_features else None}
    
    # Trim data to common length
    min_len = min(len(acc_data), len(gyro_data))
    acc_data = acc_data[:min_len]
    gyro_data = gyro_data[:min_len]
    if timestamps is not None:
        timestamps = timestamps[:min_len]
    
    # Calculate sample rate if timestamps are provided
    sample_rate = 30.0  # Default sample rate
    if timestamps is not None and len(timestamps) > 1:
        avg_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        sample_rate = 1000.0 / avg_dt
    
    # Initialize appropriate filter
    if filter_type.lower() == 'madgwick':
        orientation_filter = MadgwickFilter(sample_rate=sample_rate)
    elif filter_type.lower() == 'kalman':
        orientation_filter = KalmanFilter(sample_rate=sample_rate)
    elif filter_type.lower() == 'ekf':
        orientation_filter = ExtendedKalmanFilter(sample_rate=sample_rate)
    else:
        logger.warning(f"Unknown filter type: {filter_type}, using EKF")
        orientation_filter = ExtendedKalmanFilter(sample_rate=sample_rate)
    
    # Process data
    quaternions = []
    raw_accelerations = []  # Store raw acceleration with gravity
    linear_accelerations = []  # Store linear acceleration without gravity
    
    for i in range(len(acc_data)):
        acc = acc_data[i]
        gyro = gyro_data[i]
        
        # Calculate dt if timestamps are available
        dt = None
        if timestamps is not None and i > 0:
            dt = (timestamps[i] - timestamps[i-1]) / 1000.0
        
        # Update filter
        q = orientation_filter.update(acc, gyro, dt, is_linear_acc)
        quaternions.append(q)
        
        # Store linear acceleration (original if already linear, or computed)
        if is_linear_acc:
            linear_accelerations.append(acc)
            # Compute raw acceleration by adding gravity
            raw_acc = add_gravity(acc, q)
            raw_accelerations.append(raw_acc)
        else:
            # Input was raw acceleration, compute linear acceleration
            raw_accelerations.append(acc)
            # Compute linear acceleration by removing gravity
            rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            gravity = rot.apply([0, 0, 9.81], inverse=True)
            linear_acc = acc - gravity
            linear_accelerations.append(linear_acc)
    
    # Convert to numpy arrays
    quaternions = np.array(quaternions)
    raw_accelerations = np.array(raw_accelerations)
    linear_accelerations = np.array(linear_accelerations)
    
    # Prepare results
    results = {
        'quaternion': quaternions,
        'raw_acceleration': raw_accelerations,
        'linear_acceleration': linear_accelerations
    }
    
    # Extract features if requested
    if return_features:
        from utils.imu_fusion import extract_features_from_window
        features = extract_features_from_window({
            'quaternion': quaternions,
            'linear_acceleration': linear_accelerations,
            'gyroscope': gyro_data
        })
        results['fusion_features'] = features
    
    return results

def visualize_filter_comparison(acc_data, gyro_data, timestamps=None, is_linear_acc=True, trial_id="unknown"):
    """
    Compare different orientation filters on the same data and visualize results.
    
    Args:
        acc_data: Accelerometer data
        gyro_data: Gyroscope data
        timestamps: Optional timestamps
        is_linear_acc: Whether input acceleration is linear (gravity removed)
        trial_id: Trial identifier for output files
    
    Returns:
        Dictionary with results from different filters
    """
    filter_types = ['madgwick', 'kalman', 'ekf']
    results = {}
    
    for filter_type in filter_types:
        start_time = time.time()
        filter_results = process_imu_data(
            acc_data=acc_data,
            gyro_data=gyro_data,
            timestamps=timestamps,
            filter_type=filter_type,
            return_features=True,
            is_linear_acc=is_linear_acc
        )
        elapsed_time = time.time() - start_time
        
        results[filter_type] = {
            'quaternion': filter_results['quaternion'],
            'raw_acceleration': filter_results['raw_acceleration'],
            'linear_acceleration': filter_results['linear_acceleration'],
            'fusion_features': filter_results['fusion_features'],
            'processing_time': elapsed_time,
            'processing_rate': len(acc_data) / elapsed_time if elapsed_time > 0 else 0
        }
    
    # Create visualizations
    try:
        if timestamps is None:
            timestamps = np.arange(len(acc_data)) / 30.0
        
        # Plot quaternions
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        plt.suptitle(f'Quaternion Comparison - Trial {trial_id} (Linear Acc: {is_linear_acc})')
        
        components = ['w', 'x', 'y', 'z']
        colors = {'madgwick': 'blue', 'kalman': 'red', 'ekf': 'green'}
        
        for i, comp in enumerate(components):
            for filter_name, filter_result in results.items():
                quat = filter_result['quaternion']
                if len(quat) > 0:
                    axes[i].plot(timestamps[:len(quat)], quat[:, i], 
                                label=f'{filter_name}', color=colors[filter_name])
            axes[i].set_title(f'Quaternion {comp} component')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(VISUALIZATION_DIR, f'quaternion_comparison_{trial_id}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # Plot Euler angles
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        plt.suptitle(f'Orientation (Euler) Comparison - Trial {trial_id} (Linear Acc: {is_linear_acc})')
        
        angles = ['Roll', 'Pitch', 'Yaw']
        for filter_name, filter_result in results.items():
            quat = filter_result['quaternion']
            if len(quat) > 0:
                euler_angles = []
                for q in quat:
                    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                    euler_angles.append(r.as_euler('xyz', degrees=True))
                euler_angles = np.array(euler_angles)
                
                for i in range(3):
                    axes[i].plot(timestamps[:len(euler_angles)], euler_angles[:, i], 
                                label=f'{filter_name}', color=colors[filter_name])
        
        for i, angle in enumerate(angles):
            axes[i].set_title(f'{angle} angle')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Degrees')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(VISUALIZATION_DIR, f'euler_comparison_{trial_id}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # Compare linear acceleration estimation
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        plt.suptitle(f'Linear Acceleration Comparison - Trial {trial_id}')
        
        axes_labels = ['X', 'Y', 'Z']
        for filter_name, filter_result in results.items():
            lin_acc = filter_result['linear_acceleration']
            if len(lin_acc) > 0:
                for i in range(3):
                    axes[i].plot(timestamps[:len(lin_acc)], lin_acc[:, i], 
                                label=f'{filter_name}', color=colors[filter_name])
        
        for i, axis_label in enumerate(axes_labels):
            axes[i].set_title(f'Linear Acceleration - {axis_label} axis')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Acceleration (m/s²)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(VISUALIZATION_DIR, f'lin_acc_comparison_{trial_id}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # Performance comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        filter_names = list(results.keys())
        processing_times = [results[name]['processing_time'] for name in filter_names]
        processing_rates = [results[name]['processing_rate'] for name in filter_names]
        
        x = np.arange(len(filter_names))
        width = 0.35
        
        ax.bar(x - width/2, processing_times, width, label='Processing Time (s)')
        ax.bar(x + width/2, processing_rates, width, label='Processing Rate (samples/s)')
        ax.set_xticks(x)
        ax.set_xticklabels(filter_names)
        ax.legend()
        ax.set_title('Filter Performance Comparison')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(VISUALIZATION_DIR, f'performance_comparison_{trial_id}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Filter comparison visualizations saved to {VISUALIZATION_DIR}")
    
    except Exception as e:
        logger.error(f"Error creating filter comparison visualizations: {e}")
    
    return results

def extract_features_from_window(window_data):
    """
    Extract features from a window of IMU data.
    
    Args:
        window_data: Dictionary with quaternion, linear_acceleration, gyroscope
        
    Returns:
        Feature vector (43 elements)
    """
    quaternions = window_data.get('quaternion', np.array([]))
    acc_data = window_data.get('linear_acceleration', window_data.get('accelerometer', np.array([])))
    gyro_data = window_data.get('gyroscope', np.array([]))
    
    if len(quaternions) == 0 or len(acc_data) == 0 or len(gyro_data) == 0:
        logger.warning("Missing data for feature extraction")
        return np.zeros(43)
    
    try:
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
        if len(acc_data) > 1:
            jerk = np.diff(acc_data, axis=0)
            jerk_mag = np.linalg.norm(jerk, axis=1)
            jerk_features = [np.mean(jerk_mag), np.max(jerk_mag)]
        else:
            jerk_features = [0, 0]
        
        # Orientation features from quaternions
        euler_angles = []
        for q in quaternions:
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            euler_angles.append(r.as_euler('xyz', degrees=True))
        euler_angles = np.array(euler_angles)
        
        euler_mean = np.mean(euler_angles, axis=0)
        euler_std = np.std(euler_angles, axis=0)
        
        # Orientation change rate
        if len(euler_angles) > 1:
            angle_rates = np.diff(euler_angles, axis=0)
            angle_rate_features = [
                np.mean(np.linalg.norm(angle_rates, axis=1)), 
                np.max(np.linalg.norm(angle_rates, axis=1))
            ]
        else:
            angle_rate_features = [0, 0]
        
        # Frequency domain features
        fft_features = []
        if len(acc_data) >= 8:
            for axis in range(acc_data.shape[1]):
                fft = np.abs(np.fft.rfft(acc_data[:, axis]))
                if len(fft) > 3:
                    fft_features.extend([np.max(fft), np.mean(fft), np.var(fft)])
                else:
                    fft_features.extend([0, 0, 0])
        else:
            fft_features = [0] * 9
        
        # Combine all features
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
        return np.zeros(43)

def selective_sliding_window(data, window_size, label, fuse=False, filter_type='ekf', is_linear_acc=True):
    """
    Create windows of data for processing.
    
    Args:
        data: Dictionary with sensor data
        window_size: Window size in samples
        label: Activity label
        fuse: Whether to perform sensor fusion
        filter_type: Orientation filter type
        is_linear_acc: Whether accelerometer data is linear acceleration
        
    Returns:
        Dictionary with windowed data
    """
    from collections import defaultdict
    windowed_data = defaultdict(list)
    
    # Check if we have gyroscope data for fusion
    has_gyro = ('gyroscope' in data and 
                isinstance(data['gyroscope'], np.ndarray) and 
                len(data['gyroscope']) > 0)
    
    if fuse and not has_gyro:
        logger.warning("Fusion requested but gyroscope data not available")
        fuse = False
    
    # Check for accelerometer data
    if ('accelerometer' not in data or 
        not isinstance(data['accelerometer'], np.ndarray) or 
        len(data['accelerometer']) == 0):
        logger.warning("Missing accelerometer data, cannot create windows")
        return windowed_data
    
    # Determine if this is a fall or not for window creation strategy
    is_fall = label == 1
    
    # Create windows of accelerometer data
    from utils.loader import sliding_window
    acc_windows_candidates = sliding_window(
        data['accelerometer'], 
        is_fall=is_fall, 
        window_size=window_size, 
        stride=10 if is_fall else 32
    )
    
    if not acc_windows_candidates:
        logger.warning("No accelerometer windows created")
        return windowed_data
    
    # Define required modalities
    required_modalities = ['accelerometer']
    if fuse:
        required_modalities.append('gyroscope')
    
    # Find valid window positions
    valid_window_indices = []
    acc_window_positions = []
    
    for i, acc_window in enumerate(acc_windows_candidates):
        found = False
        for j in range(len(data['accelerometer']) - window_size + 1):
            if np.array_equal(acc_window, data['accelerometer'][j:j+window_size]):
                acc_window_positions.append((i, j))
                found = True
                break
        if not found:
            acc_window_positions.append((i, -1))
    
    # Validate windows across all required modalities
    for i, window_pos in acc_window_positions:
        if window_pos == -1:
            continue
        
        valid = True
        start_pos = window_pos
        
        for modality in required_modalities:
            if modality == 'accelerometer':
                continue
            
            if (modality not in data or 
                not isinstance(data[modality], np.ndarray) or
                start_pos + window_size > len(data[modality])):
                valid = False
                break
        
        if valid:
            valid_window_indices.append(i)
    
    if not valid_window_indices:
        logger.warning("No valid windows found across all required modalities")
        return windowed_data
    
    # Extract valid windows
    acc_windows = [acc_windows_candidates[i] for i in valid_window_indices]
    acc_positions = [acc_window_positions[i][1] for i in valid_window_indices]
    
    # Create windows for each modality
    for modality, modality_data in data.items():
        if modality in ['subject_id', 'labels'] or not isinstance(modality_data, np.ndarray):
            continue
        
        if modality == 'aligned_timestamps':
            continue
        
        if modality == 'accelerometer':
            windowed_data[modality] = np.array(acc_windows)
            continue
        
        try:
            modality_windows = []
            for start_pos in acc_positions:
                if start_pos + window_size <= len(modality_data):
                    modality_windows.append(modality_data[start_pos:start_pos+window_size])
            
            if modality_windows:
                windowed_data[modality] = np.array(modality_windows)
        
        except Exception as e:
            logger.error(f"Error creating windows for {modality}: {str(e)}")
    
    # Extract timestamps if available
    if 'aligned_timestamps' in data and isinstance(data['aligned_timestamps'], np.ndarray):
        try:
            timestamps_windows = []
            for start_pos in acc_positions:
                if start_pos + window_size <= len(data['aligned_timestamps']):
                    timestamps_windows.append(data['aligned_timestamps'][start_pos:start_pos+window_size])
            
            if timestamps_windows:
                windowed_data['aligned_timestamps'] = np.array(timestamps_windows)
        
        except Exception as e:
            logger.error(f"Error creating windows for timestamps: {str(e)}")
    
    # Perform sensor fusion if requested
    if fuse and 'accelerometer' in windowed_data and 'gyroscope' in windowed_data:
        try:
            quaternions, raw_accelerations, linear_accelerations, fusion_features = [], [], [], []
            
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                futures = []
                
                for i in range(len(windowed_data['accelerometer'])):
                    timestamps = (windowed_data['aligned_timestamps'][i] 
                                 if 'aligned_timestamps' in windowed_data and 
                                 len(windowed_data['aligned_timestamps']) > i 
                                 else None)
                    
                    futures.append(executor.submit(
                        process_imu_data,
                        acc_data=windowed_data['accelerometer'][i],
                        gyro_data=windowed_data['gyroscope'][i],
                        timestamps=timestamps,
                        filter_type=filter_type,
                        return_features=True,
                        is_linear_acc=is_linear_acc
                    ))
                
                from tqdm import tqdm
                for future in tqdm(futures, desc=f"Processing {filter_type} fusion"):
                    result = future.result()
                    quaternions.append(result['quaternion'])
                    linear_accelerations.append(result['linear_acceleration'])
                    if 'fusion_features' in result:
                        fusion_features.append(result['fusion_features'])
            
            windowed_data['quaternion'] = np.array(quaternions)
            windowed_data['linear_acceleration'] = np.array(linear_accelerations)
            if fusion_features:
                windowed_data['fusion_features'] = np.array(fusion_features)
                
        except Exception as e:
            logger.error(f"Error in fusion processing: {str(e)}")
    
    # Add labels
    windowed_data['labels'] = np.repeat(label, len(acc_windows))
    
    # Add subject ID if available
    if 'subject_id' in data:
        windowed_data['subjects'] = np.repeat(data['subject_id'], len(acc_windows))
    
    return windowed_data
