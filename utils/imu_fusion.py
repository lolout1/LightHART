import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import logging
import os
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("imu_fusion")

MAX_THREADS = 40
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(40)
filter_cache = {}

def resample_to_fixed_rate(data, timestamps, target_rate=30.0):
    if len(data) <= 1 or len(timestamps) <= 1: return data, timestamps
    start_time, end_time = timestamps[0], timestamps[-1]
    desired_times = np.arange(start_time, end_time, 1.0/target_rate)
    if len(desired_times) == 0: return data[:1], timestamps[:1]
    resampled_data = np.zeros((len(desired_times), data.shape[1]))
    for axis in range(data.shape[1]):
        try:
            interp_func = interp1d(timestamps, data[:, axis], bounds_error=False, 
                                 fill_value=(data[0, axis], data[-1, axis]))
            resampled_data[:, axis] = interp_func(desired_times)
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            if len(timestamps) > 0 and len(data) > 0:
                resampled_data[:, axis] = data[np.argmin(np.abs(timestamps[:, np.newaxis] - desired_times), axis=0), axis]
    return resampled_data, desired_times

def align_sensor_data(acc_data, gyro_data, acc_timestamps=None, gyro_timestamps=None, target_rate=30.0):
    try:
        if len(acc_data) == 0 or len(gyro_data) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        # Handle timestamp extraction if not provided
        if acc_timestamps is None: acc_timestamps = np.arange(len(acc_data))
        if gyro_timestamps is None: gyro_timestamps = np.arange(len(gyro_data))
        
        # Convert timestamps to numeric format if necessary
        if isinstance(acc_timestamps[0], np.datetime64):
            acc_timestamps = acc_timestamps.astype('datetime64[ns]').astype(np.int64) / 1e9
        if isinstance(gyro_timestamps[0], np.datetime64):
            gyro_timestamps = gyro_timestamps.astype('datetime64[ns]').astype(np.int64) / 1e9
        
        # Ensure timestamps are strictly increasing
        acc_indices = np.where(np.diff(acc_timestamps) > 0)[0] + 1
        acc_indices = np.concatenate([[0], acc_indices])
        acc_timestamps, acc_data = acc_timestamps[acc_indices], acc_data[acc_indices]
        
        gyro_indices = np.where(np.diff(gyro_timestamps) > 0)[0] + 1
        gyro_indices = np.concatenate([[0], gyro_indices])
        gyro_timestamps, gyro_data = gyro_timestamps[gyro_indices], gyro_data[gyro_indices]
        
        # Resample to fixed rate
        resampled_acc, acc_times = resample_to_fixed_rate(acc_data, acc_timestamps, target_rate)
        resampled_gyro, gyro_times = resample_to_fixed_rate(gyro_data, gyro_timestamps, target_rate)
        
        # Find common time range
        start_time = max(acc_times[0], gyro_times[0])
        end_time = min(acc_times[-1], gyro_times[-1])
        
        if start_time >= end_time:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        # Create common time base
        common_times = np.arange(start_time, end_time, 1.0/target_rate)
        if len(common_times) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        # Interpolate to common timebase
        aligned_acc = np.zeros((len(common_times), acc_data.shape[1]))
        aligned_gyro = np.zeros((len(common_times), gyro_data.shape[1]))
        
        for axis in range(acc_data.shape[1]):
            acc_interp = interp1d(acc_times, resampled_acc[:, axis], 
                                bounds_error=False, 
                                fill_value=(resampled_acc[0, axis], resampled_acc[-1, axis]))
            aligned_acc[:, axis] = acc_interp(common_times)
        
        for axis in range(gyro_data.shape[1]):
            gyro_interp = interp1d(gyro_times, resampled_gyro[:, axis],
                                  bounds_error=False, 
                                  fill_value=(resampled_gyro[0, axis], resampled_gyro[-1, axis]))
            aligned_gyro[:, axis] = gyro_interp(common_times)
        
        return aligned_acc, aligned_gyro, common_times
    
    except Exception as e:
        logger.error(f"Alignment error: {e}")
        logger.error(traceback.format_exc())
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

def apply_lowpass_filter(data, cutoff=5.0, fs=30.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered_data = np.zeros_like(data)
    for axis in range(data.shape[1]):
        filtered_data[:, axis] = filtfilt(b, a, data[:, axis])
    
    return filtered_data

def fixed_size_windows(data, window_size=128, overlap=0.5, min_windows=1):
    if len(data) < window_size:
        padded = np.zeros((window_size, data.shape[1]))
        padded[:len(data)] = data
        return [padded]
    
    stride = int(window_size * (1 - overlap))
    starts = list(range(0, len(data) - window_size + 1, stride))
    
    if len(starts) < min_windows:
        if len(data) <= window_size:
            starts = [0]
        else:
            starts = np.linspace(0, len(data) - window_size, min_windows).astype(int).tolist()
    
    return [data[start:start + window_size] for start in starts if start + window_size <= len(data)]

def save_aligned_sensor_data(subject_id, action_id, trial_id, acc_data, gyro_data, skeleton_data=None, 
                           timestamps=None, save_dir="data/aligned"):
    try:
        with file_semaphore:
            os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
            os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
            if skeleton_data is not None: os.makedirs(f"{save_dir}/skeleton", exist_ok=True)
            if timestamps is not None: os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
            
            filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
            np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
            np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
            if skeleton_data is not None: np.save(f"{save_dir}/skeleton/{filename}.npy", skeleton_data)
            if timestamps is not None: np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")

class OrientationEstimator:
    def __init__(self, freq=30.0):
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
    
    def update(self, acc, gyro, timestamp=None):
        try:
            dt = 1.0 / self.freq
            if timestamp is not None and self.last_time is not None:
                dt = timestamp - self.last_time
                self.last_time = timestamp
            elif timestamp is not None:
                self.last_time = timestamp
            
            if dt <= 0 or dt > 1.0: dt = 1.0 / self.freq
                
            new_orientation = self._update_impl(acc, gyro, dt)
            
            norm = np.linalg.norm(new_orientation)
            if norm > 1e-10:
                self.orientation_q = new_orientation / norm
            
            return self.orientation_q
        except Exception as e:
            logger.error(f"Error in orientation update: {e}")
            return self.orientation_q
    
    def _update_impl(self, acc, gyro, dt):
        raise NotImplementedError("Subclasses must implement _update_impl")
    
    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None

class MadgwickFilter(OrientationEstimator):
    def __init__(self, freq=30.0, beta=0.1):
        super().__init__(freq)
        self.beta = beta
    
    def _update_impl(self, acc, gyro, dt):
        q = self.orientation_q
        
        # Normalize accelerometer
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            qDot = 0.5 * np.array([
                -q[1] * gyro[0] - q[2] * gyro[1] - q[3] * gyro[2],
                q[0] * gyro[0] + q[2] * gyro[2] - q[3] * gyro[1],
                q[0] * gyro[1] - q[1] * gyro[2] + q[3] * gyro[0],
                q[0] * gyro[2] + q[1] * gyro[1] - q[2] * gyro[0]
            ])
            q = q + qDot * dt
            return q / np.linalg.norm(q)
        
        acc_norm = acc / acc_norm
        q0, q1, q2, q3 = q
        
        # Gradient descent algorithm
        f1 = 2.0 * (q1 * q3 - q0 * q2) - acc_norm[0]
        f2 = 2.0 * (q0 * q1 + q2 * q3) - acc_norm[1]
        f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - acc_norm[2]
        
        J_t = np.array([
            [-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
            [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
            [0.0, -4.0*q1, -4.0*q2, 0.0]
        ])
        
        grad = J_t.T @ np.array([f1, f2, f3])
        grad_norm = np.linalg.norm(grad)
        grad = grad / grad_norm if grad_norm > 0 else grad
        
        qDot = 0.5 * np.array([
            -q1 * gyro[0] - q2 * gyro[1] - q3 * gyro[2],
            q0 * gyro[0] + q2 * gyro[2] - q3 * gyro[1],
            q0 * gyro[1] - q1 * gyro[2] + q3 * gyro[0],
            q0 * gyro[2] + q1 * gyro[1] - q2 * gyro[0]
        ])
        
        qDot = qDot - self.beta * grad
        q = q + qDot * dt
        return q / np.linalg.norm(q)

class KalmanFilter(OrientationEstimator):
    def __init__(self, freq=30.0, process_noise=1e-4, measurement_noise=0.1):
        super().__init__(freq)
        
        self.state_dim = 7  # quaternion (4) + gyro bias (3)
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initialize quaternion to identity rotation
        
        # Process noise covariance matrix
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[:4, :4] *= 0.01  # Lower noise for quaternion components
        self.Q[4:, 4:] *= 10.0  # Higher noise for bias components
        
        # Measurement noise covariance matrix
        self.R = np.eye(3) * measurement_noise
        
        # Initial error covariance matrix
        self.P = np.eye(self.state_dim) * 1e-2
    
    def _update_impl(self, acc, gyro, dt):
        q = self.x[:4]
        bias = self.x[4:]
        
        # Normalize quaternion
        q_norm = np.linalg.norm(q)
        if q_norm > 0: q = q / q_norm
        
        # Correct gyro measurement with estimated bias
        gyro_corrected = gyro - bias
        
        # Create quaternion derivative from corrected gyro
        omega = np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega)
        
        # State transition matrix
        F = np.eye(self.state_dim)
        F[:4, :4] += 0.5 * dt * self._omega_matrix(gyro_corrected)
        
        # Predict state
        x_pred = self.x.copy()
        x_pred[:4] = q + q_dot * dt
        x_pred[4:] = bias  # Bias random walk model
        
        # Normalize predicted quaternion
        q_norm = np.linalg.norm(x_pred[:4])
        if q_norm > 0: x_pred[:4] = x_pred[:4] / q_norm
        
        # Predict error covariance
        P_pred = F @ self.P @ F.T + self.Q
        
        # Update step - only if we have valid accelerometer data
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            acc_normalized = acc / acc_norm
            
            # Convert quaternion to rotation matrix
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            
            # Predicted gravity direction
            g_pred = R_q @ np.array([0, 0, 1])
            
            # Measurement residual
            y = acc_normalized - g_pred
            
            # Measurement Jacobian
            H = self._compute_H_matrix(x_pred[:4])
            
            # Innovation covariance
            S = H @ P_pred @ H.T + self.R
            
            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.x = x_pred + K @ y
            
            # Update error covariance (Joseph form for numerical stability)
            I_KH = np.eye(self.state_dim) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            # No measurement update
            self.x = x_pred
            self.P = P_pred
        
        # Ensure quaternion is normalized
        q_norm = np.linalg.norm(self.x[:4])
        if q_norm > 0: self.x[:4] = self.x[:4] / q_norm
        
        return self.x[:4]
    
    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _omega_matrix(self, gyro):
        wx, wy, wz = gyro
        return np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
    
    def _quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _compute_H_matrix(self, q):
        w, x, y, z = q
        
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
        
        H = np.zeros((3, self.state_dim))
        H[:, :4] = H_q
        
        return H

class ExtendedKalmanFilter(OrientationEstimator):
    def __init__(self, freq=30.0, process_noise=1e-5, measurement_noise=0.05):
        super().__init__(freq)
        
        # State vector: [quaternion, gyro_bias]
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0  # Initialize quaternion to identity rotation
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[:4, :4] *= 0.1  # Lower noise for quaternion components
        self.Q[4:, 4:] *= 10.0  # Higher noise for bias components
        
        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise
        
        # Error covariance matrix
        self.P = np.eye(self.state_dim) * 1e-2
        
        # Reference gravity vector
        self.g_ref = np.array([0, 0, 1])
    
    def _update_impl(self, acc, gyro, dt):
        try:
            q = self.x[:4]
            bias = self.x[4:]
            
            # Normalize quaternion
            q_norm = np.linalg.norm(q)
            if q_norm > 0: q = q / q_norm
            
            # Correct gyro measurement with estimated bias
            gyro_corrected = gyro - bias
            
            # Quaternion derivative calculation
            q_dot = 0.5 * self._quaternion_product_matrix(q) @ np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
            
            # Predict next state (quaternion integration)
            q_pred = q + q_dot * dt
            q_pred = q_pred / np.linalg.norm(q_pred)
            
            # Update full state vector
            x_pred = np.zeros_like(self.x)
            x_pred[:4] = q_pred
            x_pred[4:] = bias  # Bias is assumed constant (random walk model)
            
            # Compute state transition matrix
            F = np.eye(self.state_dim)
            F[:4, :4] = self._quaternion_update_jacobian(q, gyro_corrected, dt)
            F[:4, 4:] = -0.5 * dt * self._quaternion_product_matrix(q)[:, 1:]
            
            # Propagate error covariance
            P_pred = F @ self.P @ F.T + self.Q
            
            # Update step using accelerometer (only if magnitude is close to gravity)
            acc_norm = np.linalg.norm(acc)
            if 0.5 < acc_norm < 1.5:  # More lenient check for acceleration near gravity
                acc_normalized = acc / acc_norm
                
                # Predict gravity direction from current orientation
                R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
                g_pred = R_q @ self.g_ref
                
                # Measurement residual
                z = acc_normalized
                h = g_pred
                y = z - h
                
                # Measurement Jacobian
                H = self._measurement_jacobian(x_pred[:4])
                
                # Kalman gain calculation
                S = H @ P_pred @ H.T + self.R
                K = P_pred @ H.T @ np.linalg.inv(S)
                
                # State update
                self.x = x_pred + K @ y
                
                # Covariance update (Joseph form for numerical stability)
                I_KH = np.eye(self.state_dim) - K @ H
                self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
            else:
                self.x = x_pred
                self.P = P_pred
            
            # Normalize quaternion
            self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
            
            return self.x[:4]
        
        except Exception as e:
            logger.error(f"EKF update error: {e}")
            return self.orientation_q
    
    def _quaternion_product_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
        ])
    
    def _quaternion_update_jacobian(self, q, gyro, dt):
        wx, wy, wz = gyro
        omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return np.eye(4) + 0.5 * dt * omega
    
    def _quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _measurement_jacobian(self, q):
        w, x, y, z = q
        
        H_acc = np.zeros((3, self.state_dim))
        H_acc[:3, :4] = np.array([
            [2*y, 2*z, 2*w, 2*x],
            [-2*z, 2*y, 2*x, -2*w],
            [0, -2*x, -2*y, 0]
        ])
        
        return H_acc

def get_filter_instance(subject_id, action_id, filter_type, reset=False):
    global filter_cache
    cache_key = f"{subject_id}_{action_id}_{filter_type}"
    
    if reset or cache_key not in filter_cache:
        if filter_type == 'madgwick':
            filter_instance = MadgwickFilter(beta=0.1)
        elif filter_type == 'kalman':
            filter_instance = KalmanFilter()
        elif filter_type == 'ekf':
            filter_instance = ExtendedKalmanFilter()
        else:
            filter_instance = MadgwickFilter()
        
        filter_cache[cache_key] = filter_instance
    
    return filter_cache[cache_key]

def process_imu_data(acc_data, gyro_data, timestamps=None, filter_type='madgwick', 
                    return_features=False, trial_id=None, reset_filter=False):
    if trial_id is not None:
        orientation_filter = get_filter_instance(trial_id, 0, filter_type, reset=reset_filter)
    else:
        if filter_type == 'madgwick':
            orientation_filter = MadgwickFilter()
        elif filter_type == 'kalman':
            orientation_filter = KalmanFilter()
        elif filter_type == 'ekf':
            orientation_filter = ExtendedKalmanFilter()
        else:
            orientation_filter = MadgwickFilter()
    
    try:
        quaternions = []
        
        for i in range(len(acc_data)):
            acc = acc_data[i]
            gyro = gyro_data[i]
            timestamp = timestamps[i] if timestamps is not None else None
            
            # For the first sample or if using a new filter
            if i == 0 or reset_filter:
                gravity_direction = np.array([0, 0, 9.81])
            else:
                # Use previous orientation to estimate gravity direction
                last_q = quaternions[-1]
                r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
                
            # Create normalized acceleration with gravity for orientation estimation
            acc_with_gravity = acc + gravity_direction
            norm = np.linalg.norm(acc_with_gravity)
            if norm > 1e-6:
                acc_with_gravity = acc_with_gravity / norm
            
            q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
            quaternions.append(q)
        
        results = {'quaternion': np.array(quaternions)}
        
        if return_features:
            features = extract_features_from_window(
                {'accelerometer': acc_data, 'gyroscope': gyro_data, 'quaternion': np.array(quaternions)}
            )
            results['fusion_features'] = features
        
        return results
        
    except Exception as e:
        logger.error(f"Error in IMU processing: {str(e)}")
        return {'quaternion': np.zeros((len(acc_data), 4))}

def extract_features_from_window(window_data):
    acc = window_data.get('accelerometer')
    gyro = window_data.get('gyroscope')
    quat = window_data.get('quaternion')
    
    features = []
    
    if acc is not None and len(acc) > 0:
        # Time domain features from accelerometer
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        acc_min, acc_max = np.min(acc, axis=0), np.max(acc, axis=0)
        acc_energy = np.sum(acc**2, axis=0) / len(acc)
        
        features.extend(acc_mean)
        features.extend(acc_std)
        features.extend(acc_max - acc_min)
        features.extend(acc_energy)
    
    if gyro is not None and len(gyro) > 0:
        # Time domain features from gyroscope
        gyro_mean = np.mean(gyro, axis=0)
        gyro_std = np.std(gyro, axis=0)
        gyro_min, gyro_max = np.min(gyro, axis=0), np.max(gyro, axis=0)
        gyro_energy = np.sum(gyro**2, axis=0) / len(gyro)
        
        features.extend(gyro_mean)
        features.extend(gyro_std)
        features.extend(gyro_max - gyro_min)
        features.extend(gyro_energy)
    
    if quat is not None and len(quat) > 0:
        # Quaternion features
        quat_mean = np.mean(quat, axis=0)
        quat_std = np.std(quat, axis=0)
        quat_min, quat_max = np.min(quat, axis=0), np.max(quat, axis=0)
        
        features.extend(quat_mean)
        features.extend(quat_std)
        features.extend(quat_max - quat_min)
        
        # Angular velocity from quaternions
        if len(quat) > 1:
            angular_vel = []
            for i in range(1, len(quat)):
                # Compute quaternion difference
                q1, q2 = quat[i-1], quat[i]
                # Convert to rotation objects
                r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
                r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
                # Get relative rotation
                r_diff = r2 * r1.inv()
                # Convert to axis-angle
                angle = 2 * np.arccos(np.clip(np.abs(r_diff.as_quat()[-1]), 0, 1))
                angular_vel.append(angle)
            
            features.append(np.mean(angular_vel))
            features.append(np.std(angular_vel))
            features.append(np.max(angular_vel))
    
    return np.array(features)

def process_sequence_with_filter(acc_data, gyro_data, timestamps=None, subject_id=0, action_id=0, 
                               filter_type='madgwick', filter_params=None, use_cache=True, 
                               cache_dir="processed_data", window_id=0):
    if filter_type == 'none':
        return np.zeros((len(acc_data), 4))
        
    cache_key = f"S{subject_id:02d}A{action_id:02d}W{window_id:04d}_{filter_type}"
    if use_cache:
        cache_path = os.path.join(cache_dir, f"{cache_key}.npz")
        if os.path.exists(cache_path):
            try:
                cached_data = np.load(cache_path)
                return cached_data['quaternion']
            except Exception:
                pass
    
    orientation_filter = get_filter_instance(subject_id, action_id, filter_type, filter_params)
    quaternions = np.zeros((len(acc_data), 4))
    
    for i in range(len(acc_data)):
        acc = acc_data[i]
        gyro = gyro_data[i]
        timestamp = timestamps[i] if timestamps is not None else None
        
        gravity_direction = np.array([0, 0, 9.81])
        if i > 0:
            r = Rotation.from_quat([quaternions[i-1, 1], quaternions[i-1, 2], quaternions[i-1, 3], quaternions[i-1, 0]])
            gravity_direction = r.inv().apply([0, 0, 9.81])
            
        acc_with_gravity = acc + gravity_direction
        norm = np.linalg.norm(acc_with_gravity)
        if norm > 1e-6:
            acc_with_gravity = acc_with_gravity / norm
            
        q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
        quaternions[i] = q
    
    if use_cache:
        os.makedirs(os.path.dirname(os.path.join(cache_dir, f"{cache_key}.npz")), exist_ok=True)
        try:
            np.savez_compressed(os.path.join(cache_dir, f"{cache_key}.npz"), 
                              quaternion=quaternions, 
                              window_id=window_id,
                              subject_id=subject_id,
                              action_id=action_id)
        except Exception as e:
            print(f"Warning: Failed to cache window {window_id}: {str(e)}")
    
    return quaternions

def benchmark_filters(acc_data, gyro_data, timestamps=None, filters=None):
    if filters is None:
        filters = {
            'madgwick': MadgwickFilter(beta=0.1),
            'kalman': KalmanFilter(),
            'ekf': ExtendedKalmanFilter()
        }
    
    results = {}
    
    for name, filter_obj in filters.items():
        filter_obj.reset()
        quaternions = []
        
        processing_time = time.time()
        for i in range(len(acc_data)):
            acc = acc_data[i]
            gyro = gyro_data[i]
            ts = timestamps[i] if timestamps is not None else None
            
            gravity_direction = np.array([0, 0, 9.81])
            if i > 0 and quaternions:
                last_q = quaternions[-1]
                r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
            
            acc_with_gravity = acc + gravity_direction
            norm = np.linalg.norm(acc_with_gravity)
            if norm > 1e-6:
                acc_with_gravity = acc_with_gravity / norm
            
            q = filter_obj.update(acc_with_gravity, gyro, ts)
            quaternions.append(q)
        
        processing_time = time.time() - processing_time
        
        results[name] = {
            'quaternions': np.array(quaternions),
            'processing_time': processing_time
        }
    
    return results

def preprocess_all_subjects(subjects, filter_type, output_dir, max_length=128):
    logger.info(f"Preprocessing all subjects with {filter_type} filter")
    
    from utils.dataset import SmartFallMM
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = SmartFallMM(
        root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'),
        fusion_options={'filter_type': filter_type}
    )
    
    dataset.pipe_line(
        age_group=['young'],
        modalities=['accelerometer', 'gyroscope'],
        sensors=['watch']
    )
    
    total_trials = sum(1 for subject_id in subjects 
                     for trial in dataset.matched_trials if trial.subject_id == subject_id)
    
    global filter_cache
    filter_cache = {}
    
    processed_count = 0
    from tqdm import tqdm
    with tqdm(total=total_trials, desc=f"Preprocessing all subjects ({filter_type})") as pbar:
        for subject_id in subjects:
            subject_dir = os.path.join(output_dir, f"S{subject_id:02d}")
            os.makedirs(subject_dir, exist_ok=True)
            
            subject_trials = [trial for trial in dataset.matched_trials if trial.subject_id == subject_id]
            
            for trial in subject_trials:
                processed_count += 1
                pbar.update(1)
                
                action_id = trial.action_id
                sequence_number = trial.sequence_number
                trial_id = f"S{subject_id:02d}A{action_id:02d}T{sequence_number:02d}"
                
                try:
                    if not ('accelerometer' in trial.files and 'gyroscope' in trial.files):
                        continue
                        
                    trial_data = {}
                    for modality_name, file_path in trial.files.items():
                        if modality_name in ['accelerometer', 'gyroscope']:
                            try:
                                import pandas as pd
                                file_data = pd.read_csv(file_path, header=None).dropna().bfill()
                                
                                if file_data.shape[1] > 4:
                                    timestamps = None
                                    values = file_data.iloc[:, 3:6].to_numpy(dtype=np.float32)
                                else:
                                    timestamps = file_data.iloc[:, 0].to_numpy(dtype=np.float64)
                                    values = file_data.iloc[:, 1:4].to_numpy(dtype=np.float32)
                                
                                trial_data[f"{modality_name}_values"] = values
                                trial_data[f"{modality_name}_timestamps"] = timestamps
                            except Exception as e:
                                logger.error(f"Error loading {modality_name} data: {str(e)}")
                                continue
                    
                    if 'accelerometer_values' in trial_data and 'gyroscope_values' in trial_data:
                        acc_data = trial_data['accelerometer_values']
                        gyro_data = trial_data['gyroscope_values']
                        acc_timestamps = trial_data.get('accelerometer_timestamps')
                        gyro_timestamps = trial_data.get('gyroscope_timestamps')
                        
                        aligned_acc, aligned_gyro, common_timestamps = align_sensor_data(
                            acc_data, gyro_data, acc_timestamps, gyro_timestamps
                        )
                        
                        if len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                            windows_acc = fixed_size_windows(aligned_acc, window_size=max_length, overlap=0.5)
                            windows_gyro = fixed_size_windows(aligned_gyro, window_size=max_length, overlap=0.5)
                            
                            num_windows = min(len(windows_acc), len(windows_gyro))
                            
                            for i in range(num_windows):
                                window_acc = windows_acc[i]
                                window_gyro = windows_gyro[i]
                                
                                filter_key = f"{subject_id}_{action_id}_{filter_type}_{i}"
                                if filter_key not in filter_cache:
                                    if filter_type == 'madgwick':
                                        filter_cache[filter_key] = MadgwickFilter()
                                    elif filter_type == 'kalman':
                                        filter_cache[filter_key] = KalmanFilter()
                                    elif filter_type == 'ekf':
                                        filter_cache[filter_key] = ExtendedKalmanFilter()
                                    else:
                                        filter_cache[filter_key] = MadgwickFilter()
                                
                                orientation_filter = filter_cache[filter_key]
                                
                                quaternions = []
                                for j in range(len(window_acc)):
                                    acc = window_acc[j]
                                    gyro = window_gyro[j]
                                    
                                    gravity_direction = np.array([0, 0, 9.81])
                                    if quaternions:
                                        last_q = quaternions[-1]
                                        r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                                        gravity_direction = r.inv().apply([0, 0, 9.81])
                                    
                                    acc_with_gravity = acc + gravity_direction
                                    norm = np.linalg.norm(acc_with_gravity)
                                    if norm > 1e-6:
                                        acc_with_gravity = acc_with_gravity / norm
                                    
                                    q = orientation_filter.update(acc_with_gravity, gyro)
                                    quaternions.append(q)
                                
                                window_output_file = os.path.join(subject_dir, f"{trial_id}_W{i:04d}.npz")
                                np.savez_compressed(
                                    window_output_file,
                                    accelerometer=window_acc,
                                    gyroscope=window_gyro,
                                    quaternion=np.array(quaternions),
                                    window_id=i,
                                    filter_type=filter_type
                                )
                            
                            output_file = os.path.join(subject_dir, f"{trial_id}.npz")
                            np.savez_compressed(
                                output_file,
                                accelerometer=aligned_acc,
                                gyroscope=aligned_gyro,
                                timestamps=common_timestamps,
                                filter_type=filter_type
                            )
                except Exception as e:
                    logger.error(f"Error processing trial {trial_id}: {str(e)}")
                    continue
    
    logger.info(f"Preprocessing complete: processed {processed_count}/{total_trials} trials")
