import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks
import logging, os, threading
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("imu_fusion")
thread_pool = ThreadPoolExecutor(max_workers=16)
file_semaphore = threading.Semaphore(16)
filter_cache = {}

def bandpass_filter(data, cutoff_low=0.01, cutoff_high=15.0, fs=30.0, order=2):
    nyq = 0.5 * fs
    low = cutoff_low / nyq
    high = cutoff_high / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = np.zeros_like(data)
    for axis in range(data.shape[1]):
        filtered_data[:, axis] = filtfilt(b, a, data[:, axis])
    return filtered_data

def apply_lowpass_filter(data, cutoff=8.0, fs=30.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    filtered_data = np.zeros_like(data)
    for axis in range(data.shape[1]):
        filtered_data[:, axis] = filtfilt(b, a, data[:, axis])
    return filtered_data

def improved_align_sensor_data(acc_data, gyro_data, acc_timestamps=None, gyro_timestamps=None, target_rate=30.0):
    if len(acc_data) < 3 or len(gyro_data) < 3:
        return None, None, None
        
    if acc_timestamps is None:
        acc_timestamps = np.arange(len(acc_data)) / target_rate * 1000
    if gyro_timestamps is None:
        gyro_timestamps = np.arange(len(gyro_data)) / target_rate * 1000
        
    acc_valid_indices = np.where(np.diff(acc_timestamps, prepend=acc_timestamps[0]-1) >= 0)[0]
    gyro_valid_indices = np.where(np.diff(gyro_timestamps, prepend=gyro_timestamps[0]-1) >= 0)[0]
    
    if len(acc_valid_indices) < 3 or len(gyro_valid_indices) < 3:
        return None, None, None
        
    acc_timestamps = acc_timestamps[acc_valid_indices]
    acc_data = acc_data[acc_valid_indices]
    gyro_timestamps = gyro_timestamps[gyro_valid_indices]
    gyro_data = gyro_data[gyro_valid_indices]
    
    start_time = max(acc_timestamps[0], gyro_timestamps[0])
    end_time = min(acc_timestamps[-1], gyro_timestamps[-1])
    
    if start_time >= end_time:
        return None, None, None
    
    common_times = np.linspace(start_time, end_time, int((end_time-start_time)*target_rate/1000))
    
    if len(common_times) < 3:
        return None, None, None
    
    aligned_acc = np.zeros((len(common_times), 3))
    aligned_gyro = np.zeros((len(common_times), 3))
    
    for axis in range(3):
        try:
            interp_func = interp1d(acc_timestamps, acc_data[:, axis], bounds_error=False, 
                                 fill_value=(acc_data[0, axis], acc_data[-1, axis]), kind='linear')
            aligned_acc[:, axis] = interp_func(common_times)
        except:
            idx = np.argmin(np.abs(acc_timestamps[:, np.newaxis] - common_times), axis=0)
            aligned_acc[:, axis] = acc_data[idx, axis]
    
    for axis in range(3):
        try:
            interp_func = interp1d(gyro_timestamps, gyro_data[:, axis], bounds_error=False, 
                                 fill_value=(gyro_data[0, axis], gyro_data[-1, axis]), kind='linear')
            aligned_gyro[:, axis] = interp_func(common_times)
        except:
            idx = np.argmin(np.abs(gyro_timestamps[:, np.newaxis] - common_times), axis=0)
            aligned_gyro[:, axis] = gyro_data[idx, axis]
    
    aligned_acc = bandpass_filter(aligned_acc, cutoff_low=0.01, cutoff_high=15.0, fs=target_rate)
    aligned_gyro = bandpass_filter(aligned_gyro, cutoff_low=0.01, cutoff_high=12.0, fs=target_rate)
    
    return aligned_acc, aligned_gyro, common_times

def create_sliding_windows(data, window_size=128, stride=32):
    if data is None or len(data) < window_size // 2:
        return []
        
    windows = []
    for start in range(0, max(1, len(data) - window_size + 1), stride):
        if start + window_size <= len(data):
            windows.append(data[start:start + window_size])
            
    return windows if windows else []

class MadgwickFilter:
    def __init__(self, freq=30.0, beta=0.15):
        self.freq = freq
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        
    def update(self, accel, gyro, timestamp=None):
        dt = 1.0/self.freq if timestamp is None or self.last_time is None else timestamp - self.last_time
        self.last_time = timestamp
        if dt <= 0 or dt > 1.0: dt = 1.0/self.freq
        
        q = self.q
        gx, gy, gz = gyro
        ax, ay, az = accel
        
        # Normalize accelerometer measurement
        norm = np.sqrt(ax*ax + ay*ay + az*az)
        if norm > 0:
            ax, ay, az = ax/norm, ay/norm, az/norm
            
            # Auxiliary variables to avoid repeated calculations
            _2q0 = 2.0 * q[0]
            _2q1 = 2.0 * q[1]
            _2q2 = 2.0 * q[2]
            _2q3 = 2.0 * q[3]
            _4q0 = 4.0 * q[0]
            _4q1 = 4.0 * q[1]
            _4q2 = 4.0 * q[2]
            _4q3 = 4.0 * q[3]
            _8q1 = 8.0 * q[1]
            _8q2 = 8.0 * q[2]
            q0q0 = q[0] * q[0]
            q1q1 = q[1] * q[1]
            q2q2 = q[2] * q[2]
            q3q3 = q[3] * q[3]
            
            # Gradient descent algorithm corrective step
            s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
            s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q[1] - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
            s2 = 4.0 * q0q0 * q[2] + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
            s3 = 4.0 * q1q1 * q[3] - _2q1 * ax + 4.0 * q2q2 * q[3] - _2q2 * ay
            
            # Normalize step magnitude
            norm = np.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
            
            if norm > 0.0:
                s0 /= norm
                s1 /= norm
                s2 /= norm
                s3 /= norm
            
            # Rate of change of quaternion from gyroscope
            qDot1 = 0.5 * (-q[1] * gx - q[2] * gy - q[3] * gz)
            qDot2 = 0.5 * (q[0] * gx + q[2] * gz - q[3] * gy)
            qDot3 = 0.5 * (q[0] * gy - q[1] * gz + q[3] * gx)
            qDot4 = 0.5 * (q[0] * gz + q[1] * gy - q[2] * gx)
            
            # Apply feedback step
            qDot1 -= self.beta * s0
            qDot2 -= self.beta * s1
            qDot3 -= self.beta * s2
            qDot4 -= self.beta * s3
            
            # Integrate to yield quaternion
            q[0] += qDot1 * dt
            q[1] += qDot2 * dt
            q[2] += qDot3 * dt
            q[3] += qDot4 * dt
            
            # Normalize quaternion
            norm = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
            self.q = q / norm
        
        return self.q
    
    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None

class KalmanFilter:
    def __init__(self, freq=30.0, process_noise=2e-5):
        self.freq = freq
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.P = np.eye(4) * 0.01
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(3) * 0.1
        self.last_time = None
        
    def update(self, accel, gyro, timestamp=None):
        dt = 1.0/self.freq if timestamp is None or self.last_time is None else timestamp - self.last_time
        self.last_time = timestamp
        if dt <= 0 or dt > 1.0: dt = 1.0/self.freq
        
        q = self.q
        gx, gy, gz = gyro
        
        # Process model (prediction step)
        F = np.eye(4) + 0.5 * dt * np.array([
            [0, -gx, -gy, -gz],
            [gx, 0, gz, -gy],
            [gy, -gz, 0, gx],
            [gz, gy, -gx, 0]
        ])
        
        q_pred = F @ q
        P_pred = F @ self.P @ F.T + self.Q
        
        q_pred = q_pred / np.linalg.norm(q_pred)
        
        ax, ay, az = accel
        norm = np.sqrt(ax*ax + ay*ay + az*az)
        
        # Measurement update step (when accelerometer magnitude is close to gravity)
        if norm > 0 and 0.75*9.8 < norm < 1.25*9.8:
            ax, ay, az = ax/norm, ay/norm, az/norm
            
            Rot = self._quat_to_rot(q_pred)
            g_pred = Rot @ np.array([0, 0, 1])
            y = np.array([ax, ay, az]) - g_pred
            
            H = self._compute_H(q_pred)
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            self.q = q_pred + K @ y
            self.q = self.q / np.linalg.norm(self.q)
            self.P = (np.eye(4) - K @ H) @ P_pred
        else:
            self.q = q_pred
            self.P = P_pred
            
        return self.q
    
    def _quat_to_rot(self, q):
        q0, q1, q2, q3 = q
        return np.array([
            [1-2*(q2*q2+q3*q3), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1*q1+q2*q2)]
        ])
    
    def _compute_H(self, q):
        q0, q1, q2, q3 = q
        H = np.zeros((3, 4))
        H[0, 0] = -2*q2; H[0, 1] = 2*q3; H[0, 2] = -2*q0; H[0, 3] = 2*q1
        H[1, 0] = 2*q1; H[1, 1] = 2*q0; H[1, 2] = 2*q3; H[1, 3] = 2*q2
        H[2, 0] = 0; H[2, 1] = -2*q2; H[2, 2] = -2*q1; H[2, 3] = 0
        return H
    
    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.P = np.eye(4) * 0.01
        self.last_time = None

class ExtendedKalmanFilter:
    def __init__(self, freq=30.0, process_noise=1e-5):
        self.freq = freq
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.P = np.eye(4) * 0.01
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(3) * 0.05
        self.g_ref = np.array([0, 0, 1])
        self.last_time = None
        self.acc_jerk = 0
        self.prev_acc = None
        
    def update(self, accel, gyro, timestamp=None):
        dt = 1.0/self.freq if timestamp is None or self.last_time is None else timestamp - self.last_time
        self.last_time = timestamp
        if dt <= 0 or dt > 1.0: dt = 1.0/self.freq
        
        q = self.q
        gx, gy, gz = gyro
        ax, ay, az = accel
        
        # Calculate acceleration jerk for adaptive filtering
        if self.prev_acc is not None:
            self.acc_jerk = 0.7 * self.acc_jerk + 0.3 * np.linalg.norm(np.array([ax, ay, az]) - self.prev_acc) / dt
        self.prev_acc = np.array([ax, ay, az])
        
        # Process model (prediction step) - using quaternion kinematics
        F = np.eye(4) + 0.5 * dt * np.array([
            [0, -gx, -gy, -gz],
            [gx, 0, gz, -gy],
            [gy, -gz, 0, gx],
            [gz, gy, -gx, 0]
        ])
        
        q_pred = F @ q
        P_pred = F @ self.P @ F.T + self.Q
        
        q_pred = q_pred / np.linalg.norm(q_pred)
        
        # Measurement update when accelerometer is reliable
        norm = np.sqrt(ax*ax + ay*ay + az*az)
        acc_valid = 0.75*9.8 < norm < 1.25*9.8 and self.acc_jerk < 20.0
        
        if acc_valid:
            ax, ay, az = ax/norm, ay/norm, az/norm
            
            Rot = self._quat_to_rot(q_pred)
            g_pred = Rot @ self.g_ref
            y = np.array([ax, ay, az]) - g_pred
            
            H = self._compute_H(q_pred)
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            self.q = q_pred + K @ y
            self.q = self.q / np.linalg.norm(self.q)
            self.P = (np.eye(4) - K @ H) @ P_pred
        else:
            self.q = q_pred
            self.P = P_pred
            
        return self.q
    
    def _quat_to_rot(self, q):
        q0, q1, q2, q3 = q
        return np.array([
            [1-2*(q2*q2+q3*q3), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1*q1+q2*q2)]
        ])
    
    def _compute_H(self, q):
        q0, q1, q2, q3 = q
        H = np.zeros((3, 4))
        H[0, 0] = -2*q2; H[0, 1] = 2*q3; H[0, 2] = -2*q0; H[0, 3] = 2*q1
        H[1, 0] = 2*q1; H[1, 1] = 2*q0; H[1, 2] = 2*q3; H[1, 3] = 2*q2
        H[2, 0] = 0; H[2, 1] = -2*q2; H[2, 2] = -2*q1; H[2, 3] = 0
        return H
    
    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.P = np.eye(4) * 0.01
        self.last_time = None
        self.acc_jerk = 0
        self.prev_acc = None

def get_filter_instance(subject_id, action_id, filter_type, reset=False):
    global filter_cache
    cache_key = f"{subject_id}_{action_id}_{filter_type}"
    if reset or cache_key not in filter_cache:
        if filter_type == 'madgwick': 
            filter_instance = MadgwickFilter(beta=0.15)
        elif filter_type == 'kalman': 
            filter_instance = KalmanFilter(process_noise=2e-5)
        elif filter_type == 'ekf': 
            filter_instance = ExtendedKalmanFilter(process_noise=1e-5)
        else: 
            filter_instance = MadgwickFilter(beta=0.15)
        filter_cache[cache_key] = filter_instance
    return filter_cache[cache_key]

def process_imu_data(acc_data, gyro_data, timestamps=None, filter_type='madgwick', return_features=False, trial_id=None, reset_filter=False):
    if acc_data is None or gyro_data is None or len(acc_data) < 3 or len(gyro_data) < 3:
        logger.debug(f"Invalid input data for IMU processing: acc_len={len(acc_data) if acc_data is not None else 'None'}, gyro_len={len(gyro_data) if gyro_data is not None else 'None'}")
        return None
    
    # Extract subject and action IDs from trial_id if available
    subject_id = action_id = 0
    if trial_id is not None:
        parts = trial_id.split('_')
        if len(parts) >= 2:
            try:
                subject_id = int(parts[0])
                action_id = int(parts[1])
            except:
                pass
    
    # Get or create orientation filter
    orientation_filter = get_filter_instance(subject_id, action_id, filter_type, reset=reset_filter)
    
    quaternions = []
    try:
        for i in range(len(acc_data)):
            acc = acc_data[i]
            gyro = gyro_data[i]
            timestamp = timestamps[i] if timestamps is not None else None
            
            # Apply gravity correction
            if i == 0 or reset_filter:
                gravity_direction = np.array([0, 0, 9.81])
            else:
                last_q = quaternions[-1]
                r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
            
            # Add gravity to accelerometer data for filter update
            acc_with_gravity = acc + gravity_direction
            norm = np.linalg.norm(acc_with_gravity)
            if norm > 1e-6:
                acc_with_gravity = acc_with_gravity / norm
            
            # Update orientation filter
            q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
            quaternions.append(q)
        
        # Return quaternions
        return {'quaternion': np.array(quaternions)}
    except Exception as e:
        logger.debug(f"Error processing IMU data: {e}")
        return None

def benchmark_filters(acc_data, gyro_data, timestamps=None):
    filters = {
        'madgwick': MadgwickFilter(beta=0.15), 
        'kalman': KalmanFilter(process_noise=2e-5),
        'ekf': ExtendedKalmanFilter(process_noise=1e-5)
    }
    
    results = {}
    for name, filter_obj in filters.items():
        filter_obj.reset()
        quaternions = []
        
        try:
            start_time = time.time()
            for i in range(len(acc_data)):
                acc = acc_data[i]
                gyro = gyro_data[i]
                ts = timestamps[i] if timestamps is not None else None
                
                # Apply gravity correction
                if i > 0 and quaternions:
                    last_q = quaternions[-1]
                    r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                    gravity_direction = r.inv().apply([0, 0, 9.81])
                else:
                    gravity_direction = np.array([0, 0, 9.81])
                
                acc_with_gravity = acc + gravity_direction
                norm = np.linalg.norm(acc_with_gravity)
                if norm > 1e-6:
                    acc_with_gravity = acc_with_gravity / norm
                
                q = filter_obj.update(acc_with_gravity, gyro, ts)
                quaternions.append(q)
                
            processing_time = time.time() - start_time
            
            results[name] = {
                'quaternions': np.array(quaternions),
                'processing_time': processing_time
            }
        except Exception as e:
            logger.debug(f"Error benchmarking filter {name}: {e}")
            results[name] = {
                'quaternions': np.zeros((len(acc_data), 4)),
                'processing_time': float('inf')
            }
    
    return results
