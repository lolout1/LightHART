import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
import pandas as pd
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("imu_fusion")

MAX_THREADS = 30
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(MAX_THREADS)
filter_registry = {}
filter_locks = defaultdict(threading.Lock)

def register_filter(filter_id, filter_instance):
    with filter_locks[filter_id]:
        filter_registry[filter_id] = filter_instance
    return filter_instance

def get_filter(filter_id, filter_type='madgwick', params=None, create_if_missing=True, reset=False):
    with filter_locks[filter_id]:
        if filter_id in filter_registry and not reset:
            return filter_registry[filter_id]
        if not create_if_missing:
            return None
        
        if filter_type == 'madgwick':
            beta = params.get('beta', 0.15) if params else 0.15
            filter_instance = MadgwickFilter(beta=beta)
        elif filter_type == 'kalman':
            proc_noise = params.get('process_noise', 5e-5) if params else 5e-5
            meas_noise = params.get('measurement_noise', 0.1) if params else 0.1
            filter_instance = KalmanFilter(process_noise=proc_noise, measurement_noise=meas_noise)
        elif filter_type == 'ekf':
            proc_noise = params.get('process_noise', 1e-5) if params else 1e-5
            meas_noise = params.get('measurement_noise', 0.05) if params else 0.05
            filter_instance = ExtendedKalmanFilter(process_noise=proc_noise, measurement_noise=meas_noise)
        else:
            filter_instance = MadgwickFilter()
        
        filter_registry[filter_id] = filter_instance
        return filter_instance

def clear_filters():
    for filter_id in list(filter_registry.keys()):
        with filter_locks[filter_id]:
            if filter_id in filter_registry:
                del filter_registry[filter_id]

def robust_interpolate(x, y, x_new):
    if len(x) < 2 or len(y) < 2:
        return np.full_like(x_new, y[0] if len(y) > 0 else 0.0)
    
    try:
        dy, dx = np.diff(y), np.diff(x)
        rates = np.abs(dy / np.maximum(dx, 1e-10))
        rapid_changes = rates > 2.0
        
        if not np.any(rapid_changes):
            try:
                return CubicSpline(x, y)(x_new)
            except:
                return interp1d(x, y, bounds_error=False, fill_value='extrapolate')(x_new)
        
        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        
        try:
            spline_interp = CubicSpline(x, y)
        except:
            return linear_interp(x_new)
        
        y_interp = np.zeros_like(x_new, dtype=float)
        segments, segment_start = [], None
        
        for i in range(len(rapid_changes)):
            if rapid_changes[i] and segment_start is None:
                segment_start = i
            elif not rapid_changes[i] and segment_start is not None:
                segments.append((segment_start, i))
                segment_start = None
        
        if segment_start is not None:
            segments.append((segment_start, len(rapid_changes)))
        
        linear_mask = np.zeros_like(x_new, dtype=bool)
        
        for start_idx, end_idx in segments:
            t_start = max(x[start_idx] - 0.05, x[0])
            t_end = min(x[min(end_idx, len(x)-1)] + 0.05, x[-1])
            linear_mask |= (x_new >= t_start) & (x_new <= t_end)
        
        if np.any(linear_mask):
            y_interp[linear_mask] = linear_interp(x_new[linear_mask])
        
        if np.any(~linear_mask):
            y_interp[~linear_mask] = spline_interp(x_new[~linear_mask])
        
        return y_interp
    except:
        return interp1d(x, y, bounds_error=False, fill_value='extrapolate')(x_new)

def align_sensor_data(acc_data, gyro_data, time_tolerance=0.01):
    if len(acc_data) == 0 or len(gyro_data) == 0:
        return np.array([]), np.array([]), np.array([])
    
    if isinstance(acc_data, np.ndarray) and isinstance(gyro_data, np.ndarray):
        if len(acc_data.shape) > 1 and acc_data.shape[1] >= 4:
            acc_times, gyro_times = acc_data[:, 0], gyro_data[:, 0]
            acc_values, gyro_values = acc_data[:, 1:4], gyro_data[:, 1:4]
        else:
            acc_times = np.arange(len(acc_data))
            gyro_times = np.arange(len(gyro_data))
            acc_values, gyro_values = acc_data, gyro_data
            
            if len(acc_data) == len(gyro_data):
                return acc_data, gyro_data, acc_times
            
            min_len = min(len(acc_data), len(gyro_data))
            return acc_data[:min_len], gyro_data[:min_len], acc_times[:min_len]
    else:
        try:
            if isinstance(acc_data.iloc[0, 0], str):
                acc_times = pd.to_datetime(acc_data.iloc[:, 0]).astype(np.int64) // 10**6
            else:
                acc_times = acc_data.iloc[:, 0].values
            
            if isinstance(gyro_data.iloc[0, 0], str):
                gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).astype(np.int64) // 10**6
            else:
                gyro_times = gyro_data.iloc[:, 0].values
            
            acc_values = acc_data.iloc[:, 1:4].values
            gyro_values = gyro_data.iloc[:, 1:4].values
        except:
            return np.array([]), np.array([]), np.array([])
    
    start_time = max(acc_times[0], gyro_times[0])
    end_time = min(acc_times[-1], gyro_times[-1])
    
    if end_time <= start_time:
        return np.array([]), np.array([]), np.array([])
    
    acc_sampling = np.median(np.diff(acc_times))
    gyro_sampling = np.median(np.diff(gyro_times))
    target_sampling = min(acc_sampling, gyro_sampling)
    
    common_times = np.arange(start_time, end_time, target_sampling)
    if len(common_times) < 5:
        return np.array([]), np.array([]), np.array([])
    
    acc_in_range = (acc_times >= start_time) & (acc_times <= end_time)
    gyro_in_range = (gyro_times >= start_time) & (gyro_times <= end_time)
    
    if np.sum(acc_in_range) < 3 or np.sum(gyro_in_range) < 3:
        return np.array([]), np.array([]), np.array([])
    
    aligned_acc = np.zeros((len(common_times), 3))
    aligned_gyro = np.zeros((len(common_times), 3))
    
    for axis in range(3):
        aligned_acc[:, axis] = robust_interpolate(
            acc_times[acc_in_range], 
            acc_values[acc_in_range, axis], 
            common_times
        )
        
        aligned_gyro[:, axis] = robust_interpolate(
            gyro_times[gyro_in_range], 
            gyro_values[gyro_in_range, axis], 
            common_times
        )
    
    return aligned_acc, aligned_gyro, common_times

def save_aligned_data(subject_id, action_id, trial_id, acc_data, gyro_data, timestamps=None, save_dir="data/aligned"):
    os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
    os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
    
    if timestamps is not None:
        os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
    
    filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
    
    with file_semaphore:
        np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
        np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
        
        if timestamps is not None:
            np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)

class OrientationFilter:
    def __init__(self, freq=30.0):
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
    
    def update(self, acc, gyro, timestamp=None):
        dt = 1.0 / self.freq
        
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt <= 0:
                dt = 1.0 / self.freq
        
        if timestamp is not None:
            self.last_time = timestamp
        
        new_orientation = self._update_impl(acc, gyro, dt)
        norm = np.linalg.norm(new_orientation)
        
        if norm > 1e-10:
            self.orientation_q = new_orientation / norm
        
        return self.orientation_q
    
    def _update_impl(self, acc, gyro, dt):
        pass
    
    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None

class MadgwickFilter(OrientationFilter):
    def __init__(self, freq=30.0, beta=0.15):
        super().__init__(freq)
        self.beta = beta
    
    def _update_impl(self, acc, gyro, dt):
        q = self.orientation_q
        acc_norm = np.linalg.norm(acc)
        
        if acc_norm < 1e-10:
            return q
        
        acc = acc / acc_norm
        q0, q1, q2, q3 = q
        
        f1 = 2.0 * (q1 * q3 - q0 * q2) - acc[0]
        f2 = 2.0 * (q0 * q1 + q2 * q3) - acc[1]
        f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - acc[2]
        
        J_t = np.array([
            [-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
            [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
            [0.0, -4.0*q1, -4.0*q2, 0.0]
        ])
        
        grad = J_t.T @ np.array([f1, f2, f3])
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm > 0:
            grad = grad / grad_norm
        
        qDot = 0.5 * np.array([
            -q1*gyro[0]-q2*gyro[1]-q3*gyro[2],
            q0*gyro[0]+q2*gyro[2]-q3*gyro[1],
            q0*gyro[1]-q1*gyro[2]+q3*gyro[0],
            q0*gyro[2]+q1*gyro[1]-q2*gyro[0]
        ])
        
        qDot = qDot - self.beta * grad
        q = q + qDot * dt
        
        return q / np.linalg.norm(q)

class KalmanFilter(OrientationFilter):
    def __init__(self, freq=30.0, process_noise=5e-5, measurement_noise=0.1):
        super().__init__(freq)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[:4, :4] *= 0.1
        self.Q[4:, 4:] *= 10.0
        self.R = np.eye(3) * measurement_noise
        self.P = np.eye(self.state_dim) * 1e-2
    
    def _update_impl(self, acc, gyro, dt):
        q = self.x[:4]
        bias = self.x[4:]
        
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        
        gyro_corrected = gyro - bias
        omega = np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega)
        
        F = np.eye(self.state_dim)
        F[:4, :4] += 0.5 * dt * self._omega_matrix(gyro_corrected)
        F[:4, 4:] = -0.5 * dt * self._quaternion_product_matrix(q)[:, 1:]
        
        x_pred = self.x.copy()
        x_pred[:4] = q + q_dot * dt
        x_pred[4:] = bias
        
        q_norm = np.linalg.norm(x_pred[:4])
        if q_norm > 0:
            x_pred[:4] = x_pred[:4] / q_norm
        
        P_pred = F @ self.P @ F.T + self.Q
        
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            acc = acc / acc_norm
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ np.array([0, 0, 1])
            y = acc - g_pred
            
            H = self._compute_H_matrix(x_pred[:4])
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            self.x = x_pred + K @ y
            I_KH = np.eye(self.state_dim) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            self.x = x_pred
            self.P = P_pred
        
        q_norm = np.linalg.norm(self.x[:4])
        if q_norm > 0:
            self.x[:4] = self.x[:4] / q_norm
        
        return self.x[:4]
    
    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2-x1*x2-y1*y2-z1*z2,
            w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2,
            w1*z2+x1*y2-y1*x2+z1*w2
        ])
    
    def _quaternion_product_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w]
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
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
    
    def _compute_H_matrix(self, q):
        w, x, y, z = q
        H_q = np.zeros((3, 4))
        H_q[0, 0], H_q[0, 1], H_q[0, 2], H_q[0, 3] = -2*y, 2*z, -2*w, 2*x
        H_q[1, 0], H_q[1, 1], H_q[1, 2], H_q[1, 3] = 2*x, 2*w, 2*z, 2*y
        H_q[2, 0], H_q[2, 1], H_q[2, 2], H_q[2, 3] = 0, -2*y, -2*z, 0
        
        H = np.zeros((3, self.state_dim))
        H[:, :4] = H_q
        
        return H

class ExtendedKalmanFilter(OrientationFilter):
    def __init__(self, freq=30.0, process_noise=1e-5, measurement_noise=0.05):
        super().__init__(freq)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[:4, :4] *= 0.1
        self.Q[4:, 4:] *= 5.0
        self.R = np.eye(3) * measurement_noise
        self.P = np.eye(self.state_dim) * 1e-2
        self.g_ref = np.array([0, 0, 1])
    
    def _update_impl(self, acc, gyro, dt):
        q = self.x[:4]
        bias = self.x[4:]
        
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        
        gyro_corrected = gyro - bias
        
        q_dot = 0.5 * self._quaternion_product_matrix(q) @ np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        
        q_pred = q + q_dot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)
        
        bias_pred = bias
        
        x_pred = np.zeros_like(self.x)
        x_pred[:4] = q_pred
        x_pred[4:] = bias_pred
        
        F = np.eye(self.state_dim)
        F[:4, :4] = self._quaternion_update_jacobian(q, gyro_corrected, dt)
        F[:4, 4:] = -0.5 * dt * self._quaternion_product_matrix(q)[:, 1:]
        
        P_pred = F @ self.P @ F.T + self.Q
        
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            acc = acc / acc_norm
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ self.g_ref
            
            z = acc
            h = g_pred
            H = self._measurement_jacobian(x_pred[:4])
            
            y = z - h
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            self.x = x_pred + K @ y
            I_KH = np.eye(self.state_dim) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            self.x = x_pred
            self.P = P_pred
        
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        
        return self.x[:4]
    
    def _quaternion_product_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w]
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
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
    
    def _measurement_jacobian(self, q):
        w, x, y, z = q
        H_acc = np.zeros((3, self.state_dim))
        H_acc[:3, :4] = np.array([
            [2*y, 2*z, 2*w, 2*x],
            [-2*z, 2*y, 2*x, -2*w],
            [-2*x, -2*w, 2*z, 2*y]
        ])
        return H_acc

def process_window_with_filter(acc_window, gyro_window, timestamps=None, filter_type='madgwick', 
                             filter_id=None, reset_filter=False, is_linear_acc=True, filter_params=None):
    if acc_window.shape[0] == 0 or gyro_window.shape[0] == 0:
        return {'quaternion': np.zeros((1, 4))}
    
    if acc_window.shape[0] != gyro_window.shape[0]:
        min_len = min(acc_window.shape[0], gyro_window.shape[0])
        acc_window = acc_window[:min_len]
        gyro_window = gyro_window[:min_len]
        if timestamps is not None:
            timestamps = timestamps[:min_len]
    
    if timestamps is None:
        timestamps = np.linspace(0, acc_window.shape[0] / 30.0, acc_window.shape[0])
    
    if filter_id is None:
        filter_id = f"window_{filter_type}_{np.random.randint(0, 10000)}"
    
    params = filter_params or {}
    with filter_locks[filter_id]:
        orientation_filter = get_filter(filter_id, filter_type, params=params, reset=reset_filter)
        
        if reset_filter:
            orientation_filter.reset()
        
        quaternions = np.zeros((len(acc_window), 4))
        
        for i in range(len(acc_window)):
            acc = acc_window[i]
            gyro = gyro_window[i]
            timestamp = timestamps[i] if timestamps is not None else None
            
            if is_linear_acc:
                gravity_direction = np.array([0, 0, 9.81])
                if i > 0:
                    last_q = quaternions[i-1]
                    r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                    gravity_direction = r.inv().apply([0, 0, 9.81])
                
                acc_with_gravity = acc + gravity_direction
                norm = np.linalg.norm(acc_with_gravity)
                if norm > 1e-6:
                    acc_with_gravity = acc_with_gravity / norm
            else:
                acc_with_gravity = acc
                norm = np.linalg.norm(acc_with_gravity)
                if norm > 1e-6:
                    acc_with_gravity = acc_with_gravity / norm
            
            q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
            quaternions[i] = q
    
    return {'quaternion': quaternions}

def process_windows_with_filter(window_data, filter_type='madgwick', base_filter_id=None, 
                              reset_per_window=False, is_linear_acc=True, filter_params=None):
    acc_windows = window_data.get('accelerometer')
    gyro_windows = window_data.get('gyroscope')
    
    if acc_windows is None or gyro_windows is None or acc_windows.shape[0] != gyro_windows.shape[0]:
        return {'quaternion': np.zeros((1, 1, 4))}
    
    num_windows = acc_windows.shape[0]
    window_size = acc_windows.shape[1]
    
    all_quaternions = np.zeros((num_windows, window_size, 4))
    params = filter_params or {}
    
    for i in range(num_windows):
        filter_id = f"{base_filter_id}_win_{i}" if base_filter_id else f"win_{i}_{filter_type}"
        reset = reset_per_window or i == 0
        
        result = process_window_with_filter(
            acc_windows[i], 
            gyro_windows[i], 
            None,
            filter_type=filter_type,
            filter_id=filter_id,
            reset_filter=reset,
            is_linear_acc=is_linear_acc,
            filter_params=params
        )
        
        all_quaternions[i] = result['quaternion']
    
    return {'quaternion': all_quaternions}

def process_sequential_windows(data, window_size=64, stride=32, label=None, filter_type='madgwick', 
                             filter_params=None, base_filter_id=None, stateful=True, is_linear_acc=True):
    if 'accelerometer' not in data or data['accelerometer'] is None or 'gyroscope' not in data or data['gyroscope'] is None:
        return {'labels': np.array([label])} if label is not None else {'labels': np.array([])}
    
    acc_data = data['accelerometer']
    gyro_data = data['gyroscope']
    timestamps = data.get('aligned_timestamps', None)
    
    if len(acc_data) < window_size // 2:
        return {'labels': np.array([label])} if label is not None else {'labels': np.array([])}
    
    num_windows = max(1, (len(acc_data) - window_size) // stride + 1)
    windows = defaultdict(list)
    
    if label is not None:
        windows['labels'] = []
    
    filter_id = base_filter_id if base_filter_id is not None else f"sequential_{filter_type}"
    params = filter_params or {}
    
    valid_window_count = 0
    
    for i in range(num_windows):
        start = i * stride
        end = min(start + window_size, len(acc_data))
        
        if end - start < window_size // 2:
            continue
        
        acc_window = np.zeros((window_size, acc_data.shape[1]))
        actual_length = end - start
        acc_window[:actual_length] = acc_data[start:end]
        
        gyro_window = np.zeros((window_size, gyro_data.shape[1]))
        actual_gyro_length = min(actual_length, len(gyro_data) - start)
        if actual_gyro_length == 0:
            continue
        
        gyro_window[:actual_gyro_length] = gyro_data[start:start + actual_gyro_length]
        
        ts_window = None
        if timestamps is not None:
            ts_window = np.zeros(window_size)
            actual_ts_length = min(actual_length, len(timestamps) - start)
            if actual_ts_length > 0:
                ts_window[:actual_ts_length] = timestamps[start:start + actual_ts_length]
        
        reset = not stateful or i == 0
        
        try:
            result = process_window_with_filter(
                acc_window, gyro_window, ts_window, 
                filter_type=filter_type, filter_id=filter_id, 
                reset_filter=reset, is_linear_acc=is_linear_acc,
                filter_params=params
            )
            
            windows['accelerometer'].append(acc_window)
            windows['gyroscope'].append(gyro_window)
            windows['quaternion'].append(result['quaternion'])
            if label is not None:
                windows['labels'].append(label)
            
            valid_window_count += 1
        except Exception as e:
            logger.warning(f"Failed to process window {i}: {e}")
            continue
    
    if valid_window_count == 0:
        return {'labels': np.array([label])} if label is not None else {'labels': np.array([])}
    
    for key in windows:
        if key != 'labels' and len(windows[key]) > 0:
            windows[key] = np.array(windows[key])
    
    if 'labels' in windows and isinstance(windows['labels'], list) and len(windows['labels']) > 0:
        windows['labels'] = np.array(windows['labels'])
    
    return windows

def create_filter_id(subject_id, action_id, trial_id=None, window_id=None, filter_type='madgwick'):
    base_id = f"S{subject_id}_A{action_id}"
    if trial_id is not None:
        base_id += f"_T{trial_id}"
    if window_id is not None:
        base_id += f"_W{window_id}"
    return f"{base_id}_{filter_type}"
