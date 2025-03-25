import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import time
import traceback
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import logging
import os
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("imu_fusion")

MAX_THREADS = 30
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(MAX_THREADS)
filter_registry = {}

def register_filter(filter_id, filter_instance):
    filter_registry[filter_id] = filter_instance
    return filter_instance

def get_filter(filter_id, filter_type='madgwick', params=None, create_if_missing=True, reset=False):
    if filter_id in filter_registry and not reset: return filter_registry[filter_id]
    if not create_if_missing: return None
    if filter_type == 'madgwick': filter_instance = MadgwickFilter(beta=params.get('beta', 0.1) if params else 0.1)
    elif filter_type == 'kalman': filter_instance = KalmanFilter(process_noise=params.get('process_noise', 1e-5), measurement_noise=params.get('measurement_noise', 0.1))
    elif filter_type == 'ekf': filter_instance = ExtendedKalmanFilter(process_noise=params.get('process_noise', 1e-5), measurement_noise=params.get('measurement_noise', 0.05))
    elif filter_type == 'ukf': filter_instance = UnscentedKalmanFilter(alpha=params.get('alpha', 0.1), beta=params.get('beta', 2.0), kappa=params.get('kappa', 0.0))
    elif filter_type == 'comp': filter_instance = ComplementaryFilter(alpha=params.get('alpha', 0.02))
    else: filter_instance = MadgwickFilter()
    return register_filter(filter_id, filter_instance)

def clear_filters(): filter_registry.clear()

def hybrid_interpolate(x, y, x_new, threshold=2.0, window_size=5):
    if len(x) < 2 or len(y) < 2: return np.full_like(x_new, y[0] if len(y) > 0 else 0.0)
    try:
        dy, dx = np.diff(y), np.diff(x)
        rates = np.abs(dy / np.maximum(dx, 1e-10))
        if len(rates) >= window_size: rates = savgol_filter(rates, window_size, 2)
        rapid_changes = rates > threshold
        if not np.any(rapid_changes):
            try: return CubicSpline(x, y)(x_new)
            except: return interp1d(x, y, bounds_error=False, fill_value='extrapolate')(x_new)
        if np.all(rapid_changes): return interp1d(x, y, bounds_error=False, fill_value='extrapolate')(x_new)
        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        try: spline_interp = CubicSpline(x, y)
        except: return linear_interp(x_new)
        y_interp = np.zeros_like(x_new, dtype=float)
        segments = []
        segment_start = None
        for i in range(len(rapid_changes)):
            if rapid_changes[i] and segment_start is None: segment_start = i
            elif not rapid_changes[i] and segment_start is not None:
                segments.append((segment_start, i))
                segment_start = None
        if segment_start is not None: segments.append((segment_start, len(rapid_changes)))
        linear_mask = np.zeros_like(x_new, dtype=bool)
        buffer = 0.05
        for start_idx, end_idx in segments:
            t_start = max(x[start_idx] - buffer, x[0])
            t_end = min(x[min(end_idx, len(x)-1)] + buffer, x[-1])
            linear_mask |= (x_new >= t_start) & (x_new <= t_end)
        if np.any(linear_mask): y_interp[linear_mask] = linear_interp(x_new[linear_mask])
        if np.any(~linear_mask): y_interp[~linear_mask] = spline_interp(x_new[~linear_mask])
        return y_interp
    except: return interp1d(x, y, bounds_error=False, fill_value='extrapolate')(x_new)

def align_sensor_data(acc_data, gyro_data, time_tolerance=0.01):
    if len(acc_data) == 0 or len(gyro_data) == 0: return np.array([]), np.array([]), np.array([])
    if isinstance(acc_data, np.ndarray) and isinstance(gyro_data, np.ndarray):
        acc_times = np.arange(len(acc_data))
        gyro_times = np.arange(len(gyro_data))
        if len(acc_data) == len(gyro_data): return acc_data, gyro_data, acc_times
        min_len = min(len(acc_data), len(gyro_data))
        return acc_data[:min_len], gyro_data[:min_len], acc_times[:min_len]
    try:
        if isinstance(acc_data.iloc[0, 0], str): acc_times = pd.to_datetime(acc_data.iloc[:, 0]).values
        else: acc_times = acc_data.iloc[:, 0].values
        if isinstance(gyro_data.iloc[0, 0], str): gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).values
        else: gyro_times = gyro_data.iloc[:, 0].values
    except:
        logger.error("Error extracting timestamps from data")
        return np.array([]), np.array([]), np.array([])
    start_time, end_time = max(acc_times[0], gyro_times[0]), min(acc_times[-1], gyro_times[-1])
    acc_in_range = (acc_times >= start_time) & (acc_times <= end_time)
    gyro_in_range = (gyro_times >= start_time) & (gyro_times <= end_time)
    acc_data_filtered = acc_data.iloc[acc_in_range, 1:4].values
    gyro_data_filtered = gyro_data.iloc[gyro_in_range, 1:4].values
    common_times = np.linspace(start_time, end_time, min(len(acc_data_filtered), len(gyro_data_filtered)))
    aligned_acc, aligned_gyro = np.zeros((len(common_times), 3)), np.zeros((len(common_times), 3))
    for axis in range(3):
        aligned_acc[:, axis] = hybrid_interpolate(acc_times[acc_in_range], acc_data_filtered[:, axis], common_times)
        aligned_gyro[:, axis] = hybrid_interpolate(gyro_times[gyro_in_range], gyro_data_filtered[:, axis], common_times)
    return aligned_acc, aligned_gyro, common_times

def save_aligned_data(subject_id, action_id, trial_id, acc_data, gyro_data, timestamps=None, save_dir="data/aligned"):
    os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
    os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
    if timestamps is not None: os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
    filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
    with file_semaphore:
        np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
        np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
        if timestamps is not None: np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)

def apply_adaptive_filter(acc_data, cutoff_freq=2.0, fs=30.0):
    data_length = acc_data.shape[0]
    if data_length < 15:
        filtered_data = np.zeros_like(acc_data)
        for i in range(acc_data.shape[1]):
            if data_length > 2: filtered_data[:, i] = np.convolve(acc_data[:, i], np.ones(3)/3, mode='same')
            else: filtered_data[:, i] = acc_data[:, i]
        return filtered_data
    filtered_data = np.zeros_like(acc_data)
    padlen = min(data_length - 1, 10)
    try:
        for i in range(acc_data.shape[1]):
            b, a = butter(4, cutoff_freq / (fs/2), btype='low')
            filtered_data[:, i] = filtfilt(b, a, acc_data[:, i], padlen=padlen)
    except: filtered_data = acc_data.copy()
    return filtered_data

class OrientationFilter:
    def __init__(self, freq=30.0):
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update(self, acc, gyro, timestamp=None):
        dt = 1.0 / self.freq
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt <= 0: dt = 1.0 / self.freq
        if timestamp is not None: self.last_time = timestamp
        new_orientation = self._update_impl(acc, gyro, dt)
        norm = np.linalg.norm(new_orientation)
        if norm > 1e-10: self.orientation_q = new_orientation / norm
        return self.orientation_q
        
    def _update_impl(self, acc, gyro, dt): pass
    
    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None

class MadgwickFilter(OrientationFilter):
    def __init__(self, freq=30.0, beta=0.1):
        super().__init__(freq)
        self.beta = beta
        
    def _update_impl(self, acc, gyro, dt):
        q = self.orientation_q
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10: return q
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
        if grad_norm > 0: grad = grad / grad_norm
        qDot = 0.5 * np.array([
            -q1 * gyro[0] - q2 * gyro[1] - q3 * gyro[2],
            q0 * gyro[0] + q2 * gyro[2] - q3 * gyro[1],
            q0 * gyro[1] - q1 * gyro[2] + q3 * gyro[0],
            q0 * gyro[2] + q1 * gyro[1] - q2 * gyro[0]
        ])
        qDot = qDot - self.beta * grad
        q = q + qDot * dt
        return q / np.linalg.norm(q)

class ComplementaryFilter(OrientationFilter):
    def __init__(self, freq=30.0, alpha=0.02):
        super().__init__(freq)
        self.alpha = alpha
        
    def _update_impl(self, acc, gyro, dt):
        q = self.orientation_q
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10: return q
        acc = acc / acc_norm
        acc_q = self._accel_to_quaternion(acc)
        gyro_q = self._integrate_gyro(q, gyro, dt)
        result_q = self._slerp(gyro_q, acc_q, self.alpha)
        return result_q
    
    def _accel_to_quaternion(self, acc):
        z_ref = np.array([0, 0, 1])
        rotation_axis = np.cross(z_ref, acc)
        axis_norm = np.linalg.norm(rotation_axis)
        if axis_norm < 1e-10: return np.array([1.0, 0.0, 0.0, 0.0]) if acc[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
        rotation_axis = rotation_axis / axis_norm
        angle = np.arccos(np.clip(np.dot(z_ref, acc), -1.0, 1.0))
        q = np.zeros(4)
        q[0] = np.cos(angle / 2)
        q[1:4] = rotation_axis * np.sin(angle / 2)
        return q
    
    def _integrate_gyro(self, q, gyro, dt):
        q_dot = 0.5 * np.array([
            -q[1]*gyro[0] - q[2]*gyro[1] - q[3]*gyro[2],
            q[0]*gyro[0] + q[2]*gyro[2] - q[3]*gyro[1],
            q[0]*gyro[1] - q[1]*gyro[2] + q[3]*gyro[0],
            q[0]*gyro[2] + q[1]*gyro[1] - q[2]*gyro[0]
        ])
        q_new = q + q_dot * dt
        return q_new / np.linalg.norm(q_new)
    
    def _slerp(self, q1, q2, t):
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.sum(q1 * q2)
        if dot < 0.0: q2, dot = -q2, -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q1) + (s1 * q2)

class KalmanFilter(OrientationFilter):
    def __init__(self, freq=30.0, process_noise=1e-5, measurement_noise=0.1):
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
        if q_norm > 0: q = q / q_norm
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
        if q_norm > 0: x_pred[:4] = x_pred[:4] / q_norm
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
            self.P = (np.eye(self.state_dim) - K @ H) @ P_pred
        else:
            self.x = x_pred
            self.P = P_pred
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
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
    
    def _quaternion_product_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
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
        H_q[0, 0], H_q[0, 1], H_q[0, 2], H_q[0, 3] = 2*y, 2*z, 2*w, 2*x
        H_q[1, 0], H_q[1, 1], H_q[1, 2], H_q[1, 3] = -2*z, 2*y, 2*x, -2*w
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
        self.Q[4:, 4:] *= 10.0
        self.R = np.eye(3) * measurement_noise
        self.P = np.eye(self.state_dim) * 1e-2
        self.g_ref = np.array([0, 0, 1])
        
    def _update_impl(self, acc, gyro, dt):
        q = self.x[:4]
        bias = self.x[4:]
        q_norm = np.linalg.norm(q)
        if q_norm > 0: q = q / q_norm
        gyro_corrected = gyro - bias
        q_dot = 0.5 * self._quaternion_product_matrix(q) @ np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_pred = q + q_dot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)
        x_pred = np.zeros_like(self.x)
        x_pred[:4] = q_pred
        x_pred[4:] = bias
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
            [0, -2*y, -2*z, 0]
        ])
        return H_acc

class UnscentedKalmanFilter(OrientationFilter):
    def __init__(self, freq=30.0, alpha=0.1, beta=2.0, kappa=0.0):
        super().__init__(freq)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        self.Q = np.eye(self.state_dim) * 1e-5
        self.Q[:4, :4] *= 0.1
        self.Q[4:, 4:] *= 10.0
        self.R = np.eye(3) * 0.1
        self.P = np.eye(self.state_dim) * 1e-2
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self._calculate_weights()
        self.g_ref = np.array([0, 0, 1])
        
    def _calculate_weights(self):
        n = self.state_dim
        self.num_sigma_points = 2 * n + 1
        self.Wm = np.zeros(self.num_sigma_points)
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wm[1:] = 1.0 / (2 * (n + self.lambda_))
        self.Wc = np.zeros(self.num_sigma_points)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        self.Wc[1:] = self.Wm[1:]
    
    def _generate_sigma_points(self):
        n = self.state_dim
        try: U = np.linalg.cholesky((n + self.lambda_) * self.P)
        except:
            self.P += np.eye(n) * 1e-6
            U = np.linalg.cholesky((n + self.lambda_) * self.P)
        sigma_points = np.zeros((self.num_sigma_points, n))
        sigma_points[0] = self.x
        for i in range(n):
            sigma_points[i+1] = self.x + U[i]
            sigma_points[i+1+n] = self.x - U[i]
        return sigma_points
    
    def _quaternion_normalize(self, q):
        norm = np.linalg.norm(q)
        if norm > 1e-10: return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def _process_model(self, sigma_point, gyro, dt):
        q = sigma_point[:4]
        bias = sigma_point[4:]
        q = self._quaternion_normalize(q)
        gyro_corrected = gyro - bias
        omega = np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega)
        q_pred = q + q_dot * dt
        q_pred = self._quaternion_normalize(q_pred)
        bias_pred = bias
        x_pred = np.zeros_like(sigma_point)
        x_pred[:4] = q_pred
        x_pred[4:] = bias_pred
        return x_pred
    
    def _measurement_model(self, sigma_point):
        q = sigma_point[:4]
        q = self._quaternion_normalize(q)
        R = self._quaternion_to_rotation_matrix(q)
        g_pred = R @ self.g_ref
        return g_pred
    
    def _update_impl(self, acc, gyro, dt):
        try:
            sigma_points = self._generate_sigma_points()
            sigma_points_pred = np.zeros_like(sigma_points)
            for i in range(self.num_sigma_points):
                sigma_points_pred[i] = self._process_model(sigma_points[i], gyro, dt)
            x_pred = np.zeros(self.state_dim)
            for i in range(self.num_sigma_points):
                x_pred += self.Wm[i] * sigma_points_pred[i]
            x_pred[:4] = self._quaternion_normalize(x_pred[:4])
            P_pred = np.zeros((self.state_dim, self.state_dim))
            for i in range(self.num_sigma_points):
                diff = sigma_points_pred[i] - x_pred
                diff[:4] = self._quaternion_error(sigma_points_pred[i, :4], x_pred[:4])
                P_pred += self.Wc[i] * np.outer(diff, diff)
            P_pred += self.Q
            acc_norm = np.linalg.norm(acc)
            if acc_norm < 1e-10:
                self.x = x_pred
                self.P = P_pred
                return self.x[:4]
            acc = acc / acc_norm
            z_pred = np.zeros((self.num_sigma_points, 3))
            for i in range(self.num_sigma_points):
                z_pred[i] = self._measurement_model(sigma_points_pred[i])
            z_mean = np.zeros(3)
            for i in range(self.num_sigma_points):
                z_mean += self.Wm[i] * z_pred[i]
            Pzz = np.zeros((3, 3))
            for i in range(self.num_sigma_points):
                diff = z_pred[i] - z_mean
                Pzz += self.Wc[i] * np.outer(diff, diff)
            Pzz += self.R
            Pxz = np.zeros((self.state_dim, 3))
            for i in range(self.num_sigma_points):
                diff_x = sigma_points_pred[i] - x_pred
                diff_x[:4] = self._quaternion_error(sigma_points_pred[i, :4], x_pred[:4])
                diff_z = z_pred[i] - z_mean
                Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
            try: K = Pxz @ np.linalg.inv(Pzz)
            except:
                Pzz += np.eye(3) * 1e-6
                K = Pxz @ np.linalg.inv(Pzz)
            innovation = acc - z_mean
            self.x = x_pred + K @ innovation
            self.x[:4] = self._quaternion_normalize(self.x[:4])
            self.P = P_pred - K @ Pzz @ K.T
            self.P = (self.P + self.P.T) / 2
            return self.x[:4]
        except Exception as e:
            logger.error(f"UKF update error: {e}")
            return self.orientation_q
        
    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _quaternion_error(self, q1, q2):
        q_diff = self._quaternion_multiply(q1, self._quaternion_inverse(q2))
        if q_diff[0] < 0: q_diff = -q_diff
        if abs(q_diff[0]) > 0.9999: return np.zeros(4)
        return q_diff
    
    def _quaternion_inverse(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

def process_imu_data(acc_data, gyro_data, timestamps=None, filter_type='madgwick', 
                   filter_id=None, reset_filter=False, return_features=False):
    if acc_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        return {'quaternion': np.zeros((1, 4))}
    if acc_data.shape[0] != gyro_data.shape[0]:
        min_len = min(acc_data.shape[0], gyro_data.shape[0])
        acc_data = acc_data[:min_len]
        gyro_data = gyro_data[:min_len]
        if timestamps is not None: timestamps = timestamps[:min_len]
    if timestamps is None:
        timestamps = np.linspace(0, acc_data.shape[0] / 30.0, acc_data.shape[0])
    if filter_id is None: filter_id = f"global_{filter_type}"
    orientation_filter = get_filter(filter_id, filter_type, reset=reset_filter)
    if reset_filter: orientation_filter.reset()
    quaternions = np.zeros((len(acc_data), 4))
    for i in range(len(acc_data)):
        acc = acc_data[i]
        gyro = gyro_data[i]
        timestamp = timestamps[i] if timestamps is not None else None
        gravity_direction = np.array([0, 0, 9.81])
        if i > 0:
            last_q = quaternions[i-1]
            r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
            gravity_direction = r.inv().apply([0, 0, 9.81])
        acc_with_gravity = acc + gravity_direction
        norm = np.linalg.norm(acc_with_gravity)
        if norm > 1e-6: acc_with_gravity = acc_with_gravity / norm
        q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
        quaternions[i] = q
    results = {'quaternion': quaternions}
    if return_features:
        results['fusion_features'] = extract_features(acc_data, gyro_data, quaternions)
    return results

def extract_features(acc_data, gyro_data, quat_data):
    acc_mag = np.sqrt(np.sum(acc_data**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro_data**2, axis=1))
    features = [
        np.mean(acc_mag), np.std(acc_mag), np.max(acc_mag),
        np.mean(gyro_mag), np.std(gyro_mag), np.max(gyro_mag),
    ]
    for axis in range(3):
        features.extend([
            np.mean(acc_data[:, axis]), np.std(acc_data[:, axis]),
            np.mean(gyro_data[:, axis]), np.std(gyro_data[:, axis]),
        ])
    euler_angles = np.zeros((len(quat_data), 3))
    for i, q in enumerate(quat_data):
        try:
            rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            euler_angles[i] = rot.as_euler('xyz', degrees=True)
        except: euler_angles[i] = [0, 0, 0]
    for axis in range(3):
        features.extend([
            np.mean(euler_angles[:, axis]), np.std(euler_angles[:, axis]),
            np.max(euler_angles[:, axis]) - np.min(euler_angles[:, axis]),
        ])
    if len(acc_mag) >= 16:
        try:
            from scipy.fft import rfft
            for axis in range(3):
                window = np.hanning(len(acc_data[:, axis]))
                fft_data = np.abs(rfft(acc_data[:, axis] * window))
                features.extend([
                    np.sum(fft_data), np.max(fft_data), 
                    np.argmax(fft_data), np.mean(fft_data)
                ])
                fft_data = np.abs(rfft(gyro_data[:, axis] * window))
                features.extend([
                    np.sum(fft_data), np.max(fft_data),
                    np.argmax(fft_data), np.mean(fft_data)
                ])
        except: features.extend([0] * 24)
    else: features.extend([0] * 24)
    return np.array(features, dtype=np.float32)

def process_windows_with_filter(windows_data, filter_type='madgwick', window_ids=None, 
                              base_filter_id=None, reset_per_window=False):
    acc_windows = windows_data['accelerometer']
    gyro_windows = windows_data.get('gyroscope', None)
    timestamps = windows_data.get('timestamps', None)
    if gyro_windows is None or len(gyro_windows) == 0:
        output = {'quaternion': np.zeros((len(acc_windows), acc_windows.shape[1], 4))}
        return output
    quaternions = np.zeros((len(acc_windows), acc_windows.shape[1], 4))
    if base_filter_id is None: base_filter_id = f"windows_{filter_type}"
    for i in range(len(acc_windows)):
        window_id = i if window_ids is None else window_ids[i]
        filter_id = f"{base_filter_id}_{window_id}"
        reset = reset_per_window or i == 0
        acc = acc_windows[i]
        gyro = gyro_windows[i]
        ts = timestamps[i] if timestamps is not None else None
        result = process_imu_data(acc, gyro, ts, filter_type, filter_id, reset)
        quaternions[i] = result['quaternion']
    return {'quaternion': quaternions}

def create_filter_id(subject_id, action_id, trial_id=None, window_id=None, filter_type='madgwick'):
    base_id = f"S{subject_id}_A{action_id}"
    if trial_id is not None: base_id += f"_T{trial_id}"
    if window_id is not None: base_id += f"_W{window_id}"
    return f"{base_id}_{filter_type}"
