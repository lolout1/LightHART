import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("imu_fusion")

MAX_THREADS = 30
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
filter_registry = {}
filter_locks = defaultdict(threading.Lock)

class OrientationFilter:
    def __init__(self, freq=30.0):
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.initialized = False
    
    def update(self, acc, gyro, timestamp=None):
        dt = 1.0 / self.freq
        if timestamp is not None and self.last_time is not None:
            dt = max(timestamp - self.last_time, 1e-6)
        if timestamp is not None:
            self.last_time = timestamp
        
        if not self.initialized and np.linalg.norm(acc) > 0.1:
            self._initialize_orientation(acc)
            self.initialized = True
            return self.orientation_q
        
        new_orientation = self._update_impl(acc, gyro, dt)
        norm = np.linalg.norm(new_orientation)
        if norm > 1e-10:
            self.orientation_q = new_orientation / norm
        return self.orientation_q
    
    def _initialize_orientation(self, acc):
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10: return
        acc_n = acc / acc_norm
        down = np.array([0, 0, 1])
        
        if np.allclose(acc_n, -down):
            self.orientation_q = np.array([0, 1, 0, 0])
            return
        
        axis = np.cross(acc_n, down)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-10:
            self.orientation_q = np.array([1, 0, 0, 0]) if np.dot(acc_n, down) > 0 else np.array([0, 1, 0, 0])
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(acc_n, down), -1.0, 1.0))
            sin_half, cos_half = np.sin(angle / 2), np.cos(angle / 2)
            self.orientation_q = np.array([cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half])
    
    def _update_impl(self, acc, gyro, dt):
        raise NotImplementedError("Subclasses must implement this method")
    
    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        self.initialized = False

class MadgwickFilter(OrientationFilter):
    def __init__(self, freq=30.0, beta=0.15):
        super().__init__(freq)
        self.beta = beta
    
    def _update_impl(self, acc, gyro, dt):
        q = self.orientation_q
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            return self._integrate_gyro(q, gyro, dt)
        
        acc = acc / acc_norm
        q0, q1, q2, q3 = q
        
        f1 = 2.0 * (q1 * q3 - q0 * q2) - acc[0]
        f2 = 2.0 * (q0 * q1 + q2 * q3) - acc[1]
        f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - acc[2]
        
        J_t = np.array([[-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
                        [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
                        [0.0, -4.0*q1, -4.0*q2, 0.0]])
        
        grad = J_t.T @ np.array([f1, f2, f3])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        qDot = 0.5 * np.array([-q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
                               q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
                               q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
                               q0*gyro[2] + q1*gyro[1] - q2*gyro[0]])
        
        qDot = qDot - self.beta * grad
        q_new = q + qDot * dt
        return q_new / np.linalg.norm(q_new)
    
    def _integrate_gyro(self, q, gyro, dt):
        qDot = 0.5 * np.array([-q[1]*gyro[0] - q[2]*gyro[1] - q[3]*gyro[2],
                               q[0]*gyro[0] + q[2]*gyro[2] - q[3]*gyro[1],
                               q[0]*gyro[1] - q[1]*gyro[2] + q[3]*gyro[0],
                               q[0]*gyro[2] + q[1]*gyro[1] - q[2]*gyro[0]])
        
        q_new = q + qDot * dt
        return q_new / np.linalg.norm(q_new)

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
        if q_norm > 1e-10:
            q = q / q_norm
        
        gyro_corrected = gyro - bias
        omega = np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega)
        
        F = np.eye(self.state_dim)
        F[:4, :4] += 0.5 * dt * self._omega_matrix(gyro_corrected)
        F[:4, 4:] = -0.5 * dt * self._quaternion_jacobian(q)
        
        x_pred = self.x.copy()
        x_pred[:4] = q + q_dot * dt
        x_pred[4:] = bias
        
        q_norm = np.linalg.norm(x_pred[:4])
        if q_norm > 1e-10:
            x_pred[:4] = x_pred[:4] / q_norm
        
        P_pred = F @ self.P @ F.T + self.Q
        
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            acc = acc / acc_norm
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ np.array([0, 0, 1])
            y = acc - g_pred
            
            H = self._compute_measurement_jacobian(x_pred[:4])
            
            try:
                S = H @ P_pred @ H.T + self.R
                K = P_pred @ H.T @ np.linalg.inv(S)
                self.x = x_pred + K @ y
                I_KH = np.eye(self.state_dim) - K @ H
                self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
            except:
                self.x = x_pred
                self.P = P_pred
        else:
            self.x = x_pred
            self.P = P_pred
        
        q_norm = np.linalg.norm(self.x[:4])
        if q_norm > 1e-10:
            self.x[:4] = self.x[:4] / q_norm
        
        return self.x[:4]
    
    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2])
    
    def _quaternion_jacobian(self, q):
        return np.array([[0, -q[1], -q[2], -q[3]],
                        [q[0], 0, q[3], -q[2]],
                        [q[0], -q[3], 0, q[1]],
                        [q[0], q[2], -q[1], 0]])[:, 1:]
    
    def _omega_matrix(self, gyro):
        wx, wy, wz = gyro
        return np.array([[0, -wx, -wy, -wz],
                        [wx, 0, wz, -wy],
                        [wy, -wz, 0, wx],
                        [wz, wy, -wx, 0]])
    
    def _quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
    
    def _compute_measurement_jacobian(self, q):
        w, x, y, z = q
        H_q = np.zeros((3, 4))
        H_q[0] = [-2*y, 2*z, -2*w, 2*x]
        H_q[1] = [2*x, 2*w, 2*z, 2*y]
        H_q[2] = [0, -4*y, -4*z, 0]
        
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
        if q_norm > 1e-10:
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
            
            z, h = acc, g_pred
            H = self._measurement_jacobian(x_pred[:4])
            y = z - h
            
            try:
                S = H @ P_pred @ H.T + self.R
                K = P_pred @ H.T @ np.linalg.inv(S)
                self.x = x_pred + K @ y
                I_KH = np.eye(self.state_dim) - K @ H
                self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
            except:
                self.x = x_pred
                self.P = P_pred
        else:
            self.x = x_pred
            self.P = P_pred
        
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        return self.x[:4]
    
    def _quaternion_product_matrix(self, q):
        w, x, y, z = q
        return np.array([[w, -x, -y, -z],
                        [x, w, -z, y],
                        [y, z, w, -x],
                        [z, -y, x, w]])
    
    def _quaternion_update_jacobian(self, q, gyro, dt):
        wx, wy, wz = gyro
        omega = np.array([[0, -wx, -wy, -wz],
                          [wx, 0, wz, -wy],
                          [wy, -wz, 0, wx],
                          [wz, wy, -wx, 0]])
        return np.eye(4) + 0.5 * dt * omega
    
    def _quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
    
    def _measurement_jacobian(self, q):
        w, x, y, z = q
        H_acc = np.zeros((3, self.state_dim))
        H_acc[0, :4] = [2*y, 2*z, 2*w, 2*x]
        H_acc[1, :4] = [-2*z, 2*y, 2*x, -2*w]
        H_acc[2, :4] = [-2*x, -2*w, 2*z, 2*y]
        return H_acc

class UnscentedKalmanFilter(OrientationFilter):
    def __init__(self, freq=30.0, process_noise=1e-5, measurement_noise=0.05, alpha=0.1, beta=2.0, kappa=0):
        super().__init__(freq)
        self.state_dim = 7
        self.meas_dim = 3
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[:4, :4] *= 0.1
        self.Q[4:, 4:] *= 5.0
        self.R = np.eye(self.meas_dim) * measurement_noise
        self.P = np.eye(self.state_dim) * 1e-2
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self._lambda = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.n_sigma = 2 * self.state_dim + 1
        self.wm = np.zeros(self.n_sigma)
        self.wc = np.zeros(self.n_sigma)
        self.wm[0] = self._lambda / (self.state_dim + self._lambda)
        self.wc[0] = self.wm[0] + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, self.n_sigma):
            self.wm[i] = 1.0 / (2 * (self.state_dim + self._lambda))
            self.wc[i] = self.wm[i]
        
        self.g_ref = np.array([0, 0, 1])
    
    def _update_impl(self, acc, gyro, dt):
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            q = self.x[:4].copy()
            bias = self.x[4:].copy()
            gyro_corrected = gyro - bias
            q_dot = 0.5 * self._quaternion_multiply(q, np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]]))
            q = q + q_dot * dt
            q = q / np.linalg.norm(q)
            self.x[:4] = q
            return q
        
        acc = acc / acc_norm
        sigma_points = self._generate_sigma_points()
        
        propagated_sigma = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            propagated_sigma[i] = self._process_model(sigma_points[i], gyro, dt)
        
        x_pred = np.zeros(self.state_dim)
        for i in range(self.n_sigma):
            x_pred += self.wm[i] * propagated_sigma[i]
        
        x_pred[:4] = x_pred[:4] / np.linalg.norm(x_pred[:4])
        
        P_pred = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.n_sigma):
            diff = propagated_sigma[i] - x_pred
            diff[:4] = self._quaternion_diff(propagated_sigma[i, :4], x_pred[:4])
            P_pred += self.wc[i] * np.outer(diff, diff)
        
        P_pred += self.Q
        
        meas_sigma = np.zeros((self.n_sigma, self.meas_dim))
        for i in range(self.n_sigma):
            meas_sigma[i] = self._measurement_model(propagated_sigma[i])
        
        y_pred = np.zeros(self.meas_dim)
        for i in range(self.n_sigma):
            y_pred += self.wm[i] * meas_sigma[i]
        
        Pyy = np.zeros((self.meas_dim, self.meas_dim))
        Pxy = np.zeros((self.state_dim, self.meas_dim))
        
        for i in range(self.n_sigma):
            diff_y = meas_sigma[i] - y_pred
            diff_x = propagated_sigma[i] - x_pred
            diff_x[:4] = self._quaternion_diff(propagated_sigma[i, :4], x_pred[:4])
            
            Pyy += self.wc[i] * np.outer(diff_y, diff_y)
            Pxy += self.wc[i] * np.outer(diff_x, diff_y)
        
        Pyy += self.R
        
        try:
            K = Pxy @ np.linalg.inv(Pyy)
            innovation = acc - y_pred
            self.x = x_pred + K @ innovation
            self.P = P_pred - K @ Pyy @ K.T
        except:
            self.x = x_pred
            self.P = P_pred
        
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        return self.x[:4]
    
    def _generate_sigma_points(self):
        sigma_points = np.zeros((self.n_sigma, self.state_dim))
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
        sigma_points[0] = self.x
        
        try:
            L = np.linalg.cholesky((self.state_dim + self._lambda) * self.P)
        except:
            U, s, Vh = np.linalg.svd(self.P)
            L = U @ np.diag(np.sqrt(s)) @ Vh * np.sqrt(self.state_dim + self._lambda)
        
        for i in range(self.state_dim):
            sigma_points[i+1] = self.x + L[i]
            sigma_points[i+1+self.state_dim] = self.x - L[i]
            
            sigma_points[i+1, :4] = sigma_points[i+1, :4] / np.linalg.norm(sigma_points[i+1, :4])
            sigma_points[i+1+self.state_dim, :4] = sigma_points[i+1+self.state_dim, :4] / np.linalg.norm(sigma_points[i+1+self.state_dim, :4])
        
        return sigma_points
    
    def _process_model(self, x, gyro, dt):
        q = x[:4].copy()
        bias = x[4:].copy()
        gyro_corrected = gyro - bias
        q_dot = 0.5 * self._quaternion_multiply(q, np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]]))
        q_new = q + q_dot * dt
        q_new = q_new / np.linalg.norm(q_new)
        
        x_new = np.zeros_like(x)
        x_new[:4] = q_new
        x_new[4:] = bias
        return x_new
    
    def _measurement_model(self, x):
        q = x[:4]
        R = self._quaternion_to_rotation_matrix(q)
        g_pred = R @ self.g_ref
        return g_pred
    
    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                         w1*x2 + x1*w2 + y1*z2 - z1*y2,
                         w1*y2 - x1*z2 + y1*w2 + z1*x2,
                         w1*z2 + x1*y2 - y1*x2 + z1*w2])
    
    def _quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                         [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                         [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
    
    def _quaternion_diff(self, q1, q2):
        q2_conj = np.array([q2[0], -q2[1], -q2[2], -q2[3]])
        q_diff = self._quaternion_multiply(q1, q2_conj)
        angle = 2 * np.arccos(np.clip(q_diff[0], -1.0, 1.0))
        
        if np.abs(angle) < 1e-10:
            return np.zeros(4)
        
        scale = 1.0 / np.sin(angle / 2)
        axis = q_diff[1:] * scale
        error = axis * angle
        return np.array([0, error[0], error[1], error[2]])

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
        
        params = params or {}
        
        if filter_type == 'madgwick':
            beta = params.get('beta', 0.15)
            filter_instance = MadgwickFilter(beta=beta)
        elif filter_type == 'kalman':
            proc_noise = params.get('process_noise', 5e-5)
            meas_noise = params.get('measurement_noise', 0.1)
            filter_instance = KalmanFilter(process_noise=proc_noise, measurement_noise=meas_noise)
        elif filter_type == 'ekf':
            proc_noise = params.get('process_noise', 1e-5)
            meas_noise = params.get('measurement_noise', 0.05)
            filter_instance = ExtendedKalmanFilter(process_noise=proc_noise, measurement_noise=meas_noise)
        elif filter_type == 'ukf':
            proc_noise = params.get('process_noise', 1e-5)
            meas_noise = params.get('measurement_noise', 0.05)
            alpha = params.get('alpha', 0.1)
            beta = params.get('beta', 2.0)
            kappa = params.get('kappa', 0)
            filter_instance = UnscentedKalmanFilter(process_noise=proc_noise, measurement_noise=meas_noise,
                                                   alpha=alpha, beta=beta, kappa=kappa)
        else:
            filter_instance = MadgwickFilter()
        
        filter_registry[filter_id] = filter_instance
        return filter_instance

def clear_filters():
    for filter_id in list(filter_registry.keys()):
        with filter_locks[filter_id]:
            if filter_id in filter_registry:
                del filter_registry[filter_id]

def align_sensor_data(acc_data, gyro_data=None, timestamps_acc=None, timestamps_gyro=None, target_freq=None):
    if gyro_data is None:
        logger.warning("Gyroscope data missing, using zeros")
        gyro_data = np.zeros_like(acc_data) if acc_data is not None else np.array([])
        timestamps_gyro = timestamps_acc
    
    if acc_data is None or len(acc_data) == 0 or gyro_data is None or len(gyro_data) == 0:
        return np.array([]), np.array([]), np.array([])
    
    if timestamps_acc is None:
        timestamps_acc = np.arange(len(acc_data))
    
    if timestamps_gyro is None:
        timestamps_gyro = np.arange(len(gyro_data))
    
    start_time = max(timestamps_acc[0], timestamps_gyro[0])
    end_time = min(timestamps_acc[-1], timestamps_gyro[-1])
    
    if end_time <= start_time:
        logger.warning("No overlapping time range between sensors")
        return np.array([]), np.array([]), np.array([])
    
    if target_freq is None:
        acc_intervals = np.diff(timestamps_acc)
        gyro_intervals = np.diff(timestamps_gyro)
        
        if len(acc_intervals) > 0 and len(gyro_intervals) > 0:
            acc_sampling = np.median(acc_intervals)
            gyro_sampling = np.median(gyro_intervals)
            target_interval = min(acc_sampling, gyro_sampling)
        else:
            target_interval = 1.0/30.0
    else:
        target_interval = 1.0/target_freq
    
    try:
        num_samples = max(5, int((end_time - start_time) / target_interval) + 1)
        common_times = np.linspace(start_time, end_time, num_samples)
    except Exception as e:
        logger.error(f"Error generating common timestamps: {e}")
        return np.array([]), np.array([]), np.array([])
    
    acc_in_range = (timestamps_acc >= start_time) & (timestamps_acc <= end_time)
    gyro_in_range = (timestamps_gyro >= start_time) & (timestamps_gyro <= end_time)
    
    if np.sum(acc_in_range) < 3 or np.sum(gyro_in_range) < 3:
        logger.warning("Insufficient data points in overlapping range")
        return np.array([]), np.array([]), np.array([])
    
    aligned_acc = np.zeros((len(common_times), 3))
    aligned_gyro = np.zeros((len(common_times), 3))
    
    try:
        for axis in range(3):
            aligned_acc[:, axis] = np.interp(common_times, timestamps_acc[acc_in_range], acc_data[acc_in_range, axis])
            aligned_gyro[:, axis] = np.interp(common_times, timestamps_gyro[gyro_in_range], gyro_data[gyro_in_range, axis])
    except Exception as e:
        logger.error(f"Error during interpolation: {e}")
        return np.array([]), np.array([]), np.array([])
    
    return aligned_acc, aligned_gyro, common_times

def create_filter_id(subject_id, action_id, trial_id=None, window_id=None, filter_type='madgwick'):
    base_id = f"S{subject_id}_A{action_id}"
    if trial_id is not None:
        base_id += f"_T{trial_id}"
    if window_id is not None:
        base_id += f"_W{window_id}"
    return f"{base_id}_{filter_type}"

def save_aligned_data(subject_id, action_id, trial_id, acc_data, gyro_data, timestamps=None, save_dir="data/aligned"):
    os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
    os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
    
    if timestamps is not None:
        os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
    
    filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
    
    np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
    np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
    
    if timestamps is not None:
        np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)

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
                if i > 0 and not np.all(quaternions[i-1] == 0):
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

def process_sequential_windows(data, window_size=64, stride=32, label=None, filter_type='madgwick', 
                             filter_params=None, base_filter_id=None, stateful=True, is_linear_acc=True):
    if 'accelerometer' not in data or data['accelerometer'] is None or len(data['accelerometer']) == 0 or \
       'gyroscope' not in data or data['gyroscope'] is None or len(data['gyroscope']) == 0:
        logger.warning("Missing accelerometer or gyroscope data")
        return {'labels': np.array([label])} if label is not None else {'labels': np.array([])}
    
    acc_data = data['accelerometer']
    gyro_data = data['gyroscope']
    timestamps = data.get('aligned_timestamps')
    
    if len(acc_data) < window_size // 2:
        logger.warning(f"Data too short: {len(acc_data)} samples < {window_size//2}")
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
        actual_gyro_length = min(actual_length, len(gyro_data) - start if start < len(gyro_data) else 0)
        
        if actual_gyro_length <= 0:
            continue
        
        gyro_window[:actual_gyro_length] = gyro_data[start:min(start + actual_gyro_length, len(gyro_data))]
        
        ts_window = None
        if timestamps is not None:
            ts_window = np.zeros(window_size)
            actual_ts_length = min(actual_length, len(timestamps) - start if start < len(timestamps) else 0)
            if actual_ts_length > 0:
                ts_window[:actual_ts_length] = timestamps[start:min(start + actual_ts_length, len(timestamps))]
        
        reset = not stateful or i == 0
        window_filter_id = f"{filter_id}_win_{i}" if not stateful else filter_id
        
        try:
            result = process_window_with_filter(
                acc_window, gyro_window, ts_window, 
                filter_type=filter_type, filter_id=window_filter_id, 
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
            logger.warning(f"Failed to process window {i}: {str(e)}")
            continue
    
    if valid_window_count == 0:
        return {'labels': np.array([label])} if label is not None else {'labels': np.array([])}
    
    for key in windows:
        if key != 'labels' and len(windows[key]) > 0:
            windows[key] = np.array(windows[key])
    
    if 'labels' in windows and isinstance(windows['labels'], list) and len(windows['labels']) > 0:
        windows['labels'] = np.array(windows['labels'])
    
    return windows

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
            acc_windows[i], gyro_windows[i], None,
            filter_type=filter_type, filter_id=filter_id,
            reset_filter=reset, is_linear_acc=is_linear_acc,
            filter_params=params
        )
        
        all_quaternions[i] = result['quaternion']
    
    return {'quaternion': all_quaternions}
