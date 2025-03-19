import numpy as np
from scipy.spatial.transform import Rotation
import math
import pandas as pd
import logging
import os
import time
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any
import threading
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

os.makedirs("debug_logs", exist_ok=True)
logging.basicConfig(filename=os.path.join("debug_logs", "imu_fusion.log"), level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("imu_fusion")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

MAX_THREADS = 8
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(4)

def update_thread_configuration(max_files: int, threads_per_file: int):
    global MAX_THREADS, thread_pool, file_semaphore
    new_total = max_files * threads_per_file
    if new_total != MAX_THREADS:
        thread_pool.shutdown(wait=True)
        MAX_THREADS = new_total
        thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        file_semaphore = threading.Semaphore(max_files)

def cleanup_resources():
    global thread_pool
    try:
        thread_pool.shutdown(wait=False)
        thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")

def align_sensor_data(acc_data: np.ndarray, gyro_data: np.ndarray, time_tolerance: float = 0.05):
    start_time = time.time()
    try:
        if isinstance(acc_data, pd.DataFrame):
            if isinstance(acc_data.iloc[0, 0], str):
                acc_times = pd.to_datetime(acc_data.iloc[:, 0]).values
            else:
                acc_times = acc_data.iloc[:, 0].values
            acc_values = acc_data.iloc[:, 1:4].values
        else:
            acc_values = acc_data
            acc_times = np.linspace(0, len(acc_values) / 50.0, len(acc_values))
            
        if isinstance(gyro_data, pd.DataFrame):
            if isinstance(gyro_data.iloc[0, 0], str):
                gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).values
            else:
                gyro_times = gyro_data.iloc[:, 0].values
            gyro_values = gyro_data.iloc[:, 1:4].values
        else:
            gyro_values = gyro_data
            gyro_times = np.linspace(0, len(gyro_values) / 50.0, len(gyro_values))

        if isinstance(acc_times[0], np.datetime64):
            acc_times_sec = np.array([(t - acc_times[0]).total_seconds() for t in acc_times])
            gyro_times_sec = np.array([(t - acc_times[0]).total_seconds() for t in gyro_times])
            start_time_point = max(acc_times_sec[0], gyro_times_sec[0])
            end_time_point = min(acc_times_sec[-1], gyro_times_sec[-1])
        else:
            start_time_point = max(acc_times[0], gyro_times[0])
            end_time_point = min(acc_times[-1], gyro_times[-1])
        
        if start_time_point >= end_time_point:
            return np.array([]), np.array([]), np.array([])
        
        sample_rate = 50.0
        duration = end_time_point - start_time_point
        num_samples = int(duration * sample_rate)
        
        if num_samples < 5:
            return np.array([]), np.array([]), np.array([])
        
        common_times = np.linspace(start_time_point, end_time_point, num_samples)
        aligned_acc = np.zeros((num_samples, 3))
        aligned_gyro = np.zeros((num_samples, 3))
        
        if isinstance(acc_times[0], np.datetime64):
            acc_times_sec = np.array([(t - acc_times[0]).total_seconds() for t in acc_times])
            gyro_times_sec = np.array([(t - acc_times[0]).total_seconds() for t in gyro_times])
        else:
            acc_times_sec = acc_times
            gyro_times_sec = gyro_times
        
        for axis in range(3):
            aligned_acc[:, axis] = np.interp(common_times, acc_times_sec, acc_values[:, axis])
            aligned_gyro[:, axis] = np.interp(common_times, gyro_times_sec, gyro_values[:, axis])
        
        return aligned_acc, aligned_gyro, common_times
    
    except Exception as e:
        logger.error(f"Sensor alignment failed: {str(e)}")
        return np.array([]), np.array([]), np.array([])

class OrientationFilter:
    def __init__(self, freq: float = 30.0):
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_gyro = np.zeros(3)
        self.initialized = False
        self.name = "Base OrientationFilter"
    
    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        if not self.initialized and np.linalg.norm(acc) > 0.1:
            self._initialize_from_accel(acc)
            self.initialized = True
        
        dt = 1.0 / self.freq
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            self.last_time = timestamp
        elif timestamp is not None:
            self.last_time = timestamp
            
        dt = max(0.001, min(dt, 0.1))
        
        if np.allclose(gyro, self.last_gyro, atol=1e-7) and dt < 0.01:
            return self.orientation_q
            
        self.last_gyro = np.copy(gyro)
        
        return self._update_impl(acc, gyro, dt)
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")
    
    def _initialize_from_accel(self, acc: np.ndarray) -> None:
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            return
            
        acc_normalized = acc / acc_norm
        z_axis = np.array([0, 0, 1])
        cross = np.cross(z_axis, acc_normalized)
        cross_norm = np.linalg.norm(cross)
        
        if cross_norm < 1e-10:
            if acc_normalized[2] > 0:
                self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                self.orientation_q = np.array([0.0, 1.0, 0.0, 0.0])
            return
        
        axis = cross / cross_norm
        angle = np.arccos(np.dot(z_axis, acc_normalized))
        
        self.orientation_q = np.array([
            np.cos(angle/2),
            axis[0] * np.sin(angle/2),
            axis[1] * np.sin(angle/2),
            axis[2] * np.sin(angle/2)
        ])
        
        self.orientation_q = self.orientation_q / np.linalg.norm(self.orientation_q)
    
    def reset(self) -> None:
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        self.last_gyro = np.zeros(3)
        self.initialized = False

class MadgwickFilter(OrientationFilter):
    def __init__(self, freq: float = 30.0, beta: float = 0.1):
        super().__init__(freq)
        self.beta = beta
        self.name = "Madgwick"
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        q = self.orientation_q
        
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            return q
        
        acc_norm = acc / acc_norm
        
        q0, q1, q2, q3 = q
        
        f = np.array([
            2.0 * (q1*q3 - q0*q2) - acc_norm[0],
            2.0 * (q0*q1 + q2*q3) - acc_norm[1],
            2.0 * (0.5 - q1*q1 - q2*q2) - acc_norm[2]
        ])
        
        J = np.array([
            [-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
            [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
            [0.0, -4.0*q1, -4.0*q2, 0.0]
        ])
        
        grad = J.T @ f
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
        
        qDot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])
        
        qDot = qDot - self.beta * grad
        
        q = q + qDot * dt
        
        q = q / np.linalg.norm(q)
        
        self.orientation_q = q
        
        return q

class ComplementaryFilter(OrientationFilter):
    def __init__(self, freq: float = 30.0, alpha: float = 0.02):
        super().__init__(freq)
        self.alpha = alpha
        self.name = "Complementary"
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        q = self.orientation_q
        
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            return q
        
        acc_normalized = acc / acc_norm
        
        acc_q = self._accel_to_quaternion(acc_normalized)
        
        gyro_q = self._integrate_gyro(q, gyro, dt)
        
        result_q = self._slerp(gyro_q, acc_q, self.alpha)
        
        result_q = result_q / np.linalg.norm(result_q)
        self.orientation_q = result_q
        
        return result_q
    
    def _accel_to_quaternion(self, acc: np.ndarray) -> np.ndarray:
        z_ref = np.array([0, 0, 1])
        
        rotation_axis = np.cross(z_ref, acc)
        axis_norm = np.linalg.norm(rotation_axis)
        
        if axis_norm < 1e-10:
            if acc[2] > 0:
                return np.array([1.0, 0.0, 0.0, 0.0])
            else:
                return np.array([0.0, 1.0, 0.0, 0.0])
                
        rotation_axis = rotation_axis / axis_norm
        
        angle = np.arccos(np.clip(np.dot(z_ref, acc), -1.0, 1.0))
        
        q = np.zeros(4)
        q[0] = np.cos(angle / 2)
        q[1:4] = rotation_axis * np.sin(angle / 2)
        
        return q
    
    def _integrate_gyro(self, q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        q_dot = 0.5 * np.array([
            -q[1]*gyro[0] - q[2]*gyro[1] - q[3]*gyro[2],
            q[0]*gyro[0] + q[2]*gyro[2] - q[3]*gyro[1],
            q[0]*gyro[1] - q[1]*gyro[2] + q[3]*gyro[0],
            q[0]*gyro[2] + q[1]*gyro[1] - q[2]*gyro[0]
        ])
        
        q_new = q + q_dot * dt
        
        return q_new / np.linalg.norm(q_new)
    
    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        dot = np.sum(q1 * q2)
        
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
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
    def __init__(self, freq: float = 30.0):
        super().__init__(freq)
        
        self.state = np.zeros(7)
        self.state[0] = 1.0
        
        self.P = np.eye(7) * 0.01
        self.Q = np.eye(7) * 0.001
        self.R = np.eye(3) * 0.1
        
        self.name = "Kalman"
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        q = self.state[:4]
        bias = self.state[4:7]
        
        q = q / np.linalg.norm(q)
        
        gyro_corrected = gyro - bias
        
        q_dot = 0.5 * np.array([
            -q[1]*gyro_corrected[0] - q[2]*gyro_corrected[1] - q[3]*gyro_corrected[2],
            q[0]*gyro_corrected[0] + q[2]*gyro_corrected[2] - q[3]*gyro_corrected[1],
            q[0]*gyro_corrected[1] - q[1]*gyro_corrected[2] + q[3]*gyro_corrected[0],
            q[0]*gyro_corrected[2] + q[1]*gyro_corrected[1] - q[2]*gyro_corrected[0]
        ])
        
        q_pred = q + q_dot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)
        
        state_pred = np.zeros(7)
        state_pred[:4] = q_pred
        state_pred[4:7] = bias
        
        F = np.eye(7)
        
        P_pred = F @ self.P @ F.T + self.Q
        
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 0.1 and abs(acc_norm - 9.81) < 3.0:
            acc_normalized = acc / acc_norm
            
            R_q = self._quaternion_to_rotation_matrix(q_pred)
            expected_gravity = R_q @ np.array([0, 0, 1])
            
            residual = acc_normalized - expected_gravity
            
            H = np.zeros((3, 7))
            H[:3, :4] = self._gravity_jacobian(q_pred)
            
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            self.state = state_pred + K @ residual
            
            I_KH = np.eye(7) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            self.state = state_pred
            self.P = P_pred
        
        self.state[:4] = self.state[:4] / np.linalg.norm(self.state[:4])
        
        self.orientation_q = self.state[:4]
        
        return self.orientation_q
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        yy = y * y
        yz = y * z
        yw = y * w
        zz = z * z
        zw = z * w
        
        R = np.array([
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
        ])
        
        return R
    
    def _gravity_jacobian(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        
        J = np.zeros((3, 4))
        
        J[0, 0] = -2*y
        J[0, 1] = 2*z
        J[0, 2] = -2*w
        J[0, 3] = 2*x
        
        J[1, 0] = 2*x
        J[1, 1] = 2*w
        J[1, 2] = 2*z
        J[1, 3] = 2*y
        
        J[2, 0] = 0
        J[2, 1] = -2*y
        J[2, 2] = -2*z
        J[2, 3] = 0
        
        return J

class ExtendedKalmanFilter(OrientationFilter):
    def __init__(self, freq: float = 30.0):
        super().__init__(freq)
        
        self.state_dim = 7
        
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        
        self.Q = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5])
        
        self.R_base = np.eye(3) * 0.05
        self.R = self.R_base.copy()
        
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])
        
        self.g_ref = np.array([0, 0, 1])
        
        self.acc_history = []
        self.max_history = 10
        
        self.name = "EKF"
            
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        try:
            q = self.x[:4]
            bias = self.x[4:]
            
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            
            gyro_corrected = gyro - bias
            
            acc_norm = np.linalg.norm(acc)
            self.acc_history.append(acc_norm)
            if len(self.acc_history) > self.max_history:
                self.acc_history.pop(0)
                
            if len(self.acc_history) >= 3:
                acc_var = np.var(self.acc_history)
                dynamic_factor = 1.0 + 10.0 * min(acc_var, 1.0)
                self.R = self.R_base * dynamic_factor
            
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
            
            if acc_norm > 1e-10:
                if 0.5 < acc_norm < 3.0:  
                    acc_normalized = acc / acc_norm
                    
                    R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
                    g_pred = R_q @ self.g_ref
                    
                    y = acc_normalized - g_pred
                    
                    H = self._measurement_jacobian(x_pred[:4])
                    
                    S = H @ P_pred @ H.T + self.R
                    
                    try:
                        K = P_pred @ H.T @ np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        K = P_pred @ H.T @ np.linalg.pinv(S)
                    
                    self.x = x_pred + K @ y
                    
                    I_KH = np.eye(self.state_dim) - K @ H
                    self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
                else:
                    self.x = x_pred
                    self.P = P_pred
            else:
                self.x = x_pred
                self.P = P_pred
            
            self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
            
            self.orientation_q = self.x[:4]
            
            return self.orientation_q
            
        except Exception as e:
            logger.error(f"EKF update error: {e}")
            return self.orientation_q
    
    def _quaternion_product_matrix(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
        ])
    
    def _quaternion_update_jacobian(self, q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        wx, wy, wz = gyro
        omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return np.eye(4) + 0.5 * dt * omega
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        yy = y * y
        yz = y * z
        yw = y * w
        zz = z * z
        zw = z * w
        
        return np.array([
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
        ])
    
    def _measurement_jacobian(self, q: np.ndarray) -> np.ndarray:
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
        H[:3, :4] = H_q
        
        return H

class UnscentedKalmanFilter(OrientationFilter):
    def __init__(self, freq: float = 30.0, alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0):
        super().__init__(freq)
        
        self.state_dim = 7
        self.meas_dim = 3
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.n_sigma = 2 * self.state_dim + 1
        
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        
        self.P = np.eye(self.state_dim) * 0.01
        
        self.Q = np.eye(self.state_dim) * 0.001
        self.Q[:4, :4] *= 0.0001
        self.Q[4:, 4:] *= 0.001
        
        self.R = np.eye(self.meas_dim) * 0.1
        
        self._recalculate_weights()
        
        self.g_ref = np.array([0, 0, 1])
        
        self.acc_history = []
        self.max_history = 10
        
        self.name = "UKF"
    
    def _recalculate_weights(self):
        self.lambda_ukf = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        
        self.Wm = np.zeros(self.n_sigma)
        self.Wm[0] = self.lambda_ukf / (self.state_dim + self.lambda_ukf)
        self.Wm[1:] = 1.0 / (2.0 * (self.state_dim + self.lambda_ukf))
        
        self.Wc = np.zeros(self.n_sigma)
        self.Wc[0] = self.Wm[0] + (1.0 - self.alpha**2 + self.beta)
        self.Wc[1:] = self.Wm[1:]
    
    def _generate_sigma_points(self):
        try:
            L = np.linalg.cholesky((self.state_dim + self.lambda_ukf) * self.P)
        except np.linalg.LinAlgError:
            U, S, Vh = np.linalg.svd(self.P)
            L = U @ np.diag(np.sqrt(S)) @ Vh * np.sqrt(self.state_dim + self.lambda_ukf)
        
        sigma_points = np.zeros((self.n_sigma, self.state_dim))
        
        sigma_points[0] = self.x
        
        for i in range(self.state_dim):
            sigma_points[i+1] = self.x + L[i]
            sigma_points[i+1+self.state_dim] = self.x - L[i]
        
        for i in range(self.n_sigma):
            q_norm = np.linalg.norm(sigma_points[i, :4])
            if q_norm > 0:
                sigma_points[i, :4] /= q_norm
        
        return sigma_points
    
    def _quaternion_propagate(self, q, gyro, dt):
        q_dot = 0.5 * np.array([
            -q[1]*gyro[0] - q[2]*gyro[1] - q[3]*gyro[2],
            q[0]*gyro[0] + q[2]*gyro[2] - q[3]*gyro[1],
            q[0]*gyro[1] - q[1]*gyro[2] + q[3]*gyro[0],
            q[0]*gyro[2] + q[1]*gyro[1] - q[2]*gyro[0]
        ])
        
        q_new = q + q_dot * dt
        
        return q_new / np.linalg.norm(q_new)
    
    def _process_model(self, sigma_point, gyro, dt):
        q = sigma_point[:4]
        bias = sigma_point[4:7]
        
        gyro_corrected = gyro - bias
        
        q_new = self._quaternion_propagate(q, gyro_corrected, dt)
        
        new_sigma_point = np.zeros_like(sigma_point)
        new_sigma_point[:4] = q_new
        new_sigma_point[4:7] = bias
        
        return new_sigma_point
    
    def _measurement_model(self, sigma_point):
        q = sigma_point[:4]
        
        R = self._quaternion_to_rotation_matrix(q)
        
        gravity_dir = R @ self.g_ref
        
        return gravity_dir
    
    def _quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q
        
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        yy = y * y
        yz = y * z
        yw = y * w
        zz = z * z
        zw = z * w
        
        return np.array([
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
        ])
    
    def _mean_quaternion(self, quaternions, weights):
        q_mean = quaternions[0].copy()
        
        for _ in range(3):
            e = np.zeros((len(quaternions), 3))
            for i, q in enumerate(quaternions):
                if np.dot(q, q_mean) < 0:
                    q = -q
                
                e_q = np.zeros(4)
                e_q[0] = q[0] * q_mean[0] + q[1] * q_mean[1] + q[2] * q_mean[2] + q[3] * q_mean[3]
                e_q[1] = q[0] * q_mean[1] - q[1] * q_mean[0] - q[2] * q_mean[3] + q[3] * q_mean[2]
                e_q[2] = q[0] * q_mean[2] + q[1] * q_mean[3] - q[2] * q_mean[0] - q[3] * q_mean[1]
                e_q[3] = q[0] * q_mean[3] - q[1] * q_mean[2] + q[2] * q_mean[1] - q[3] * q_mean[0]
                
                if e_q[0] > 0.9999:
                    e[i] = 2 * e_q[1:4]
                else:
                    angle = 2 * np.arccos(np.clip(e_q[0], -1.0, 1.0))
                    axis = e_q[1:4] / np.sin(angle/2)
                    e[i] = angle * axis
            
            e_mean = np.zeros(3)
            for i in range(len(quaternions)):
                e_mean += weights[i] * e[i]
            
            e_norm = np.linalg.norm(e_mean)
            if e_norm < 1e-10:
                break
            
            eq = np.zeros(4)
            eq[0] = np.cos(e_norm/2)
            eq[1:4] = np.sin(e_norm/2) * e_mean / e_norm
            
            q_new = np.zeros(4)
            q_new[0] = eq[0] * q_mean[0] - eq[1] * q_mean[1] - eq[2] * q_mean[2] - eq[3] * q_mean[3]
            q_new[1] = eq[0] * q_mean[1] + eq[1] * q_mean[0] + eq[2] * q_mean[3] - eq[3] * q_mean[2]
            q_new[2] = eq[0] * q_mean[2] - eq[1] * q_mean[3] + eq[2] * q_mean[0] + eq[3] * q_mean[1]
            q_new[3] = eq[0] * q_mean[3] + eq[1] * q_mean[2] - eq[2] * q_mean[1] + eq[3] * q_mean[0]
            
            q_mean = q_new / np.linalg.norm(q_new)
        
        return q_mean
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        try:
            acc_norm = np.linalg.norm(acc)
            self.acc_history.append(acc_norm)
            if len(self.acc_history) > self.max_history:
                self.acc_history.pop(0)
                
            if len(self.acc_history) >= 3:
                acc_var = np.var(self.acc_history)
                dynamic_factor = 1.0 + 15.0 * min(acc_var, 1.0)
                self.R = np.eye(self.meas_dim) * 0.1 * dynamic_factor
            
            if acc_norm < 1e-6:
                q = self.x[:4]
                bias = self.x[4:7]
                gyro_corrected = gyro - bias
                q_new = self._quaternion_propagate(q, gyro_corrected, dt)
                self.x[:4] = q_new
                self.orientation_q = q_new
                return q_new
            
            if not (0.5 < acc_norm < 3.0 and acc_var < 0.5):
                q = self.x[:4]
                bias = self.x[4:7]
                gyro_corrected = gyro - bias
                q_new = self._quaternion_propagate(q, gyro_corrected, dt)
                self.x[:4] = q_new
                self.orientation_q = q_new
                return q_new
                
            sigma_points = self._generate_sigma_points()
            
            sigma_points_pred = np.zeros_like(sigma_points)
            for i in range(self.n_sigma):
                sigma_points_pred[i] = self._process_model(sigma_points[i], gyro, dt)
            
            x_pred = np.zeros(self.state_dim)
            
            quaternions = sigma_points_pred[:, :4]
            x_pred[:4] = self._mean_quaternion(quaternions, self.Wm)
            
            for i in range(4, self.state_dim):
                x_pred[i] = np.sum(self.Wm * sigma_points_pred[:, i])
            
            P_pred = np.zeros((self.state_dim, self.state_dim))
            for i in range(self.n_sigma):
                error = np.zeros(self.state_dim)
                
                q = sigma_points_pred[i, :4]
                q_mean = x_pred[:4]
                
                if np.dot(q, q_mean) < 0:
                    q = -q
                
                q_mean_inv = np.array([q_mean[0], -q_mean[1], -q_mean[2], -q_mean[3]])
                q_err = np.zeros(4)
                q_err[0] = q[0]*q_mean_inv[0] - q[1]*q_mean_inv[1] - q[2]*q_mean_inv[2] - q[3]*q_mean_inv[3]
                q_err[1] = q[0]*q_mean_inv[1] + q[1]*q_mean_inv[0] + q[2]*q_mean_inv[3] - q[3]*q_mean_inv[2]
                q_err[2] = q[0]*q_mean_inv[2] - q[1]*q_mean_inv[3] + q[2]*q_mean_inv[0] + q[3]*q_mean_inv[1]
                q_err[3] = q[0]*q_mean_inv[3] + q[1]*q_mean_inv[2] - q[2]*q_mean_inv[1] + q[3]*q_mean_inv[0]
                
                if abs(q_err[0] - 1.0) < 1e-6:
                    error[:3] = np.zeros(3)
                else:
                    angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
                    axis = q_err[1:4] / np.sin(angle/2)
                    error[:3] = angle * axis
                
                error[3:] = sigma_points_pred[i, 4:] - x_pred[4:]
                
                delta = error.reshape(-1, 1)
                P_pred += self.Wc[i] * delta @ delta.T
            
            P_pred += self.Q
            
            z_pred = np.zeros((self.n_sigma, self.meas_dim))
            for i in range(self.n_sigma):
                z_pred[i] = self._measurement_model(sigma_points_pred[i])
            
            z_mean = np.zeros(self.meas_dim)
            for i in range(self.meas_dim):
                z_mean[i] = np.sum(self.Wm * z_pred[:, i])
            
            z = acc / acc_norm
            
            S = np.zeros((self.meas_dim, self.meas_dim))
            for i in range(self.n_sigma):
                delta = (z_pred[i] - z_mean).reshape(-1, 1)
                S += self.Wc[i] * delta @ delta.T
            
            S += self.R
            
            Pxz = np.zeros((self.state_dim, self.meas_dim))
            for i in range(self.n_sigma):
                error = np.zeros(self.state_dim)
                
                q = sigma_points_pred[i, :4]
                q_mean = x_pred[:4]
                
                if np.dot(q, q_mean) < 0:
                    q = -q
                
                q_mean_inv = np.array([q_mean[0], -q_mean[1], -q_mean[2], -q_mean[3]])
                q_err = np.zeros(4)
                q_err[0] = q[0]*q_mean_inv[0] - q[1]*q_mean_inv[1] - q[2]*q_mean_inv[2] - q[3]*q_mean_inv[3]
                q_err[1] = q[0]*q_mean_inv[1] + q[1]*q_mean_inv[0] + q[2]*q_mean_inv[3] - q[3]*q_mean_inv[2]
                q_err[2] = q[0]*q_mean_inv[2] - q[1]*q_mean_inv[3] + q[2]*q_mean_inv[0] + q[3]*q_mean_inv[1]
                q_err[3] = q[0]*q_mean_inv[3] + q[1]*q_mean_inv[2] - q[2]*q_mean_inv[1] + q[3]*q_mean_inv[0]
                
                if abs(q_err[0] - 1.0) < 1e-6:
                    error[:3] = np.zeros(3)
                else:
                    angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
                    axis = q_err[1:4] / np.sin(angle/2)
                    error[:3] = angle * axis
                
                error[3:] = sigma_points_pred[i, 4:] - x_pred[4:]
                
                delta_z = (z_pred[i] - z_mean).reshape(1, -1)
                Pxz += self.Wc[i] * error.reshape(-1, 1) @ delta_z
            
            try:
                K = Pxz @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = Pxz @ np.linalg.pinv(S)
            
            innovation = z - z_mean
            
            correction = K @ innovation
            
            angle = np.linalg.norm(correction[:3])
            if angle > 1e-10:
                axis = correction[:3] / angle
                dq = np.zeros(4)
                dq[0] = np.cos(angle/2)
                dq[1:4] = axis * np.sin(angle/2)
                
                q_new = np.zeros(4)
                q_new[0] = dq[0]*x_pred[0] - dq[1]*x_pred[1] - dq[2]*x_pred[2] - dq[3]*x_pred[3]
                q_new[1] = dq[0]*x_pred[1] + dq[1]*x_pred[0] + dq[2]*x_pred[3] - dq[3]*x_pred[2]
                q_new[2] = dq[0]*x_pred[2] - dq[1]*x_pred[3] + dq[2]*x_pred[0] + dq[3]*x_pred[1]
                q_new[3] = dq[0]*x_pred[3] + dq[1]*x_pred[2] - dq[2]*x_pred[1] + dq[3]*x_pred[0]
                
                self.x[:4] = q_new / np.linalg.norm(q_new)
            else:
                self.x[:4] = x_pred[:4]
            
            self.x[4:] = x_pred[4:] + correction[3:]
            
            self.P = P_pred - K @ S @ K.T
            
            self.P = 0.5 * (self.P + self.P.T)
            
            self.orientation_q = self.x[:4]
            
            return self.orientation_q
            
        except Exception as e:
            logger.error(f"UKF update error: {e}")
            return self.orientation_q

def extract_features_from_window(window_data: Dict[str, np.ndarray]) -> np.ndarray:
    try:
        quaternions = window_data.get('quaternion', np.array([]))
        acc_data = window_data.get('linear_acceleration', window_data.get('accelerometer', np.array([])))
        gyro_data = window_data.get('angular_velocity', window_data.get('gyroscope', np.array([])))

        if len(quaternions) == 0 or len(acc_data) == 0 or len(gyro_data) == 0:
            logger.warning("Insufficient data for feature extraction")
            return np.zeros(43)

        acc_mean = np.mean(acc_data, axis=0)
        acc_std = np.std(acc_data, axis=0)
        acc_max = np.max(acc_data, axis=0)
        acc_min = np.min(acc_data, axis=0)

        acc_mag = np.linalg.norm(acc_data, axis=1)
        acc_mag_mean = np.mean(acc_mag)
        acc_mag_std = np.std(acc_mag)
        acc_mag_max = np.max(acc_mag)

        gyro_mean = np.mean(gyro_data, axis=0)
        gyro_std = np.std(gyro_data, axis=0)
        gyro_max = np.max(np.abs(gyro_data), axis=0)

        jerk_features = []
        if len(acc_data) > 1:
            jerk = np.diff(acc_data, axis=0, prepend=acc_data[0].reshape(1, -1))
            jerk_mag = np.linalg.norm(jerk, axis=1)
            jerk_mag_mean = np.mean(jerk_mag)
            jerk_mag_max = np.max(jerk_mag)
            jerk_features = [jerk_mag_mean, jerk_mag_max]
        else:
            jerk_features = [0, 0]

        euler_angles = []
        for q in quaternions:
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            euler_angles.append(r.as_euler('xyz', degrees=True))

        euler_angles = np.array(euler_angles)

        euler_mean = np.mean(euler_angles, axis=0)
        euler_std = np.std(euler_angles, axis=0)

        angle_rate_features = []
        if len(euler_angles) > 1:
            angle_rates = np.diff(euler_angles, axis=0, prepend=euler_angles[0].reshape(1, -1))
            angle_rate_mag = np.linalg.norm(angle_rates, axis=1)
            angle_rate_mean = np.mean(angle_rate_mag)
            angle_rate_max = np.max(angle_rate_mag)
            angle_rate_features = [angle_rate_mean, angle_rate_max]
        else:
            angle_rate_features = [0, 0]

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

def process_imu_data(acc_data: np.ndarray, gyro_data: np.ndarray, timestamps: Optional[np.ndarray] = None,
                    filter_type: str = 'madgwick', return_features: bool = False) -> Dict[str, np.ndarray]:
    start_time = time.time()
    
    if not isinstance(acc_data, np.ndarray) or not isinstance(gyro_data, np.ndarray):
        logger.error(f"Invalid input types: acc={type(acc_data)}, gyro={type(gyro_data)}")
        return {
            'quaternion': np.zeros((1, 4)),
            'linear_acceleration': np.zeros((1, 3)) 
        }
    
    if acc_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        logger.error("Empty input data")
        return {
            'quaternion': np.zeros((1, 4)),
            'linear_acceleration': np.zeros((1, 3))
        }
    
    if acc_data.shape[0] != gyro_data.shape[0]:
        min_len = min(acc_data.shape[0], gyro_data.shape[0])
        logger.warning(f"Data length mismatch: acc={acc_data.shape[0]}, gyro={gyro_data.shape[0]}, truncating to {min_len}")
        acc_data = acc_data[:min_len]
        gyro_data = gyro_data[:min_len]
        if timestamps is not None:
            timestamps = timestamps[:min_len]
    
    try:
        if timestamps is None:
            timestamps = np.linspace(0, acc_data.shape[0] / 30.0, acc_data.shape[0])
        
        gyro_max = np.max(np.abs(gyro_data))
        if gyro_max > 20.0:
            logger.info(f"Converting gyroscope data from deg/s to rad/s (max value: {gyro_max})")
            gyro_data = gyro_data * np.pi / 180.0
        
        if filter_type == 'madgwick':
            orientation_filter = MadgwickFilter()
        elif filter_type == 'comp':
            orientation_filter = ComplementaryFilter()
        elif filter_type == 'kalman':
            orientation_filter = KalmanFilter()
        elif filter_type == 'ekf':
            orientation_filter = ExtendedKalmanFilter()
        elif filter_type == 'ukf':
            orientation_filter = UnscentedKalmanFilter()
        else:
            logger.warning(f"Unknown filter type '{filter_type}', using Madgwick")
            orientation_filter = MadgwickFilter()
            filter_type = 'madgwick'
        
        logger.info(f"Applying {filter_type} filter to IMU data ({len(acc_data)} samples)")
        
        quaternions = []
        linear_accelerations = []
        
        for i in range(len(acc_data)):
            acc = acc_data[i]
            gyro = gyro_data[i]
            timestamp = timestamps[i] if timestamps is not None else None
            
            q = orientation_filter.update(acc, gyro, timestamp)
            quaternions.append(q)
            
            if i > 0:
                R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                gravity = R.apply([0, 0, 9.81])
                lin_acc = acc - gravity
            else:
                lin_acc = acc - np.array([0, 0, 9.81])
            
            linear_accelerations.append(lin_acc)
        
        quaternions = np.array(quaternions)
        linear_accelerations = np.array(linear_accelerations)
        
        results = {
            'quaternion': quaternions,
            'linear_acceleration': linear_accelerations
        }
        
        if return_features:
            window_data = {
                'quaternion': quaternions,
                'linear_acceleration': linear_accelerations,
                'accelerometer': acc_data,
                'gyroscope': gyro_data
            }
            features = extract_features_from_window(window_data)
            results['fusion_features'] = features
        
        elapsed_time = time.time() - start_time
        logger.info(f"IMU processing with {filter_type} filter completed in {elapsed_time:.2f}s")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in IMU processing: {str(e)}")
        
        sample_size = max(1, len(acc_data) if isinstance(acc_data, np.ndarray) else 1)
        return {
            'quaternion': np.zeros((sample_size, 4)),
            'linear_acceleration': np.zeros((sample_size, 3)) 
        }

def save_aligned_sensor_data(subject_id: int, action_id: int, trial_id: int, acc_data: np.ndarray, gyro_data: np.ndarray,
                          skeleton_data: Optional[np.ndarray] = None, timestamps: Optional[np.ndarray] = None,
                          save_dir: str = "data/aligned") -> None:
    try:
        with file_semaphore:
            os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
            os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
            
            if skeleton_data is not None:
                os.makedirs(f"{save_dir}/skeleton", exist_ok=True)
            
            if timestamps is not None:
                os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
            
            filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
            
            np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
            np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
            
            if skeleton_data is not None:
                np.save(f"{save_dir}/skeleton/{filename}.npy", skeleton_data)
            
            if timestamps is not None:
                np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)
            
            logger.debug(f"Saved aligned data for {filename} to {save_dir}")
        
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")
