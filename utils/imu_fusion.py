import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Union, Optional
import pandas as pd
import time
import traceback
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import logging
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

log_dir = "debug_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "imu_fusion.log"), level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("imu_fusion")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

MAX_THREADS = 40
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(40)

def setup_gpu_environment():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            logger.info(f"Found {num_gpus} GPUs. Using GPUs 0 and 1 for processing")
            return True, [0, 1]
        elif num_gpus == 1:
            logger.info("Found 1 GPU. Using GPU 0 for processing")
            return True, [0]
        else:
            logger.warning("No GPUs found, falling back to CPU processing")
            return False, []
    else:
        logger.warning("PyTorch CUDA not available, using CPU processing")
        return False, []

USE_GPU, GPU_DEVICES = setup_gpu_environment()

def align_sensor_data(acc_data: np.ndarray, gyro_data: np.ndarray, time_tolerance: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_time = time.time()
    logger.info(f"Starting sensor alignment: acc shape={acc_data.shape}, gyro shape={gyro_data.shape}")
    
    if isinstance(acc_data, pd.DataFrame):
        if isinstance(acc_data.iloc[0, 0], str):
            acc_times = pd.to_datetime(acc_data.iloc[:, 0]).values
        else:
            acc_times = acc_data.iloc[:, 0].values
        
        if isinstance(gyro_data.iloc[0, 0], str):
            gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).values
        else:
            gyro_times = gyro_data.iloc[:, 0].values
    else:
        acc_times = acc_data[:, 0]
        gyro_times = gyro_data[:, 0]

    start_time_point = max(acc_times[0], gyro_times[0])
    
    acc_start_idx = np.searchsorted(acc_times, start_time_point)
    gyro_start_idx = np.searchsorted(gyro_times, start_time_point)
    
    if isinstance(acc_data, pd.DataFrame):
        acc_data_filtered = acc_data.iloc[acc_start_idx:].reset_index(drop=True)
        gyro_data_filtered = gyro_data.iloc[gyro_start_idx:].reset_index(drop=True)
        
        if isinstance(acc_data_filtered.iloc[0, 0], str):
            acc_times = pd.to_datetime(acc_data_filtered.iloc[:, 0]).values
        else:
            acc_times = acc_data_filtered.iloc[:, 0].values
            
        acc_data_filtered = acc_data_filtered.iloc[:, 1:4].values
        gyro_data_filtered = gyro_data_filtered.iloc[:, 1:4].values
    else:
        acc_data_filtered = acc_data[acc_start_idx:, 1:4]
        gyro_data_filtered = gyro_data[gyro_start_idx:, 1:4]
        acc_times = acc_times[acc_start_idx:]
        gyro_times = gyro_times[gyro_start_idx:]

    acc_times_np = np.array([t.astype('int64') if hasattr(t, 'astype') else t for t in acc_times])
    gyro_times_np = np.array([t.astype('int64') if hasattr(t, 'astype') else t for t in gyro_times])
    
    aligned_acc, aligned_gyro, aligned_times = [], [], []
    
    if len(acc_times) > 1000:
        def process_chunk(start_idx, end_idx):
            local_acc, local_gyro, local_times = [], [], []
            if isinstance(acc_times[0], np.datetime64):
                tolerance_ns = np.timedelta64(int(time_tolerance * 1e9), 'ns')
            else:
                tolerance_ns = time_tolerance
                
            for i in range(start_idx, end_idx):
                time_diffs = np.abs(gyro_times_np - acc_times_np[i])
                closest_idx = np.argmin(time_diffs)
                
                if time_diffs[closest_idx] <= tolerance_ns:
                    local_acc.append(acc_data_filtered[i])
                    local_gyro.append(gyro_data_filtered[closest_idx])
                    local_times.append(acc_times[i])
                    
            return local_acc, local_gyro, local_times
        
        chunk_size = max(1, len(acc_times) // MAX_THREADS)
        futures = []
        
        for start_idx in range(0, len(acc_times), chunk_size):
            end_idx = min(start_idx + chunk_size, len(acc_times))
            futures.append(thread_pool.submit(process_chunk, start_idx, end_idx))
        
        with tqdm(total=len(futures), desc="Aligning sensor data") as pbar:
            for future in as_completed(futures):
                chunk_acc, chunk_gyro, chunk_times = future.result()
                aligned_acc.extend(chunk_acc)
                aligned_gyro.extend(chunk_gyro)
                aligned_times.extend(chunk_times)
                pbar.update(1)
    else:
        if isinstance(acc_times[0], np.datetime64):
            tolerance_ns = np.timedelta64(int(time_tolerance * 1e9), 'ns')
        else:
            tolerance_ns = time_tolerance
            
        for i, acc_time in enumerate(acc_times):
            time_diffs = np.abs(gyro_times - acc_time)
            closest_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_idx] <= tolerance_ns:
                aligned_acc.append(acc_data_filtered[i])
                aligned_gyro.append(gyro_data_filtered[closest_idx])
                aligned_times.append(acc_time)

    aligned_acc = np.array(aligned_acc)
    aligned_gyro = np.array(aligned_gyro)
    aligned_times = np.array(aligned_times)

    elapsed_time = time.time() - start_time
    logger.info(f"Alignment complete: {len(aligned_acc)} matched samples in {elapsed_time:.2f}s")
    
    return aligned_acc, aligned_gyro, aligned_times

def save_aligned_sensor_data(subject_id: int, action_id: int, trial_id: int,
                          acc_data: np.ndarray, gyro_data: np.ndarray,
                          skeleton_data: Optional[np.ndarray] = None,
                          timestamps: Optional[np.ndarray] = None,
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
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")

def hybrid_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, threshold: float = 2.0, window_size: int = 5) -> np.ndarray:
    if len(x) < 2 or len(y) < 2:
        return np.full_like(x_new, y[0] if len(y) > 0 else 0.0)

    try:
        dy = np.diff(y)
        dx = np.diff(x)
        rates = np.abs(dy / np.maximum(dx, 1e-10))
        if len(rates) >= window_size:
            rates = savgol_filter(rates, window_size, 2)
        rapid_changes = rates > threshold

        if not np.any(rapid_changes):
            try:
                cs = CubicSpline(x, y)
                return cs(x_new)
            except Exception as e:
                linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
                return linear_interp(x_new)

        if np.all(rapid_changes):
            linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            return linear_interp(x_new)

        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        try:
            spline_interp = CubicSpline(x, y)
        except Exception as e:
            return linear_interp(x_new)

        y_interp = np.zeros_like(x_new, dtype=float)
        segments = []
        segment_start = None
        for i in range(len(rapid_changes)):
            if rapid_changes[i] and segment_start is None:
                segment_start = i
            elif not rapid_changes[i] and segment_start is not None:
                segments.append((segment_start, i))
                segment_start = None

        if segment_start is not None:
            segments.append((segment_start, len(rapid_changes)))

        linear_mask = np.zeros_like(x_new, dtype=bool)
        buffer = 0.05
        for start_idx, end_idx in segments:
            t_start = max(x[start_idx] - buffer, x[0])
            t_end = min(x[min(end_idx, len(x)-1)] + buffer, x[-1])
            linear_mask |= (x_new >= t_start) & (x_new <= t_end)

        if np.any(linear_mask):
            y_interp[linear_mask] = linear_interp(x_new[linear_mask])
        if np.any(~linear_mask):
            y_interp[~linear_mask] = spline_interp(x_new[~linear_mask])

        return y_interp
    except Exception as e:
        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        return linear_interp(x_new)

class OrientationEstimator:
    def __init__(self, freq: float = 30.0):
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: float = None) -> np.ndarray:
        try:
            dt = 1.0 / self.freq
            if timestamp is not None and self.last_time is not None:
                dt = timestamp - self.last_time
                self.last_time = timestamp
            elif timestamp is not None:
                self.last_time = timestamp

            new_orientation = self._update_impl(acc, gyro, dt)
            
            norm = np.linalg.norm(new_orientation)
            if norm > 1e-10:
                self.orientation_q = new_orientation / norm
            
            return self.orientation_q
        except Exception as e:
            logger.error(f"Error in orientation update: {e}")
            return self.orientation_q

    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement _update_impl")

    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None

class MadgwickFilter(OrientationEstimator):
    def __init__(self, freq: float = 30.0, beta: float = 0.1):
        super().__init__(freq)
        self.beta = beta

    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        q = self.orientation_q
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            return q
        acc_norm = acc / acc_norm
        
        q0, q1, q2, q3 = q
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

class ComplementaryFilter(OrientationEstimator):
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
        return result_q / np.linalg.norm(result_q)
    
    def _accel_to_quaternion(self, acc: np.ndarray) -> np.ndarray:
        z_ref = np.array([0, 0, 1])
        rotation_axis = np.cross(z_ref, acc)
        axis_norm = np.linalg.norm(rotation_axis)
        
        if axis_norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0]) if acc[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
        
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

class KalmanFilter(OrientationEstimator):
    def __init__(self, freq: float = 30.0):
        super().__init__(freq)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        self.Q = np.eye(self.state_dim) * 1e-4
        self.Q[:4, :4] *= 1e-6
        self.Q[4:, 4:] *= 1e-3
        self.R = np.eye(3) * 0.1
        self.P = np.eye(self.state_dim) * 1e-2
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
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
        
        x_pred = self.x.copy()
        x_pred[:4] = q + q_dot * dt
        x_pred[4:] = bias
        
        q_norm = np.linalg.norm(x_pred[:4])
        if q_norm > 0:
            x_pred[:4] = x_pred[:4] / q_norm
        
        P_pred = F @ self.P @ F.T + self.Q
        
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-10:
            acc_norm = acc / acc_norm
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ np.array([0, 0, 1])
            y = acc_norm - g_pred
            H = self._compute_H_matrix(x_pred[:4])
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            self.x = x_pred + K @ y
            self.P = (np.eye(self.state_dim) - K @ H) @ P_pred
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
        H_q[0, 0] = -2*y; H_q[0, 1] = 2*z; H_q[0, 2] = -2*w; H_q[0, 3] = 2*x
        H_q[1, 0] = 2*x; H_q[1, 1] = 2*w; H_q[1, 2] = 2*z; H_q[1, 3] = 2*y
        H_q[2, 0] = 0; H_q[2, 1] = -2*y; H_q[2, 2] = -2*z; H_q[2, 3] = 0
        
        H = np.zeros((3, self.state_dim))
        H[:, :4] = H_q
        return H

class ExtendedKalmanFilter(OrientationEstimator):
    def __init__(self, freq: float = 30.0):
        super().__init__(freq)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        self.Q = np.eye(self.state_dim) * 1e-5
        self.Q[:4, :4] *= 1e-6
        self.Q[4:, 4:] *= 1e-4
        self.R = np.eye(3) * 0.1
        self.P = np.eye(self.state_dim) * 1e-2
        self.g_ref = np.array([0, 0, 1])
        
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        try:
            q = self.x[:4]
            bias = self.x[4:]
            
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            
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
                acc_norm = acc / acc_norm
                R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
                g_pred = R_q @ self.g_ref
                z = acc_norm
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
            [0, -2*y, -2*z, 0]
        ])
        return H_acc

def apply_adaptive_filter(acc_data: np.ndarray, cutoff_freq: float = 2.0, fs: float = 30.0) -> np.ndarray:
    data_length = acc_data.shape[0]
    if data_length < 15:
        filtered_data = np.zeros_like(acc_data)
        for i in range(acc_data.shape[1]):
            if data_length > 2:
                filtered_data[:, i] = np.convolve(acc_data[:, i], np.ones(3)/3, mode='same')
            else:
                filtered_data[:, i] = acc_data[:, i]
        return filtered_data

    filtered_data = np.zeros_like(acc_data)
    padlen = min(data_length - 1, 10)
    try:
        for i in range(acc_data.shape[1]):
            b, a = butter(4, cutoff_freq / (fs/2), btype='low')
            filtered_data[:, i] = filtfilt(b, a, acc_data[:, i], padlen=padlen)
    except Exception as e:
        filtered_data = acc_data.copy()
    return filtered_data

def ensure_3d_vector(v, default_value=0.0):
    if v is None:
        return np.array([default_value, default_value, default_value])
    v_array = np.asarray(v)
    if v_array.size == 0:
        return np.array([default_value, default_value, default_value])
    if v_array.shape[-1] == 3:
        return v_array
    if v_array.ndim == 1:
        if v_array.size == 1:
            return np.array([v_array[0], default_value, default_value])
        if v_array.size == 2:
            return np.array([v_array[0], v_array[1], default_value])
        return v_array[:3]
    if v_array.ndim == 2 and v_array.shape[0] == 1:
        if v_array.shape[1] == 1:
            return np.array([v_array[0, 0], default_value, default_value])
        if v_array.shape[1] == 2:
            return np.array([v_array[0, 0], v_array[0, 1], default_value])
        return v_array[0, :3]
    return np.array([default_value, default_value, default_value])

def process_imu_batch(batch_index, acc_data, gyro_data, timestamps, filter_type, return_features):
    try:
        result = process_imu_data(acc_data, gyro_data, timestamps, filter_type, return_features)
        return batch_index, result
    except Exception as e:
        logger.error(f"Error processing batch {batch_index}: {str(e)}")
        return batch_index, None

def process_imu_data(acc_data: np.ndarray, gyro_data: np.ndarray,
                    timestamps: Optional[np.ndarray] = None,
                    filter_type: str = 'madgwick',
                    return_features: bool = False) -> Dict[str, np.ndarray]:
    start_time = time.time()
    
    if not isinstance(acc_data, np.ndarray) or not isinstance(gyro_data, np.ndarray):
        logger.error(f"Invalid input types: acc={type(acc_data)}, gyro={type(gyro_data)}")
        return {'quaternion': np.zeros((1, 4))}
    
    valid_filters = ['madgwick', 'comp', 'kalman', 'ekf', 'ukf']
    if filter_type not in valid_filters:
        logger.warning(f"Unknown filter type: {filter_type}, defaulting to 'madgwick'")
        filter_type = 'madgwick'
    
    if acc_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        logger.error("Empty input data")
        return {'quaternion': np.zeros((1, 4))}
    
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
            logger.info("Using Madgwick filter for orientation estimation")
        elif filter_type == 'comp':
            orientation_filter = ComplementaryFilter()
            logger.info("Using Complementary filter for orientation estimation")
        elif filter_type == 'kalman':
            orientation_filter = KalmanFilter()
            logger.info("Using Kalman filter for orientation estimation")
        elif filter_type == 'ekf':
            orientation_filter = ExtendedKalmanFilter()
            logger.info("Using Extended Kalman filter for orientation estimation")
        else:
            orientation_filter = MadgwickFilter()
            
        logger.info(f"Processing {len(acc_data)} samples with {filter_type} filter")
        
        quaternions = []
        
        for i in range(len(acc_data)):
            acc = ensure_3d_vector(acc_data[i])
            gyro = ensure_3d_vector(gyro_data[i])
            timestamp = timestamps[i] if timestamps is not None else None
            
            gravity_direction = np.array([0, 0, 9.81])
            if i > 0 and len(quaternions) > 0:
                last_q = quaternions[-1]
                r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
                
            acc_with_gravity = np.zeros(3)
            acc_with_gravity[:len(acc)] = acc
            acc_with_gravity = acc_with_gravity + gravity_direction
            acc_with_gravity_norm = np.linalg.norm(acc_with_gravity)
            if acc_with_gravity_norm > 1e-10:
                acc_with_gravity = acc_with_gravity / acc_with_gravity_norm
            else:
                acc_with_gravity = np.array([0, 0, 1])
            
            q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
            quaternions.append(q)
        
        quaternions = np.array(quaternions)
        
        results = {'quaternion': quaternions}
        
        if return_features:
            features = extract_features_from_window({
                'quaternion': quaternions,
                'accelerometer': acc_data,
                'gyroscope': gyro_data
            })
            results['fusion_features'] = features
        
        elapsed_time = time.time() - start_time
        logger.info(f"IMU processing with {filter_type} filter completed in {elapsed_time:.2f}s")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in IMU processing: {str(e)}")
        logger.error(traceback.format_exc())
        sample_size = max(1, len(acc_data) if isinstance(acc_data, np.ndarray) else 1)
        return {'quaternion': np.zeros((sample_size, 4))}

def extract_features_from_window(data: Dict[str, np.ndarray]) -> np.ndarray:
    try:
        acc_data = data['accelerometer']
        gyro_data = data.get('gyroscope')
        quat_data = data.get('quaternion')
        
        features = []
        
        if acc_data is None:
            logger.error("Accelerometer data required for feature extraction")
            return np.zeros(65, dtype=np.float32)
            
        window_length = len(acc_data)
        
        acc_data_3d = np.zeros((window_length, 3))
        for i in range(min(window_length, acc_data.shape[0])):
            acc_data_3d[i, :min(3, acc_data.shape[1] if len(acc_data.shape) > 1 else 1)] = ensure_3d_vector(acc_data[i])
        acc_data = acc_data_3d
        
        acc_mag = np.sqrt(np.sum(acc_data**2, axis=1))
        
        features.extend([
            np.mean(acc_mag), np.std(acc_mag), np.max(acc_mag), np.min(acc_mag),
            np.percentile(acc_mag, 25), np.percentile(acc_mag, 75), np.max(acc_mag) - np.min(acc_mag),
        ])
        
        for axis in range(3):
            axis_data = acc_data[:, axis]
            features.extend([np.mean(axis_data), np.std(axis_data), np.max(axis_data), np.min(axis_data)])
        
        if window_length >= 16:
            for axis in range(3):
                axis_data = acc_data[:, axis]
                window = np.hanning(len(axis_data))
                windowed_data = axis_data * window
                fft_data = np.abs(np.fft.rfft(windowed_data))
                
                if len(fft_data) >= 5:
                    features.extend([
                        np.sum(fft_data), np.max(fft_data), np.argmax(fft_data),
                        np.mean(fft_data), np.std(fft_data),
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
        else:
            features.extend([0] * 15)
        
        if quat_data is not None and len(quat_data) > 0:
            euler_angles = np.zeros((len(quat_data), 3))
            
            for i, q in enumerate(quat_data):
                try:
                    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                    euler_angles[i] = rot.as_euler('xyz', degrees=True)
                except Exception as e:
                    euler_angles[i] = [0, 0, 0]
            
            for axis in range(3):
                axis_data = euler_angles[:, axis]
                features.extend([
                    np.mean(axis_data), np.std(axis_data), np.max(axis_data) - np.min(axis_data),
                ])
            
            if gyro_data is not None and len(gyro_data) > 0:
                gyro_data_3d = np.zeros((len(gyro_data), 3))
                for i in range(min(len(gyro_data), gyro_data.shape[0])):
                    gyro_data_3d[i, :min(3, gyro_data.shape[1] if len(gyro_data.shape) > 1 else 1)] = ensure_3d_vector(gyro_data[i])
                
                gyro_mag = np.sqrt(np.sum(gyro_data_3d**2, axis=1))
                
                features.extend([np.mean(gyro_mag), np.max(gyro_mag), np.std(gyro_mag)])
                
                for i in range(3):
                    try:
                        if i < gyro_data_3d.shape[1]:
                            corr = np.corrcoef(euler_angles[:, i], gyro_data_3d[:, i])[0, 1]
                            features.append(corr if not np.isnan(corr) else 0)
                        else:
                            features.append(0)
                    except:
                        features.append(0)
        else:
            features.extend([0] * 9)
            if gyro_data is not None:
                features.extend([0] * 6)
        
        feature_vector = np.array(features, dtype=np.float32)
        
        expected_length = 65
        if len(feature_vector) < expected_length:
            padding = np.zeros(expected_length - len(feature_vector), dtype=np.float32)
            feature_vector = np.concatenate([feature_vector, padding])
        elif len(feature_vector) > expected_length:
            feature_vector = feature_vector[:expected_length]
        
        return feature_vector
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        logger.error(traceback.format_exc())
        return np.zeros(65, dtype=np.float32)
