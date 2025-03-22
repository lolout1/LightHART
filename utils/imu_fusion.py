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
from collections import defaultdict

log_dir = "debug_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "imu_fusion.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("imu_fusion")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

MAX_THREADS = 40
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(40)
filter_cache = {}

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

def align_sensor_data(acc_data: np.ndarray, gyro_data: np.ndarray,
                     time_tolerance: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_time = time.time()
    logger.info(f"Starting sensor alignment: acc shape={acc_data.shape}, gyro shape={gyro_data.shape}")

    if isinstance(acc_data.iloc[0, 0], str):
        logger.debug("Converting accelerometer timestamps from string to datetime")
        acc_times = pd.to_datetime(acc_data.iloc[:, 0]).values
    else:
        acc_times = acc_data.iloc[:, 0].values

    if isinstance(gyro_data.iloc[0, 0], str):
        logger.debug("Converting gyroscope timestamps from string to datetime")
        gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).values
    else:
        gyro_times = gyro_data.iloc[:, 0].values

    start_time_point = max(acc_times[0], gyro_times[0])
    logger.debug(f"Common start time: {start_time_point}")

    acc_start_idx = np.searchsorted(acc_times, start_time_point)
    gyro_start_idx = np.searchsorted(gyro_times, start_time_point)

    logger.debug(f"Trimming data: acc from {acc_start_idx}, gyro from {gyro_start_idx}")
    acc_data_filtered = acc_data.iloc[acc_start_idx:].reset_index(drop=True)
    gyro_data_filtered = gyro_data.iloc[gyro_start_idx:].reset_index(drop=True)

    if isinstance(acc_data_filtered.iloc[0, 0], str):
        acc_times = pd.to_datetime(acc_data_filtered.iloc[:, 0]).values
    else:
        acc_times = acc_data_filtered.iloc[:, 0].values

    acc_times_np = np.array([t.astype('int64') if hasattr(t, 'astype') else t for t in acc_times])
    gyro_times_np = np.array([t.astype('int64') if hasattr(t, 'astype') else t for t in gyro_times])
    
    aligned_acc = []
    aligned_gyro = []
    aligned_times = []
    
    if len(acc_times) > 1000:
        logger.debug(f"Using parallel processing for {len(acc_times)} timestamps")
        
        def process_chunk(start_idx, end_idx):
            local_acc = []
            local_gyro = []
            local_times = []
            
            if isinstance(acc_times[0], np.datetime64):
                tolerance_ns = np.timedelta64(int(time_tolerance * 1e9), 'ns')
            else:
                tolerance_ns = time_tolerance
                
            for i in range(start_idx, end_idx):
                time_diffs = np.abs(gyro_times_np - acc_times_np[i])
                closest_idx = np.argmin(time_diffs)
                
                if time_diffs[closest_idx] <= tolerance_ns:
                    local_acc.append(acc_data_filtered.iloc[i, 1:4].values)
                    local_gyro.append(gyro_data_filtered.iloc[closest_idx, 1:4].values)
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
        logger.debug(f"Using sequential processing for {len(acc_times)} timestamps")
        
        if isinstance(acc_times[0], np.datetime64):
            tolerance_ns = np.timedelta64(int(time_tolerance * 1e9), 'ns')
        else:
            tolerance_ns = time_tolerance
            
        for i, acc_time in enumerate(acc_times):
            time_diffs = np.abs(gyro_times - acc_time)
            closest_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_idx] <= tolerance_ns:
                aligned_acc.append(acc_data_filtered.iloc[i, 1:4].values)
                aligned_gyro.append(gyro_data_filtered.iloc[closest_idx, 1:4].values)
                aligned_times.append(acc_time)

    aligned_acc = np.array(aligned_acc)
    aligned_gyro = np.array(aligned_gyro)
    aligned_times = np.array(aligned_times)

    elapsed_time = time.time() - start_time
    logger.info(f"Alignment complete: {len(aligned_acc)} matched samples in {elapsed_time:.2f}s")
    logger.debug(f"Aligned data shapes: acc={aligned_acc.shape}, gyro={aligned_gyro.shape}")

    if len(aligned_acc) > 0:
        logger.debug(f"Acc min/max/mean: {np.min(aligned_acc):.3f}/{np.max(aligned_acc):.3f}/{np.mean(aligned_acc):.3f}")
        logger.debug(f"Gyro min/max/mean: {np.min(aligned_gyro):.3f}/{np.max(aligned_gyro):.3f}/{np.mean(aligned_gyro):.3f}")

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
            
            logger.debug(f"Saved aligned data for {filename} to {save_dir}")
        
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")

def hybrid_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray,
                      threshold: float = 2.0, window_size: int = 5) -> np.ndarray:
    if len(x) < 2 or len(y) < 2:
        logger.warning("Not enough points for interpolation")
        return np.full_like(x_new, y[0] if len(y) > 0 else 0.0)

    try:
        dy = np.diff(y)
        dx = np.diff(x)

        rates = np.abs(dy / np.maximum(dx, 1e-10))

        if len(rates) >= window_size:
            rates = savgol_filter(rates, window_size, 2)

        rapid_changes = rates > threshold

        if not np.any(rapid_changes):
            logger.debug("Using cubic spline interpolation for entire signal")
            try:
                cs = CubicSpline(x, y)
                return cs(x_new)
            except Exception as e:
                logger.warning(f"Cubic spline failed: {e}, falling back to linear")
                linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
                return linear_interp(x_new)

        if np.all(rapid_changes):
            logger.debug("Using linear interpolation for entire signal")
            linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            return linear_interp(x_new)

        logger.debug(f"Using hybrid interpolation: {np.sum(rapid_changes)}/{len(rapid_changes)} points have rapid changes")

        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        try:
            spline_interp = CubicSpline(x, y)
        except Exception as e:
            logger.warning(f"Cubic spline failed: {e}, using linear for all points")
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
        logger.error(f"Hybrid interpolation failed: {e}")
        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        return linear_interp(x_new)

class OrientationEstimator:
    def __init__(self, freq: float = 30.0):
        self.freq = freq
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        logger.debug(f"Initialized {self.__class__.__name__} with freq={freq}Hz")

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
            logger.error(traceback.format_exc())
            return self.orientation_q

    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement _update_impl")

    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        logger.debug(f"Reset {self.__class__.__name__} filter state")

class MadgwickFilter(OrientationEstimator):
    def __init__(self, freq: float = 30.0, beta: float = 0.1):
        super().__init__(freq)
        self.beta = beta
        logger.debug(f"Initialized MadgwickFilter with beta={beta}")

    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        q = self.orientation_q

        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            logger.warning("Zero acceleration detected, skipping orientation update")
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

        q = q / np.linalg.norm(q)

        logger.debug(f"Updated orientation: q=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
        return q

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
        
        result_q = result_q / np.linalg.norm(result_q)
        
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
        
        logger.debug(f"Initialized KalmanFilter with state_dim={self.state_dim}")
    
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
        
        logger.debug(f"Updated orientation: q=[{self.x[0]:.4f}, {self.x[1]:.4f}, "
                    f"{self.x[2]:.4f}, {self.x[3]:.4f}]")
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
        
        logger.debug(f"Initialized ExtendedKalmanFilter with state_dim={self.state_dim}")
    
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
            
            logger.debug(f"Updated orientation: q=[{self.x[0]:.4f}, {self.x[1]:.4f}, "
                         f"{self.x[2]:.4f}, {self.x[3]:.4f}]")
            return self.x[:4]
        
        except Exception as e:
            logger.error(f"EKF update error: {e}")
            logger.error(traceback.format_exc())
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

class UnscentedKalmanFilter(OrientationEstimator):
    def __init__(self, freq: float = 30.0, alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0):
        super().__init__(freq)
        
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        
        self.Q = np.eye(self.state_dim) * 1e-5
        self.Q[:4, :4] *= 1e-6
        self.Q[4:, 4:] *= 1e-4
        
        self.R = np.eye(3) * 0.1
        
        self.P = np.eye(self.state_dim) * 1e-2
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.lambda_ = self.alpha * self.alpha * (self.state_dim + self.kappa) - self.state_dim
        
        self._calculate_weights()
        
        self.g_ref = np.array([0, 0, 1])
        
        logger.debug(f"Initialized UnscentedKalmanFilter with state_dim={self.state_dim}, "
                    f"alpha={alpha}, beta={beta}, kappa={kappa}")
    
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
        
        U = np.linalg.cholesky((n + self.lambda_) * self.P)
        
        sigma_points = np.zeros((self.num_sigma_points, n))
        sigma_points[0] = self.x
        
        for i in range(n):
            sigma_points[i+1] = self.x + U[i]
            sigma_points[i+1+n] = self.x - U[i]
        
        return sigma_points
    
    def _quaternion_normalize(self, q):
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            return q / norm
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
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
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
            
            acc_norm = acc / acc_norm
            
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
            
            K = Pxz @ np.linalg.inv(Pzz)
            
            innovation = acc_norm - z_mean
            self.x = x_pred + K @ innovation
            
            self.x[:4] = self._quaternion_normalize(self.x[:4])
            
            self.P = P_pred - K @ Pzz @ K.T
            
            logger.debug(f"Updated orientation: q=[{self.x[0]:.4f}, {self.x[1]:.4f}, "
                        f"{self.x[2]:.4f}, {self.x[3]:.4f}]")
            return self.x[:4]
            
        except Exception as e:
            logger.error(f"UKF update error: {e}")
            logger.error(traceback.format_exc())
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
        
        if q_diff[0] < 0:
            q_diff = -q_diff
        
        if abs(q_diff[0]) > 0.9999:
            return np.zeros(4)
        
        return q_diff
    
    def _quaternion_inverse(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

def apply_adaptive_filter(acc_data: np.ndarray, cutoff_freq: float = 2.0, fs: float = 30.0) -> np.ndarray:
    logger.debug(f"Applying adaptive filter with cutoff={cutoff_freq}Hz, fs={fs}Hz")

    data_length = acc_data.shape[0]

    if data_length < 15:
        logger.warning(f"Input data too small for Butterworth filtering (length={data_length}), using simple smoothing")
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
        logger.error(f"Filtering failed: {str(e)}, returning original data")
        filtered_data = acc_data.copy()

    return filtered_data

def get_or_create_filter(filter_type, filter_params=None):
    global filter_cache
    cache_key = f"global_{filter_type}"
    
    if cache_key not in filter_cache:
        if filter_type == 'madgwick':
            beta = 0.15
            if filter_params and 'beta' in filter_params:
                beta = filter_params['beta']
            filter_instance = MadgwickFilter(beta=beta)
        elif filter_type == 'kalman':
            filter_instance = KalmanFilter()
            if filter_params:
                process_noise = filter_params.get('process_noise', 5e-5)
                measurement_noise = filter_params.get('measurement_noise', 0.1)
                filter_instance.Q[:4, :4] *= process_noise
                filter_instance.Q[4:, 4:] *= process_noise * 10
                filter_instance.R *= measurement_noise
        elif filter_type == 'ekf':
            filter_instance = ExtendedKalmanFilter()
            if filter_params:
                process_noise = filter_params.get('process_noise', 1e-5)
                measurement_noise = filter_params.get('measurement_noise', 0.05)
                filter_instance.Q[:4, :4] *= process_noise
                filter_instance.Q[4:, 4:] *= process_noise * 10
                filter_instance.R *= measurement_noise
        elif filter_type == 'ukf':
            alpha = 0.15
            beta = 2.0
            kappa = 1.0
            if filter_params:
                alpha = filter_params.get('alpha', alpha)
                beta = filter_params.get('beta', beta)
                kappa = filter_params.get('kappa', kappa)
            filter_instance = UnscentedKalmanFilter(alpha=alpha, beta=beta, kappa=kappa)
        else:
            filter_instance = MadgwickFilter(beta=0.15)
        filter_cache[cache_key] = filter_instance
    return filter_cache[cache_key]

def process_sequence_with_filter(acc_data, gyro_data, timestamps=None, filter_type='madgwick', 
                               filter_params=None, use_cache=True, cache_dir="processed_data", window_id=0):
    window_cache_key = f"W{window_id:04d}_{filter_type}"
    if use_cache:
        cache_path = os.path.join(cache_dir, f"{window_cache_key}.npz")
        if os.path.exists(cache_path):
            try:
                cached_data = np.load(cache_path)
                return cached_data['quaternion']
            except Exception as e:
                logger.warning(f"Error loading cached window: {e}")
    
    orientation_filter = get_or_create_filter(filter_type, filter_params)
    quaternions = np.zeros((len(acc_data), 4))
    
    for i in range(len(acc_data)):
        acc = acc_data[i]
        gyro = gyro_data[i]
        timestamp = timestamps[i] if timestamps is not None else None
        
        gravity_direction = np.array([0, 0, 9.81])
        if i > 0:
            from scipy.spatial.transform import Rotation
            last_q = quaternions[i-1]
            r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
            gravity_direction = r.inv().apply([0, 0, 9.81])
            
        acc_with_gravity = acc + gravity_direction
        norm = np.linalg.norm(acc_with_gravity)
        if norm > 1e-6:
            acc_with_gravity = acc_with_gravity / norm
            
        q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
        quaternions[i] = q
    
    if use_cache:
        os.makedirs(os.path.dirname(os.path.join(cache_dir, f"{window_cache_key}.npz")), exist_ok=True)
        np.savez_compressed(os.path.join(cache_dir, f"{window_cache_key}.npz"), quaternion=quaternions, window_id=window_id)
    
    return quaternions

def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int, peaks: Union[List[int], np.ndarray],
                           label: int, fuse: bool, filter_type: str = 'madgwick', filter_params=None, 
                           use_cache=True, cache_dir="processed_data") -> Dict[str, np.ndarray]:
    start_time = time.time()
    logger.info(f"Creating {len(peaks)} sequential windows with {filter_type} fusion (continuous filter state)")
    
    windowed_data = defaultdict(list)
    has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and len(data['gyroscope']) > 0
    has_acc = 'accelerometer' in data and data['accelerometer'] is not None and len(data['accelerometer']) > 0
    
    if not has_acc:
        logger.error("Missing accelerometer data - required for processing")
        return {'labels': np.array([label])}
    
    if fuse and not has_gyro:
        logger.warning("Fusion requested but gyroscope data not available")
        fuse = False
    
    windows_created = 0
    from tqdm import tqdm
    
    for window_idx, peak in enumerate(tqdm(peaks, desc="Processing windows sequentially")):
        start = max(0, peak - window_size // 2)
        end = min(len(data['accelerometer']), start + window_size)
        
        if end - start < window_size // 2:
            logger.debug(f"Skipping window at peak {peak}: too small ({end-start} < {window_size//2})")
            continue
        
        try:
            window_data = {}
            for modality, modality_data in data.items():
                if modality != 'labels' and modality_data is not None and len(modality_data) > 0:
                    try:
                        if modality == 'aligned_timestamps':
                            if len(modality_data.shape) == 1:
                                window_data_array = modality_data[start:min(end, len(modality_data))]
                                if len(window_data_array) < window_size:
                                    padded = np.zeros(window_size, dtype=window_data_array.dtype)
                                    padded[:len(window_data_array)] = window_data_array
                                    window_data_array = padded
                            else:
                                window_data_array = modality_data[start:min(end, len(modality_data)), :]
                                if window_data_array.shape[0] < window_size:
                                    padded = np.zeros((window_size, window_data_array.shape[1]), dtype=window_data_array.dtype)
                                    padded[:window_data_array.shape[0]] = window_data_array
                                    window_data_array = padded
                        else:
                            window_data_array = modality_data[start:min(end, len(modality_data)), :]
                            if window_data_array.shape[0] < window_size:
                                padded = np.zeros((window_size, window_data_array.shape[1]), dtype=window_data_array.dtype)
                                padded[:window_data_array.shape[0]] = window_data_array
                                window_data_array = padded
                        window_data[modality] = window_data_array
                    except Exception as e:
                        logger.error(f"Error extracting {modality} window: {str(e)}")
                        if modality == 'accelerometer': window_data[modality] = np.zeros((window_size, 3))
                        elif modality == 'gyroscope': window_data[modality] = np.zeros((window_size, 3))
                        elif modality == 'quaternion': window_data[modality] = np.zeros((window_size, 4))
                        else: window_data[modality] = None
            
            if fuse and 'accelerometer' in window_data and 'gyroscope' in window_data:
                try:
                    acc_window = window_data['accelerometer']
                    gyro_window = window_data['gyroscope']
                    timestamps = None
                    if 'aligned_timestamps' in window_data:
                        timestamps = window_data['aligned_timestamps']
                        if len(timestamps.shape) > 1:
                            timestamps = timestamps[:, 0] if timestamps.shape[1] > 0 else None
                    
                    window_data['quaternion'] = process_sequence_with_filter(
                        acc_data=acc_window, gyro_data=gyro_window, timestamps=timestamps,
                        filter_type=filter_type, filter_params=filter_params,
                        use_cache=use_cache, cache_dir=cache_dir, window_id=peak
                    )
                except Exception as e:
                    logger.error(f"Error in fusion processing: {str(e)}")
                    window_data['quaternion'] = np.zeros((window_size, 4))
            else:
                window_data['quaternion'] = np.zeros((window_size, 4))
            
            if 'quaternion' not in window_data or window_data['quaternion'] is None:
                window_data['quaternion'] = np.zeros((window_size, 4))
            elif window_data['quaternion'].shape[0] != window_size:
                temp = np.zeros((window_size, 4))
                if window_data['quaternion'].shape[0] < window_size:
                    temp[:window_data['quaternion'].shape[0]] = window_data['quaternion']
                else:
                    temp = window_data['quaternion'][:window_size]
                window_data['quaternion'] = temp
            
            for modality, modality_window in window_data.items():
                if modality_window is not None:
                    windowed_data[modality].append(modality_window)
            
            windows_created += 1
        except Exception as e:
            logger.error(f"Error processing window at peak {peak}: {str(e)}")
    
    for modality in windowed_data:
        if modality != 'labels' and len(windowed_data[modality]) > 0:
            try:
                windowed_data[modality] = np.array(windowed_data[modality])
            except Exception as e:
                logger.error(f"Error converting {modality} windows to array: {str(e)}")
                if modality == 'quaternion':
                    windowed_data[modality] = np.zeros((windows_created, window_size, 4))
    
    windowed_data['labels'] = np.repeat(label, windows_created)
    
    if fuse and ('quaternion' not in windowed_data or len(windowed_data['quaternion']) == 0):
        logger.warning("No quaternion data in final windows, adding zeros")
        if 'accelerometer' in windowed_data and len(windowed_data['accelerometer']) > 0:
            num_windows = len(windowed_data['accelerometer'])
            windowed_data['quaternion'] = np.zeros((num_windows, window_size, 4))
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processed {windows_created} windows in {elapsed_time:.2f}s with continuous filter state")
    return windowed_data

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
        return {
            'quaternion': np.zeros((1, 4))
        }
    
    if acc_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        logger.error("Empty input data")
        return {
            'quaternion': np.zeros((1, 4))
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
        
        orientation_filter = get_or_create_filter(filter_type)
        logger.info(f"Processing {len(acc_data)} samples with {filter_type} filter")
        
        quaternions = []
        
        for i in range(len(acc_data)):
            acc = acc_data[i]
            gyro = gyro_data[i]
            timestamp = timestamps[i] if timestamps is not None else None
            
            gravity_direction = np.array([0, 0, 9.81])
            if i > 0 and len(quaternions) > 0:
                last_q = quaternions[-1]
                r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
                
            acc_with_gravity = acc + gravity_direction
            acc_with_gravity = acc_with_gravity / np.linalg.norm(acc_with_gravity)
            
            q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
            quaternions.append(q)
        
        quaternions = np.array(quaternions)
        
        results = {
            'quaternion': quaternions
        }
        
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
        return {
            'quaternion': np.zeros((sample_size, 4))
        }

def extract_features_from_window(data: Dict[str, np.ndarray]) -> np.ndarray:
    try:
        acc_data = data['accelerometer']
        gyro_data = data.get('gyroscope')
        quat_data = data.get('quaternion')
        
        features = []
        
        if acc_data is None:
            logger.error("Accelerometer data required for feature extraction")
            return np.zeros(32)
            
        window_length = len(acc_data)
        
        acc_mag = np.sqrt(np.sum(acc_data**2, axis=1))
        
        features.extend([
            np.mean(acc_mag),
            np.std(acc_mag),
            np.max(acc_mag),
            np.min(acc_mag),
            np.percentile(acc_mag, 25),
            np.percentile(acc_mag, 75),
            np.max(acc_mag) - np.min(acc_mag),
        ])
        
        for axis in range(3):
            axis_data = acc_data[:, axis]
            features.extend([
                np.mean(axis_data),
                np.std(axis_data),
                np.max(axis_data),
                np.min(axis_data),
            ])
        
        if window_length >= 16:
            for axis in range(3):
                axis_data = acc_data[:, axis]
                
                window = np.hanning(len(axis_data))
                windowed_data = axis_data * window
                
                fft_data = np.abs(np.fft.rfft(windowed_data))
                
                if len(fft_data) >= 5:
                    features.extend([
                        np.sum(fft_data),
                        np.max(fft_data),
                        np.argmax(fft_data),
                        np.mean(fft_data),
                        np.std(fft_data),
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
                    logger.warning(f"Error converting quaternion to Euler angles: {e}")
                    euler_angles[i] = [0, 0, 0]
            
            for axis in range(3):
                axis_data = euler_angles[:, axis]
                features.extend([
                    np.mean(axis_data),
                    np.std(axis_data),
                    np.max(axis_data) - np.min(axis_data),
                ])
        else:
            features.extend([0] * 9)
        
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
