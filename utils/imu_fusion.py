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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("imu_fusion")

MAX_THREADS = 40
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(40)
filter_cache = {}
def get_or_create_filter(subject_id, action_id, filter_type, filter_params=None):
    from utils.imu_fusion import MadgwickFilter, KalmanFilter, ExtendedKalmanFilter
    key = f"{subject_id}_{action_id}_{filter_type}"
    if key not in FILTER_INSTANCES:
        print(f"Creating new filter instance for {key}")
        if filter_type == 'madgwick':
            beta = filter_params.get('beta', 0.1) if filter_params else 0.1
            FILTER_INSTANCES[key] = MadgwickFilter(beta=beta)
        elif filter_type == 'kalman':
            process_noise = filter_params.get('process_noise', 5e-5) if filter_params else 5e-5
            measurement_noise = filter_params.get('measurement_noise', 0.1) if filter_params else 0.1
            FILTER_INSTANCES[key] = KalmanFilter(process_noise=process_noise, measurement_noise=measurement_noise)
        elif filter_type == 'ekf':
            process_noise = filter_params.get('process_noise', 1e-5) if filter_params else 1e-5
            measurement_noise = filter_params.get('measurement_noise', 0.05) if filter_params else 0.05
            FILTER_INSTANCES[key] = ExtendedKalmanFilter(process_noise=process_noise, measurement_noise=measurement_noise)
        else:
            FILTER_INSTANCES[key] = MadgwickFilter()
    return FILTER_INSTANCES[key]
def align_sensor_data(acc_data: np.ndarray, gyro_data: np.ndarray, time_tolerance: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        logger.info(f"Starting sensor alignment: acc shape={acc_data.shape}, gyro shape={gyro_data.shape}")
        if len(acc_data) == 0 or len(gyro_data) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        if hasattr(acc_data, 'iloc'):
            try:
                if isinstance(acc_data.iloc[0, 0], str):
                    acc_times = pd.to_datetime(acc_data.iloc[:, 0]).values
                else:
                    acc_times = acc_data.iloc[:, 0].values
                acc_values = acc_data.iloc[:, 1:4].values
            except Exception as e:
                logger.error(f"Invalid accelerometer data format")
                return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        else:
            acc_times = np.arange(len(acc_data))
            acc_values = acc_data
        
        if hasattr(gyro_data, 'iloc'):
            try:
                if isinstance(gyro_data.iloc[0, 0], str):
                    gyro_times = pd.to_datetime(gyro_data.iloc[:, 0]).values
                else:
                    gyro_times = gyro_data.iloc[:, 0].values
                gyro_values = gyro_data.iloc[:, 1:4].values
            except Exception as e:
                logger.error(f"Invalid gyroscope data format")
                return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        else:
            gyro_times = np.arange(len(gyro_data))
            gyro_values = gyro_data
            
        if len(acc_times) < 2 or len(gyro_times) < 2:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
            
        start_time_point = max(acc_times[0], gyro_times[0])
        
        aligned_acc = []
        aligned_gyro = []
        aligned_times = []
        
        if isinstance(acc_times[0], np.datetime64):
            tolerance_ns = np.timedelta64(int(time_tolerance * 1e9), 'ns')
        else:
            tolerance_ns = time_tolerance
            
        acc_times_np = np.array([t.astype('int64') if hasattr(t, 'astype') else t for t in acc_times])
        gyro_times_np = np.array([t.astype('int64') if hasattr(t, 'astype') else t for t in gyro_times])
        
        # Try interpolation approach for better alignment
        try:
            # Create timestamps for aligned data - use the higher frequency dataset
            if len(acc_times) >= len(gyro_times):
                aligned_times = acc_times
                # Interpolate gyro data to match acc timestamps
                gyro_interp = interp1d(gyro_times_np, gyro_values, axis=0, bounds_error=False, fill_value="extrapolate")
                aligned_gyro = gyro_interp(acc_times_np)
                aligned_acc = acc_values
            else:
                aligned_times = gyro_times
                # Interpolate acc data to match gyro timestamps
                acc_interp = interp1d(acc_times_np, acc_values, axis=0, bounds_error=False, fill_value="extrapolate")
                aligned_acc = acc_interp(gyro_times_np)
                aligned_gyro = gyro_values
                
            # Filter out any NaN values that might have been created
            valid_indices = ~(np.isnan(aligned_acc).any(axis=1) | np.isnan(aligned_gyro).any(axis=1))
            aligned_acc = aligned_acc[valid_indices]
            aligned_gyro = aligned_gyro[valid_indices]
            aligned_times = np.array(aligned_times)[valid_indices]
            
            if len(aligned_acc) > 0:
                return aligned_acc, aligned_gyro, aligned_times
        except Exception as e:
            logger.warning(f"Interpolation failed, falling back to nearest match: {str(e)}")
            
        # Fallback to nearest match approach
        for i, acc_time in enumerate(acc_times):
            time_diffs = np.abs(gyro_times - acc_time)
            closest_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_idx] <= tolerance_ns:
                try:
                    acc_val = acc_values[i] if i < len(acc_values) else np.zeros(3)
                    gyro_val = gyro_values[closest_idx] if closest_idx < len(gyro_values) else np.zeros(3)
                    
                    if len(acc_val) >= 3 and len(gyro_val) >= 3:
                        aligned_acc.append(acc_val[:3])
                        aligned_gyro.append(gyro_val[:3])
                        aligned_times.append(acc_time)
                except IndexError:
                    continue
        
        if not aligned_acc:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        return np.array(aligned_acc), np.array(aligned_gyro), np.array(aligned_times)
    except Exception as e:
        logger.error(f"Error in sensor alignment: {str(e)}")
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

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
            
            if dt <= 0 or dt > 1.0:  # Sanity check for time delta
                dt = 1.0 / self.freq
                
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
        
        # Normalize accelerometer data
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            # Create quaternion derivative from gyroscope only
            qDot = 0.5 * np.array([
                -q[1] * gyro[0] - q[2] * gyro[1] - q[3] * gyro[2],
                q[0] * gyro[0] + q[2] * gyro[2] - q[3] * gyro[1],
                q[0] * gyro[1] - q[1] * gyro[2] + q[3] * gyro[0],
                q[0] * gyro[2] + q[1] * gyro[1] - q[2] * gyro[0]
            ])
            
            # Integrate to yield quaternion
            q = q + qDot * dt
            return q / np.linalg.norm(q)
        
        acc_norm = acc / acc_norm
        
        q0, q1, q2, q3 = q
        
        # Estimate direction of gravity from current orientation
        f1 = 2.0 * (q1 * q3 - q0 * q2) - acc_norm[0]
        f2 = 2.0 * (q0 * q1 + q2 * q3) - acc_norm[1]
        f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - acc_norm[2]
        
        # Compute and normalize gradient
        J_t = np.array([
            [-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
            [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
            [0.0, -4.0*q1, -4.0*q2, 0.0]
        ])
        
        grad = J_t.T @ np.array([f1, f2, f3])
        grad_norm = np.linalg.norm(grad)
        grad = grad / grad_norm if grad_norm > 0 else grad
        
        # Compute quaternion derivative from gyroscope
        qDot = 0.5 * np.array([
            -q1 * gyro[0] - q2 * gyro[1] - q3 * gyro[2],
            q0 * gyro[0] + q2 * gyro[2] - q3 * gyro[1],
            q0 * gyro[1] - q1 * gyro[2] + q3 * gyro[0],
            q0 * gyro[2] + q1 * gyro[1] - q2 * gyro[0]
        ])
        
        # Subtract gradient component
        qDot = qDot - self.beta * grad
        
        # Integrate to yield quaternion
        q = q + qDot * dt
        q = q / np.linalg.norm(q)
        
        return q

class KalmanFilter(OrientationEstimator):
    def __init__(self, freq: float = 30.0, process_noise: float = 1e-4, measurement_noise: float = 0.1):
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
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        q = self.x[:4]
        bias = self.x[4:]
        
        # Normalize quaternion
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        
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
        if q_norm > 0:
            x_pred[:4] = x_pred[:4] / q_norm
        
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
    def __init__(self, freq: float = 30.0, process_noise: float = 1e-5, measurement_noise: float = 0.05):
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
    
    def _update_impl(self, acc: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        try:
            q = self.x[:4]
            bias = self.x[4:]
            
            # Normalize quaternion
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                q = q / q_norm
            
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

class ComplementaryFilter(OrientationEstimator):
    def __init__(self, freq: float = 30.0, alpha: float = 0.02):
        super().__init__(freq)
        self.alpha = alpha
    
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
        elif filter_type == 'comp':
            filter_instance = ComplementaryFilter()
        else:
            filter_instance = MadgwickFilter()
        
        filter_cache[cache_key] = filter_instance
    
    return filter_cache[cache_key]

def process_imu_data(acc_data: np.ndarray, gyro_data: np.ndarray, timestamps=None, 
                   filter_type='madgwick', return_features=False, trial_id=None, reset_filter=False):
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
        
        return {'quaternion': np.array(quaternions)}
        
    except Exception as e:
        logger.error(f"Error in IMU processing: {str(e)}")
        return {'quaternion': np.zeros((len(acc_data), 4))}

def extract_features_from_window(window_data):
    """Extract features from sensor data window for fusion"""
    acc = window_data.get('accelerometer')
    gyro = window_data.get('gyroscope')
    quat = window_data.get('quaternion')
    
    features = []
    
    if acc is not None and len(acc) > 0:
        # Time domain features from accelerometer
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        acc_min = np.min(acc, axis=0)
        acc_max = np.max(acc, axis=0)
        acc_energy = np.sum(acc**2, axis=0) / len(acc)
        
        features.extend(acc_mean)
        features.extend(acc_std)
        features.extend(acc_max - acc_min)
        features.extend(acc_energy)
    
    if gyro is not None and len(gyro) > 0:
        # Time domain features from gyroscope
        gyro_mean = np.mean(gyro, axis=0)
        gyro_std = np.std(gyro, axis=0)
        gyro_min = np.min(gyro, axis=0)
        gyro_max = np.max(gyro, axis=0)
        gyro_energy = np.sum(gyro**2, axis=0) / len(gyro)
        
        features.extend(gyro_mean)
        features.extend(gyro_std)
        features.extend(gyro_max - gyro_min)
        features.extend(gyro_energy)
    
    if quat is not None and len(quat) > 0:
        # Quaternion features
        quat_mean = np.mean(quat, axis=0)
        quat_std = np.std(quat, axis=0)
        quat_min = np.min(quat, axis=0)
        quat_max = np.max(quat, axis=0)
        
        features.extend(quat_mean)
        features.extend(quat_std)
        features.extend(quat_max - quat_min)
        
        # Angular velocity from quaternions
        if len(quat) > 1:
            angular_vel = []
            for i in range(1, len(quat)):
                # Compute quaternion difference
                q1 = quat[i-1]
                q2 = quat[i]
                # Convert to rotation objects
                r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
                r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
                # Get relative rotation
                r_diff = r2 * r1.inv()
                # Convert to axis-angle
                angle = 2 * np.arccos(np.clip(np.abs(r_diff.as_quat()[-1]), 0, 1))
                angular_vel.append(angle)
            
            # Angular velocity statistics
            if angular_vel:
                features.append(np.mean(angular_vel))
                features.append(np.std(angular_vel))
                features.append(np.max(angular_vel))
    
    return np.array(features)

def process_sequence_with_filter(acc_data, gyro_data, timestamps=None, subject_id=0, action_id=0, 
                              filter_type='madgwick', filter_params=None, use_cache=True, 
                              cache_dir="processed_data", window_id=0):
    cache_key = f"S{subject_id:02d}A{action_id:02d}W{window_id:04d}_{filter_type}"
    if use_cache:
        cache_path = os.path.join(cache_dir, f"{cache_key}.npz")
        if os.path.exists(cache_path):
            try:
                cached_data = np.load(cache_path)
                return cached_data['quaternion']
            except Exception:
                pass
    
    # Get or create a persistent filter instance for this subject-action pair
    # This ensures state is preserved across windows from the same sequence
    orientation_filter = get_or_create_filter(subject_id, action_id, filter_type, filter_params)
    quaternions = np.zeros((len(acc_data), 4))
    
    # Process each sample in the window, maintaining state between samples
    for i in range(len(acc_data)):
        acc = acc_data[i]
        gyro = gyro_data[i]
        timestamp = timestamps[i] if timestamps is not None else None
        
        # Use estimated gravity from previous orientation
        gravity_direction = np.array([0, 0, 9.81])
        if i > 0:
            r = Rotation.from_quat([quaternions[i-1, 1], quaternions[i-1, 2], quaternions[i-1, 3], quaternions[i-1, 0]])
            gravity_direction = r.inv().apply([0, 0, 9.81])
            
        acc_with_gravity = acc + gravity_direction
        norm = np.linalg.norm(acc_with_gravity)
        if norm > 1e-6:
            acc_with_gravity = acc_with_gravity / norm
            
        # Update filter with current sensor data, preserving state
        q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
        quaternions[i] = q
    
    # Cache the results for this window
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
def preprocess_all_subjects(subjects, filter_type, output_dir, max_length=64):
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
    
    # Count total trials for progress reporting
    total_trials = sum(1 for subject_id in subjects for trial in dataset.matched_trials if trial.subject_id == subject_id)
    
    # Global filter cache to maintain state across processing
    global FILTER_INSTANCES
    FILTER_INSTANCES = {}
    
    processed_count = 0
    with tqdm(total=total_trials, desc=f"Preprocessing all subjects ({filter_type})") as pbar:
        for subject_id in subjects:
            subject_dir = os.path.join(output_dir, f"S{subject_id:02d}")
            os.makedirs(subject_dir, exist_ok=True)
            
            # Get all trials for this subject
            subject_trials = [trial for trial in dataset.matched_trials if trial.subject_id == subject_id]
            
            for trial in subject_trials:
                processed_count += 1
                pbar.update(1)
                
                action_id = trial.action_id
                trial_id = f"S{subject_id:02d}A{action_id:02d}"
                
                trial_data = {}
                try:
                    if not ('accelerometer' in trial.files and 'gyroscope' in trial.files):
                        continue
                        
                    for modality_name, file_path in trial.files.items():
                        if modality_name in ['accelerometer', 'gyroscope']:
                            try:
                                file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
                                if file_data.shape[1] > 4:
                                    cols = file_data.shape[1] - 3
                                    file_data = file_data.iloc[:, 3:]
                                else:
                                    cols = 3
                                
                                if file_data.shape[0] > 2:
                                    data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
                                else:
                                    data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
                                
                                trial_data[modality_name] = data
                            except Exception as e:
                                logger.error(f"Error loading {modality_name} data: {str(e)}")
                                continue
                    
                    if 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                        acc_data = trial_data['accelerometer']
                        gyro_data = trial_data['gyroscope']
                        
                        # Align sensor data
                        aligned_acc, aligned_gyro, timestamps = align_sensor_data(acc_data, gyro_data)
                        
                        if len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                            # Create or get filter for this trial
                            filter_key = f"{subject_id}_{action_id}_{filter_type}"
                            if filter_key not in FILTER_INSTANCES:
                                if filter_type == 'madgwick':
                                    FILTER_INSTANCES[filter_key] = MadgwickFilter()
                                elif filter_type == 'kalman':
                                    FILTER_INSTANCES[filter_key] = KalmanFilter()
                                elif filter_type == 'ekf':
                                    FILTER_INSTANCES[filter_key] = ExtendedKalmanFilter()
                                else:
                                    FILTER_INSTANCES[filter_key] = MadgwickFilter()
                            
                            orientation_filter = FILTER_INSTANCES[filter_key]
                            
                            # Process data in windows
                            window_size = max_length
                            num_windows = (len(aligned_acc) + window_size - 1) // window_size
                            
                            all_quaternions = []
                            for i in range(num_windows):
                                start_idx = i * window_size
                                end_idx = min(start_idx + window_size, len(aligned_acc))
                                
                                window_acc = aligned_acc[start_idx:end_idx]
                                window_gyro = aligned_gyro[start_idx:end_idx]
                                window_time = timestamps[start_idx:end_idx] if timestamps is not None else None
                                
                                # Process window
                                quaternions = []
                                for j in range(len(window_acc)):
                                    acc = window_acc[j]
                                    gyro = window_gyro[j]
                                    t = window_time[j] if window_time is not None else None
                                    
                                    gravity_direction = np.array([0, 0, 9.81])
                                    if all_quaternions and len(all_quaternions) > 0:
                                        last_q = all_quaternions[-1]
                                        r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                                        gravity_direction = r.inv().apply([0, 0, 9.81])
                                    
                                    acc_with_gravity = acc + gravity_direction
                                    norm = np.linalg.norm(acc_with_gravity)
                                    if norm > 1e-6:
                                        acc_with_gravity = acc_with_gravity / norm
                                    
                                    # Use the same filter instance to maintain state
                                    q = orientation_filter.update(acc_with_gravity, gyro, t)
                                    quaternions.append(q)
                                    all_quaternions.append(q)
                                
                                # Save window data
                                window_output_file = os.path.join(subject_dir, f"{trial_id}_W{i:04d}.npz")
                                np.savez_compressed(
                                    window_output_file,
                                    accelerometer=window_acc,
                                    gyroscope=window_gyro,
                                    quaternion=np.array(quaternions),
                                    timestamps=window_time,
                                    window_id=i,
                                    filter_type=filter_type
                                )
                            
                            # Also save the complete sequence data
                            output_file = os.path.join(subject_dir, f"{trial_id}.npz")
                            np.savez_compressed(
                                output_file,
                                accelerometer=aligned_acc,
                                gyroscope=aligned_gyro,
                                quaternion=np.array(all_quaternions),
                                timestamps=timestamps,
                                filter_type=filter_type
                            )
                except Exception as e:
                    logger.error(f"Error processing trial {trial_id}: {str(e)}")
                    continue
    
    logger.info(f"Preprocessing complete: processed {processed_count}/{total_trials} trials")
