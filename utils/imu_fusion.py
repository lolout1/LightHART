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
        q = q / np.linalg.norm(q)
        
        return q

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
            
            gravity_direction = np.array([0, 0, 9.81])
            if i > 0 and len(quaternions) > 0:
                last_q = quaternions[-1]
                r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
                
            acc_with_gravity = acc + gravity_direction
            acc_with_gravity = acc_with_gravity / np.linalg.norm(acc_with_gravity)
            
            q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
            quaternions.append(q)
        
        return {'quaternion': np.array(quaternions)}
        
    except Exception as e:
        logger.error(f"Error in IMU processing: {str(e)}")
        return {'quaternion': np.zeros((len(acc_data), 4))}

def preprocess_all_subjects(subjects, filter_type, output_dir, max_length=64):
    logger.info(f"Preprocessing all subjects with {filter_type} filter")
    
    from utils.dataset import prepare_smartfallmm
    import argparse
    
    args = argparse.Namespace()
    args.dataset = 'smartfallmm'
    args.dataset_args = {
        'mode': 'sliding_window',
        'max_length': max_length,
        'task': 'fd',
        'modalities': ['accelerometer', 'gyroscope'],
        'age_group': ['young'],
        'sensors': ['watch'],
        'fusion_options': {
            'enabled': True,
            'filter_type': filter_type,
            'acc_threshold': 3.0,
            'gyro_threshold': 1.0,
            'visualize': False,
            'save_aligned': True,
            'use_cache': True,
            'cache_dir': output_dir,
        }
    }
    
    builder = prepare_smartfallmm(args)
    
    for subject_id in tqdm(subjects, desc=f"Preprocessing subjects ({filter_type})"):
        subject_dir = os.path.join(output_dir, f"S{subject_id:02d}")
        os.makedirs(subject_dir, exist_ok=True)
        
        subject_trials = [trial for trial in builder.dataset.matched_trials if trial.subject_id == subject_id]
        
        for trial in tqdm(subject_trials, desc=f"Subject {subject_id} trials", leave=False):
            action_id = trial.action_id
            trial_id = f"S{subject_id:02d}A{action_id:02d}"
            
            trial_data = {}
            try:
                for modality_name, file_path in trial.files.items():
                    file_type = file_path.split('.')[-1]
                    if file_type == 'csv':
                        try:
                            file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
                        except:
                            file_data = pd.read_csv(file_path, index_col=False, header=None, sep=';').dropna().bfill()
                        
                        if 'skeleton' in file_path:
                            cols = 96
                        else:
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
                
                if 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                    acc_data = trial_data['accelerometer']
                    gyro_data = trial_data['gyroscope']
                    
                    # Align sensor data
                    aligned_acc, aligned_gyro, timestamps = align_sensor_data(acc_data, gyro_data)
                    
                    if len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                        # Process with orientation filter
                        result = process_imu_data(
                            aligned_acc, aligned_gyro, timestamps,
                            filter_type=filter_type, 
                            trial_id=trial_id, 
                            reset_filter=True
                        )
                        
                        # Save processed data
                        output_file = os.path.join(subject_dir, f"{trial_id}.npz")
                        np.savez_compressed(
                            output_file,
                            accelerometer=aligned_acc,
                            gyroscope=aligned_gyro,
                            quaternion=result['quaternion'],
                            timestamps=timestamps,
                            filter_type=filter_type
                        )
            except Exception as e:
                logger.error(f"Error processing trial {trial_id}: {str(e)}")
                continue
    
    logger.info(f"Preprocessing complete for all subjects with {filter_type} filter")
