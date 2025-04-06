import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, sosfilt, find_peaks
import logging,os,time,traceback,threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("imu_fusion")
MAX_THREADS = 16
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(16)
filter_cache = {}
def resample_to_fixed_rate(data, timestamps, target_rate=30.0):
    if len(data) <= 1 or timestamps is None or len(timestamps) <= 1: return None, None
    start_time, end_time = timestamps[0], timestamps[-1]
    desired_times = np.arange(start_time, end_time, 1.0/target_rate)
    if len(desired_times) == 0: return None, None
    resampled_data = np.zeros((len(desired_times), data.shape[1]))
    for axis in range(data.shape[1]):
        try:
            interp_func = interp1d(timestamps, data[:, axis], bounds_error=False, fill_value=(data[0, axis], data[-1, axis]), kind='linear')
            resampled_data[:, axis] = interp_func(desired_times)
        except:
            if len(timestamps) > 0 and len(data) > 0:
                idx = np.argmin(np.abs(timestamps[:, np.newaxis] - desired_times), axis=0)
                resampled_data[:, axis] = data[idx, axis]
            else: return None, None
    return resampled_data, desired_times
def align_sensor_data(acc_data, gyro_data, acc_timestamps=None, gyro_timestamps=None, target_rate=30.0):
    try:
        if len(acc_data) < 3 or len(gyro_data) < 3: return None, None, None
        if acc_timestamps is None: acc_timestamps = np.arange(len(acc_data))/target_rate
        if gyro_timestamps is None: gyro_timestamps = np.arange(len(gyro_data))/target_rate
        if isinstance(acc_timestamps[0], np.datetime64):
            acc_timestamps = acc_timestamps.astype('datetime64[ns]').astype(np.int64) / 1e9
        if isinstance(gyro_timestamps[0], np.datetime64):
            gyro_timestamps = gyro_timestamps.astype('datetime64[ns]').astype(np.int64) / 1e9
        acc_indices = np.where(np.diff(acc_timestamps) > 0)[0] + 1
        acc_indices = np.concatenate([[0], acc_indices])
        acc_timestamps, acc_data = acc_timestamps[acc_indices], acc_data[acc_indices]
        gyro_indices = np.where(np.diff(gyro_timestamps) > 0)[0] + 1
        gyro_indices = np.concatenate([[0], gyro_indices])
        gyro_timestamps, gyro_data = gyro_timestamps[gyro_indices], gyro_data[gyro_indices]
        resampled_acc, acc_times = resample_to_fixed_rate(acc_data, acc_timestamps, target_rate)
        resampled_gyro, gyro_times = resample_to_fixed_rate(gyro_data, gyro_timestamps, target_rate)
        if resampled_acc is None or resampled_gyro is None: return None, None, None
        start_time = max(acc_times[0], gyro_times[0])
        end_time = min(acc_times[-1], gyro_times[-1])
        if start_time >= end_time: return None, None, None
        common_times = np.arange(start_time, end_time, 1.0/target_rate)
        if len(common_times) < 3: return None, None, None
        aligned_acc = np.zeros((len(common_times), acc_data.shape[1]))
        aligned_gyro = np.zeros((len(common_times), gyro_data.shape[1]))
        for axis in range(acc_data.shape[1]):
            try:
                acc_interp = interp1d(acc_times, resampled_acc[:, axis], bounds_error=False, kind='linear',
                                    fill_value=(resampled_acc[0, axis], resampled_acc[-1, axis]))
                aligned_acc[:, axis] = acc_interp(common_times)
            except:
                idx = np.argmin(np.abs(acc_times[:, np.newaxis] - common_times), axis=0)
                if idx.size > 0 and resampled_acc.size > 0: aligned_acc[:, axis] = resampled_acc[idx, axis]
                else: return None, None, None
        for axis in range(gyro_data.shape[1]):
            try:
                gyro_interp = interp1d(gyro_times, resampled_gyro[:, axis], bounds_error=False, kind='linear',
                                      fill_value=(resampled_gyro[0, axis], resampled_gyro[-1, axis]))
                aligned_gyro[:, axis] = gyro_interp(common_times)
            except:
                idx = np.argmin(np.abs(gyro_times[:, np.newaxis] - common_times), axis=0)
                if idx.size > 0 and resampled_gyro.size > 0: aligned_gyro[:, axis] = resampled_gyro[idx, axis]
                else: return None, None, None
        aligned_acc = apply_lowpass_filter(aligned_acc, cutoff=10.0, fs=target_rate)
        aligned_gyro = apply_lowpass_filter(aligned_gyro, cutoff=8.0, fs=target_rate)
        return aligned_acc, aligned_gyro, common_times
    except: return None, None, None
def apply_lowpass_filter(data, cutoff=8.0, fs=30.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    filtered_data = np.zeros_like(data)
    for axis in range(data.shape[1]): filtered_data[:, axis] = sosfilt(sos, data[:, axis])
    return filtered_data
def fixed_size_windows(data, window_size=128, overlap=0.5, min_windows=1, stride=64):
    if data is None or len(data) < window_size // 2: return []
    if len(data) < window_size:
        padded = np.zeros((window_size, data.shape[1]))
        padded[:len(data)] = data
        return [padded]
    start_indices = range(0, len(data) - window_size + 1, stride)
    if len(start_indices) < min_windows:
        if len(data) <= window_size: start_indices = [0]
        else: start_indices = np.linspace(0, len(data) - window_size, min_windows).astype(int).tolist()
    return [data[start:start + window_size] for start in start_indices if start + window_size <= len(data)]
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
    except: pass
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
            elif timestamp is not None: self.last_time = timestamp
            if dt <= 0 or dt > 1.0: dt = 1.0 / self.freq
            new_orientation = self._update_impl(acc, gyro, dt)
            norm = np.linalg.norm(new_orientation)
            if norm > 1e-10: self.orientation_q = new_orientation / norm
            return self.orientation_q
        except: return self.orientation_q
    def _update_impl(self, acc, gyro, dt): raise NotImplementedError()
    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
class MadgwickFilter(OrientationEstimator):
    def __init__(self, freq=30.0, beta=0.15, zeta=0.02):
        super().__init__(freq)
        self.beta = beta
        self.zeta = zeta
        self.gyro_bias = np.zeros(3)
    def _update_impl(self, acc, gyro, dt):
        q = self.orientation_q
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-6:
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
        acc_magnitude = np.linalg.norm(acc)
        adaptive_beta = self.beta
        if acc_magnitude > 1.3 * 9.81:
            adaptive_beta = min(0.3, self.beta * (acc_magnitude / 9.81))
        gyro_corrected = gyro - self.gyro_bias
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
            -q1 * gyro_corrected[0] - q2 * gyro_corrected[1] - q3 * gyro_corrected[2],
            q0 * gyro_corrected[0] + q2 * gyro_corrected[2] - q3 * gyro_corrected[1],
            q0 * gyro_corrected[1] - q1 * gyro_corrected[2] + q3 * gyro_corrected[0],
            q0 * gyro_corrected[2] + q1 * gyro_corrected[1] - q2 * gyro_corrected[0]
        ])
        qDot = qDot - adaptive_beta * grad
        q = q + qDot * dt
        if acc_magnitude < 1.1 * 9.81 and acc_magnitude > 0.9 * 9.81:
            qErr = 2.0 * np.array([
                q0 * grad[0] + q3 * grad[1] - q2 * grad[2],
                q1 * grad[0] + q2 * grad[1] + q3 * grad[2],
                -q2 * grad[0] + q1 * grad[1] - q0 * grad[2]
            ])
            self.gyro_bias += self.zeta * qErr * dt
        return q / np.linalg.norm(q)
class KalmanFilter(OrientationEstimator):
    def __init__(self, freq=30.0, process_noise=2e-5, measurement_noise=0.1):
        super().__init__(freq)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[:4, :4] *= 0.005
        self.Q[4:, 4:] *= 5.0
        self.R_base = np.eye(3) * measurement_noise
        self.R = self.R_base.copy()
        self.P = np.eye(self.state_dim) * 1e-2
        self.prev_acc = None
        self.acc_change_rate = 0
    def _update_impl(self, acc, gyro, dt):
        q = self.x[:4]
        bias = self.x[4:]
        q_norm = np.linalg.norm(q)
        if q_norm > 0: q = q / q_norm
        if self.prev_acc is not None:
            self.acc_change_rate = 0.8 * self.acc_change_rate + 0.2 * np.linalg.norm(acc - self.prev_acc) / dt
        self.prev_acc = acc.copy()
        acc_magnitude = np.linalg.norm(acc)
        noise_scale = 1.0
        if acc_magnitude > 1.3 * 9.81 or self.acc_change_rate > 10.0:
            noise_scale = min(10.0, (acc_magnitude / 9.81) * (1.0 + 0.5 * self.acc_change_rate/10.0))
        self.R = self.R_base * noise_scale
        gyro_corrected = gyro - bias
        omega = np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega)
        F = np.eye(self.state_dim)
        F[:4, :4] += 0.5 * dt * self._omega_matrix(gyro_corrected)
        F[:4, 4:] = -0.5 * dt * np.column_stack([
            np.zeros((4, 1)),
            self._quaternion_derivative_matrix(q)
        ])
        x_pred = self.x.copy()
        x_pred[:4] = q + q_dot * dt
        x_pred[4:] = bias
        q_norm = np.linalg.norm(x_pred[:4])
        if q_norm > 0: x_pred[:4] = x_pred[:4] / q_norm
        P_pred = F @ self.P @ F.T + self.Q
        acc_norm = np.linalg.norm(acc)
        if 0.5 * 9.81 < acc_norm < 1.5 * 9.81:
            acc_normalized = acc / acc_norm
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ np.array([0, 0, 1])
            y = acc_normalized - g_pred
            H = self._compute_H_matrix(x_pred[:4])
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            self.x = x_pred + K @ y
            I_KH = np.eye(self.state_dim) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else: self.x, self.P = x_pred, P_pred
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
    def _quaternion_derivative_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [-x, -y, -z],
            [w, -z, y],
            [z, w, -x],
            [-y, x, w]
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
    def __init__(self, freq=30.0, process_noise=1e-5, measurement_noise=0.05):
        super().__init__(freq)
        self.state_dim = 7
        self.x = np.zeros(self.state_dim)
        self.x[0] = 1.0
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[:4, :4] *= 0.05
        self.Q[4:, 4:] *= 5.0
        self.R_base = np.eye(3) * measurement_noise
        self.R = self.R_base.copy()
        self.P = np.eye(self.state_dim) * 1e-2
        self.g_ref = np.array([0, 0, 1])
        self.prev_acc = None
        self.acc_jerk = 0
    def _update_impl(self, acc, gyro, dt):
        try:
            q = self.x[:4]
            bias = self.x[4:]
            q_norm = np.linalg.norm(q)
            if q_norm > 0: q = q / q_norm
            acc_magnitude = np.linalg.norm(acc)
            if self.prev_acc is not None:
                self.acc_jerk = 0.7 * self.acc_jerk + 0.3 * np.linalg.norm(acc - self.prev_acc) / dt
            self.prev_acc = acc.copy()
            noise_scale = 1.0
            if acc_magnitude > 1.3 * 9.81 or self.acc_jerk > 15.0:
                noise_scale = min(15.0, (acc_magnitude / 9.81) * (1.0 + 0.3 * self.acc_jerk/10.0))
            self.R = self.R_base * noise_scale
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
            acc_valid = 0.5 * 9.81 < acc_magnitude < 1.5 * 9.81
            if self.acc_jerk > 25.0: acc_valid = False
            if acc_valid:
                acc_normalized = acc / acc_magnitude
                R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
                g_pred = R_q @ self.g_ref
                z = acc_normalized
                h = g_pred
                y = z - h
                H = self._measurement_jacobian(x_pred[:4])
                S = H @ P_pred @ H.T + self.R
                K = P_pred @ H.T @ np.linalg.inv(S)
                self.x = x_pred + K @ y
                I_KH = np.eye(self.state_dim) - K @ H
                self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
            else: self.x, self.P = x_pred, P_pred
            self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])
            return self.x[:4]
        except: return self.orientation_q
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
        if filter_type == 'madgwick': filter_instance = MadgwickFilter(beta=0.15)
        elif filter_type == 'kalman': filter_instance = KalmanFilter(process_noise=2e-5)
        elif filter_type == 'ekf': filter_instance = ExtendedKalmanFilter(process_noise=1e-5)
        else: filter_instance = MadgwickFilter(beta=0.15)
        filter_cache[cache_key] = filter_instance
    return filter_cache[cache_key]
def process_imu_data(acc_data, gyro_data, timestamps=None, filter_type='madgwick', 
                    return_features=False, trial_id=None, reset_filter=False):
    if acc_data is None or gyro_data is None or len(acc_data) < 3 or len(gyro_data) < 3:
        return None
    if trial_id is not None:
        orientation_filter = get_filter_instance(trial_id, 0, filter_type, reset=reset_filter)
    else:
        if filter_type == 'madgwick': orientation_filter = MadgwickFilter(beta=0.15)
        elif filter_type == 'kalman': orientation_filter = KalmanFilter()
        elif filter_type == 'ekf': orientation_filter = ExtendedKalmanFilter()
        else: orientation_filter = MadgwickFilter(beta=0.15)
    try:
        quaternions = []
        for i in range(len(acc_data)):
            acc = acc_data[i]
            gyro = gyro_data[i]
            timestamp = timestamps[i] if timestamps is not None else None
            if i == 0 or reset_filter: gravity_direction = np.array([0, 0, 9.81])
            else:
                last_q = quaternions[-1]
                r = Rotation.from_quat([last_q[1], last_q[2], last_q[3], last_q[0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
            acc_with_gravity = acc + gravity_direction
            norm = np.linalg.norm(acc_with_gravity)
            if norm > 1e-6: acc_with_gravity = acc_with_gravity / norm
            q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
            quaternions.append(q)
        results = {'quaternion': np.array(quaternions)}
        if return_features:
            features = extract_features_from_window(
                {'accelerometer': acc_data, 'gyroscope': gyro_data, 'quaternion': np.array(quaternions)}
            )
            results['fusion_features'] = features
        return results
    except: return None
idef extract_features_from_window(window_data):
    acc = window_data.get('accelerometer')
    gyro = window_data.get('gyroscope')
    quat = window_data.get('quaternion')
    features = []
    if acc is not None and len(acc) > 2:
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        acc_min, acc_max = np.min(acc, axis=0), np.max(acc, axis=0)
        acc_range = acc_max - acc_min
        acc_energy = np.sum(acc**2, axis=0) / len(acc)
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))
        acc_mag_mean = np.mean(acc_mag)
        acc_mag_std = np.std(acc_mag)
        acc_mag_max = np.max(acc_mag)
        acc_peaks = np.sum(acc_mag > 2.0 * 9.81) / len(acc_mag)
        acc_jerk = np.mean(np.abs(np.diff(acc, axis=0)), axis=0) if len(acc) > 3 else np.zeros(acc.shape[1])
        features.extend(acc_mean)
        features.extend(acc_std)
        features.extend(acc_range)
        features.extend(acc_energy)
        features.append(acc_mag_mean)
        features.append(acc_mag_std)
        features.append(acc_mag_max)
        features.append(acc_peaks)
        features.extend(acc_jerk)
    if gyro is not None and len(gyro) > 2:
        gyro_mean = np.mean(gyro, axis=0)
        gyro_std = np.std(gyro, axis=0)
        gyro_min, gyro_max = np.min(gyro, axis=0), np.max(gyro, axis=0)
        gyro_range = gyro_max - gyro_min
        gyro_energy = np.sum(gyro**2, axis=0) / len(gyro)
        gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))
        gyro_mag_mean = np.mean(gyro_mag)
        gyro_mag_std = np.std(gyro_mag)
        gyro_mag_max = np.max(gyro_mag)
        gyro_peaks = np.sum(gyro_mag > 2.0) / len(gyro_mag)
        gyro_jerk = np.mean(np.abs(np.diff(gyro, axis=0)), axis=0) if len(gyro) > 3 else np.zeros(gyro.shape[1])
        features.extend(gyro_mean)
        features.extend(gyro_std)
        features.extend(gyro_range)
        features.extend(gyro_energy)
        features.append(gyro_mag_mean)
        features.append(gyro_mag_std)
        features.append(gyro_mag_max)
        features.append(gyro_peaks)
        features.extend(gyro_jerk)
    if quat is not None and len(quat) > 2:
        quat_mean = np.mean(quat, axis=0)
        quat_std = np.std(quat, axis=0)
        quat_range = np.max(quat, axis=0) - np.min(quat, axis=0)
        features.extend(quat_mean)
        features.extend(quat_std)
        features.extend(quat_range)
        if len(quat) > 3:
            angular_vel = []
            for i in range(1, len(quat)):
                q1, q2 = quat[i-1], quat[i]
                try:
                    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
                    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
                    r_diff = r2 * r1.inv()
                    angle = 2 * np.arccos(np.clip(np.abs(r_diff.as_quat()[-1]), 0, 1))
                    angular_vel.append(angle)
                except:
                    angular_vel.append(0.0)
            if angular_vel:
                ang_vel_mean = np.mean(angular_vel)
                ang_vel_std = np.std(angular_vel)
                ang_vel_max = np.max(angular_vel)
                ang_vel_peaks = np.sum(np.array(angular_vel) > 0.2) / len(angular_vel)
                rot_mag = np.sum(angular_vel)
                features.append(ang_vel_mean)
                features.append(ang_vel_std)
                features.append(ang_vel_max)
                features.append(ang_vel_peaks)
                features.append(rot_mag)
    if acc is not None and gyro is not None and len(acc) > 2 and len(gyro) > 2:
        for i in range(min(acc.shape[1], gyro.shape[1])):
            try:
                corr = np.corrcoef(acc[:, i], gyro[:, i])[0, 1]
                features.append(corr if not np.isnan(corr) else 0)
            except:
                features.append(0)
        try:
            acc_mag = np.sqrt(np.sum(acc**2, axis=1))
            gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))
            acc_thresh = np.mean(acc_mag) + 2*np.std(acc_mag)
            gyro_thresh = np.mean(gyro_mag) + 2*np.std(gyro_mag)
            acc_peaks = (acc_mag > acc_thresh).astype(int)
            gyro_peaks = (gyro_mag > gyro_thresh).astype(int)
            peak_sync = np.sum(acc_peaks & gyro_peaks) / max(1, np.sum(acc_peaks | gyro_peaks))
            features.append(peak_sync)
        except:
            features.append(0)
    return np.array(features)

def process_sequence_with_filter(acc_data, gyro_data, timestamps=None, subject_id=0, action_id=0, 
                               filter_type='madgwick', filter_params=None, use_cache=False, 
                               cache_dir="processed_data", window_id=0):
    if acc_data is None or gyro_data is None or len(acc_data) < 3 or len(gyro_data) < 3:
        return None
    if filter_type == 'none':
        return np.zeros((len(acc_data), 4))
    cache_key = f"S{subject_id:02d}A{action_id:02d}W{window_id:04d}_{filter_type}"
    cache_path = os.path.join(cache_dir, f"{cache_key}.npz")
    if use_cache and os.path.exists(cache_path):
        try:
            cached_data = np.load(cache_path)
            return cached_data['quaternion']
        except:
            pass
    orientation_filter = get_filter_instance(subject_id, action_id, filter_type, filter_params)
    quaternions = np.zeros((len(acc_data), 4))
    for i in range(len(acc_data)):
        acc = acc_data[i]
        gyro = gyro_data[i]
        timestamp = timestamps[i] if timestamps is not None else None
        gravity_direction = np.array([0, 0, 9.81])
        if i > 0:
            try:
                r = Rotation.from_quat([quaternions[i-1, 1], quaternions[i-1, 2], quaternions[i-1, 3], quaternions[i-1, 0]])
                gravity_direction = r.inv().apply([0, 0, 9.81])
            except:
                pass
        acc_with_gravity = acc + gravity_direction
        norm = np.linalg.norm(acc_with_gravity)
        if norm > 1e-6:
            acc_with_gravity = acc_with_gravity / norm
        q = orientation_filter.update(acc_with_gravity, gyro, timestamp)
        quaternions[i] = q
    if use_cache:
        cache_dir_path = os.path.dirname(cache_path)
        try:
            if not os.path.exists(cache_dir_path):
                os.makedirs(cache_dir_path, exist_ok=True)
            np.savez_compressed(cache_path, 
                               quaternion=quaternions, 
                               window_id=window_id,
                               subject_id=subject_id,
                               action_id=action_id)
        except:
            pass
    return quaternions

def selective_sliding_window(data, window_size=128, stride=20, height=None, distance=None):
    if data is None or len(data) < window_size // 2:
        return []
    try:
        if data.shape[1] >= 3:
            sig_magnitude = np.sqrt(np.sum(data[:, :3]**2, axis=1))
            if height is None:
                mean_mag = np.mean(sig_magnitude)
                std_mag = np.std(sig_magnitude)
                height = mean_mag + 1.5 * std_mag
            if distance is None:
                distance = window_size // 4
            peaks, _ = find_peaks(sig_magnitude, height=height, distance=distance)
            if len(peaks) == 0:
                return fixed_size_windows(data, window_size=window_size, stride=stride)
            windows = []
            for peak in peaks:
                start = max(0, peak - window_size // 2)
                end = min(len(data), start + window_size)
                if end - start < window_size:
                    if start == 0:
                        end = min(len(data), window_size)
                    else:
                        start = max(0, end - window_size)
                if end - start == window_size:
                    windows.append(data[start:end])
            if not windows:
                return fixed_size_windows(data, window_size=window_size, stride=stride)
            return windows
        return fixed_size_windows(data, window_size=window_size, stride=stride)
    except:
        return fixed_size_windows(data, window_size=window_size, stride=stride)

def benchmark_filters(acc_data, gyro_data, timestamps=None, filters=None):
    if acc_data is None or gyro_data is None or len(acc_data) < 3 or len(gyro_data) < 3:
        return {}
    if filters is None:
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
        except:
            results[name] = {
                'quaternions': np.zeros((len(acc_data), 4)),
                'processing_time': 0.0
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
                            except:
                                continue
                    if 'accelerometer_values' in trial_data and 'gyroscope_values' in trial_data:
                        acc_data = trial_data['accelerometer_values']
                        gyro_data = trial_data['gyroscope_values']
                        acc_timestamps = trial_data.get('accelerometer_timestamps')
                        gyro_timestamps = trial_data.get('gyroscope_timestamps')
                        aligned_acc, aligned_gyro, common_timestamps = align_sensor_data(
                            acc_data, gyro_data, acc_timestamps, gyro_timestamps
                        )
                        if aligned_acc is not None and aligned_gyro is not None and len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                            windows_acc = fixed_size_windows(aligned_acc, window_size=max_length, stride=20)
                            windows_gyro = fixed_size_windows(aligned_gyro, window_size=max_length, stride=20)
                            num_windows = min(len(windows_acc), len(windows_gyro))
                            for i in range(num_windows):
                                window_acc = windows_acc[i]
                                window_gyro = windows_gyro[i]
                                filter_key = f"{subject_id}_{action_id}_{filter_type}_{i}"
                                if filter_key not in filter_cache:
                                    if filter_type == 'madgwick':
                                        filter_cache[filter_key] = MadgwickFilter(beta=0.15)
                                    elif filter_type == 'kalman':
                                        filter_cache[filter_key] = KalmanFilter(process_noise=2e-5)
                                    elif filter_type == 'ekf':
                                        filter_cache[filter_key] = ExtendedKalmanFilter(process_noise=1e-5)
                                    else:
                                        filter_cache[filter_key] = MadgwickFilter(beta=0.15)
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
                except:
                    continue
    logger.info(f"Preprocessing complete: processed {processed_count}/{total_trials} trials")
