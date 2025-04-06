import numpy as np
from scipy.spatial.transform import Rotation
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("imu_fusion")
thread_pool = ThreadPoolExecutor(max_workers=4)

def save_aligned_sensor_data(subject_id, action_id, trial_id, acc_data, gyro_data, quaternions=None, timestamps=None, save_dir="data/aligned"):
    try:
        os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
        os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
        if quaternions is not None: os.makedirs(f"{save_dir}/quaternion", exist_ok=True)
        filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
        np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
        np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
        if quaternions is not None: np.save(f"{save_dir}/quaternion/{filename}.npy", quaternions)
        if timestamps is not None:
            os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
            np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)
        logger.info(f"Saved aligned data for {filename}")
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")

def bandpass_filter(data, lowcut=0.5, highcut=15.0, fs=30.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(0.001, min(0.999, low))
    high = max(low + 0.001, min(0.999, high))
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    return filtered

def align_sensor_data(acc_df, gyro_df, target_freq=30.0):
    logger.info(f"Starting sensor alignment with target frequency {target_freq}Hz")
    start_time = time.time()
    try:
        acc_timestamps = acc_df['timestamp'].values
        gyro_timestamps = gyro_df['timestamp'].values
        acc_data = acc_df.iloc[:, 1:4].values
        gyro_data = gyro_df.iloc[:, 1:4].values
        if len(acc_timestamps) < 3 or len(gyro_timestamps) < 3:
            logger.warning("Insufficient data points for alignment")
            return None, None, None
        start_time_point = max(acc_timestamps[0], gyro_timestamps[0])
        end_time_point = min(acc_timestamps[-1], gyro_timestamps[-1])
        if start_time_point >= end_time_point:
            logger.warning("No temporal overlap between sensors")
            return None, None, None
        duration_sec = (end_time_point - start_time_point)
        if duration_sec <= 0:
            logger.warning(f"Invalid duration: {duration_sec} seconds")
            return None, None, None
        n_samples = max(10, int(duration_sec * target_freq))
        common_timestamps = np.linspace(start_time_point, end_time_point, n_samples)
        aligned_acc = np.zeros((n_samples, 3))
        aligned_gyro = np.zeros((n_samples, 3))
        for axis in range(3):
            acc_interp = interp1d(acc_timestamps, acc_data[:, axis], bounds_error=False, fill_value="extrapolate")
            aligned_acc[:, axis] = acc_interp(common_timestamps)
            gyro_interp = interp1d(gyro_timestamps, gyro_data[:, axis], bounds_error=False, fill_value="extrapolate")
            aligned_gyro[:, axis] = gyro_interp(common_timestamps)
        safe_freq = max(5.0, min(1000.0, target_freq))
        aligned_acc = bandpass_filter(aligned_acc, lowcut=0.1, highcut=min(safe_freq/2.1, 15.0), fs=safe_freq)
        aligned_gyro = bandpass_filter(aligned_gyro, lowcut=0.1, highcut=min(safe_freq/2.1, 12.0), fs=safe_freq)
        elapsed = time.time() - start_time
        logger.info(f"Sensor alignment complete: {n_samples} aligned samples in {elapsed:.2f}s")
        return aligned_acc, aligned_gyro, common_timestamps
    except Exception as e:
        logger.error(f"Error during sensor alignment: {str(e)}")
        return None, None, None

def hybrid_interpolate(time1, data1, time2, data2, target_time=None, method='linear'):
    if target_time is None: target_time = time1
    try:
        f1 = interp1d(time1, data1, kind=method, axis=0, bounds_error=False, fill_value="extrapolate")
        interp_data1 = f1(target_time)
    except:
        f1 = interp1d(time1, data1, kind='linear', axis=0, bounds_error=False, fill_value="extrapolate")
        interp_data1 = f1(target_time)
    try:
        f2 = interp1d(time2, data2, kind=method, axis=0, bounds_error=False, fill_value="extrapolate")
        interp_data2 = f2(target_time)
    except:
        f2 = interp1d(time2, data2, kind='linear', axis=0, bounds_error=False, fill_value="extrapolate")
        interp_data2 = f2(target_time)
    return interp_data1, interp_data2

def cleanup_resources():
    global thread_pool
    if 'thread_pool' in globals() and thread_pool is not None:
        thread_pool.shutdown(wait=True)

def update_thread_configuration(max_workers=None):
    global thread_pool
    if 'thread_pool' in globals() and thread_pool is not None:
        thread_pool.shutdown(wait=True)
    if max_workers is None:
        import os
        max_workers = min(os.cpu_count(), 8)
    thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"Thread pool reconfigured with {max_workers} workers")

# --- IMU Filter Implementations (only madgwick, kalman, and ekf) ---

class MadgwickFilter:
    def __init__(self, beta=0.1, sample_rate=30.0):
        self.beta = beta
        self.sample_rate = sample_rate
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    def update(self, acc, gyro, dt=None):
        if dt is None: dt = 1.0 / self.sample_rate
        q = self.quaternion
        if np.linalg.norm(acc) < 1e-10:
            acc_norm = np.array([0, 0, 1])
        else:
            acc_norm = acc / np.linalg.norm(acc)
        q0, q1, q2, q3 = q
        f = np.array([
            2*(q1*q3 - q0*q2) - acc_norm[0],
            2*(q0*q1 + q2*q3) - acc_norm[1],
            2*(0.5 - q1**2 - q2**2) - acc_norm[2]
        ])
        J = np.array([
            [-2*q2, 2*q3, -2*q0, 2*q1],
            [2*q1, 2*q0, 2*q3, 2*q2],
            [0, -4*q1, -4*q2, 0]
        ])
        gradient = J.T @ f
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 0:
            gradient = gradient / gradient_norm
        qDot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])
        qDot = qDot - self.beta * gradient
        q = q + qDot * dt
        q = q / np.linalg.norm(q)
        self.quaternion = q
        return q
    def reset(self):
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

class KalmanFilter:
    def __init__(self, sample_rate=30.0):
        self.sample_rate = sample_rate
        self.state = np.zeros(7)
        self.state[0] = 1.0
        self.P = np.diag([1e-2]*4 + [1e-3]*3)
        self.Q = np.diag([1e-5]*4 + [1e-4]*3)
        self.R = np.eye(3) * 0.1
    def update(self, acc, gyro, dt=None):
        if dt is None: dt = 1.0 / self.sample_rate
        q = self.state[:4]
        bias = self.state[4:]
        q = q / np.linalg.norm(q)
        gyro_corrected = gyro - bias
        q_dot = 0.5 * self._quaternion_multiply(q, np.array([0, *gyro_corrected]))
        F = np.eye(7)
        F[:4, :4] += dt * 0.5 * self._omega_matrix(gyro_corrected)
        x_pred = np.zeros(7)
        x_pred[:4] = q + q_dot * dt
        x_pred[4:] = bias
        x_pred[:4] = x_pred[:4] / np.linalg.norm(x_pred[:4])
        P_pred = F @ self.P @ F.T + self.Q
        acc_norm = np.linalg.norm(acc)
        if 0.5 < acc_norm < 1.5:
            acc_unit = acc / acc_norm
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ np.array([0, 0, 1])
            y = acc_unit - g_pred
            H = self._compute_H_matrix(x_pred[:4])
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            self.state = x_pred + K @ y
            self.P = (np.eye(7) - K @ H) @ P_pred
        else:
            self.state = x_pred
            self.P = P_pred
        self.state[:4] = self.state[:4] / np.linalg.norm(self.state[:4])
        return self.state[:4]
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
            [1 - 2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
    def _compute_H_matrix(self, q):
        w, x, y, z = q
        H_q = np.zeros((3, 4))
        H_q[0, :] = [-2*y, 2*z, -2*w, 2*x]
        H_q[1, :] = [2*x, 2*w, 2*z, 2*y]
        H_q[2, :] = [0, -2*y, -2*z, 0]
        H = np.zeros((3, 7))
        H[:, :4] = H_q
        return H
    def reset(self):
        self.state = np.zeros(7)
        self.state[0] = 1.0
        self.P = np.diag([1e-2]*4 + [1e-3]*3)

class ExtendedKalmanFilter:
    def __init__(self, sample_rate=30.0):
        self.sample_rate = sample_rate
        self.state = np.zeros(7)
        self.state[0] = 1.0
        self.P = np.diag([1e-2]*4 + [1e-4]*3)
        self.Q = np.diag([1e-6]*4 + [1e-5]*3)
        self.R_base = np.eye(3) * 0.05
        self.R = self.R_base.copy()
        self.g_ref = np.array([0, 0, 1])
        self.acc_history = []
        self.max_history = 10
    def update(self, acc, gyro, dt=None):
        if dt is None: dt = 1.0/self.sample_rate
        q = self.state[:4]
        bias = self.state[4:]
        q = q/np.linalg.norm(q)
        gyro_corrected = gyro - bias
        acc_norm = np.linalg.norm(acc)
        self.acc_history.append(acc_norm)
        if len(self.acc_history) > self.max_history:
            self.acc_history.pop(0)
        if len(self.acc_history) >= 3:
            acc_var = np.var(self.acc_history)
            dynamic_factor = 1.0 + 10.0 * min(acc_var, 1.0)
            self.R = self.R_base * dynamic_factor
        q_dot = 0.5 * self._quaternion_product_matrix(q) @ np.array([0, *gyro_corrected])
        q_pred = q + q_dot * dt
        q_pred = q_pred/np.linalg.norm(q_pred)
        x_pred = np.zeros(7)
        x_pred[:4] = q_pred
        x_pred[4:] = bias
        F = np.eye(7)
        F[:4, :4] = self._quaternion_update_jacobian(q, gyro_corrected, dt)
        F[:4, 4:] = -0.5*dt*self._quaternion_product_matrix(q)[:, 1:]
        P_pred = F @ self.P @ F.T + self.Q
        if 0.5 < acc_norm < 3.0:
            acc_normalized = acc/acc_norm
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ self.g_ref
            y = acc_normalized - g_pred
            H = self._measurement_jacobian(x_pred[:4])
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            self.state = x_pred + K @ y
            I_KH = np.eye(7) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            self.state = x_pred
            self.P = P_pred
        self.state[:4] = self.state[:4]/np.linalg.norm(self.state[:4])
        return self.state[:4]
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
        xx, xy, xz, xw = x*x, x*y, x*z, x*w
        yy, yz, yw = y*y, y*z, y*w
        zz, zw = z*z, z*w
        return np.array([
            [1-2*(yy+zz), 2*(xy-zw), 2*(xz+yw)],
            [2*(xy+zw), 1-2*(xx+zz), 2*(yz-xw)],
            [2*(xz-yw), 2*(yz+xw), 1-2*(xx+yy)]
        ])
    def _measurement_jacobian(self, q):
        w, x, y, z = q
        H_q = np.zeros((3, 4))
        H_q[0, :] = [-2*y, 2*z, -2*w, 2*x]
        H_q[1, :] = [2*x, 2*w, 2*z, 2*y]
        H_q[2, :] = [0, -2*y, -2*z, 0]
        H = np.zeros((3, 7))
        H[:, :4] = H_q
        return H
    def reset(self):
        self.state = np.zeros(7)
        self.state[0] = 1.0
        self.P = np.diag([1e-2]*4 + [1e-4]*3)
        self.acc_history = []

def extract_features_from_window(window_data):
    quaternions = window_data.get('quaternion', np.array([]))
    acc_data = window_data.get('linear_acceleration', window_data.get('accelerometer', np.array([])))
    gyro_data = window_data.get('gyroscope', np.array([]))
    if len(quaternions) == 0 or len(acc_data) == 0 or len(gyro_data) == 0:
        logger.warning("Missing data for feature extraction")
        return np.zeros(43)
    try:
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
        if len(acc_data) > 1:
            jerk = np.diff(acc_data, axis=0)
            jerk_mag = np.linalg.norm(jerk, axis=1)
            jerk_features = [np.mean(jerk_mag), np.max(jerk_mag)]
        else:
            jerk_features = [0, 0]
        euler_angles = []
        for q in quaternions:
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            euler_angles.append(r.as_euler('xyz', degrees=True))
        euler_angles = np.array(euler_angles)
        euler_mean = np.mean(euler_angles, axis=0)
        euler_std = np.std(euler_angles, axis=0)
        if len(euler_angles) > 1:
            angle_rates = np.diff(euler_angles, axis=0)
            angle_rate_features = [np.mean(np.linalg.norm(angle_rates, axis=1)), np.max(np.linalg.norm(angle_rates, axis=1))]
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

def process_imu_data(acc_data, gyro_data, timestamps=None, filter_type='ekf', return_features=False, is_linear_acc=True):
    logger.info(f"Processing IMU data: filter={filter_type}")
    if len(acc_data) == 0 or len(gyro_data) == 0:
        logger.error("Empty input data")
        return {'quaternion': np.zeros((0, 4)),
                'linear_acceleration': np.zeros((0, 3)),
                'fusion_features': np.zeros(43) if return_features else None}
    min_len = min(len(acc_data), len(gyro_data))
    acc_data = acc_data[:min_len]
    gyro_data = gyro_data[:min_len]
    if timestamps is not None:
        timestamps = timestamps[:min_len]
    sample_rate = 30.0
    if timestamps is not None and len(timestamps) > 1:
        avg_dt = (timestamps[-1]-timestamps[0])/(len(timestamps)-1)
        sample_rate = 1000.0/avg_dt
    if filter_type.lower() == 'madgwick':
        orientation_filter = MadgwickFilter(sample_rate=sample_rate)
    elif filter_type.lower() == 'kalman':
        orientation_filter = KalmanFilter(sample_rate=sample_rate)
    elif filter_type.lower() == 'ekf':
        orientation_filter = ExtendedKalmanFilter(sample_rate=sample_rate)
    else:
        logger.warning(f"Unknown filter type: {filter_type}, using EKF")
        orientation_filter = ExtendedKalmanFilter(sample_rate=sample_rate)
    quaternions = []
    linear_accelerations = []
    for i in range(len(acc_data)):
        acc = acc_data[i]
        gyro = gyro_data[i]
        dt = None
        if timestamps is not None and i > 0:
            dt = (timestamps[i]-timestamps[i-1])/1000.0
        q = orientation_filter.update(acc, gyro, dt)
        quaternions.append(q)
        linear_accelerations.append(acc)
    quaternions = np.array(quaternions)
    linear_accelerations = np.array(linear_accelerations)
    results = {'quaternion': quaternions, 'linear_acceleration': linear_accelerations}
    if return_features:
        features = extract_features_from_window({'quaternion': quaternions,
                                                 'linear_acceleration': linear_accelerations,
                                                 'gyroscope': gyro_data})
        results['fusion_features'] = features
    return results

def compare_filters(acc_data, gyro_data, timestamps=None):
    filter_types = ['madgwick', 'kalman', 'ekf']
    results = {}
    for filter_type in filter_types:
        start_time = time.time()
        filter_results = process_imu_data(
            acc_data=acc_data,
            gyro_data=gyro_data,
            timestamps=timestamps,
            filter_type=filter_type,
            return_features=True
        )
        elapsed_time = time.time() - start_time
        results[filter_type] = {
            'quaternion': filter_results['quaternion'],
            'linear_acceleration': filter_results['linear_acceleration'],
            'fusion_features': filter_results['fusion_features'],
            'processing_time': elapsed_time,
            'processing_rate': len(acc_data) / elapsed_time
        }
    return results

