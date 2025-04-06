# utils/imu_fusion.py
import numpy as np
from scipy.spatial.transform import Rotation
import logging
import os
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("imu_fusion")

# Thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=4)

def save_aligned_sensor_data(subject_id, action_id, trial_id, 
                            acc_data, gyro_data, quaternions=None, 
                            timestamps=None, save_dir="data/aligned"):
    """Save aligned sensor data to disk"""
    try:
        # Create directories
        os.makedirs(f"{save_dir}/accelerometer", exist_ok=True)
        os.makedirs(f"{save_dir}/gyroscope", exist_ok=True)
        
        if quaternions is not None:
            os.makedirs(f"{save_dir}/quaternion", exist_ok=True)
        
        # Create filename
        filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}"
        
        # Save numpy arrays
        np.save(f"{save_dir}/accelerometer/{filename}.npy", acc_data)
        np.save(f"{save_dir}/gyroscope/{filename}.npy", gyro_data)
        
        if quaternions is not None:
            np.save(f"{save_dir}/quaternion/{filename}.npy", quaternions)
        
        if timestamps is not None:
            os.makedirs(f"{save_dir}/timestamps", exist_ok=True)
            np.save(f"{save_dir}/timestamps/{filename}.npy", timestamps)
            
        logger.info(f"Saved aligned data for {filename}")
    except Exception as e:
        logger.error(f"Error saving aligned data: {e}")

class MadgwickFilter:
    """Madgwick orientation filter for IMU sensor fusion"""
    
    def __init__(self, beta=0.1, sample_rate=30.0):
        self.beta = beta  # Filter gain
        self.sample_rate = sample_rate
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    
    def update(self, acc, gyro, dt=None):
        """
        Update orientation using accelerometer and gyroscope data
        
        Args:
            acc: Accelerometer reading [x, y, z] in m/s²
            gyro: Gyroscope reading [x, y, z] in rad/s
            dt: Time step in seconds (optional)
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        if dt is None:
            dt = 1.0 / self.sample_rate
            
        q = self.quaternion
        
        # Normalize accelerometer
        if np.linalg.norm(acc) < 1e-10:
            acc_norm = np.array([0, 0, 1])  # Default if accelerometer reading is too small
        else:
            acc_norm = acc / np.linalg.norm(acc)
            
        # Quaternion elements
        q0, q1, q2, q3 = q
        
        # Reference direction of Earth's gravity
        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q1*q3 - q0*q2) - acc_norm[0],
            2*(q0*q1 + q2*q3) - acc_norm[1],
            2*(0.5 - q1**2 - q2**2) - acc_norm[2]
        ])
        
        # Jacobian matrix
        J = np.array([
            [-2*q2, 2*q3, -2*q0, 2*q1],
            [2*q1, 2*q0, 2*q3, 2*q2],
            [0, -4*q1, -4*q2, 0]
        ])
        
        # Gradient
        gradient = J.T @ f
        gradient_norm = np.linalg.norm(gradient)
        
        if gradient_norm > 0:
            gradient = gradient / gradient_norm
            
        # Gyroscope quaternion derivative
        qDot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])
        
        # Combine with gradient
        qDot = qDot - self.beta * gradient
        
        # Integrate to get updated quaternion
        q = q + qDot * dt
        
        # Normalize
        q = q / np.linalg.norm(q)
        
        self.quaternion = q
        return q
        
    def reset(self):
        """Reset filter to initial state"""
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

class KalmanFilter:
    """Standard Kalman filter for orientation estimation"""
    
    def __init__(self, sample_rate=30.0):
        self.sample_rate = sample_rate
        
        # State: quaternion (4), gyro_bias (3)
        self.state = np.zeros(7)
        self.state[0] = 1.0  # Initial quaternion
        
        # Covariance matrix
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])
        
        # Process noise
        self.Q = np.diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])
        
        # Measurement noise
        self.R = np.eye(3) * 0.1
    
    def update(self, acc, gyro, dt=None):
        """
        Update orientation using Kalman filter
        
        Args:
            acc: Accelerometer reading [x, y, z] in m/s²
            gyro: Gyroscope reading [x, y, z] in rad/s
            dt: Time step in seconds (optional)
            
        Returns:
            Updated quaternion [w, x, y, z]
        """
        if dt is None:
            dt = 1.0 / self.sample_rate
            
        # Extract state components
        q = self.state[:4]  # Quaternion
        bias = self.state[4:]  # Gyro bias
        
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        
        # Correct gyro with bias
        gyro_corrected = gyro - bias
        
        # Prediction step - integrate quaternion
        q_dot = 0.5 * self._quaternion_multiply(q, np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]]))
        
        # State transition matrix
        F = np.eye(7)
        F[:4, :4] += dt * 0.5 * self._omega_matrix(gyro_corrected)
        
        # Predict state
        x_pred = np.zeros(7)
        x_pred[:4] = q + q_dot * dt
        x_pred[4:] = bias  # Bias assumed constant
        
        # Normalize quaternion
        x_pred[:4] = x_pred[:4] / np.linalg.norm(x_pred[:4])
        
        # Predict covariance
        P_pred = F @ self.P @ F.T + self.Q
        
        # Measurement update if acceleration is valid
        acc_norm = np.linalg.norm(acc)
        if 0.5 < acc_norm < 1.5:  # Near gravity
            # Normalize accelerometer
            acc_unit = acc / acc_norm
            
            # Expected gravity direction from orientation
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ np.array([0, 0, 1])  # Gravity reference
            
            # Innovation
            y = acc_unit - g_pred
            
            # Measurement matrix
            H = self._compute_H_matrix(x_pred[:4])
            
            # Innovation covariance
            S = H @ P_pred @ H.T + self.R
            
            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.state = x_pred + K @ y
            
            # Update covariance
            self.P = (np.eye(7) - K @ H) @ P_pred
        else:
            # No measurement update
            self.state = x_pred
            self.P = P_pred
            
        # Normalize quaternion
        self.state[:4] = self.state[:4] / np.linalg.norm(self.state[:4])
        
        return self.state[:4]
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _omega_matrix(self, gyro):
        """Create omega matrix for quaternion differentiation"""
        wx, wy, wz = gyro
        return np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def _compute_H_matrix(self, q):
        """Compute linearized measurement matrix"""
        w, x, y, z = q

        # Jacobian of gravity vector with respect to quaternion
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

        # Full measurement matrix
        H = np.zeros((3, 7))
        H[:, :4] = H_q

        return H

    def reset(self):
        """Reset filter to initial state"""
        self.state = np.zeros(7)
        self.state[0] = 1.0
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])

class ExtendedKalmanFilter:
    """Extended Kalman Filter for IMU orientation tracking"""

    def __init__(self, sample_rate=30.0):
        self.sample_rate = sample_rate

        # State: quaternion (4), gyro_bias (3)
        self.state = np.zeros(7)
        self.state[0] = 1.0  # Initial quaternion

        # Error covariance
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])

        # Process noise
        self.Q = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5])

        # Measurement noise - adaptive
        self.R_base = np.eye(3) * 0.05
        self.R = self.R_base.copy()

        # Gravity reference
        self.g_ref = np.array([0, 0, 1])  # Normalized gravity

        # For adaptive noise
        self.acc_history = []
        self.max_history = 10

    def update(self, acc, gyro, dt=None):
        """
        Update orientation using Extended Kalman Filter

        Args:
            acc: Accelerometer reading [x, y, z] in m/s²
            gyro: Gyroscope reading [x, y, z] in rad/s
            dt: Time step in seconds (optional)

        Returns:
            Updated quaternion [w, x, y, z]
        """
        if dt is None:
            dt = 1.0 / self.sample_rate

        # Extract state
        q = self.state[:4]  # Quaternion
        bias = self.state[4:]  # Gyro bias

        # Normalize quaternion
        q = q / np.linalg.norm(q)

        # Correct gyro with bias
        gyro_corrected = gyro - bias

        # Adaptive measurement noise
        acc_norm = np.linalg.norm(acc)
        self.acc_history.append(acc_norm)
        if len(self.acc_history) > self.max_history:
            self.acc_history.pop(0)

        # Increase noise during dynamics
        if len(self.acc_history) >= 3:
            acc_var = np.var(self.acc_history)
            dynamic_factor = 1.0 + 10.0 * min(acc_var, 1.0)
            self.R = self.R_base * dynamic_factor

        # Prediction step - quaternion integration
        q_dot = 0.5 * self._quaternion_product_matrix(q) @ np.array([0, gyro_corrected[0], gyro_corrected[1], gyro_corrected[2]])
        q_pred = q + q_dot * dt
        q_pred = q_pred / np.linalg.norm(q_pred)

        # Predicted state
        x_pred = np.zeros(7)
        x_pred[:4] = q_pred
        x_pred[4:] = bias  # Bias assumed constant

        # Jacobian of state transition
        F = np.eye(7)
        F[:4, :4] = self._quaternion_update_jacobian(q, gyro_corrected, dt)
        F[:4, 4:] = -0.5 * dt * self._quaternion_product_matrix(q)[:, 1:]

        # Predict covariance
        P_pred = F @ self.P @ F.T + self.Q

        # Measurement update
        if 0.5 < acc_norm < 3.0:  # Near gravity
            # Normalize accelerometer
            acc_normalized = acc / acc_norm

            # Expected gravity from orientation
            R_q = self._quaternion_to_rotation_matrix(x_pred[:4])
            g_pred = R_q @ self.g_ref

            # Innovation
            y = acc_normalized - g_pred

            # Measurement Jacobian
            H = self._measurement_jacobian(x_pred[:4])

            # Innovation covariance
            S = H @ P_pred @ H.T + self.R

            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(S)

            # Update state
            self.state = x_pred + K @ y

            # Update covariance - Joseph form
            I_KH = np.eye(7) - K @ H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            # No measurement update
            self.state = x_pred
            self.P = P_pred

        # Normalize quaternion
        self.state[:4] = self.state[:4] / np.linalg.norm(self.state[:4])

        return self.state[:4]

    def _quaternion_product_matrix(self, q):
        """Matrix for quaternion multiplication"""
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
        ])

    def _quaternion_update_jacobian(self, q, gyro, dt):
        """Jacobian of quaternion update"""
        wx, wy, wz = gyro
        omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return np.eye(4) + 0.5 * dt * omega

    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
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

    def _measurement_jacobian(self, q):
        """Compute measurement Jacobian"""
        w, x, y, z = q

        # Jacobian of gravity with respect to quaternion
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

        # Full measurement matrix
        H = np.zeros((3, 7))
        H[:3, :4] = H_q

        return H

    def reset(self):
        """Reset filter to initial state"""
        self.state = np.zeros(7)
        self.state[0] = 1.0
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])
        self.acc_history = []

def extract_features_from_window(window_data):
    """
    Extract features from window of IMU data

    Args:
        window_data: Dictionary with quaternion, linear_acceleration, gyroscope

    Returns:
        Feature vector
    """
    # Extract data
    quaternions = window_data.get('quaternion', np.array([]))
    acc_data = window_data.get('linear_acceleration', window_data.get('accelerometer', np.array([])))
    gyro_data = window_data.get('gyroscope', np.array([]))

    # Check for missing data
    if len(quaternions) == 0 or len(acc_data) == 0 or len(gyro_data) == 0:
        logger.warning("Missing data for feature extraction")
        return np.zeros(43)  # Return zeros

    try:
        # Statistical features from acceleration
        acc_mean = np.mean(acc_data, axis=0)
        acc_std = np.std(acc_data, axis=0)
        acc_max = np.max(acc_data, axis=0)
        acc_min = np.min(acc_data, axis=0)

        # Magnitude of acceleration
        acc_mag = np.linalg.norm(acc_data, axis=1)
        acc_mag_mean = np.mean(acc_mag)
        acc_mag_std = np.std(acc_mag)
        acc_mag_max = np.max(acc_mag)

        # Gyroscope features
        gyro_mean = np.mean(gyro_data, axis=0)
        gyro_std = np.std(gyro_data, axis=0)
        gyro_max = np.max(np.abs(gyro_data), axis=0)

        # Jerk (derivative of acceleration)
        jerk_features = []
        if len(acc_data) > 1:
            jerk = np.diff(acc_data, axis=0)
            jerk_mag = np.linalg.norm(jerk, axis=1)
            jerk_mag_mean = np.mean(jerk_mag)
            jerk_mag_max = np.max(jerk_mag)
            jerk_features = [jerk_mag_mean, jerk_mag_max]
        else:
            jerk_features = [0, 0]

        # Convert quaternions to Euler angles
        euler_angles = []
        for q in quaternions:
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy format
            euler_angles.append(r.as_euler('xyz', degrees=True))

        euler_angles = np.array(euler_angles)

        # Orientation features
        euler_mean = np.mean(euler_angles, axis=0)
        euler_std = np.std(euler_angles, axis=0)

        # Orientation change rate
        angle_rate_features = []
        if len(euler_angles) > 1:
            angle_rates = np.diff(euler_angles, axis=0)
            angle_rate_mag = np.linalg.norm(angle_rates, axis=1)
            angle_rate_mean = np.mean(angle_rate_mag)
            angle_rate_max = np.max(angle_rate_mag)
            angle_rate_features = [angle_rate_mean, angle_rate_max]
        else:
            angle_rate_features = [0, 0]

        # Add frequency domain features
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

        # Combine all features
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

def process_imu_data(acc_data, gyro_data, timestamps=None, filter_type='ekf', return_features=False):
    """
    Process IMU data with sensor fusion

    Args:
        acc_data: Accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        timestamps: Optional timestamps
        filter_type: Type of filter ('madgwick', 'kalman', 'ekf')
        return_features: Whether to compute derived features

    Returns:
        Dictionary with processed data
    """
    logger.info(f"Processing IMU data: filter={filter_type}")

    # Input validation
    if len(acc_data) == 0 or len(gyro_data) == 0:
        logger.error("Empty input data")
        return {
            'quaternion': np.zeros((0, 4)),
            'linear_acceleration': np.zeros((0, 3)),
            'fusion_features': np.zeros(43) if return_features else None
        }

    # Match data lengths
    min_len = min(len(acc_data), len(gyro_data))
    acc_data = acc_data[:min_len]
    gyro_data = gyro_data[:min_len]

    if timestamps is not None:
        timestamps = timestamps[:min_len]

    # Convert to uniform sample rate if timestamps provided
    sample_rate = 30.0  # Default
    if timestamps is not None and len(timestamps) > 1:
        avg_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        sample_rate = 1000.0 / avg_dt  # Convert to Hz assuming timestamps in ms

    # Initialize orientation filter
    if filter_type.lower() == 'madgwick':
        orientation_filter = MadgwickFilter(sample_rate=sample_rate)
    elif filter_type.lower() == 'kalman':
        orientation_filter = KalmanFilter(sample_rate=sample_rate)
    elif filter_type.lower() == 'ekf':
        orientation_filter = ExtendedKalmanFilter(sample_rate=sample_rate)
    else:
        logger.warning(f"Unknown filter type: {filter_type}, using EKF")
        orientation_filter = ExtendedKalmanFilter(sample_rate=sample_rate)

    # Process data
    quaternions = []
    linear_accelerations = []

    for i in range(len(acc_data)):
        # Current samples
        acc = acc_data[i]
        gyro = gyro_data[i]

        # Calculate time delta if timestamps available
        dt = None
        if timestamps is not None and i > 0:
            dt = (timestamps[i] - timestamps[i-1]) / 1000.0  # ms to seconds

        # Update orientation
        q = orientation_filter.update(acc, gyro, dt)
        quaternions.append(q)

        # Store linear acceleration
        linear_accelerations.append(acc)

    # Convert to arrays
    quaternions = np.array(quaternions)
    linear_accelerations = np.array(linear_accelerations)

    # Prepare results
    results = {
        'quaternion': quaternions,
        'linear_acceleration': linear_accelerations
    }

    # Extract features if requested
    if return_features:
        features = extract_features_from_window({
            'quaternion': quaternions,
            'linear_acceleration': linear_accelerations,
            'gyroscope': gyro_data
        })
        results['fusion_features'] = features

    return results

def compare_filters(acc_data, gyro_data, timestamps=None):
    """
    Compare performance of different orientation filters

    Args:
        acc_data: Accelerometer data
        gyro_data: Gyroscope data
        timestamps: Optional timestamps

    Returns:
        Dictionary with results for each filter
    """
    filter_types = ['madgwick', 'kalman', 'ekf']
    results = {}

    for filter_type in filter_types:
        start_time = time.time()

        # Process with this filter
        filter_results = process_imu_data(
            acc_data=acc_data,
            gyro_data=gyro_data,
            timestamps=timestamps,
            filter_type=filter_type,
            return_features=True
        )

        elapsed_time = time.time() - start_time

        # Store results with timing
        results[filter_type] = {
            'quaternion': filter_results['quaternion'],
            'linear_acceleration': filter_results['linear_acceleration'],
            'fusion_features': filter_results['fusion_features'],
            'processing_time': elapsed_time,
            'processing_rate': len(acc_data) / elapsed_time  # samples/sec
        }

    return results
