import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Union, Optional
import pandas as pd
import time
import traceback
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

# Target resampling rate for Android compatibility
TARGET_SAMPLE_RATE = 30.0  # Hz (fixed rate for consistent training/inference)
WINDOW_SIZE = 64  # samples
STRIDE_LENGTH = 10  # samples

def fix_monotonic_timestamps(timestamps):
    """
    Ensure timestamps are strictly monotonically increasing
    """
    if len(timestamps) <= 1:
        return timestamps
    
    # Convert to numeric format if needed
    if isinstance(timestamps[0], (str, pd.Timestamp)):
        try:
            ts_numeric = pd.to_datetime(timestamps).astype(np.int64).values
        except:
            # If conversion fails, try treating as seconds
            try:
                ts_numeric = np.array([float(ts) for ts in timestamps])
            except:
                # If all else fails, use indices
                logger.warning("Could not convert timestamps to numeric values, using indices")
                return np.arange(len(timestamps))
    else:
        ts_numeric = np.array(timestamps)
    
    # Find non-monotonic or duplicate timestamps
    diffs = np.diff(ts_numeric)
    problematic_indices = np.where(diffs <= 0)[0]
    
    if len(problematic_indices) > 0:
        logger.debug(f"Found {len(problematic_indices)} non-monotonic timestamp(s)")
        
        # Fix non-monotonic timestamps with minimal changes
        for idx in problematic_indices:
            # Add a small increment to make it strictly greater than the previous
            min_increment = 1  # Use 1ns for datetime64 or 1e-6 for float seconds
            if isinstance(timestamps[0], (str, pd.Timestamp)):
                # For datetime timestamps
                ts_numeric[idx+1] = ts_numeric[idx] + min_increment
            else:
                # For numeric timestamps, use small fraction of typical diff
                typical_diff = np.median(diffs[diffs > 0])
                min_increment = max(1e-6, typical_diff * 0.001)
                ts_numeric[idx+1] = ts_numeric[idx] + min_increment
    
    # Convert back to original format if needed
    if isinstance(timestamps[0], pd.Timestamp):
        return pd.to_datetime(ts_numeric)
    elif isinstance(timestamps[0], str):
        # Try to maintain original string format
        if 'T' in timestamps[0]:  # ISO format
            return [pd.Timestamp(ts).isoformat() for ts in pd.to_datetime(ts_numeric)]
        else:  # Assume standard format
            return [pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in pd.to_datetime(ts_numeric)]
    else:
        return ts_numeric
def last_value_resample(data, timestamps, target_timestamps):
    """
    Resample data using last-value-forward approach (Android-compatible)
    This method only uses past information, suitable for real-time processing
    """
    if len(data) == 0 or len(timestamps) == 0:
        return np.zeros((len(target_timestamps), data.shape[1] if len(data.shape) > 1 else 1))
        
    # Ensure timestamps are monotonically increasing
    timestamps = fix_monotonic_timestamps(timestamps)
    
    # Convert to numpy arrays if they aren't already
    data = np.array(data)
    timestamps = np.array(timestamps)
    target_timestamps = np.array(target_timestamps)
    
    # Initialize output array
    if len(data.shape) > 1:
        resampled = np.zeros((len(target_timestamps), data.shape[1]))
    else:
        resampled = np.zeros(len(target_timestamps))
    
    # Find the first valid data point
    if target_timestamps[0] < timestamps[0]:
        # For timestamps before the first data point, use the first value
        idx = 0
    else:
        # Find the first data point that occurred before or at the first target timestamp
        idx = np.searchsorted(timestamps, target_timestamps[0], side='right') - 1
        idx = max(0, idx)
    
    # If the resampling target starts before data, use the first data point for initial values
    if idx == 0:
        if len(data.shape) > 1:
            resampled[0] = data[0]
        else:
            resampled[0] = data[0]
    
    # Resample each target timestamp
    last_idx = idx
    for i, target_time in enumerate(target_timestamps):
        # Find the last measurement that occurred before or at the target time
        while last_idx + 1 < len(timestamps) and timestamps[last_idx + 1] <= target_time:
            last_idx += 1
            
        # Use the last valid measurement
        if len(data.shape) > 1:
            resampled[i] = data[last_idx]
        else:
            resampled[i] = data[last_idx]
    
    return resampled


def align_sensor_data(acc_data: np.ndarray, gyro_data: np.ndarray, time_tolerance: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align accelerometer and gyroscope data using Android-compatible approach
    
    Args:
        acc_data: Accelerometer data, either raw numpy array or pandas DataFrame with timestamps in first column
        gyro_data: Gyroscope data, similar format to acc_data
        time_tolerance: Maximum allowed time difference for alignment
        
    Returns:
        Tuple of (aligned_acc, aligned_gyro, aligned_timestamps)
    """
    try:
        logger.info(f"Starting sensor alignment: acc shape={acc_data.shape}, gyro shape={gyro_data.shape}")
        if len(acc_data) == 0 or len(gyro_data) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        # Extract timestamps and values from acc_data
        if hasattr(acc_data, 'iloc'):
            try:
                # DataFrame with timestamps in first column
                if isinstance(acc_data.iloc[0, 0], str):
                    acc_times = pd.to_datetime(acc_data.iloc[:, 0]).values
                else:
                    acc_times = acc_data.iloc[:, 0].values
                acc_values = acc_data.iloc[:, 1:4].values
            except Exception as e:
                logger.error(f"Invalid accelerometer data format")
                return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        else:
            # Raw numpy array - synthetic timestamps
            acc_times = np.arange(len(acc_data))
            acc_values = acc_data
        
        # Extract timestamps and values from gyro_data
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
            # Raw numpy array - synthetic timestamps
            gyro_times = np.arange(len(gyro_data))
            gyro_values = gyro_data
            
        if len(acc_times) < 2 or len(gyro_times) < 2:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        # Fix any issues with monotonicity in timestamps
        acc_times = fix_monotonic_timestamps(acc_times)
        gyro_times = fix_monotonic_timestamps(gyro_times)
            
        # Normalize timestamps to common reference for comparison
        if isinstance(acc_times[0], (pd.Timestamp, np.datetime64)):
            # Convert datetime to numeric (nanoseconds)
            acc_times_numeric = pd.to_datetime(acc_times).astype(np.int64).values
            gyro_times_numeric = pd.to_datetime(gyro_times).astype(np.int64).values
            
            # Normalize to seconds from first timestamp for better numerical stability
            start_time = min(acc_times_numeric[0], gyro_times_numeric[0])
            acc_times_sec = (acc_times_numeric - start_time) / 1e9
            gyro_times_sec = (gyro_times_numeric - start_time) / 1e9
        else:
            # Already numeric - normalize to seconds if needed
            if max(acc_times) > 1e10 or max(gyro_times) > 1e10:  # Likely nanoseconds or similar
                start_time = min(acc_times[0], gyro_times[0])
                acc_times_sec = (acc_times - start_time) / 1e9
                gyro_times_sec = (gyro_times - start_time) / 1e9
            else:  # Assume already in sensible units
                acc_times_sec = acc_times
                gyro_times_sec = gyro_times
        
        # Determine target timebase using fixed rate (30Hz) spanning common time range
        common_start = max(acc_times_sec[0], gyro_times_sec[0])
        common_end = min(acc_times_sec[-1], gyro_times_sec[-1])
        
        if common_end <= common_start:
            logger.warning("No common time range between accelerometer and gyroscope data")
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        # Create uniform 30Hz timestamps spanning the common range
        target_fs = 30.0  # 30 Hz fixed rate
        num_samples = int((common_end - common_start) * target_fs) + 1
        uniform_times = np.linspace(common_start, common_end, num_samples)
        
        # Perform last-value-forward resampling (Android-compatible)
        aligned_acc = np.zeros((len(uniform_times), 3))
        aligned_gyro = np.zeros((len(uniform_times), 3))
        
        # Resample accelerometer using last-value-forward
        acc_idx = 0
        for i, t in enumerate(uniform_times):
            # Find last accelerometer reading before this time
            while acc_idx + 1 < len(acc_times_sec) and acc_times_sec[acc_idx + 1] <= t:
                acc_idx += 1
            aligned_acc[i] = acc_values[acc_idx]
        
        # Resample gyroscope using last-value-forward
        gyro_idx = 0
        for i, t in enumerate(uniform_times):
            # Find last gyroscope reading before this time
            while gyro_idx + 1 < len(gyro_times_sec) and gyro_times_sec[gyro_idx + 1] <= t:
                gyro_idx += 1
            aligned_gyro[i] = gyro_values[gyro_idx]
        
        return aligned_acc, aligned_gyro, uniform_times
    
    except Exception as e:
        logger.error(f"Error in sensor alignment: {str(e)}")
        logger.error(traceback.format_exc())
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

class SimpleComplementaryFilter:
    """
    Lightweight and efficient complementary filter for orientation estimation
    Suitable for real-time Android implementation
    """
    def __init__(self, alpha=0.98, sample_rate=TARGET_SAMPLE_RATE):
        self.alpha = alpha
        self.sample_rate = sample_rate
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion (w,x,y,z)
        self.last_time = None
    
    def reset(self):
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        
    def update(self, acc, gyro, timestamp=None):
        # Calculate time delta
        dt = 1.0 / self.sample_rate
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            self.last_time = timestamp
        elif timestamp is not None:
            self.last_time = timestamp
        
        if dt <= 0 or dt > 1.0:  # Sanity check for time delta
            dt = 1.0 / self.sample_rate
        
        # Normalize accelerometer to get gravity direction
        acc_norm = np.linalg.norm(acc)
        if acc_norm < 1e-10:
            acc_normalized = np.array([0, 0, 1])  # Default to "up" when acceleration is zero
        else:
            acc_normalized = acc / acc_norm
            
        # Convert gravity direction to quaternion (this is the acc-based orientation)
        acc_quat = self._gravity_to_quaternion(acc_normalized)
        
        # Integrate gyroscope to get orientation change
        gyro_quat = self._integrate_gyro(gyro, dt)
        gyro_orientation = self._quaternion_multiply(self.orientation, gyro_quat)
        
        # Complementary filter: combine accel and gyro with simple weighted average
        # This is much more efficient than a full Madgwick or Mahony filter
        w_gyro = self.alpha
        w_acc = 1.0 - self.alpha
        
        # Adjust weights based on acceleration magnitude
        # When acceleration is far from 1g, trust gyro more
        g_error = abs(acc_norm - 9.81) / 9.81
        if g_error > 0.1:  # More than 10% from gravity
            w_gyro = min(0.99, self.alpha + 0.1 * g_error)
            w_acc = 1.0 - w_gyro
            
        # Simple weighted average in quaternion space
        # For efficiency, we skip proper SLERP and just do direct weighting
        self.orientation = self._quaternion_normalize(
            w_gyro * gyro_orientation + w_acc * acc_quat
        )
        
        return self.orientation
        
    def _gravity_to_quaternion(self, gravity):
        """Convert gravity vector to orientation quaternion"""
        # Find rotation from [0,0,1] to gravity vector
        z_axis = np.array([0, 0, 1])
        
        # Use cross product to find rotation axis
        rotation_axis = np.cross(z_axis, gravity)
        axis_norm = np.linalg.norm(rotation_axis)
        
        if axis_norm < 1e-10:
            # Vectors are parallel, no rotation needed
            if gravity[2] > 0:
                return np.array([1.0, 0.0, 0.0, 0.0])  # Identity
            else:
                return np.array([0.0, 1.0, 0.0, 0.0])  # 180° around X
                
        rotation_axis = rotation_axis / axis_norm
        
        # Find rotation angle
        dot_product = np.dot(z_axis, gravity)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Convert to quaternion
        sin_half_angle = np.sin(angle / 2)
        cos_half_angle = np.cos(angle / 2)
        
        return np.array([
            cos_half_angle,
            rotation_axis[0] * sin_half_angle,
            rotation_axis[1] * sin_half_angle,
            rotation_axis[2] * sin_half_angle
        ])
    
    def _integrate_gyro(self, gyro, dt):
        """Convert angular velocity to quaternion change"""
        # For small rotations, we can approximate
        angle = np.linalg.norm(gyro) * dt
        if angle < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
            
        axis = gyro / np.linalg.norm(gyro)
        
        sin_half_angle = np.sin(angle / 2)
        cos_half_angle = np.cos(angle / 2)
        
        return np.array([
            cos_half_angle,
            axis[0] * sin_half_angle,
            axis[1] * sin_half_angle,
            axis[2] * sin_half_angle
        ])
    
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
    
    def _quaternion_normalize(self, q):
        """Normalize quaternion to unit length"""
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm

# Legacy filter classes maintained for backward compatibility
class MadgwickFilter:
    def __init__(self, freq: float = TARGET_SAMPLE_RATE, beta: float = 0.1):
        self.freq = freq
        self.beta = beta
        self.last_time = None
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        # For real-time use, we'll actually use the simpler filter
        self._real_filter = SimpleComplementaryFilter(alpha=0.98, sample_rate=freq)
    
    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: float = None) -> np.ndarray:
        # Delegate to the simpler filter for better real-time performance
        self.orientation_q = self._real_filter.update(acc, gyro, timestamp)
        return self.orientation_q
        
    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None
        self._real_filter.reset()

# Keep KalmanFilter and ExtendedKalmanFilter classes for backward compatibility
# but internally they now use SimpleComplementaryFilter for better real-time performance
class KalmanFilter:
    def __init__(self, freq: float = TARGET_SAMPLE_RATE, process_noise: float = 1e-4, measurement_noise: float = 0.1):
        self._real_filter = SimpleComplementaryFilter(alpha=0.98, sample_rate=freq)
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
    
    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: float = None) -> np.ndarray:
        self.orientation_q = self._real_filter.update(acc, gyro, timestamp)
        return self.orientation_q
        
    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self._real_filter.reset()

class ExtendedKalmanFilter:
    def __init__(self, freq: float = TARGET_SAMPLE_RATE, process_noise: float = 1e-5, measurement_noise: float = 0.05):
        self._real_filter = SimpleComplementaryFilter(alpha=0.98, sample_rate=freq)
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
    
    def update(self, acc: np.ndarray, gyro: np.ndarray, timestamp: float = None) -> np.ndarray:
        self.orientation_q = self._real_filter.update(acc, gyro, timestamp)
        return self.orientation_q
        
    def reset(self):
        self.orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self._real_filter.reset()

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
        elif filter_type == 'simple':
            filter_instance = SimpleComplementaryFilter()
        else:
            filter_instance = SimpleComplementaryFilter()
        
        filter_cache[cache_key] = filter_instance
    
    return filter_cache[cache_key]

def extract_windows(data, window_size=WINDOW_SIZE, stride=STRIDE_LENGTH):
    """
    Extract overlapping windows from time series data
    
    Args:
        data: Time series data [samples, features]
        window_size: Number of samples per window
        stride: Number of samples to advance for each window
        
    Returns:
        windows: Array of windowed data [num_windows, window_size, features]
    """
    if len(data) < window_size:
        # Not enough data for even one window
        return np.array([])
        
    # Calculate number of windows
    num_windows = (len(data) - window_size) // stride + 1
    
    # Special case: when data is exactly window_size
    if len(data) == window_size:
        return np.array([data])
    
    # Special case: not enough data for striding
    if num_windows <= 0:
        return np.array([data[:window_size]])
    
    # Extract windows
    windows = []
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        if end_idx <= len(data):
            windows.append(data[start_idx:end_idx])
    
    return np.array(windows)

def process_imu_data(acc_data: np.ndarray, gyro_data: np.ndarray, timestamps=None, 
                   filter_type='complementary', return_features=False, trial_id=None, reset_filter=False):
    """
    Process IMU data with Android-optimized orientation filter
    
    Args:
        acc_data: Accelerometer data [n, 3]
        gyro_data: Gyroscope data [n, 3]
        timestamps: Optional timestamps for sensor fusion
        filter_type: Type of orientation filter to use ('complementary' recommended for Android)
        return_features: Whether to return additional features
        trial_id: Optional identifier for filter persistence
        reset_filter: Whether to reset the filter state
        
    Returns:
        Dictionary with orientation data and optional features
    """
    if trial_id is not None:
        orientation_filter = get_filter_instance(trial_id, 0, filter_type, reset=reset_filter)
    else:
        # Use SimpleComplementaryFilter by default for Android compatibility
        if filter_type == 'madgwick':
            orientation_filter = MadgwickFilter()
        elif filter_type == 'kalman':
            orientation_filter = KalmanFilter()
        elif filter_type == 'ekf':
            orientation_filter = ExtendedKalmanFilter()
        else:
            orientation_filter = SimpleComplementaryFilter()
    
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
                
            # Process with the orientation filter
            q = orientation_filter.update(acc, gyro, timestamp)
            quaternions.append(q)
        
        results = {'quaternion': np.array(quaternions)}
        
        if return_features:
            from utils.imu_fusion import extract_features_from_window
            features = extract_features_from_window({
                'quaternion': np.array(quaternions),
                'accelerometer': acc_data,
                'gyroscope': gyro_data
            })
            results['fusion_features'] = features
        
        return results
        
    except Exception as e:
        logger.error(f"Error in IMU processing: {str(e)}")
        logger.error(traceback.format_exc())
        return {'quaternion': np.zeros((len(acc_data), 4))}

def preprocess_all_subjects(subjects, filter_type, output_dir, max_length=64):
    """Preprocess all subjects with Android-optimized approach"""
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
    global filter_cache
    filter_cache = {}
    
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
                                # Improved CSV loading with better separator detection
                                with open(file_path, 'r') as f:
                                    first_line = f.readline().strip()
                                    sep = ';' if ';' in first_line else ','
                                
                                file_data = pd.read_csv(file_path, index_col=False, header=None, sep=sep).dropna().bfill()
                                
                                # Extract timestamps and data
                                if file_data.shape[1] > 4:
                                    # Handle extra columns
                                    timestamps = file_data.iloc[:, 0]
                                    data = file_data.iloc[:, 1:4].to_numpy(dtype=np.float32)
                                else:
                                    timestamps = file_data.iloc[:, 0]
                                    data = file_data.iloc[:, 1:].to_numpy(dtype=np.float32)
                                
                                trial_data[f"{modality_name}_timestamps"] = timestamps
                                trial_data[modality_name] = data
                            except Exception as e:
                                logger.error(f"Error loading {modality_name} data: {str(e)}")
                                continue
                    
                    if 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                        acc_data = trial_data['accelerometer']
                        gyro_data = trial_data['gyroscope']
                        acc_timestamps = trial_data.get('accelerometer_timestamps')
                        gyro_timestamps = trial_data.get('gyroscope_timestamps')
                        
                        # Align sensor data using Android-compatible method
                        aligned_acc, aligned_gyro, aligned_timestamps = align_sensor_data(
                            pd.DataFrame({0: acc_timestamps, 1: acc_data[:, 0], 2: acc_data[:, 1], 3: acc_data[:, 2]}),
                            pd.DataFrame({0: gyro_timestamps, 1: gyro_data[:, 0], 2: gyro_data[:, 1], 3: gyro_data[:, 2]})
                        )
                        
                        if len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                            # Use SimpleComplementaryFilter for best Android performance
                            filter_instance = SimpleComplementaryFilter()
                            
                            # Process in fixed windows
                            window_size = max_length
                            stride = 10  # Use stride of 10 as requested
                            
                            all_quaternions = []
                            windowed_data = {
                                'accelerometer': [],
                                'gyroscope': [],
                                'quaternion': [],
                                'timestamps': []
                            }
                            
                            # Create windows with stride
                            for i in range(0, len(aligned_acc) - window_size + 1, stride):
                                window_acc = aligned_acc[i:i + window_size]
                                window_gyro = aligned_gyro[i:i + window_size]
                                window_time = aligned_timestamps[i:i + window_size]
                                
                                # Calculate orientation for this window
                                window_quaternions = []
                                for j in range(len(window_acc)):
                                    q = filter_instance.update(window_acc[j], window_gyro[j], window_time[j])
                                    window_quaternions.append(q)
                                    all_quaternions.append(q)
                                
                                # Store window data
                                windowed_data['accelerometer'].append(window_acc)
                                windowed_data['gyroscope'].append(window_gyro)
                                windowed_data['quaternion'].append(np.array(window_quaternions))
                                windowed_data['timestamps'].append(window_time)
                                
                                # Save individual window
                                window_output_file = os.path.join(subject_dir, f"{trial_id}_W{len(windowed_data['accelerometer'])-1:04d}.npz")
                                np.savez_compressed(
                                    window_output_file,
                                    accelerometer=window_acc,
                                    gyroscope=window_gyro,
                                    quaternion=np.array(window_quaternions),
                                    timestamps=window_time,
                                    window_id=len(windowed_data['accelerometer'])-1,
                                    filter_type=filter_type
                                )
                            
                            # Also save the complete sequence data
                            output_file = os.path.join(subject_dir, f"{trial_id}.npz")
                            np.savez_compressed(
                                output_file,
                                accelerometer=aligned_acc,
                                gyroscope=aligned_gyro,
                                quaternion=np.array(all_quaternions[:len(aligned_acc)]),
                                timestamps=aligned_timestamps,
                                filter_type=filter_type
                            )
                except Exception as e:
                    logger.error(f"Error processing trial {trial_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
    
    logger.info(f"Preprocessing complete: processed {processed_count}/{total_trials} trials")
def extract_features_from_window(window_data):
    """
    Extract features from sensor data window optimized for real-time Android
    Focus on computationally efficient features that are discriminative for falls
    """
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
        
        # Energy and magnitude features (good for fall detection)
        acc_energy = np.sum(acc**2, axis=0) / len(acc)
        acc_magnitude = np.sqrt(np.sum(acc**2, axis=1))
        acc_magnitude_mean = np.mean(acc_magnitude)
        acc_magnitude_std = np.std(acc_magnitude)
        acc_magnitude_max = np.max(acc_magnitude)
        
        # Compute signal magnitude area (SMA) - efficient and effective for activity recognition
        acc_sma = np.sum(np.abs(acc)) / len(acc)
        
        # Add all features
        features.extend(acc_mean)
        features.extend(acc_std)
        features.extend(acc_max - acc_min)
        features.extend(acc_energy)
        features.append(acc_magnitude_mean)
        features.append(acc_magnitude_std)
        features.append(acc_magnitude_max)
        features.append(acc_sma)
    
    if gyro is not None and len(gyro) > 0:
        # Time domain features from gyroscope
        gyro_mean = np.mean(gyro, axis=0)
        gyro_std = np.std(gyro, axis=0)
        gyro_min = np.min(gyro, axis=0)
        gyro_max = np.max(gyro, axis=0)
        
        # Energy and magnitude features
        gyro_energy = np.sum(gyro**2, axis=0) / len(gyro)
        gyro_magnitude = np.sqrt(np.sum(gyro**2, axis=1))
        gyro_magnitude_mean = np.mean(gyro_magnitude)
        gyro_magnitude_std = np.std(gyro_magnitude)
        gyro_magnitude_max = np.max(gyro_magnitude)
        
        # Compute SMA for gyro
        gyro_sma = np.sum(np.abs(gyro)) / len(gyro)
        
        # Add all features
        features.extend(gyro_mean)
        features.extend(gyro_std)
        features.extend(gyro_max - gyro_min)
        features.extend(gyro_energy)
        features.append(gyro_magnitude_mean)
        features.append(gyro_magnitude_std)
        features.append(gyro_magnitude_max)
        features.append(gyro_sma)
    
    # Cross-sensor correlation (highly valuable for fall detection)
    if acc is not None and gyro is not None and len(acc) > 0 and len(gyro) > 0:
        # Cross-correlation between accelerometer and gyroscope axes
        for i in range(3):
            for j in range(3):
                corr = np.corrcoef(acc[:, i], gyro[:, j])[0, 1]
                features.append(corr if not np.isnan(corr) else 0)
    
    if quat is not None and len(quat) > 0:
        # Simple quaternion features (optimized for real-time)
        quat_mean = np.mean(quat, axis=0)
        quat_std = np.std(quat, axis=0)
        
        # Extract orientation change rate (important for fall detection)
        orientation_change = np.zeros(len(quat)-1)
        for i in range(1, len(quat)):
            # Angular difference between consecutive quaternions
            dot_product = np.abs(np.sum(quat[i] * quat[i-1]))
            dot_product = np.clip(dot_product, -1.0, 1.0)
            orientation_change[i-1] = np.arccos(dot_product) * 2.0
        
        # Orientation change statistics
        if len(orientation_change) > 0:
            orientation_mean = np.mean(orientation_change)
            orientation_std = np.std(orientation_change)
            orientation_max = np.max(orientation_change)
            
            features.append(orientation_mean)
            features.append(orientation_std)
            features.append(orientation_max)
        
        # Add quaternion features
        features.extend(quat_mean)
        features.extend(quat_std)
    
    return np.array(features)

def stateful_process_imu_data(acc_data, gyro_data, timestamps=None, filter_type='simple', trial_id=None, reset_filter=False, return_features=False):
    """Stateful IMU processing for window-by-window processing"""
    if trial_id is None:
        return process_imu_data(acc_data, gyro_data, timestamps, filter_type, return_features)
    
    orientation_filter = get_filter_instance(trial_id, 0, filter_type, reset=reset_filter)
    
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
        
        results = {'quaternion': np.array(quaternions)}
        
        if return_features:
            features = extract_features_from_window({
                'quaternion': np.array(quaternions),
                'accelerometer': acc_data,
                'gyroscope': gyro_data
            })
            results['fusion_features'] = features
        
        return results
        
    except Exception as e:
        logger.error(f"Error in stateful IMU processing: {str(e)}")
        return {'quaternion': np.zeros((len(acc_data), 4))}

def preprocess_all_subjects(subjects, filter_type, output_dir, max_length=WINDOW_SIZE):
    """Preprocess all subjects using fixed-rate resampling and window extraction"""
    logger.info(f"Preprocessing all subjects with {filter_type} filter")
    
    from utils.dataset import SmartFallMM
    from tqdm.auto import tqdm
    
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
    
    for subject_id in tqdm(subjects, desc=f"Preprocessing ({filter_type})"):
        subject_dir = os.path.join(output_dir, f"S{subject_id:02d}")
        os.makedirs(subject_dir, exist_ok=True)
        
        subject_trials = [trial for trial in dataset.matched_trials if trial.subject_id == subject_id]
        
        for trial in tqdm(subject_trials, desc=f"Subject {subject_id}", leave=False):
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
                            
                            # Extract timestamp if present
                            if file_data.shape[1] > cols:
                                timestamps = file_data.iloc[:, 0].to_numpy()
                                trial_data[f"{modality_name}_timestamps"] = timestamps
                            
                            trial_data[modality_name] = data
                        except Exception as e:
                            logger.error(f"Error loading {modality_name} data: {e}")
                            continue
                
                if 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                    acc_data = trial_data['accelerometer']
                    gyro_data = trial_data['gyroscope']
                    acc_timestamps = trial_data.get('accelerometer_timestamps')
                    gyro_timestamps = trial_data.get('gyroscope_timestamps')
                    
                    # Align data to fixed rate using Android-compatible method
                    aligned_acc, aligned_gyro, timestamps = align_sensor_data(
                        acc_data, gyro_data, acc_timestamps, gyro_timestamps
                    )
                    
                    if len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                        # Process with simple filter for orientation
                        result = process_imu_data(
                            aligned_acc, aligned_gyro, timestamps,
                            filter_type=filter_type, 
                            trial_id=trial_id, 
                            reset_filter=True
                        )
                        
                        # Save results for training
                        output_file = os.path.join(subject_dir, f"{trial_id}.npz")
                        np.savez_compressed(
                            output_file,
                            accelerometer=aligned_acc,
                            gyroscope=aligned_gyro,
                            quaternion=result['quaternion'],
                            timestamps=timestamps,
                            filter_type=filter_type
                        )
                        
                        # Extract windows for training
                        windows_acc = extract_windows(aligned_acc, window_size=max_length, stride=STRIDE_LENGTH)
                        windows_gyro = extract_windows(aligned_gyro, window_size=max_length, stride=STRIDE_LENGTH)
                        windows_quat = extract_windows(result['quaternion'], window_size=max_length, stride=STRIDE_LENGTH)
                        
                        # Save each window
                        for w_idx in range(len(windows_acc)):
                            window_file = os.path.join(subject_dir, f"{trial_id}_W{w_idx:04d}.npz")
                            np.savez_compressed(
                                window_file,
                                accelerometer=windows_acc[w_idx],
                                gyroscope=windows_gyro[w_idx],
                                quaternion=windows_quat[w_idx],
                                window_id=w_idx,
                                filter_type=filter_type
                            )
            except Exception as e:
                logger.error(f"Error processing trial {trial_id}: {str(e)}")
                continue
    
    logger.info(f"Preprocessing complete for all subjects with {filter_type} filter")
