from typing import Any, List, Tuple, Dict, Union, Optional
from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, butter, filtfilt
from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

FILTER_STATES = {}
VISUALIZATION_DIR = 'visualization_output'
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def csvloader(file_path):
    try:
        file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        timestamps = file_data.iloc[2:, 0].to_numpy(dtype=np.float64)
        values = file_data.iloc[2:, 1:4].to_numpy(dtype=np.float32)
        return values, timestamps
    except Exception as e:
        print(f"Error loading CSV {file_path}: {str(e)}")
        return np.array([]), np.array([])

def matloader(file_path, **kwargs):
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        raise ValueError(f"Unsupported key {key} for matlab file")
    return loadmat(file_path)[key], None

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def bandpass_filter(data, lowcut=0.5, highcut=15.0, fs=30.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    return filtered

def add_gravity(linear_acc, q):
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    gravity = rot.apply([0, 0, 9.81], inverse=True)
    return linear_acc + gravity

def visualize_alignment(acc_orig, gyro_orig, acc_aligned, gyro_aligned,
                        acc_timestamps, gyro_timestamps, aligned_timestamps, trial_id):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    plt.suptitle(f'Sensor Alignment - Trial {trial_id}')
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        axes[i, 0].plot(acc_timestamps, acc_orig[:, i], 'b.', alpha=0.3, label='Original')
        axes[i, 0].plot(aligned_timestamps, acc_aligned[:, i], 'r-', label='Aligned')
        axes[i, 0].set_title(f'Accelerometer {axis_name}-axis')
        axes[i, 0].set_xlabel('Time (ms)')
        axes[i, 0].set_ylabel('Acceleration (m/s²)')
        axes[i, 0].legend()
        axes[i, 1].plot(gyro_timestamps, gyro_orig[:, i], 'b.', alpha=0.3, label='Original')
        axes[i, 1].plot(aligned_timestamps, gyro_aligned[:, i], 'r-', label='Aligned')
        axes[i, 1].set_title(f'Gyroscope {axis_name}-axis')
        axes[i, 1].set_xlabel('Time (ms)')
        axes[i, 1].set_ylabel('Angular velocity (rad/s)')
        axes[i, 1].legend()
    plt.tight_layout()
    output_path = os.path.join(VISUALIZATION_DIR, f'alignment_{trial_id}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    acc_mag_orig = np.sqrt(np.sum(acc_orig**2, axis=1))
    acc_mag_aligned = np.sqrt(np.sum(acc_aligned**2, axis=1))
    gyro_mag_orig = np.sqrt(np.sum(gyro_orig**2, axis=1))
    gyro_mag_aligned = np.sqrt(np.sum(gyro_aligned**2, axis=1))
    ax1.plot(acc_timestamps, acc_mag_orig, 'b.', alpha=0.3, label='Original')
    ax1.plot(aligned_timestamps, acc_mag_aligned, 'r-', label='Aligned')
    ax1.set_title('Acceleration Magnitude')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Magnitude (m/s²)')
    ax1.legend()
    ax2.plot(gyro_timestamps, gyro_mag_orig, 'b.', alpha=0.3, label='Original')
    ax2.plot(aligned_timestamps, gyro_mag_aligned, 'r-', label='Aligned')
    ax2.set_title('Angular Velocity Magnitude')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Magnitude (rad/s)')
    ax2.legend()
    plt.tight_layout()
    output_path = os.path.join(VISUALIZATION_DIR, f'magnitude_{trial_id}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visualizations saved to {VISUALIZATION_DIR}")

def align_sensors(acc_data, gyro_data, acc_timestamps, gyro_timestamps, target_freq=30.0):
    if len(acc_data) < 3 or len(gyro_data) < 3:
        return None, None, None
    start_time = max(acc_timestamps[0], gyro_timestamps[0])
    end_time = min(acc_timestamps[-1], gyro_timestamps[-1])
    if start_time >= end_time:
        return None, None, None
    common_times = np.linspace(start_time, end_time, int((end_time - start_time) * target_freq / 1000))
    if len(common_times) < 10:
        return None, None, None
    aligned_acc = np.zeros((len(common_times), 3))
    aligned_gyro = np.zeros((len(common_times), 3))
    for axis in range(3):
        try:
            acc_interp = interp1d(acc_timestamps, acc_data[:, axis], bounds_error=False, fill_value="extrapolate")
            aligned_acc[:, axis] = acc_interp(common_times)
            gyro_interp = interp1d(gyro_timestamps, gyro_data[:, axis], bounds_error=False, fill_value="extrapolate")
            aligned_gyro[:, axis] = gyro_interp(common_times)
        except:
            pass
    aligned_acc = bandpass_filter(aligned_acc, lowcut=0.1, highcut=15.0, fs=target_freq)
    aligned_gyro = bandpass_filter(aligned_gyro, lowcut=0.1, highcut=12.0, fs=target_freq)
    return aligned_acc, aligned_gyro, common_times

def selective_sliding_window(data, is_fall=False, window_size=128, stride=32):
    # Implementation as needed ...
    return []

class Processor(ABC):
    def __init__(self, file_path, mode, max_length, label, **kwargs):
        if mode not in ['sliding_window', 'avg_pool']:
            raise ValueError(f"Processing mode {mode} is undefined")
        self.label = label
        self.mode = mode
        self.max_length = max_length
        self.file_path = file_path
        self.input_shape = []
        self.kwargs = kwargs
        self.trial_id = os.path.basename(file_path).split('.')[0] if file_path else None
        self.target_freq = kwargs.get('target_freq', 30.0)
        self.filter_type = kwargs.get('filter_type', None)
        self.visualize = kwargs.get('visualize', False)
        if self.trial_id and self.filter_type and self.trial_id not in FILTER_STATES:
            FILTER_STATES[self.trial_id] = {
                'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
                'timestamp': None,
                'type': self.filter_type
            }
    def set_input_shape(self, sequence):
        self.input_shape = sequence.shape
    def _import_loader(self, file_path):
        file_type = file_path.split('.')[-1]
        if file_type not in ['csv', 'mat']:
            raise ValueError(f"Unsupported file type {file_type}")
        return LOADER_MAP[file_type]
    def load_file(self, file_path):
        loader = self._import_loader(file_path)
        data, timestamps = loader(file_path, **self.kwargs)
        self.set_input_shape(data)
        return data, timestamps
    def avg_pool(self, sequence, window_size=5, max_length=512):
        shape = sequence.shape
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        stride = (sequence.shape[2] // max_length) + 1 if max_length < sequence.shape[2] else 1
        pooled = F.avg_pool1d(sequence, kernel_size=window_size, stride=stride)
        return pooled.squeeze(0).numpy().transpose(1, 0).reshape(-1, *shape[1:])
    def pad_sequence(self, sequence, max_length):
        shape = list(self.input_shape)
        shape[0] = max_length
        pooled = self.avg_pool(sequence, max_length=max_length)
        padded = np.zeros(shape, sequence.dtype)
        actual_length = min(len(pooled), max_length)
        padded[:actual_length] = pooled[:actual_length]
        return padded
    def process(self, data, timestamps=None):
        if self.mode == 'avg_pool':
            return self.pad_sequence(data, self.max_length)
        else:
            is_fall = self.label == 1
            windows = selective_sliding_window(data, is_fall=is_fall,
                                               window_size=self.max_length,
                                               stride=10 if is_fall else 32)
            if self.trial_id and self.trial_id in FILTER_STATES:
                quaternion = FILTER_STATES[self.trial_id]['quaternion']
                processed_windows = []
                for window in windows:
                    window_with_gravity = np.array([add_gravity(sample, quaternion) for sample in window])
                    processed_windows.append(window_with_gravity)
                return processed_windows
            return windows

