import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

class AccelerometerProcessor:
    def __init__(self, target_freq=50, window_size=128, window_overlap=0.5):
        self.target_freq = target_freq
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.butter_cutoff = 10.0
        
    def load_and_preprocess(self, file_path):
        data = pd.read_csv(file_path, header=None)
        timestamps = pd.to_datetime(data.iloc[:, 0])
        acc_values = data.iloc[:, -3:].values
        resampled_acc = self._resample_to_uniform(timestamps, acc_values)
        filtered_acc = self._apply_butterworth(resampled_acc)
        normalized_acc = self._normalize(filtered_acc)
        return normalized_acc
    
    def _resample_to_uniform(self, timestamps, values):
        seconds = [(t - timestamps[0]).total_seconds() for t in timestamps]
        duration = seconds[-1]
        num_points = int(duration * self.target_freq)
        uniform_times = np.linspace(0, duration, num_points)
        resampled_values = np.zeros((num_points, values.shape[1]))
        for i in range(values.shape[1]):
            interpolator = interp1d(seconds, values[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
            resampled_values[:, i] = interpolator(uniform_times)
        return resampled_values
    
    def _apply_butterworth(self, data, order=4):
        nyquist = 0.5 * self.target_freq
        normal_cutoff = self.butter_cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    
    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
        return normalized
    
    def segment_windows(self, data, window_size=None, overlap=None):
        if window_size is None:
            window_size = self.window_size
        if overlap is None:
            overlap = self.window_overlap
        stride = int(window_size * (1 - overlap))
        num_windows = (data.shape[0] - window_size) // stride + 1
        windows = []
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            windows.append(data[start_idx:end_idx])
        return np.array(windows)
    
    def enhance_features(self, windows):
        enhanced_windows = []
        for window in windows:
            magnitude = np.sqrt(np.sum(window**2, axis=1))[:, np.newaxis]
            enhanced_window = np.concatenate([window, magnitude], axis=1)
            enhanced_windows.append(enhanced_window)
        return np.array(enhanced_windows)
