import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

class AccelerometerProcessor:
    def __init__(self, target_freq=25, window_size=64, window_overlap=0.5):
        self.target_freq = target_freq
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.butter_cutoff = 7.5
        self.butter_order = 4
        self.b, self.a = self._get_butterworth_coeffs()
        
    def _get_butterworth_coeffs(self):
        nyquist = 0.5 * self.target_freq
        normal_cutoff = self.butter_cutoff / nyquist
        b, a = butter(self.butter_order, normal_cutoff, btype='low', analog=False)
        return b, a
        
    def load_and_preprocess(self, file_path):
        data = pd.read_csv(file_path, header=None)
        if data.shape[0] < 10:
            raise ValueError(f"Insufficient data rows: {data.shape[0]}")
                
        timestamps = pd.to_datetime(data.iloc[:, 0])
        acc_values = data.iloc[:, -3:].values
        
        if len(timestamps) != len(timestamps.unique()):
            timestamps = self._fix_duplicate_timestamps(timestamps)
            
        time_diffs = np.diff(timestamps.astype(np.int64)) / 1e9
        if np.max(time_diffs) > 5:
            gaps = np.where(time_diffs > 5)[0]
            segments = np.split(np.arange(len(timestamps)), gaps + 1)
            largest_segment_idx = np.argmax([len(s) for s in segments])
            
            mask = np.zeros(len(timestamps), dtype=bool)
            mask[segments[largest_segment_idx]] = True
            timestamps = timestamps[mask]
            acc_values = acc_values[mask]
            
        if len(timestamps) < 25:
            raise ValueError(f"Insufficient samples after preprocessing: {len(timestamps)}")
            
        resampled_acc = self._resample_to_uniform(timestamps, acc_values)
        filtered_acc = self._apply_butterworth(resampled_acc)
        normalized_acc = self._normalize(filtered_acc)
        
        return normalized_acc
    
    def _fix_duplicate_timestamps(self, timestamps):
        fixed_timestamps = timestamps.copy()
        seen = {}
        for i, ts in enumerate(timestamps):
            if ts in seen:
                fixed_timestamps[i] = ts + pd.Timedelta(microseconds=1000*(i-seen[ts]))
            seen[ts] = i
        return fixed_timestamps
    
    def _resample_to_uniform(self, timestamps, values):
        seconds = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        
        if not np.all(np.diff(seconds) >= 0):
            bad_indices = np.where(np.diff(seconds) < 0)[0]
            for idx in bad_indices:
                seconds[idx + 1] = seconds[idx] + 0.001
        
        duration = seconds[-1]
        if duration <= 0:
            raise ValueError(f"Invalid duration: {duration}")
            
        num_points = max(int(duration * self.target_freq), 25)
        uniform_times = np.linspace(0, duration, num_points)
        
        resampled_values = np.zeros((num_points, values.shape[1]))
        for i in range(values.shape[1]):
            try:
                interpolator = interp1d(seconds, values[:, i], kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                resampled_values[:, i] = interpolator(uniform_times)
            except Exception:
                resampled_values[:, i] = np.interp(uniform_times, seconds, values[:, i])
                
        return resampled_values
    
    def _apply_butterworth(self, data):
        filtered_data = filtfilt(self.b, self.a, data, axis=0)
        return filtered_data
    
    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < 1e-8] = 1.0
        normalized = (data - mean) / std
        return normalized
    
    def segment_windows(self, data, window_size=None, overlap=None):
        if window_size is None:
            window_size = self.window_size
        if overlap is None:
            overlap = self.window_overlap
            
        stride = int(window_size * (1 - overlap))
        num_windows = max(1, (data.shape[0] - window_size) // stride + 1)
        windows = []
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = min(start_idx + window_size, data.shape[0])
            
            window = data[start_idx:end_idx]
            if window.shape[0] < window_size:
                padding = np.zeros((window_size - window.shape[0], window.shape[1]))
                window = np.vstack([window, padding])
                
            windows.append(window)
            
        return np.array(windows)
    
    def enhance_features(self, windows):
        enhanced_windows = []
        for window in windows:
            magnitude = np.sqrt(np.sum(window**2, axis=1))[:, np.newaxis]
            enhanced_window = np.concatenate([window, magnitude], axis=1)
            enhanced_windows.append(enhanced_window)
        return np.array(enhanced_windows)
