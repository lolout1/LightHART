import torch
import numpy as np
from scipy.signal import butter, filtfilt, resample
from scipy.interpolate import interp1d
import logging
import pandas as pd
import os

logger = logging.getLogger(__name__)

class AccelerometerProcessor:
    def __init__(self, 
                 target_length=128,
                 target_hz=50, 
                 lowpass_cutoff=7.5, 
                 filter_order=4,
                 debug=False):
        self.target_length = target_length
        self.target_hz = target_hz
        self.lowpass_cutoff = lowpass_cutoff
        self.filter_order = filter_order
        self.debug = debug
    
    def _butterworth_filter(self, data):
        nyquist = 0.5 * self.target_hz
        normal_cutoff = self.lowpass_cutoff / nyquist
        b, a = butter(self.filter_order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)
    
    def _compute_smv(self, data):
        if self.debug:
            logger.info(f"Computing SMV from data shape: {data.shape}")
            
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        smv = np.sqrt(x**2 + y**2 + z**2).reshape(-1, 1)
        
        if self.debug:
            logger.info(f"SMV shape: {smv.shape}, min: {smv.min():.4f}, max: {smv.max():.4f}")
            
        return np.hstack((smv, data))
    
    def _resample_sequence(self, data, timestamps=None):
        if len(data) == 0:
            return np.array([])
            
        if timestamps is not None and len(timestamps) > 1:
            try:
                if self.debug:
                    logger.info(f"Resampling with timestamps: {len(timestamps)} points")
                    dt = np.diff(timestamps)
                    logger.info(f"Timestamp intervals - min: {dt.min():.4f}ms, max: {dt.max():.4f}ms, mean: {dt.mean():.4f}ms")
                
                t_norm = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
                t_target = np.linspace(0, 1, self.target_length)
                
                resampled = np.zeros((self.target_length, data.shape[1]))
                for i in range(data.shape[1]):
                    f = interp1d(t_norm, data[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
                    resampled[:, i] = f(t_target)
                
                return resampled
            except Exception as e:
                logger.warning(f"Error in timestamp-based resampling: {e}")
                
        if len(data) < self.target_length:
            logger.warning(f"Data length ({len(data)}) < target length ({self.target_length}), skipping")
            return np.array([])
                
        if self.debug:
            logger.info(f"Using uniform resampling from {len(data)} to {self.target_length} points")
        return resample(data, self.target_length)
    
    def _normalize(self, data):
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            channel = data[:, i]
            mean = np.mean(channel)
            std = np.std(channel)
            
            if std < 1e-10:
                std = 1.0
                
            normalized = (channel - mean) / std
            normalized = np.clip(normalized, -5.0, 5.0)
            result[:, i] = normalized
            
        return result
    
    def load_from_csv(self, file_path):
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return np.array([]), None
            
        if os.path.getsize(file_path) == 0:
            logger.warning(f"Empty file: {file_path}")
            return np.array([]), None
            
        try:
            data = []
            timestamps = []
            error_count = 0
            max_errors = 5  # Maximum number of errors before skipping file
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                for i, line in enumerate(lines):
                    try:
                        parts = line.strip().split(',')
                        if len(parts) < 4:  # Need at least timestamp + x,y,z
                            error_count += 1
                            continue
                            
                        try:
                            # Try to parse timestamp
                            timestamp = pd.to_datetime(parts[0]).timestamp() * 1000  # ms
                            values = [float(x) for x in parts[1:4]]  # x, y, z
                            
                            timestamps.append(timestamp)
                            data.append(values)
                        except (ValueError, TypeError):
                            # If first column is not timestamp, try parsing all as floats
                            try:
                                values = [float(x) for x in parts[:3]]  # Try first 3 columns
                                data.append(values)
                            except ValueError:
                                error_count += 1
                    except Exception as e:
                        error_count += 1
                        if error_count > max_errors:
                            logger.warning(f"Too many errors in {file_path}, skipping")
                            return np.array([]), None
            
            if len(data) == 0:
                logger.warning(f"No valid data found in {file_path}")
                return np.array([]), None
                
            data_array = np.array(data)
            
            if len(timestamps) == len(data):
                timestamps_array = np.array(timestamps)
                return data_array, timestamps_array
            else:
                return data_array, None
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return np.array([]), None
    
    def process(self, raw_data, timestamps=None):
        if len(raw_data) == 0:
            return torch.zeros((self.target_length, 4), dtype=torch.float32)
            
        if self.debug:
            logger.info(f"Processing raw data shape: {raw_data.shape}")
            if timestamps is not None:
                logger.info(f"With timestamps shape: {timestamps.shape}")
        
        if raw_data.shape[1] < 3:
            logger.warning(f"Expected at least 3 channels, got {raw_data.shape[1]}")
            return torch.zeros((self.target_length, 4), dtype=torch.float32)
        
        data = raw_data[:, :3] if raw_data.shape[1] > 3 else raw_data
        
        data_with_smv = self._compute_smv(data)
        
        resampled = self._resample_sequence(data_with_smv, timestamps)
        if len(resampled) == 0:
            return torch.zeros((self.target_length, 4), dtype=torch.float32)
            
        filtered = self._butterworth_filter(resampled)
        
        normalized = self._normalize(filtered)
        
        tensor = torch.tensor(normalized, dtype=torch.float32)
        
        if self.debug:
            logger.info(f"Final processed tensor shape: {tensor.shape}")
            
        return tensor

class SlidingWindowProcessor:
    def __init__(self, 
                 window_size=128, 
                 stride=10,
                 processor=None):
        self.window_size = window_size
        self.stride = stride
        self.processor = processor or AccelerometerProcessor(target_length=window_size)
    
    def _extract_windows(self, data, timestamps=None, is_fall=False):
        if len(data) < self.window_size:
            return [], []
        
        windows = []
        window_timestamps = []
        
        if is_fall:
            acc_mag = np.sqrt(np.sum(data[:, :3]**2, axis=1))
            threshold = np.mean(acc_mag) + 1.5 * np.std(acc_mag)
            
            peaks = []
            for i in range(10, len(acc_mag) - 10):
                if (acc_mag[i] > threshold and 
                    acc_mag[i] == np.max(acc_mag[max(0, i-10):min(len(acc_mag), i+10)])):
                    peaks.append(i)
            
            if not peaks:
                peaks = [len(data)//2]
            
            for peak in peaks:
                start = max(0, peak - self.window_size//2)
                end = start + self.window_size
                
                if end > len(data):
                    continue
                
                windows.append(data[start:end])
                if timestamps is not None:
                    window_timestamps.append(timestamps[start:end])
                
        else:
            for start in range(0, len(data) - self.window_size + 1, self.stride):
                windows.append(data[start:start + self.window_size])
                if timestamps is not None:
                    window_timestamps.append(timestamps[start:start + self.window_size])
        
        return windows, window_timestamps
    
    def process_directory(self, directory, is_fall=False):
        processed_tensors = []
        file_names = []
        
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return processed_tensors, file_names
            
        for file_name in os.listdir(directory):
            if not file_name.endswith('.csv'):
                continue
                
            file_path = os.path.join(directory, file_name)
            data, timestamps = self.processor.load_from_csv(file_path)
            
            if len(data) == 0:
                continue
                
            tensors = self.process(data, timestamps, is_fall)
            
            if len(tensors) > 0:
                processed_tensors.append(tensors)
                file_names.append(file_name)
                
        return processed_tensors, file_names
    
    def process(self, data, timestamps=None, is_fall=False):
        windows, window_timestamps = self._extract_windows(data, timestamps, is_fall)
        
        processed_windows = []
        for i, window in enumerate(windows):
            ts = window_timestamps[i] if window_timestamps else None
            processed = self.processor.process(window, ts)
            if processed.shape[0] == self.window_size:
                processed_windows.append(processed)
            
        if processed_windows:
            return torch.stack(processed_windows)
        else:
            return torch.zeros((0, self.window_size, 4), dtype=torch.float32)
