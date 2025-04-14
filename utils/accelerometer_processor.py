# utils/accelerometer_processor.py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfilt
from scipy.interpolate import interp1d
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AccelerometerProcessor')

class AccelerometerProcessor:
    def __init__(self, target_freq=25, window_size=64, window_overlap=0.5):
        self.target_freq = target_freq
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.butter_cutoff = 7.5
        self.butter_order = 4
        self.sos = self._get_butterworth_coeffs()
        
    def _get_butterworth_coeffs(self):
        """Generate Butterworth filter coefficients using second-order sections for better numerical stability"""
        nyquist = 0.5 * self.target_freq
        normal_cutoff = self.butter_cutoff / nyquist
        from scipy.signal import butter
        sos = butter(self.butter_order, normal_cutoff, btype='low', analog=False, output='sos')
        return sos
        
    def load_and_preprocess(self, file_path):
        """Load and preprocess accelerometer data with robust error handling"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Load data
            data = pd.read_csv(file_path, header=None)
            if data.shape[0] < 10:
                raise ValueError(f"Insufficient data rows: {data.shape[0]}")
            
            # Extract timestamps and values
            try:
                timestamps = pd.to_datetime(data.iloc[:, 0], errors='coerce')
            except Exception as e:
                logger.error(f"Error parsing timestamps: {str(e)}")
                raise ValueError(f"Invalid timestamp format in {file_path}")
                
            # Check for NaT values in timestamps
            if timestamps.isna().any():
                bad_indices = timestamps.isna().to_numpy().nonzero()[0]
                logger.warning(f"Found {len(bad_indices)} invalid timestamps at rows {bad_indices[:5]}...")
                # Drop rows with invalid timestamps
                valid_mask = ~timestamps.isna()
                timestamps = timestamps[valid_mask]
                data = data[valid_mask]
                
            # Extract accelerometer values
            acc_values = data.iloc[:, -3:].values
            if acc_values.shape[0] < 10:
                raise ValueError(f"Insufficient valid data rows after cleaning: {acc_values.shape[0]}")
            
            # Fix any duplicate timestamps
            timestamps, acc_values = self._fix_duplicate_timestamps(timestamps, acc_values)
            
            # Check for time reversals or other anomalies
            seconds = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
            if not np.all(np.diff(seconds) >= 0):
                # Fix time reversals
                seconds, acc_values = self._fix_time_reversals(seconds, acc_values)
            
            # Calculate the duration
            duration = seconds[-1] - seconds[0]
            if duration <= 0:
                # This can happen if all timestamps are the same or corrupted
                raise ValueError(f"Invalid duration: {duration}")
                
            # Check for gaps in the data
            time_diffs = np.diff(seconds)
            if np.max(time_diffs) > 5:  # Gap larger than 5 seconds
                # Process data segments separately and keep the longest one
                segments = self._segment_data_by_gaps(seconds, acc_values, gap_threshold=5)
                if not segments:
                    raise ValueError("No valid data segments found after gap detection")
                seconds, acc_values = segments[0]  # Use the longest segment
            
            # Make sure we have enough data for resampling
            if len(seconds) < 5:
                raise ValueError(f"Insufficient samples after preprocessing: {len(seconds)}")
                
            # Resample to uniform time steps
            resampled_acc = self._resample_to_uniform(seconds, acc_values, duration)
            
            # Apply filtering
            filtered_acc = self._apply_butterworth(resampled_acc)
            
            # Normalize
            normalized_acc = self._normalize(filtered_acc)
            
            return normalized_acc
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    def _fix_duplicate_timestamps(self, timestamps, values):
        """Fix duplicate timestamps by adding small offsets"""
        # Identify duplicates
        dupes = timestamps.duplicated()
        
        if dupes.any():
            # Create a new timestamps series
            fixed_timestamps = timestamps.copy()
            dupe_count = dupes.sum()
            
            # For each duplicate, add a microsecond offset
            offset_micro = 1000  # 1ms offset
            last_time = None
            offset_count = 0
            
            for i in range(len(timestamps)):
                if i > 0 and timestamps[i] == last_time:
                    offset_count += 1
                    fixed_timestamps.iloc[i] = last_time + pd.Timedelta(microseconds=offset_micro * offset_count)
                else:
                    last_time = timestamps[i]
                    offset_count = 0
            
            logger.info(f"Fixed {dupe_count} duplicate timestamps")
            return fixed_timestamps, values
        
        return timestamps, values
    
    def _fix_time_reversals(self, seconds, values):
        """Fix time reversals by enforcing monotonically increasing time"""
        # Find indices where time goes backward
        bad_indices = np.where(np.diff(seconds) <= 0)[0]
        
        if len(bad_indices) > 0:
            logger.warning(f"Found {len(bad_indices)} time reversals")
            
            # Create corrected seconds array
            corrected_seconds = seconds.copy()
            
            # Set small positive increments for problematic points
            for idx in bad_indices:
                corrected_seconds[idx + 1] = corrected_seconds[idx] + 0.001  # Add 1ms
            
            # Ensure monotonicity
            for i in range(1, len(corrected_seconds)):
                if corrected_seconds[i] <= corrected_seconds[i-1]:
                    corrected_seconds[i] = corrected_seconds[i-1] + 0.001
            
            return corrected_seconds, values
        
        return seconds, values
    
    def _segment_data_by_gaps(self, seconds, values, gap_threshold=5):
        """Segment data by large gaps and return segments sorted by length (largest first)"""
        # Find large gaps
        large_gaps = np.where(np.diff(seconds) > gap_threshold)[0]
        
        # Create segments
        segments = []
        start_idx = 0
        
        for gap_idx in large_gaps:
            end_idx = gap_idx + 1
            if end_idx - start_idx >= 5:  # Minimum segment size
                segment_seconds = seconds[start_idx:end_idx]
                segment_values = values[start_idx:end_idx]
                segments.append((segment_seconds, segment_values))
            start_idx = end_idx
        
        # Add the last segment
        if len(seconds) - start_idx >= 5:
            segment_seconds = seconds[start_idx:]
            segment_values = values[start_idx:]
            segments.append((segment_seconds, segment_values))
        
        # Sort segments by length (largest first)
        segments.sort(key=lambda x: len(x[0]), reverse=True)
        
        return segments
    
    def _resample_to_uniform(self, seconds, values, duration=None):
        """Resample accelerometer data to uniform time points"""
        if duration is None:
            duration = seconds[-1] - seconds[0]
        
        # Determine number of points based on target frequency
        num_points = max(int(duration * self.target_freq), 10)
        uniform_times = np.linspace(seconds[0], seconds[-1], num_points)
        
        # Create output array
        resampled_values = np.zeros((num_points, values.shape[1]))
        
        # Perform interpolation for each axis
        for i in range(values.shape[1]):
            try:
                interpolator = interp1d(seconds, values[:, i], kind='linear', 
                                       bounds_error=False, fill_value='extrapolate')
                resampled_values[:, i] = interpolator(uniform_times)
            except Exception as e:
                logger.warning(f"Error in interpolation: {str(e)}. Falling back to numpy interp.")
                resampled_values[:, i] = np.interp(uniform_times, seconds, values[:, i])
                
        return resampled_values
    
    def _apply_butterworth(self, data):
        """Apply Butterworth filter for noise reduction"""
        # Use second-order sections (sos) for numerical stability
        filtered_data = sosfilt(self.sos, data, axis=0)
        return filtered_data
    
    def _normalize(self, data):
        """Normalize data using mean and standard deviation"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Avoid division by zero
        std[std < 1e-8] = 1.0
        normalized = (data - mean) / std
        return normalized
    
    def segment_windows(self, data, window_size=None, overlap=None):
        """Segment data into overlapping windows"""
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
        """Enhance features by adding magnitude information"""
        enhanced_windows = []
        for window in windows:
            # Calculate magnitude (resultant)
            magnitude = np.sqrt(np.sum(window**2, axis=1))[:, np.newaxis]
            enhanced_window = np.concatenate([window, magnitude], axis=1)
            enhanced_windows.append(enhanced_window)
        return np.array(enhanced_windows)
