# utils/loader.py - Key changes
import os
from typing import List, Dict, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("loader")

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', fusion_options=None, **kwargs):
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        
        # Default fusion options if none provided
        self.fusion_options = fusion_options or {
            'enabled': False,
            'filter_type': 'madgwick',
            'visualize': False,
            'save_aligned': False
        }
        
        # Create directory for aligned data
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        for dir_name in ["accelerometer", "gyroscope", "quaternion"]:
            os.makedirs(os.path.join(self.aligned_data_dir, dir_name), exist_ok=True)
            
        logger.info(f"Initialized DatasetBuilder: mode={mode}, task={task}, fusion={self.fusion_options}")

    def load_file(self, file_path):
        try:
            # Determine file type and use appropriate loader
            file_type = file_path.split('.')[-1]
            if file_type not in ['csv', 'mat']:
                raise ValueError(f"Unsupported file type {file_type}")
                
            # Use appropriate loader
            if file_type == 'csv':
                # Read CSV data with special handling for different formats
                file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
                
                # Determine number of columns based on data type
                if 'skeleton' in file_path:
                    cols = 96  # Skeleton data has 32 joints × 3 coordinates
                else:
                    # Check if this is a meta sensor file or other inertial data
                    if file_data.shape[1] > 4:
                        # Meta sensor format: epoch, time, elapsed time, x, y, z
                        cols = file_data.shape[1] - 3
                        file_data = file_data.iloc[:, 3:]  # Skip first 3 columns
                    else:
                        cols = 3  # Standard inertial data has 3 axes (x, y, z)
                
                # Extract data, skipping header rows if present
                if file_data.shape[0] > 2:
                    activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
                else:
                    activity_data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
                    
                return activity_data
            else:  # MAT file
                key = self.kwargs.get('key', None)
                if key not in ['d_iner', 'd_skel']:
                    raise ValueError(f"Unsupported {key} for matlab file")
                
                from scipy.io import loadmat
                return loadmat(file_path)[key]
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def align_sensors(self, acc_data, gyro_data, target_freq=30.0):
        """
        Aligns accelerometer and gyroscope data to a common timeline
        
        Args:
            acc_data: Accelerometer data
            gyro_data: Gyroscope data
            target_freq: Target frequency in Hz
            
        Returns:
            Tuple of (aligned_acc, aligned_gyro, timestamps)
        """
        logger.info(f"Aligning sensor data with target frequency {target_freq}Hz")
        
        # Create synthetic timestamps if not available
        # Assuming variable sampling rate for inertial sensors
        n_acc_samples = acc_data.shape[0]
        n_gyro_samples = gyro_data.shape[0]
        
        # Create timestamps assuming 30ms varying intervals for inertial data
        acc_timestamps = np.cumsum(np.random.normal(30, 5, n_acc_samples))
        gyro_timestamps = np.cumsum(np.random.normal(30, 5, n_gyro_samples))
        
        # Start from 0
        acc_timestamps = acc_timestamps - acc_timestamps[0]
        gyro_timestamps = gyro_timestamps - gyro_timestamps[0]
        
        # Determine alignment range
        start_time = max(acc_timestamps[0], gyro_timestamps[0])
        end_time = min(acc_timestamps[-1], gyro_timestamps[-1])
        
        if start_time >= end_time:
            logger.warning("No temporal overlap between sensors")
            return None, None, None
            
        # Create common timeline
        duration = end_time - start_time
        n_samples = int(duration * target_freq / 1000)  # Convert ms to samples
        
        if n_samples < 10:  # Minimum viable data
            logger.warning(f"Overlap too short: {duration:.2f}ms")
            return None, None, None
            
        common_timestamps = np.linspace(start_time, end_time, n_samples)
        
        # Interpolate each axis
        aligned_acc = np.zeros((n_samples, 3))
        aligned_gyro = np.zeros((n_samples, 3))
        
        for axis in range(3):
            # Linear interpolation is more robust for variable rate data
            acc_interp = interp1d(acc_timestamps, acc_data[:, axis], 
                                 bounds_error=False, fill_value="extrapolate")
            aligned_acc[:, axis] = acc_interp(common_timestamps)
            
            gyro_interp = interp1d(gyro_timestamps, gyro_data[:, axis], 
                                  bounds_error=False, fill_value="extrapolate")
            aligned_gyro[:, axis] = gyro_interp(common_timestamps)
        
        logger.info(f"Sensor alignment complete: {n_samples} aligned samples")
        
        return aligned_acc, aligned_gyro, common_timestamps

    def apply_sensor_fusion(self, acc_data, gyro_data, timestamps=None, filter_type='madgwick'):
        """
        Apply sensor fusion to combine accelerometer and gyroscope data
        
        Args:
            acc_data: Accelerometer data
            gyro_data: Gyroscope data
            timestamps: Optional timestamps
            filter_type: Type of fusion filter ('madgwick', 'kalman', 'ekf')
            
        Returns:
            Dictionary with quaternions and linear acceleration
        """
        from utils.imu_fusion import process_imu_data
        
        # Process using IMU fusion
        return process_imu_data(
            acc_data=acc_data,
            gyro_data=gyro_data,
            timestamps=timestamps,
            filter_type=filter_type,
            return_features=True
        )

    def process_peaks(self, data, label):
        """
        Process data using peak detection for fall events
        
        Args:
            data: Sensor data dictionary
            label: Activity label
            
        Returns:
            Processed windows
        """
        # Calculate magnitude for peak detection
        acc_data = data['accelerometer']
        sqrt_sum = np.sqrt(np.sum(acc_data**2, axis=1))
        
        # Adjust parameters based on label
        if label == 1:  # Fall
            peaks, _ = find_peaks(sqrt_sum, height=12, distance=10)
        else:  # Non-fall
            peaks, _ = find_peaks(sqrt_sum, height=10, distance=20)
            
        if len(peaks) == 0:
            # If no peaks found, use point of maximum acceleration
            peaks = [np.argmax(sqrt_sum)]
            
        # Create windows around detected peaks
        windows = []
        for peak in peaks:
            start = max(0, peak - self.max_length // 2)
            end = min(len(acc_data), start + self.max_length)
            
            # Adjust window if needed
            if end - start < self.max_length:
                if start == 0:
                    # Pad at the end
                    window = np.zeros((self.max_length, acc_data.shape[1]))
                    window[:end-start] = acc_data[start:end]
                else:
                    # Adjust start position
                    start = max(0, end - self.max_length)
                    window = acc_data[start:end]
            else:
                window = acc_data[start:end]
                
            windows.append(window)
            
        # Ensure we have at least one window
        if not windows and len(acc_data) > 0:
            # Create a window from the middle of the sequence
            mid = len(acc_data) // 2
            start = max(0, mid - self.max_length // 2)
            end = min(len(acc_data), start + self.max_length)
            
            window = np.zeros((self.max_length, acc_data.shape[1]))
            window[:end-start] = acc_data[start:end]
            windows.append(window)
            
        return windows, peaks

    def process(self, data, label, fuse=False, filter_type='madgwick'):
        """
        Process data using either average pooling or peak-based windows
        
        Args:
            data: Dictionary of sensor data
            label: Activity label
            fuse: Whether to apply sensor fusion
            filter_type: Type of filter to use
            
        Returns:
            Dictionary of processed data
        """
        if self.mode == 'avg_pool':
            # Process full sequence with average pooling
            processed_data = {}
            
            for modality, modality_data in data.items():
                if modality != 'labels' and len(modality_data) > 0:
                    # Pad sequence to fixed length
                    seq_len = modality_data.shape[0]
                    if seq_len > self.max_length:
                        # Apply average pooling to reduce length
                        stride = (seq_len // self.max_length) + 1
                        from torch.nn import functional as F
                        import torch
                        modality_tensor = torch.tensor(modality_data, dtype=torch.float32)
                        modality_tensor = modality_tensor.transpose(0, 1).unsqueeze(0)
                        pooled = F.avg_pool1d(modality_tensor, kernel_size=1, stride=stride)
                        pooled = pooled.squeeze(0).transpose(0, 1).numpy()
                        # Trim to exact length
                        if pooled.shape[0] > self.max_length:
                            pooled = pooled[:self.max_length]
                        processed_data[modality] = pooled
                    else:
                        # Pad sequence
                        padded = np.zeros((self.max_length, modality_data.shape[1]))
                        padded[:seq_len] = modality_data
                        processed_data[modality] = padded
            
            # Add label
            processed_data['labels'] = np.array([label])
            
            # Apply fusion if requested
            if fuse and 'accelerometer' in processed_data and 'gyroscope' in processed_data:
                # Extract timestamps if available
                timestamps = data.get('timestamps', None)
                
                # Process with IMU fusion
                fusion_result = self.apply_sensor_fusion(
                    processed_data['accelerometer'],
                    processed_data['gyroscope'],
                    timestamps,
                    filter_type
                )
                
                # Add fusion results
                processed_data.update(fusion_result)
                
            return processed_data
        else:
            # Use peak detection and create windows
            windows, peaks = self.process_peaks(data, label)
            
            # Process all created windows
            processed_windows = {
                'accelerometer': [],
                'gyroscope': [],
                'quaternion': [],
                'linear_acceleration': [],
                'fusion_features': [],
                'labels': []
            }
            
            for window in windows:
                # Process this window with fusion if requested
                if fuse and 'gyroscope' in data:
                    gyro_window = data['gyroscope'][peaks[0]-self.max_length//2:peaks[0]+self.max_length//2]
                    if len(gyro_window) < self.max_length:
                        gyro_window = np.zeros((self.max_length, data['gyroscope'].shape[1]))
                    
                    # Apply fusion
                    fusion_result = self.apply_sensor_fusion(
                        window, 
                        gyro_window,
                        None,  # No timestamps for window
                        filter_type
                    )
                    
                    # Add results
                    processed_windows['accelerometer'].append(window)
                    processed_windows['gyroscope'].append(gyro_window)
                    processed_windows['quaternion'].append(fusion_result['quaternion'])
                    processed_windows['linear_acceleration'].append(fusion_result['linear_acceleration'])
                    processed_windows['fusion_features'].append(fusion_result['fusion_features'])
                    processed_windows['labels'].append(label)
                else:
                    # Just add the window without fusion
                    processed_windows['accelerometer'].append(window)
                    processed_windows['labels'].append(label)
            
            # Convert lists to numpy arrays
            for key in processed_windows:
                if len(processed_windows[key]) > 0:
                    processed_windows[key] = np.array(processed_windows[key])
            
            return processed_windows

    def _process_trial(self, trial, label, fuse, filter_type):
        """
        Process a single trial with error handling
        
        Args:
            trial: Trial object
            label: Label for this activity
            fuse: Whether to apply fusion
            filter_type: Type of filter to use
            
        Returns:
            Processed trial data
        """
        try:
            # Load data from each modality
            trial_data = {}
            for modality, file_path in trial.files.items():
                modality_data = self.load_file(file_path)
                trial_data[modality] = modality_data
            
            # Align accelerometer and gyroscope if both exist
            if 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                aligned_acc, aligned_gyro, timestamps = self.align_sensors(
                    trial_data['accelerometer'], 
                    trial_data['gyroscope']
                )
                
                if aligned_acc is not None:
                    trial_data['accelerometer'] = aligned_acc
                    trial_data['gyroscope'] = aligned_gyro
                    trial_data['timestamps'] = timestamps
                    
                    # Save aligned data if requested
                    if self.fusion_options.get('save_aligned', False):
                        from utils.imu_fusion import save_aligned_sensor_data
                        save_aligned_sensor_data(
                            trial.subject_id,
                            trial.action_id,
                            trial.sequence_number,
                            aligned_acc,
                            aligned_gyro,
                            None,  # No quaternions yet
                            timestamps
                        )
            
            # Process data with fusion if requested
            processed_data = self.process(trial_data, label, fuse, filter_type)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing trial: {str(e)}")
            return None

    def make_dataset(self, subjects, fuse=False, filter_type='madgwick', visualize=False):
        """
        Create dataset for specified subjects with optional fusion
        
        Args:
            subjects: List of subject IDs to include
            fuse: Whether to apply sensor fusion
            filter_type: Fusion filter type
            visualize: Whether to generate visualizations
        """
        logger.info(f"Making dataset for subjects={subjects}, fusion={fuse}, filter={filter_type}")
        
        self.data = defaultdict(list)
        self.fuse = fuse
        
        # Process each trial with threading for better performance
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=min(8, len(self.dataset.matched_trials))) as executor:
            # Dictionary to track futures
            future_to_trial = {}
            
            # Submit tasks
            for trial in self.dataset.matched_trials:
                if trial.subject_id not in subjects:
                    continue
                
                # Determine label based on task
                if self.task == 'fd':
                    label = int(trial.action_id > 9)  # 1 for fall, 0 for ADL
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1  # Activity recognition
                
                # Submit task
                future = executor.submit(
                    self._process_trial, 
                    trial, 
                    label, 
                    fuse, 
                    filter_type
                )
                future_to_trial[future] = trial
            
            # Collect results with progress tracking
            for future in tqdm(future_to_trial, desc="Processing trials"):
                trial = future_to_trial[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        # Add to dataset
                        for key, value in result.items():
                            if len(value) > 0:
                                self.data[key].append(value)
                except Exception as e:
                    logger.error(f"Error processing trial {trial.subject_id}-{trial.action_id}: {str(e)}")
        
        # Concatenate data
        for key in self.data:
            if all(isinstance(x, np.ndarray) for x in self.data[key]):
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                    logger.info(f"Concatenated {key} data with shape {self.data[key].shape}")
                except:
                    logger.warning(f"Could not concatenate {key} data")
        
    def normalization(self):
        """
        Normalize data for model training
        
        Returns:
            Dictionary of normalized data
        """
        from sklearn.preprocessing import StandardScaler
        
        # Normalize each modality separately
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion', 'linear_acceleration']:
                        # Reshape for standardization
                        original_shape = value.shape
                        reshaped = value.reshape(-1, original_shape[-1])
                        
                        # Standardize
                        normalized = StandardScaler().fit_transform(reshaped)
                        
                        # Reshape back
                        self.data[key] = normalized.reshape(original_shape)
                    elif key == 'fusion_features':
                        # Already extracted features
                        self.data[key] = StandardScaler().fit_transform(value)
                except Exception as e:
                    logger.error(f"Error normalizing {key} data: {str(e)}")
        
        return self.data
