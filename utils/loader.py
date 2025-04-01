import os
import time
import traceback
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

from utils.imu_fusion import (
    align_sensor_data,
    fixed_size_windows,
    process_imu_data,
    extract_features_from_window,
    benchmark_filters,
    preprocess_all_subjects,
    apply_lowpass_filter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loader")
filter_cache = {}

MAX_THREADS = 40
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(40)

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Load a CSV file and extract the sensor data.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional keyword arguments
        
    Returns:
        Numpy array with the sensor data and timestamps if available
    """
    try:
        try:
            file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        except:
            file_data = pd.read_csv(file_path, index_col=False, header=None, sep=';').dropna().bfill()
        
        # Extract timestamps if available
        timestamps = None
        if 'skeleton' in file_path:
            cols = 96
            values = file_data.iloc[:, :cols].to_numpy(dtype=np.float32)
        else:
            if file_data.shape[1] > 4:
                # Meta sensor format with timestamps
                cols = file_data.shape[1] - 3
                timestamps = file_data.iloc[:, :3].to_numpy(dtype=np.float64)
                values = file_data.iloc[:, 3:].to_numpy(dtype=np.float32)
            else:
                # Watch/phone format with timestamp in first column
                timestamps = file_data.iloc[:, 0].to_numpy(dtype=np.float64)
                values = file_data.iloc[:, 1:].to_numpy(dtype=np.float32)
        
        if file_data.shape[1] < cols and 'skeleton' not in file_path:
            logger.warning(f"File has fewer columns than expected: {file_data.shape[1]} < {cols}")
            missing_cols = cols - file_data.shape[1]
            values_padded = np.zeros((values.shape[0], cols))
            values_padded[:, :values.shape[1]] = values
            values = values_padded
        
        if values.shape[0] > 2:
            values = values[2:]
        
        return values, timestamps
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise

def matloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Load a MAT file and extract the sensor data.
    
    Args:
        file_path: Path to the MAT file
        **kwargs: Additional keyword arguments
        
    Returns:
        Numpy array with the sensor data
    """
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        raise ValueError(f"Unsupported {key} for matlab file")
    try:
        from scipy.io import loadmat
        data = loadmat(file_path)[key]
        return data, None
    except Exception as e:
        logger.error(f"Error loading MAT {file_path}: {str(e)}")
        raise

LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

def load_and_align_data(accelerometer_path: str, gyroscope_path: str, target_rate: float = 30.0) -> Dict[str, np.ndarray]:
    """
    Load and align accelerometer and gyroscope data.
    
    Args:
        accelerometer_path: Path to accelerometer data file
        gyroscope_path: Path to gyroscope data file
        target_rate: Target sampling rate in Hz
        
    Returns:
        Dictionary with aligned sensor data
    """
    try:
        # Load data
        acc_data, acc_timestamps = csvloader(accelerometer_path)
        gyro_data, gyro_timestamps = csvloader(gyroscope_path)
        
        # Align data
        aligned_acc, aligned_gyro, common_timestamps = align_sensor_data(
            acc_data, gyro_data, acc_timestamps, gyro_timestamps, target_rate
        )
        
        return {
            'accelerometer': aligned_acc,
            'gyroscope': aligned_gyro,
            'timestamps': common_timestamps
        }
    except Exception as e:
        logger.error(f"Error loading and aligning data: {str(e)}")
        return {}

def create_windows_from_aligned_data(aligned_data: Dict[str, np.ndarray], window_size: int = 128, 
                                    overlap: float = 0.5, min_windows: int = 1) -> List[Dict[str, np.ndarray]]:
    """
    Create windows from aligned sensor data.
    
    Args:
        aligned_data: Dictionary with aligned sensor data
        window_size: Size of each window
        overlap: Overlap ratio between consecutive windows
        min_windows: Minimum number of windows to create
        
    Returns:
        List of windows, each as a dictionary with sensor data
    """
    try:
        if 'accelerometer' not in aligned_data or 'gyroscope' not in aligned_data:
            return []
        
        acc_data = aligned_data['accelerometer']
        gyro_data = aligned_data['gyroscope']
        timestamps = aligned_data.get('timestamps')
        
        # Create windows
        acc_windows = fixed_size_windows(acc_data, window_size, overlap, min_windows)
        gyro_windows = fixed_size_windows(gyro_data, window_size, overlap, min_windows)
        
        if timestamps is not None:
            timestamps_windows = fixed_size_windows(timestamps.reshape(-1, 1), window_size, overlap, min_windows)
            timestamps_windows = [ts.flatten() for ts in timestamps_windows]
        else:
            timestamps_windows = [None] * len(acc_windows)
        
        # Create window dictionaries
        windows = []
        for i in range(min(len(acc_windows), len(gyro_windows))):
            window = {
                'accelerometer': acc_windows[i],
                'gyroscope': gyro_windows[i]
            }
            if timestamps_windows[i] is not None:
                window['timestamps'] = timestamps_windows[i]
            windows.append(window)
        
        return windows
    except Exception as e:
        logger.error(f"Error creating windows: {str(e)}")
        return []

def process_all_windows(windows: List[Dict[str, np.ndarray]], filter_type: str = 'none', 
                       return_features: bool = False, trial_id: Optional[str] = None) -> List[Dict[str, np.ndarray]]:
    """
    Process all windows with orientation estimation.
    
    Args:
        windows: List of windows, each as a dictionary with sensor data
        filter_type: Type of filter to use
        return_features: Whether to extract additional features
        trial_id: Identifier for the trial
        
    Returns:
        List of processed windows
    """
    processed_windows = []
    
    for i, window in enumerate(windows):
        if 'accelerometer' not in window or 'gyroscope' not in window:
            continue
        
        acc_data = window['accelerometer']
        gyro_data = window['gyroscope']
        timestamps = window.get('timestamps')
        
        # Apply low-pass filter to reduce noise
        acc_data = apply_lowpass_filter(acc_data, cutoff=5.0, fs=30.0)
        gyro_data = apply_lowpass_filter(gyro_data, cutoff=5.0, fs=30.0)
        
        # Process with orientation filter if requested
        if filter_type != 'none':
            window_id = f"{trial_id}_W{i}" if trial_id else str(i)
            result = process_imu_data(
                acc_data, gyro_data, timestamps, 
                filter_type=filter_type,
                return_features=return_features,
                trial_id=window_id,
                reset_filter=(i == 0)  # Reset filter for first window
            )
            
            processed_window = window.copy()
            processed_window.update(result)
        else:
            processed_window = window.copy()
        
        processed_windows.append(processed_window)
    
    return processed_windows

class DatasetBuilder:
    """Class for building datasets from raw sensor data."""
    
    def __init__(self, dataset, mode, max_length, task='fd', fusion_options=None, **kwargs):
        if mode not in ['avg_pool', 'fixed_window']:
            logger.warning(f"Unsupported processing method {mode}, using 'fixed_window' instead")
            mode = 'fixed_window'
        
        self.dataset = dataset
        self.data = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fusion_options = fusion_options or {}
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        
        for dir_name in ["accelerometer", "gyroscope", "skeleton"]:
            os.makedirs(os.path.join(self.aligned_data_dir, dir_name), exist_ok=True)
        
        if fusion_options:
            self.fusion_enabled = fusion_options.get('enabled', False)
            self.filter_type = fusion_options.get('filter_type', 'none')
            self.use_stateful = fusion_options.get('process_per_window', False) == False
            self.target_rate = fusion_options.get('target_rate', 30.0)
            self.window_overlap = fusion_options.get('window_overlap', 0.5)
        else:
            self.fusion_enabled = False
            self.filter_type = 'none'
            self.use_stateful = False
            self.target_rate = 30.0
            self.window_overlap = 0.5
    
    def load_file(self, file_path):
        """Load a file using the appropriate loader."""
        try:
            file_type = file_path.split('.')[-1]
            if file_type not in ['csv', 'mat']:
                raise ValueError(f"Unsupported file type {file_type}")
            
            loader = LOADER_MAP[file_type]
            data, timestamps = loader(file_path, **self.kwargs)
            return data, timestamps
        
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def process_trial(self, trial, label, visualize=False, save_aligned=False):
        """
        Process a single trial.
        
        Args:
            trial: Trial object with file paths
            label: Label for the trial
            visualize: Whether to visualize the data
            save_aligned: Whether to save aligned data
            
        Returns:
            Dictionary with processed data
        """
        try:
            trial_id = f"S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}"
            
            if not ('accelerometer' in trial.files and 'gyroscope' in trial.files):
                logger.warning(f"Missing required modalities for trial {trial_id}")
                return None
            
            # Load data
            acc_data, acc_timestamps = self.load_file(trial.files['accelerometer'])
            gyro_data, gyro_timestamps = self.load_file(trial.files['gyroscope'])
            
            # Align data
            aligned_acc, aligned_gyro, common_timestamps = align_sensor_data(
                acc_data, gyro_data, acc_timestamps, gyro_timestamps, self.target_rate
            )
            
            if len(aligned_acc) == 0 or len(aligned_gyro) == 0:
                logger.warning(f"Failed to align data for trial {trial_id}")
                return None
            
            # Save aligned data if requested
            if save_aligned:
                from utils.imu_fusion import save_aligned_sensor_data
                save_aligned_sensor_data(
                    trial.subject_id,
                    trial.action_id,
                    trial.sequence_number,
                    aligned_acc,
                    aligned_gyro,
                    None,  # No skeleton data
                    common_timestamps,
                    self.aligned_data_dir
                )
            
            # Create windows
            windows = fixed_size_windows(
                aligned_acc, 
                window_size=self.max_length,
                overlap=self.window_overlap,
                min_windows=1
            )
            
            gyro_windows = fixed_size_windows(
                aligned_gyro,
                window_size=self.max_length,
                overlap=self.window_overlap,
                min_windows=1
            )
            
            # Ensure same number of windows
            num_windows = min(len(windows), len(gyro_windows))
            
            # Process windows
            processed_data = {
                'accelerometer': [],
                'gyroscope': [],
                'labels': []
            }
            
            if self.fusion_enabled and self.filter_type != 'none':
                processed_data['quaternion'] = []
            
            for i in range(num_windows):
                acc_window = windows[i]
                gyro_window = gyro_windows[i]
                
                # Apply low-pass filter to reduce noise
                acc_window = apply_lowpass_filter(acc_window, cutoff=5.0, fs=self.target_rate)
                gyro_window = apply_lowpass_filter(gyro_window, cutoff=5.0, fs=self.target_rate)
                
                processed_data['accelerometer'].append(acc_window)
                processed_data['gyroscope'].append(gyro_window)
                processed_data['labels'].append(label)
                
                # Apply orientation filter if enabled
                if self.fusion_enabled and self.filter_type != 'none':
                    window_id = f"{trial_id}_W{i}"
                    result = process_imu_data(
                        acc_window, gyro_window, None,
                        filter_type=self.filter_type,
                        return_features=False,
                        trial_id=window_id,
                        reset_filter=(i == 0)
                    )
                    processed_data['quaternion'].append(result['quaternion'])
            
            # Convert lists to arrays
            for key in processed_data:
                if processed_data[key]:
                    processed_data[key] = np.array(processed_data[key])
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Error processing trial: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def make_dataset(self, subjects: List[int], fuse: bool, filter_type: str = 'none',
                    visualize: bool = False, save_aligned: bool = False):
        """
        Build a dataset from trials for specified subjects.
        
        Args:
            subjects: List of subject identifiers
            fuse: Whether to apply sensor fusion
            filter_type: Type of filter to use
            visualize: Whether to visualize the data
            save_aligned: Whether to save aligned data
        """
        self.data = {}
        self.fusion_enabled = fuse
        self.filter_type = filter_type
        
        if hasattr(self, 'fusion_options'):
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)
        
        # Reset filter cache
        global filter_cache
        filter_cache = {}
        
        # Process all trials
        with ThreadPoolExecutor(max_workers=min(8, len(self.dataset.matched_trials))) as executor:
            future_to_trial = {}
            
            for trial in self.dataset.matched_trials:
                if trial.subject_id not in subjects:
                    continue
                
                # Determine label based on task
                if self.task == 'fd':
                    # Fall detection: 1 for falls (action_id > 9), 0 for ADLs
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    # Age classification
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    # Activity recognition
                    label = trial.action_id - 1
                
                # Process trial in parallel
                future = executor.submit(
                    self.process_trial,
                    trial,
                    label,
                    visualize,
                    save_aligned
                )
                future_to_trial[future] = trial
            
            # Collect results
            count = 0
            processed_count = 0
            skipped_count = 0
            
            for future in tqdm.tqdm(as_completed(future_to_trial), total=len(future_to_trial), desc="Processing trials"):
                trial = future_to_trial[future]
                count += 1
                
                try:
                    trial_data = future.result()
                    if trial_data is not None:
                        # Add trial data to dataset
                        for key, value in trial_data.items():
                            if key not in self.data:
                                self.data[key] = []
                            self.data[key].append(value)
                        processed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error processing trial {trial.subject_id}_{trial.action_id}: {str(e)}")
                    skipped_count += 1
        
        # Concatenate data
        for key in self.data:
            if self.data[key]:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logger.error(f"Error concatenating {key} data: {str(e)}")
        
        logger.info(f"Processed {processed_count} trials, skipped {skipped_count} trials")

    def normalization(self) -> Dict[str, np.ndarray]:
        """
        Apply normalization to the dataset.
        
        Returns:
            Dictionary with normalized data
        """
        from sklearn.preprocessing import StandardScaler
        
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion', 'linear_acceleration'] and len(value.shape) >= 2:
                        num_samples, length = value.shape[:2]
                        reshaped_data = value.reshape(num_samples * length, -1)
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                        self.data[key] = norm_data.reshape(value.shape)
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        self.data[key] = StandardScaler().fit_transform(value)
                except Exception as e:
                    logger.error(f"Error normalizing {key} data: {str(e)}")
        
        return self.data
