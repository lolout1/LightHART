# utils/enhanced_dataset_builder.py

import os
import numpy as np
import pandas as pd
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.processor.base_quat import (
    parse_watch_csv, 
    create_skeleton_timestamps
)
from utils.enhanced_imu_fusion import (
    EnhancedExtendedKalmanIMU
)
from utils.enhanced_alignment import (
    enhanced_align_modalities,
    extract_orientation_from_skeleton
)

logger = logging.getLogger("EnhancedDatasetBuilder")

class EnhancedDatasetBuilder:
    """
    Enhanced dataset builder that handles asynchronous modalities and drift correction.
    Optimized for fall detection with SmartFallMM dataset.
    """
    
    def __init__(
        self,
        dataset,
        mode: str = 'variable_time',
        max_length: int = 128,
        task: str = 'fd',
        window_size_sec: float = 4.0,
        stride_sec: float = 1.0,
        imu_fusion: str = 'ekf',
        align_method: str = 'enhanced',
        wrist_idx: int = 9,  # Index of the wrist joint
        **kwargs
    ):
        """
        Initialize enhanced dataset builder.
        
        Args:
            dataset: SmartFallMM dataset object with matched_trials attribute
            mode: Data processing mode ('variable_time', 'fixed')
            max_length: Maximum sequence length for fixed mode
            task: Task type ('fd' for fall detection)
            window_size_sec: Window size in seconds
            stride_sec: Stride size in seconds
            imu_fusion: Fusion method ('standard', 'ekf', 'ukf')
            align_method: Method for aligning IMU and skeleton ('enhanced', 'dtw', 'basic')
            wrist_idx: Index of the wrist joint (default: 9)
            **kwargs: Additional arguments
        """
        self.dataset = dataset
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.imu_fusion = imu_fusion
        self.align_method = align_method
        self.wrist_idx = wrist_idx
        self.kwargs = kwargs
        
        # Default fusion parameters
        self.fusion_params = {
            'process_noise': 0.01,
            'measurement_noise': 0.1,
            'gyro_bias_noise': 0.01,
            'drift_correction_weight': kwargs.get('drift_correction_weight', 0.3)
        }
        
        # Fall detection specific settings
        self.fall_specific_features = kwargs.get('fall_specific_features', True)
        
        # Calibration settings
        self.do_calibration = kwargs.get('calibrate_filter', True)
        self.calibration_samples = kwargs.get('calibration_samples', 5)
        
        # Caching settings
        self.cache_dir = kwargs.get('cache_dir', './.cache_enhanced')
        self.use_cache = kwargs.get('use_cache', True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Error handling strategy
        self.skel_error_strategy = kwargs.get('skel_error_strategy', 'best_effort')
        
        # Init data containers
        self.data = {}
        self.processed_data = {'labels': []}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def _trial_label(self, trial) -> int:
        """
        Get label for a trial.
        
        Args:
            trial: Trial object with action_id attribute
            
        Returns:
            Label as integer (0 for ADL, 1 for fall in fd task)
        """
        if self.task == 'fd':
            return int(trial.action_id > 9)  # 1 for fall, 0 for ADL
        elif self.task == 'age':
            return int(trial.subject_id < 29 or trial.subject_id > 46)
        else:
            return trial.action_id - 1  # Activity classification (multi-class)
    
    def get_cache_filename(self, trial, subjects) -> str:
        """
        Generate cache filename for a trial.
        
        Args:
            trial: Trial object
            subjects: List of subject IDs
            
        Returns:
            Cache file path
        """
        sstr = '_'.join(map(str, subjects))
        fusion = self.imu_fusion
        wrist = self.wrist_idx
        align = self.align_method
        
        return os.path.join(
            self.cache_dir,
            f"enhanced_s{trial.subject_id}_a{trial.action_id}_t{trial.sequence_number}_sub{sstr}_fuse{fusion}_align{align}.npz"
        )
    
    def load_and_align_data(self, accel_path, gyro_path, skel_path=None):
        """
        Load accelerometer, gyroscope, and skeleton data and align them.
        
        Args:
            accel_path: Path to accelerometer data file
            gyro_path: Path to gyroscope data file
            skel_path: Path to skeleton data file (optional)
            
        Returns:
            Dictionary with aligned data and reference orientations
        """
        # Load accelerometer data
        accel_data = parse_watch_csv(accel_path)
        if accel_data.shape[0] == 0:
            logger.warning(f"Empty accelerometer data: {accel_path}")
            return None
        
        # Load gyroscope data
        gyro_data = parse_watch_csv(gyro_path) if gyro_path else np.zeros((0, 4))
        if gyro_data.shape[0] == 0:
            logger.warning(f"Empty gyroscope data: {gyro_path}")
            # Can proceed with just accelerometer if needed
        
        # Load skeleton data if provided
        skel_data = None
        if skel_path:
            try:
                df = pd.read_csv(skel_path, header=None).dropna(how='all').fillna(0)
                skel_array = df.values.astype(np.float32)
                
                # Skeleton data doesn't have time column
                skel_data = skel_array
            except Exception as e:
                logger.error(f"Error loading skeleton data: {e}")
                if self.skel_error_strategy == 'drop_trial':
                    return None
        
        # Extract timestamps from accelerometer
        accel_timestamps = accel_data[:, 0]
        
        # Align gyroscope with accelerometer if both exist
        if gyro_data.shape[0] > 0:
            # Interpolate gyroscope to match accelerometer timestamps
            from scipy.interpolate import interp1d
            
            # Check if timestamps differ
            gyro_timestamps = gyro_data[:, 0]
            
            if not np.array_equal(accel_timestamps, gyro_timestamps):
                try:
                    gyro_interp = interp1d(
                        gyro_timestamps,
                        gyro_data[:, 1:],
                        axis=0,
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    
                    # Get common time range
                    t_min = max(accel_timestamps[0], gyro_timestamps[0])
                    t_max = min(accel_timestamps[-1], gyro_timestamps[-1])
                    
                    # Filter to common range
                    valid_mask = (accel_timestamps >= t_min) & (accel_timestamps <= t_max)
                    valid_accel_data = accel_data[valid_mask]
                    valid_timestamps = accel_timestamps[valid_mask]
                    
                    # Interpolate gyroscope
                    valid_gyro_values = gyro_interp(valid_timestamps)
                    valid_gyro_data = np.column_stack([valid_timestamps, valid_gyro_values])
                    
                    # Update data
                    accel_data = valid_accel_data
                    gyro_data = valid_gyro_data
                    accel_timestamps = valid_timestamps
                except Exception as e:
                    logger.error(f"Error interpolating gyroscope data: {e}")
                    # Proceed with unaligned data if needed
        
        # If no skeleton data, return IMU-only
        if skel_data is None or skel_data.shape[0] == 0:
            return {
                'accel_data': accel_data,
                'gyro_data': gyro_data,
                'accel_timestamps': accel_timestamps,
                'skel_data': None,
                'aligned': False,
                'reference_orientations': None,
                'reference_timestamps': None
            }
        
        # Align skeleton and IMU data
        try:
            # Generate skeleton timestamps (at 30 fps)
            skel_timestamps = create_skeleton_timestamps(skel_data, fps=30.0)
            
            # Align modalities using enhanced method
            alignment_result = enhanced_align_modalities(
                imu_data=accel_data[:, 1:4],  # Use acceleration values only
                skel_data=skel_data,
                imu_timestamps=accel_timestamps,
                skel_fps=30.0,
                wrist_idx=self.wrist_idx,
                return_all=True
            )
            
            if alignment_result['success']:
                # Extract aligned data
                aligned_imu = alignment_result['aligned_imu']
                aligned_skel = alignment_result['aligned_skel']
                aligned_timestamps = alignment_result['aligned_timestamps']
                
                # Get reference orientations for drift correction
                ref_timestamps = alignment_result['reference_timestamps']
                ref_orientations = alignment_result['reference_orientations']
                
                # For each aligned IMU sample, find corresponding gyro sample
                if gyro_data.shape[0] > 0:
                    aligned_gyro = np.zeros((len(aligned_timestamps), 3))
                    
                    for i, ts in enumerate(aligned_timestamps):
                        # Find closest gyro timestamp
                        gyro_idx = np.argmin(np.abs(gyro_data[:, 0] - ts))
                        aligned_gyro[i] = gyro_data[gyro_idx, 1:4]
                    
                    # Create gyro data with aligned timestamps
                    aligned_gyro_data = np.column_stack([aligned_timestamps, aligned_gyro])
                else:
                    aligned_gyro_data = None
                
                # Recreate aligned accelerometer data with timestamps
                aligned_accel_data = np.column_stack([aligned_timestamps, aligned_imu])
                
                return {
                    'accel_data': aligned_accel_data,
                    'gyro_data': aligned_gyro_data,
                    'accel_timestamps': aligned_timestamps,
                    'skel_data': aligned_skel,
                    'aligned': True,
                    'reference_orientations': ref_orientations,
                    'reference_timestamps': ref_timestamps
                }
            else:
                logger.warning("Enhanced alignment failed, using original data")
                return {
                    'accel_data': accel_data,
                    'gyro_data': gyro_data,
                    'accel_timestamps': accel_timestamps,
                    'skel_data': skel_data,
                    'aligned': False,
                    'reference_orientations': None,
                    'reference_timestamps': None
                }
        except Exception as e:
            logger.error(f"Error in modality alignment: {e}")
            return {
                'accel_data': accel_data,
                'gyro_data': gyro_data,
                'accel_timestamps': accel_timestamps,
                'skel_data': skel_data,
                'aligned': False,
                'reference_orientations': None,
                'reference_timestamps': None
            }
    
    def process_with_fusion(self, accel_data, gyro_data, accel_timestamps=None, 
                          reference_orientations=None, reference_timestamps=None):
        """
        Process IMU data with fusion and drift correction.
        
        Args:
            accel_data: Accelerometer data with timestamps
            gyro_data: Gyroscope data with timestamps
            accel_timestamps: Timestamps for accelerometer
            reference_orientations: Optional reference orientations from skeleton
            reference_timestamps: Optional timestamps for reference orientations
            
        Returns:
            Fused data with orientation features
        """
        # Extract timestamps if not provided
        if accel_timestamps is None:
            accel_timestamps = accel_data[:, 0]
        
        # Extract values
        accel_values = accel_data[:, 1:4]
        
        if gyro_data is not None and gyro_data.shape[0] > 0:
            gyro_values = gyro_data[:, 1:4]
        else:
            # Create zero gyro values if not available
            gyro_values = np.zeros_like(accel_values)
        
        # Create Kalman filter - optimized for fall detection
        kalman_filter = EnhancedExtendedKalmanIMU(
            dt=1/30.0,  # Default sample rate
            process_noise=self.fusion_params['process_noise'],
            measurement_noise=self.fusion_params['measurement_noise'],
            gyro_bias_noise=self.fusion_params['gyro_bias_noise'],
            drift_correction_weight=self.fusion_params['drift_correction_weight']
        )
        
        # Set reference data for drift correction if available
        if reference_orientations is not None and reference_timestamps is not None:
            kalman_filter.set_reference_data(reference_timestamps, reference_orientations)
        
        # Process the sequence with drift correction
        fused_data = kalman_filter.process_sequence(accel_values, gyro_values, accel_timestamps)
        
        # Add timestamp column
        return np.column_stack([accel_timestamps, fused_data])
    
    def create_windows(self, data, is_skeleton=False):
        """
        Create windows from data.
        
        Args:
            data: Array with timestamps in first column
            is_skeleton: Whether the data is skeleton
            
        Returns:
            List of windows
        """
        if data is None or data.shape[0] == 0:
            return []
        
        # Get timestamps
        timestamps = data[:, 0]
        min_t = timestamps[0]
        max_t = timestamps[-1]
        
        windows = []
        t_start = min_t
        
        while t_start + self.window_size_sec <= max_t + 1e-9:
            # Extract data within time window
            mask = (timestamps >= t_start) & (timestamps < t_start + self.window_size_sec)
            window_data = data[mask]
            
            if is_skeleton:
                # For skeleton, keep variable length
                if window_data.shape[0] >= 5:  # Minimum sample threshold
                    windows.append(window_data)
            else:
                # For IMU, resample to fixed length
                if window_data.shape[0] >= 5:
                    # Resample to fixed length
                    if window_data.shape[0] != self.max_length:
                        indices = np.linspace(0, window_data.shape[0] - 1, self.max_length).astype(int)
                        resampled = window_data[indices]
                        windows.append(resampled)
                    else:
                        windows.append(window_data)
            
            t_start += self.stride_sec
        
        return windows
    
    def process_trial(self, trial, subjects=None):
        """
        Process a single trial.
        
        Args:
            trial: Trial object
            subjects: List of subject IDs (for caching)
            
        Returns:
            Tuple of (trial_data_dict, labels) or None if trial should be skipped
        """
        # Skip if subject not in the specified list
        if subjects and trial.subject_id not in subjects:
            return None
            
        # Check for required accelerometer data
        if 'accelerometer' not in trial.files:
            logger.warning(f"No accelerometer data for trial: {trial.subject_id}-{trial.action_id}-{trial.sequence_number}")
            return None
        
        # Get trial label
        label = self._trial_label(trial)
        
        # Check cache
        cache_file = self.get_cache_filename(trial, subjects)
        if self.use_cache and os.path.exists(cache_file):
            try:
                loaded = np.load(cache_file, allow_pickle=True)
                return (loaded['trial_dict'][()], loaded['labels'].tolist())
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        
        # Load and align data
        a_path = trial.files['accelerometer']
        g_path = trial.files.get('gyroscope', None)
        s_path = trial.files.get('skeleton', None)
        
        alignment_result = self.load_and_align_data(a_path, g_path, s_path)
        
        if alignment_result is None:
            logger.warning(f"Alignment failed for trial: {trial.subject_id}-{trial.action_id}-{trial.sequence_number}")
            return None
        
        # Process with fusion
        fused_data = self.process_with_fusion(
            alignment_result['accel_data'],
            alignment_result['gyro_data'],
            alignment_result['accel_timestamps'],
            alignment_result['reference_orientations'],
            alignment_result['reference_timestamps']
        )
        
        # Create windows
        imu_windows = self.create_windows(fused_data, is_skeleton=False)
        
        if not imu_windows:
            logger.warning(f"No valid windows for trial: {trial.subject_id}-{trial.action_id}-{trial.sequence_number}")
            return None
        
        # Create skeleton windows if available and aligned
        trial_dict = {'fused_imu': imu_windows}
        labels = [label] * len(imu_windows)
        
        if alignment_result['skel_data'] is not None and alignment_result['aligned']:
            # Add time column to skeleton data
            skel_with_time = np.column_stack([
                alignment_result['accel_timestamps'],
                alignment_result['skel_data']
            ])
            
            skel_windows = self.create_windows(skel_with_time, is_skeleton=True)
            
            if skel_windows:
                # Match number of windows
                min_windows = min(len(imu_windows), len(skel_windows))
                trial_dict['fused_imu'] = imu_windows[:min_windows]
                trial_dict['skeleton'] = skel_windows[:min_windows]
                labels = [label] * min_windows
        
        # Save to cache
        if self.use_cache:
            try:
                np.savez_compressed(
                    cache_file, 
                    trial_dict=trial_dict, 
                    labels=np.array(labels)
                )
            except Exception as e:
                logger.warning(f"Error saving cache: {e}")
        
        return (trial_dict, labels)
    
    def make_dataset(self, subjects, max_workers=4):
        """
        Build dataset for specified subjects.
        
        Args:
            subjects: List of subject IDs to include
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary with processed data
        """
        self.data.clear()
        self.processed_data = {'labels': []}
        
        # Generate tasks for each trial
        tasks = [(t, subjects) for t in self.dataset.matched_trials]
        
        start = time.time()
        logger.info(f"Building enhanced dataset: {len(tasks)} trials, fusion={self.imu_fusion}")
        
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(self.process_trial, tri, subjects) for tri in self.dataset.matched_trials]
            
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                    
                trial_dict, labs = res
                
                # Merge trial results into processed data
                for k in trial_dict:
                    if k not in self.processed_data:
                        self.processed_data[k] = []
                    self.processed_data[k].extend(trial_dict[k])
                    
                self.processed_data['labels'].extend(labs)
        
        end = time.time()
        
        # Print statistics
        data_stats = {k: len(v) for k, v in self.processed_data.items()}
        logger.info(f"Dataset built in {end-start:.2f}s: {data_stats}")
        
        return self.processed_data
