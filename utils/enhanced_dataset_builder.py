# utils/enhanced_dataset_builder.py (modified)

import os
import numpy as np
import pandas as pd
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from utils.processor.base_quat import (
    parse_watch_csv, 
    create_skeleton_timestamps
)
from utils.imu_fusion_robust import (  # Note we're using our new robust version
    StandardKalmanIMU, 
    ExtendedKalmanIMU, 
    UnscentedKalmanIMU,
    calibrate_filter, 
    extract_orientation_from_skeleton,
    fuse_inertial_modalities
)
from utils.enhanced_alignment import robust_align_modalities

# Configure logging
logger = logging.getLogger("EnhancedDatasetBuilder")

class EnhancedDatasetBuilder:
    """
    Enhanced dataset builder with robust error handling and flexible fusion options.
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
        align_method: str = 'dtw',
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
            align_method: Method for aligning IMU and skeleton
            wrist_idx: Index of the wrist joint
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
        
        logger.info(f"Initialized EnhancedDatasetBuilder: mode={mode}, fusion={imu_fusion}")
    
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
    
    def get_calibration_cache_filename(self) -> str:
        """
        Generate calibration filename.
        
        Returns:
            Calibration cache file path
        """
        return os.path.join(
            self.cache_dir,
            f"kalman_params_{self.imu_fusion}_wrist{self.wrist_idx}.json"
        )
    
    def load_calibration_parameters(self) -> bool:
        """
        Load calibration parameters from cache.
        
        Returns:
            True if parameters were loaded, False otherwise
        """
        path = self.get_calibration_cache_filename()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    params = json.load(f)
                self.fusion_params = params
                logger.info(f"Loaded calibrated parameters: {params}")
                return True
            except Exception as e:
                logger.warning(f"Error loading calibration parameters: {e}")
        return False
    
    def save_calibration_parameters(self) -> bool:
        """
        Save calibration parameters to cache.
        
        Returns:
            True if parameters were saved, False otherwise
        """
        path = self.get_calibration_cache_filename()
        try:
            with open(path, 'w') as f:
                json.dump(self.fusion_params, f)
            logger.info(f"Saved calibration parameters to {path}")
            return True
        except Exception as e:
            logger.warning(f"Error saving calibration parameters: {e}")
            return False
    
    def get_representative_trials(self, subjects, n=5) -> list:
        """
        Get a balanced set of representative trials for calibration.
        
        Args:
            subjects: List of subject IDs to include
            n: Number of trials to select
            
        Returns:
            List of selected trials
        """
        # Get falls and non-falls separately
        fall_trials = []
        nonfall_trials = []
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue
                
            # Must have accelerometer, gyroscope and skeleton
            keys = trial.files.keys()
            if not all(k in keys for k in ['accelerometer', 'gyroscope', 'skeleton']):
                continue
                
            # Check file existence
            if not all(os.path.exists(trial.files[k]) for k in ['accelerometer', 'gyroscope', 'skeleton']):
                continue
                
            # Get label
            label = self._trial_label(trial)
            if label == 1:  # Fall
                fall_trials.append(trial)
            else:  # ADL
                nonfall_trials.append(trial)
        
        # Shuffle trials for randomness
        np.random.shuffle(fall_trials)
        np.random.shuffle(nonfall_trials)
        
        # Select balanced set
        n_fall = min(n//2, len(fall_trials))
        n_nonfall = min(n-n_fall, len(nonfall_trials))
        
        if n_fall == 0 and fall_trials:
            n_fall = min(n, len(fall_trials))
        if n_nonfall == 0 and nonfall_trials:
            n_nonfall = min(n, len(nonfall_trials))
        
        selected = fall_trials[:n_fall] + nonfall_trials[:n_nonfall]
        np.random.shuffle(selected)
        
        logger.info(f"Selected {len(selected)} representative trials for calibration ({n_fall} falls, {n_nonfall} ADLs)")
        return selected
    
    def calibrate_filter_parameters(self, subjects) -> None:
        """
        Calibrate filter parameters using skeleton as ground truth.
        
        Args:
            subjects: List of subject IDs to include
        """
        # Try to load from cache first
        if self.load_calibration_parameters():
            return
            
        logger.info(f"Calibrating {self.imu_fusion} filter parameters...")
        
        # Get representative trials
        trials = self.get_representative_trials(subjects, self.calibration_samples)
        if not trials:
            logger.warning("No suitable trials for calibration. Using default parameters.")
            return
        
        # Process each trial to calibrate
        all_params = []
        for trial in trials:
            try:
                logger.info(f"Calibrating with trial S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}")
                
                # Load data
                accel_path = trial.files['accelerometer']
                gyro_path = trial.files['gyroscope']
                skel_path = trial.files['skeleton']
                
                accel_data = parse_watch_csv(accel_path)
                gyro_data = parse_watch_csv(gyro_path)
                
                # Load skeleton data
                try:
                    df = pd.read_csv(skel_path, header=None).dropna(how='all').fillna(0)
                    skel_data = df.values.astype(np.float32)
                except Exception as e:
                    logger.warning(f"Error loading skeleton data: {e}")
                    continue
                
                if accel_data.shape[0] < 50 or gyro_data.shape[0] < 50 or skel_data.shape[0] < 50:
                    logger.warning("Insufficient data for calibration")
                    continue
                
                # Create skeleton timestamps if needed
                if skel_data.shape[1] == 96:  # No time column
                    skel_timestamps = np.arange(skel_data.shape[0]) / 30.0  # 30 fps
                else:
                    skel_timestamps = skel_data[:, 0]
                    skel_data = skel_data[:, 1:]  # Remove time column
                
                # Align data using robust alignment
                aligned_accel, aligned_skel, aligned_ts = robust_align_modalities(
                    accel_data[:, 1:4],  # Skip time column
                    skel_data,
                    accel_data[:, 0],
                    method=self.align_method,
                    wrist_idx=self.wrist_idx
                )
                
                if aligned_accel.shape[0] < 50 or aligned_skel.shape[0] < 50:
                    logger.warning("Insufficient aligned data for calibration")
                    continue
                
                # Get gyro values aligned to same timestamps
                from scipy.interpolate import interp1d
                
                # Extract gyro values (skip time column)
                gyro_values = gyro_data[:, 1:4]
                gyro_timestamps = gyro_data[:, 0]
                
                # Interpolate gyro to aligned timestamps
                try:
                    gyro_interp = interp1d(
                        gyro_timestamps, 
                        gyro_values,
                        axis=0,
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    aligned_gyro = gyro_interp(aligned_ts)
                except Exception as e:
                    logger.warning(f"Gyro interpolation failed: {e}, using zeros")
                    aligned_gyro = np.zeros_like(aligned_accel)
                
                # Calibrate filter
                _, params = calibrate_filter(
                    aligned_accel,
                    aligned_gyro,
                    aligned_skel,
                    filter_type=self.imu_fusion,
                    timestamps=aligned_ts,
                    wrist_idx=self.wrist_idx
                )
                
                all_params.append(params)
                logger.info(f"Trial parameters: {params}")
                
            except Exception as e:
                logger.warning(f"Error calibrating trial: {e}")
                continue
        
        # Calculate average parameters
        if all_params:
            avg_params = np.mean(all_params, axis=0)
            self.fusion_params = {
                'process_noise': float(avg_params[0]),
                'measurement_noise': float(avg_params[1]),
                'gyro_bias_noise': float(avg_params[2]),
                'drift_correction_weight': self.fusion_params['drift_correction_weight']
            }
            logger.info(f"Calibrated parameters: {self.fusion_params}")
            self.save_calibration_parameters()
        else:
            logger.warning("Calibration failed. Using default parameters.")
    
    def load_and_align_data(self, accel_path, gyro_path, skel_path=None):
        """
        Load accelerometer, gyroscope, and skeleton data and align them.
        
        Args:
            accel_path: Path to accelerometer data file
            gyro_path: Path to gyroscope data file
            skel_path: Path to skeleton data file (optional)
            
        Returns:
            Dictionary with aligned data
        """
        result = {
            'accel_data': None,
            'gyro_data': None,
            'skel_data': None,
            'aligned': False,
            'reference_orientations': None,
            'reference_timestamps': None
        }
        
        # Load accelerometer data (required)
        accel_data = parse_watch_csv(accel_path)
        if accel_data.shape[0] == 0:
            logger.warning(f"Empty accelerometer data: {accel_path}")
            return None
        
        result['accel_data'] = accel_data
        result['accel_timestamps'] = accel_data[:, 0]
        
        # Load gyroscope data (optional)
        if gyro_path and os.path.exists(gyro_path):
            gyro_data = parse_watch_csv(gyro_path)
            if gyro_data.shape[0] > 0:
                result['gyro_data'] = gyro_data
            else:
                logger.warning(f"Empty gyroscope data: {gyro_path}")
        else:
            logger.warning("No gyroscope data provided")
        
        # No skeleton data - return early
        if not skel_path or not os.path.exists(skel_path):
            return result
        
        # Load skeleton data
        try:
            df = pd.read_csv(skel_path, header=None).dropna(how='all').fillna(0)
            skel_array = df.values.astype(np.float32)
            
            # Add time column if not present
            if skel_array.shape[1] == 96:  # No time column
                skel_ts = np.arange(skel_array.shape[0]) / 30.0  # 30 fps
                result['skel_data'] = skel_array
            else:
                skel_ts = skel_array[:, 0]
                result['skel_data'] = skel_array[:, 1:]  # Remove time column
                
            result['skel_timestamps'] = skel_ts
            
            # Try alignment
            if result['skel_data'] is not None and result['skel_data'].shape[0] > 0:
                # Extract accel values (skip time column)
                accel_values = accel_data[:, 1:4]
                accel_timestamps = accel_data[:, 0]
                
                # Align modalities
                aligned_accel, aligned_skel, aligned_ts = robust_align_modalities(
                    accel_values,
                    result['skel_data'],
                    accel_timestamps,
                    method=self.align_method,
                    wrist_idx=self.wrist_idx
                )
                
                if aligned_accel.shape[0] > 10 and aligned_skel.shape[0] > 10:
                    # Update aligned data
                    result['aligned_accel'] = aligned_accel
                    result['aligned_skel'] = aligned_skel
                    result['aligned_timestamps'] = aligned_ts
                    result['aligned'] = True
                    
                    # Extract reference orientations
                    reference_orientations = extract_orientation_from_skeleton(
                        aligned_skel, wrist_idx=self.wrist_idx
                    )
                    
                    result['reference_orientations'] = reference_orientations
                    result['reference_timestamps'] = aligned_ts
                    
                    # Also align gyroscope if available
                    if 'gyro_data' in result and result['gyro_data'] is not None:
                        gyro_data = result['gyro_data']
                        gyro_values = gyro_data[:, 1:4]
                        gyro_timestamps = gyro_data[:, 0]
                        
                        # Interpolate gyro to aligned timestamps
                        try:
                            from scipy.interpolate import interp1d
                            
                            gyro_interp = interp1d(
                                gyro_timestamps,
                                gyro_values,
                                axis=0,
                                bounds_error=False,
                                fill_value="extrapolate"
                            )
                            
                            aligned_gyro = gyro_interp(aligned_ts)
                            result['aligned_gyro'] = aligned_gyro
                        except Exception as e:
                            logger.warning(f"Gyro interpolation failed: {e}, using zeros")
                            result['aligned_gyro'] = np.zeros_like(aligned_accel)
                else:
                    logger.warning("Alignment failed")
        
        except Exception as e:
            logger.warning(f"Error loading or aligning skeleton data: {e}")
            if self.skel_error_strategy == 'drop_trial':
                return None
        
        return result
    
    def process_with_fusion(self, data_dict):
        """
        Process data with IMU fusion.
        
        Args:
            data_dict: Dictionary from load_and_align_data
            
        Returns:
            Dictionary with fused windows
        """
        if data_dict is None or 'accel_data' not in data_dict or data_dict['accel_data'] is None:
            logger.warning("No accelerometer data for fusion")
            return {}
        
        # Get accelerometer data
        accel_data = data_dict['accel_data']
        
        # Get gyroscope data if available
        gyro_data = data_dict.get('gyro_data', None)
        
        # Get reference data if available
        reference_orientations = data_dict.get('reference_orientations', None)
        reference_timestamps = data_dict.get('reference_timestamps', None)
        
        # Process data through IMU fusion
        result = fuse_inertial_modalities(
            {'accelerometer': [accel_data], 'gyroscope': [gyro_data] if gyro_data is not None else []},
            fusion_method=self.imu_fusion,
            use_gyro=gyro_data is not None,
            reference_orientations=reference_orientations,
            reference_timestamps=reference_timestamps
        )
        
        # Create windows
        fused_windows = []
        if 'fused_imu' in result and result['fused_imu']:
            fused_data = result['fused_imu'][0]
            
            # Use sliding windows to create fixed-length segments
            from utils.processor.base import sliding_windows_by_time_fixed
            
            fused_windows = sliding_windows_by_time_fixed(
                fused_data,
                window_size_sec=self.window_size_sec,
                stride_sec=self.stride_sec,
                fixed_count=self.max_length
            )
        
        # Also create skeleton windows if available
        skel_windows = []
        if 'aligned' in data_dict and data_dict['aligned'] and 'aligned_skel' in data_dict:
            aligned_skel = data_dict['aligned_skel']
            aligned_ts = data_dict['aligned_timestamps']
            
            # Add time column
            skel_with_time = np.column_stack([aligned_ts, aligned_skel])
            
            # Create variable-length windows
            from utils.processor.base import sliding_windows_by_time
            
            skel_windows = sliding_windows_by_time(
                skel_with_time,
                window_size_sec=self.window_size_sec,
                stride_sec=self.stride_sec
            )
        
        return {
            'fused_imu': fused_windows,
            'skeleton': skel_windows
        }
    
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
        fusion_result = self.process_with_fusion(alignment_result)
        
        # Check if we have any windows
        if not fusion_result or 'fused_imu' not in fusion_result or not fusion_result['fused_imu']:
            logger.warning(f"No valid windows for trial: {trial.subject_id}-{trial.action_id}-{trial.sequence_number}")
            return None
        
        # Create labels for each window
        imu_windows = fusion_result['fused_imu']
        skel_windows = fusion_result.get('skeleton', [])
        
        # Match counts if both modalities are present
        if skel_windows:
            min_windows = min(len(imu_windows), len(skel_windows))
            imu_windows = imu_windows[:min_windows]
            skel_windows = skel_windows[:min_windows]
            labels = [label] * min_windows
        else:
            labels = [label] * len(imu_windows)
        
        # Assemble result dictionary
        trial_dict = {'fused_imu': imu_windows}
        if skel_windows:
            trial_dict['skeleton'] = skel_windows
        
        # Cache result
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
        
        # Calibrate filter parameters if needed
        if self.do_calibration and self.imu_fusion in ['standard', 'ekf', 'ukf']:
            self.calibrate_filter_parameters(subjects)
        
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
