"""
Dataset builder and loader for quaternion-enhanced fall detection.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from utils.processor.base_quat import (
    parse_watch_csv, 
    create_skeleton_timestamps,
    sliding_windows_by_time_fixed, 
    sliding_windows_by_time, 
    robust_align_modalities
)
from utils.imu_fusion import (
    StandardKalmanIMU, 
    ExtendedKalmanIMU, 
    UnscentedKalmanIMU,
    calibrate_filter, 
    extract_orientation_from_skeleton
)

class DatasetBuilderQuat:
    """
    Dataset builder with quaternion-based IMU fusion.
    
    Builds windows of fused data with orientation information from wrist IMU data.
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
        Initialize dataset builder.
        
        Args:
            dataset: SmartFallMM dataset object with matched_trials attribute
            mode: Data processing mode ('variable_time', 'fixed')
            max_length: Maximum sequence length for fixed mode
            task: Task type ('fd' for fall detection)
            window_size_sec: Window size in seconds
            stride_sec: Stride size in seconds
            imu_fusion: Fusion method ('standard', 'ekf', 'ukf')
            align_method: Method for aligning IMU and skeleton ('dtw', 'simple')
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
        
        # Default filter parameters
        self.filter_params = {
            'process_noise': 0.01,
            'measurement_noise': 0.1,
            'gyro_bias_noise': 0.01
        }
        self.calibrated = False
        
        # Calibration settings
        self.do_calibration = kwargs.get('calibrate_filter', True)
        self.calibration_samples = kwargs.get('calibration_samples', 5)
        
        # Caching settings
        self.cache_dir = kwargs.get('cache_dir', './.cache_quat')
        self.use_cache = kwargs.get('use_cache', True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Error handling strategy
        self.skel_error_strategy = kwargs.get('skel_error_strategy', 'drop_trial')
        
        # Init data containers
        self.data = {}
        self.processed_data = {'labels': []}
    
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
        
        return os.path.join(
            self.cache_dir,
            f"s{trial.subject_id}_a{trial.action_id}_t{trial.sequence_number}_sub{sstr}_fuse{fusion}_wrist{wrist}.npz"
        )
    
    def get_calibration_filename(self) -> str:
        """
        Generate calibration parameters filename.
        
        Returns:
            Calibration file path
        """
        return os.path.join(
            self.cache_dir, 
            f"kalman_params_{self.imu_fusion}_wrist{self.wrist_idx}.json"
        )
    
    def load_calibration_params(self) -> bool:
        """
        Load calibration parameters from cache.
        
        Returns:
            True if parameters were loaded successfully, False otherwise
        """
        path = self.get_calibration_filename()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    par = json.load(f)
                self.filter_params = par
                self.calibrated = True
                print(f"Loaded calibration parameters: {par}")
                return True
            except Exception as e:
                print(f"Error loading calibration: {e}")
        return False
    
    def save_calibration_params(self) -> bool:
        """
        Save calibration parameters to cache.
        
        Returns:
            True if parameters were saved successfully, False otherwise
        """
        path = self.get_calibration_filename()
        try:
            with open(path, 'w') as f:
                json.dump(self.filter_params, f)
            print(f"Saved calibration parameters to {path}")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def get_representative_trials(self, subjects, n=5) -> List:
        """
        Get representative trials for calibration.
        
        Args:
            subjects: List of subject IDs to include
            n: Number of trials to select
            
        Returns:
            List of selected trials
        """
        fall, nonfall = [], []
        
        for tri in self.dataset.matched_trials:
            if tri.subject_id not in subjects:
                continue
                
            k = tri.files.keys()
            if ('accelerometer' not in k) or ('gyroscope' not in k) or ('skeleton' not in k):
                continue
                
            lab = self._trial_label(tri)
            if lab == 1:
                fall.append(tri)
            else:
                nonfall.append(tri)
        
        # Shuffle and select a balanced set
        np.random.shuffle(fall)
        np.random.shuffle(nonfall)
        
        nf = min(n//2, len(fall))
        nn = min(n-nf, len(nonfall))
        
        selected = fall[:nf] + nonfall[:nn]
        np.random.shuffle(selected)
        
        return selected
    
    def calibrate_filter_params(self, subjects) -> None:
        """
        Calibrate filter parameters using representative trials.
        
        Args:
            subjects: List of subject IDs to include in calibration
        """
        if self.load_calibration_params():
            return
            
        print("Calibrating filter parameters...")
        reps = self.get_representative_trials(subjects, self.calibration_samples)
        
        if not reps:
            print("No representative trials found, using default parameters")
            return
            
        all_params = []
        for tri in reps:
            try:
                # Load trial data
                accel_fp = tri.files['accelerometer']
                gyro_fp = tri.files['gyroscope']
                skel_fp = tri.files['skeleton']
                
                # Parse data
                a_data = parse_watch_csv(accel_fp)
                g_data = parse_watch_csv(gyro_fp)
                
                # Load skeleton data
                try:
                    df = pd.read_csv(skel_fp, header=None).dropna(how='all').fillna(0)
                    s_data = df.values.astype(np.float32)
                    
                    # Add time if needed
                    if s_data.shape[1] == 96:  # No time column
                        s_data = create_skeleton_timestamps(s_data)
                except:
                    print(f"Error loading skeleton: {skel_fp}")
                    continue
                
                if a_data.shape[0] == 0 or g_data.shape[0] == 0 or s_data.shape[0] == 0:
                    continue
                
                # Calibrate filter
                _, params = calibrate_filter(
                    a_data[:, 1:], 
                    g_data[:, 1:], 
                    s_data, 
                    self.imu_fusion, 
                    a_data[:, 0],
                    wrist_idx=self.wrist_idx
                )
                
                all_params.append(params)
                print(f"Trial {tri.subject_id}-{tri.action_id}-{tri.sequence_number} params: {params}")
                
            except Exception as e:
                print(f"Error calibrating trial: {e}")
                continue
        
        if all_params:
            mean_params = np.mean(all_params, axis=0)
            self.filter_params = {
                'process_noise': float(mean_params[0]),
                'measurement_noise': float(mean_params[1]),
                'gyro_bias_noise': float(mean_params[2])
            }
            self.calibrated = True
            self.save_calibration_params()
            print(f"Calibrated parameters: {self.filter_params}")
    
    def make_dataset(self, subjects, max_workers=12) -> Dict:
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
            self.calibrate_filter_params(subjects)
        
        # Generate tasks for each trial
        tasks = [(t, subjects) for t in self.dataset.matched_trials]
        
        start = time.time()
        print(f"Building dataset: {len(tasks)} trials, fusion={self.imu_fusion}, wrist_idx={self.wrist_idx}")
        
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(self.process_trial, tri, subjects) for (tri, subjects) in tasks]
            
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
        print(f"Dataset built in {end-start:.2f}s: {data_stats}")
        
        return self.processed_data
    
    def process_trial(self, trial, subjects) -> Optional[Tuple[Dict, List]]:
        """
        Process a single trial.
        
        Args:
            trial: Trial object
            subjects: List of subject IDs
            
        Returns:
            Tuple of (trial_data_dict, labels) or None if trial should be skipped
        """
        # Skip if subject not in the specified list
        if trial.subject_id not in subjects:
            return None
            
        # Check for required accelerometer data
        if 'accelerometer' not in trial.files:
            return None
        
        # Check cache
        cache_file = self.get_cache_filename(trial, subjects)
        if self.use_cache and os.path.exists(cache_file):
            try:
                loaded = np.load(cache_file, allow_pickle=True)
                return (loaded['trial_dict'][()], loaded['labels'].tolist())
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Get trial label
        label = self._trial_label(trial)
        
        # Load accelerometer data
        a_fp = trial.files['accelerometer']
        g_fp = trial.files.get('gyroscope', '')
        s_fp = trial.files.get('skeleton', '')
        
        a_data = parse_watch_csv(a_fp)
        if a_data.shape[0] == 0:
            return None
        
        # Load gyroscope data if available
        has_gyro = False
        g_data = np.zeros((0, 4), dtype=np.float32)
        if g_fp and os.path.exists(g_fp):
            tmp = parse_watch_csv(g_fp)
            if tmp.shape[0] > 0:
                has_gyro = True
                g_data = tmp
        
        # Function to fuse accelerometer and gyroscope data
        def do_fusion(acc, gyr):
            """Fuse accelerometer and gyroscope data using selected filter."""
            from scipy.interpolate import interp1d
            
            a_ts = acc[:, 0]  # Timestamps
            a_xyz = acc[:, 1:]  # Values
            
            g_ts = gyr[:, 0]
            g_xyz = gyr[:, 1:]
            
            # Interpolate gyroscope data to match accelerometer timestamps
            if not np.array_equal(a_ts, g_ts):
                f = interp1d(g_ts, g_xyz, axis=0, bounds_error=False, fill_value='extrapolate')
                g_aligned = f(a_ts)
            else:
                g_aligned = g_xyz
            
            # Select filter type
            if self.imu_fusion == 'standard':
                fil = StandardKalmanIMU(
                    dt=self.kwargs.get('dt', 1/30),
                    process_noise=self.filter_params['process_noise'],
                    measurement_noise=self.filter_params['measurement_noise'],
                    gyro_bias_noise=self.filter_params['gyro_bias_noise']
                )
            elif self.imu_fusion == 'ekf':
                fil = ExtendedKalmanIMU(
                    dt=self.kwargs.get('dt', 1/30),
                    process_noise=self.filter_params['process_noise'],
                    measurement_noise=self.filter_params['measurement_noise'],
                    gyro_bias_noise=self.filter_params['gyro_bias_noise']
                )
            else:  # ukf
                fil = UnscentedKalmanIMU(
                    dt=self.kwargs.get('dt', 1/30),
                    process_noise=self.filter_params['process_noise'],
                    measurement_noise=self.filter_params['measurement_noise'],
                    gyro_bias_noise=self.filter_params['gyro_bias_noise']
                )
            
            # Process the sequence and get fused data
            fused = fil.process_sequence(a_xyz, g_aligned, timestamps=a_ts)
            
            # Add time column back
            return np.column_stack([a_ts, fused])
        
        # Use original accelerometer data if no gyroscope data
        final_imu = a_data
        wname = 'accelerometer'
        
        # Fuse if gyroscope data is available
        if has_gyro:
            try:
                fused_imu = do_fusion(a_data, g_data)
                if fused_imu.shape[0] > 10:
                    final_imu = fused_imu
                    wname = 'fused_imu'
            except Exception as e:
                print(f"Fusion error: {e}")
        
        # Load skeleton data if available
        has_skel = False
        s_data = np.zeros((0, 97), dtype=np.float32)  # Time + 96 features
        
        if s_fp and os.path.exists(s_fp):
            try:
                # Load skeleton data
                df = pd.read_csv(s_fp, header=None).dropna(how='all').fillna(0)
                arr = df.values.astype(np.float32)
                
                # Add time column if missing
                if arr.shape[1] == 96:  # No time column
                    t = np.arange(arr.shape[0]) / 30.0  # Assume 30 fps
                    t = t.reshape(-1, 1)
                    s_data = np.hstack([t, arr])
                else:
                    s_data = arr
                
                if s_data.shape[0] > 0:
                    has_skel = True
            except Exception as e:
                print(f"Error loading skeleton: {e}")
        
        # Align skeleton and IMU data if both are available
        if has_skel:
            try:
                ali_imu, ali_skel, nts = robust_align_modalities(
                    final_imu, s_data, final_imu[:, 0], 
                    method=self.align_method,
                    wrist_idx=self.wrist_idx
                )
                
                if ali_imu.shape[0] > 0 and ali_skel.shape[0] > 0:
                    final_imu = ali_imu
                    s_data = ali_skel
                else:
                    if self.skel_error_strategy == 'drop_trial':
                        return None
                    has_skel = False
            except Exception as e:
                print(f"Alignment error: {e}")
                if self.skel_error_strategy == 'drop_trial':
                    return None
                has_skel = False
        
        # Create windows
        trial_dict = {}
        
        def do_window(data, is_skel=False):
            """Create windows from data sequence."""
            if is_skel:
                # Variable-length windows for skeleton
                return sliding_windows_by_time(
                    data, 
                    window_size_sec=self.window_size_sec,
                    stride_sec=self.stride_sec
                )
            else:
                # Fixed-length windows for IMU
                return sliding_windows_by_time_fixed(
                    data,
                    window_size_sec=self.window_size_sec,
                    stride_sec=self.stride_sec,
                    fixed_count=128
                )
        
        # Create IMU windows
        imu_wins = do_window(final_imu, is_skel=False)
        if len(imu_wins) == 0:
            return None
            
        trial_dict[wname] = imu_wins
        labs = [label] * len(imu_wins)
        
        # Create skeleton windows if available
        if has_skel:
            sk_wins = do_window(s_data, is_skel=True)
            if len(sk_wins) > 0:
                # Limit to minimum of IMU and skeleton windows
                L = min(len(imu_wins), len(sk_wins))
                trial_dict[wname] = imu_wins[:L]
                trial_dict['skeleton'] = sk_wins[:L]
                labs = [label] * L
            else:
                if self.skel_error_strategy == 'drop_trial':
                    return None
        
        # Save to cache
        if self.use_cache:
            try:
                np.savez_compressed(
                    cache_file, 
                    trial_dict=trial_dict, 
                    labels=np.array(labs)
                )
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        return (trial_dict, labs)
