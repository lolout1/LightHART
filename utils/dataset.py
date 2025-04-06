import os
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, butter, filtfilt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dataset")

class ModalityFile:
    def __init__(self, subject_id, action_id, sequence_number, file_path):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

class Modality:
    def __init__(self, name):
        self.name = name
        self.files = []
    
    def add_file(self, subject_id, action_id, sequence_number, file_path):
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)

class MatchedTrial:
    def __init__(self, subject_id, action_id, sequence_number):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files = {}
    
    def add_file(self, modality_name, file_path):
        self.files[modality_name] = file_path

class SmartFallMM:
    def __init__(self, root_dir, fusion_options=None):
        self.root_dir = root_dir
        self.age_groups = {"old": {}, "young": {}}
        self.matched_trials = []
        self.selected_sensors = {}
        self.fusion_options = fusion_options or {}
        self.target_sample_rate = self.fusion_options.get('target_sample_rate', 30.0)
        self.window_size = self.fusion_options.get('window_size', 128)
        self.window_overlap = self.fusion_options.get('window_overlap', 0.5)
        self.invalid_files = []
        self.filter_states = {}

    def add_modality(self, age_group, modality_name):
        if age_group not in self.age_groups:
            return
        self.age_groups[age_group][modality_name] = Modality(modality_name)

    def select_sensor(self, modality_name, sensor_name=None):
        if modality_name == "skeleton":
            self.selected_sensors[modality_name] = None
        else:
            if sensor_name is None:
                return
            self.selected_sensors[modality_name] = sensor_name

    def load_files(self):
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    if modality_name in self.selected_sensors:
                        sensor_name = self.selected_sensors[modality_name]
                        modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                    else:
                        continue

                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        try:
                            if file.endswith('.csv'):
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                        except Exception as e:
                            logger.debug(f"Error processing file {file}: {e}")
                            self.invalid_files.append(os.path.join(root, file))

    def match_trials(self):
        trial_dict = {}
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    key = (modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][modality_name] = modality_file.file_path

        required_modalities = list(self.age_groups['young'].keys())
        
        for key, files_dict in trial_dict.items():
            if 'accelerometer' in files_dict and ('gyroscope' in files_dict or 'gyroscope' not in required_modalities):
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)
                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)
                self.matched_trials.append(matched_trial)
                
        logger.info(f"Matched {len(self.matched_trials)} trials across modalities")

    def pipe_line(self, age_group, modalities, sensors):
        for age in age_group: 
            for modality in modalities:
                self.add_modality(age, modality)
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else: 
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)
        self.load_files()
        self.match_trials()
        logger.info(f"Loaded {len(self.matched_trials)} matched trials")

    def parse_csv_file(self, file_path):
        if not os.path.exists(file_path):
            logger.debug(f"File does not exist: {file_path}")
            return None, None
            
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                
            sep = ',' if ',' in first_line else ';'
            skip_rows = 0
            
            # Try to determine if there's a header by checking first line content
            if any(word in first_line.lower() for word in ['time', 'date', 'timestamp']):
                skip_rows = 1
            
            # Read data with the detected parameters
            data_frame = pd.read_csv(file_path, sep=sep, header=None, skiprows=skip_rows)
            
            # Clean data: remove NaN, duplicated timestamps
            data_frame = data_frame.dropna()
            
            # Special handling for skeleton data
            if 'skeleton' in file_path:
                if data_frame.shape[1] < 96:
                    logger.debug(f"Skeleton data incomplete: {file_path} - columns: {data_frame.shape[1]}")
                    padded = np.zeros((data_frame.shape[0], 96))
                    padded[:, :data_frame.shape[1]] = data_frame.values
                    return np.arange(len(padded)), padded
                return np.arange(len(data_frame)), data_frame.values
            
            # For accelerometer/gyroscope data
            if data_frame.empty or data_frame.shape[0] < 3:
                logger.debug(f"Not enough data points in file: {file_path}")
                return None, None
            
            # Parse timestamps and values
            timestamp_col = 0
            value_cols = list(range(1, min(4, data_frame.shape[1])))
            
            # Try to parse timestamp column
            try:
                # Check if timestamp is datetime format
                if isinstance(data_frame.iloc[0, timestamp_col], str) and any(c in data_frame.iloc[0, timestamp_col] for c in ['-', ':', '/']):
                    # Convert timestamps to milliseconds
                    timestamps = pd.to_datetime(data_frame.iloc[:, timestamp_col], errors='coerce')
                    # Make timestamps relative to first timestamp
                    timestamps = (timestamps - timestamps.iloc[0]).dt.total_seconds() * 1000
                else:
                    # Try to convert directly to numeric
                    timestamps = pd.to_numeric(data_frame.iloc[:, timestamp_col], errors='coerce')
            except Exception as e:
                logger.debug(f"Error parsing timestamps in {file_path}: {e}")
                # Use synthetic timestamps as fallback
                timestamps = np.arange(len(data_frame)) * (1000/30)  # 30 Hz
            
            # Extract values from columns 1-3
            try:
                values = data_frame.iloc[:, value_cols].astype(float).values
                # Pad if we have fewer than 3 columns
                if values.shape[1] < 3:
                    padded = np.zeros((values.shape[0], 3))
                    padded[:, :values.shape[1]] = values
                    values = padded
            except Exception as e:
                logger.debug(f"Error extracting values from {file_path}: {e}")
                return None, None
            
            # Final validation
            if len(values) < 3 or np.isnan(values).any() or np.isinf(values).any():
                logger.debug(f"Invalid values in {file_path}")
                return None, None
            
            # Drop duplicate timestamps
            unique_idx = np.where(np.diff(timestamps) > 0)[0]
            unique_idx = np.append(unique_idx, len(timestamps)-1)  # Add last point
            unique_idx = np.insert(unique_idx, 0, 0)  # Add first point
            
            if len(unique_idx) < 3:
                logger.debug(f"Not enough unique timestamps in {file_path}")
                return None, None
                
            return timestamps[unique_idx], values[unique_idx]
            
        except Exception as e:
            logger.debug(f"Exception parsing {file_path}: {e}")
            return None, None

    def align_sensor_data(self, acc_file, gyro_file):
        acc_timestamps, acc_values = self.parse_csv_file(acc_file)
        gyro_timestamps, gyro_values = self.parse_csv_file(gyro_file)
        
        if acc_timestamps is None or gyro_timestamps is None or acc_values is None or gyro_values is None:
            trial_id = os.path.basename(acc_file).split('.')[0] if acc_file else "unknown"
            logger.debug(f"Failed to extract data for trial {trial_id}")
            return None, None, None
            
        # Find common time range
        start_time = max(acc_timestamps[0], gyro_timestamps[0])
        end_time = min(acc_timestamps[-1], gyro_timestamps[-1])
        
        if start_time >= end_time:
            trial_id = os.path.basename(acc_file).split('.')[0]
            logger.debug(f"No common time range for trial {trial_id}")
            return None, None, None
        
        # Create common time base with target sample rate
        common_times = np.linspace(start_time, end_time, int((end_time-start_time)*self.target_sample_rate/1000))
        
        if len(common_times) < 10:  # Ensure we have enough points
            trial_id = os.path.basename(acc_file).split('.')[0]
            logger.debug(f"Too few common time points for trial {trial_id}")
            return None, None, None
        
        # Resample to common time base using linear interpolation
        aligned_acc = np.zeros((len(common_times), 3))
        aligned_gyro = np.zeros((len(common_times), 3))
        
        for axis in range(3):
            if axis < acc_values.shape[1]:
                try:
                    interp_func = interp1d(acc_timestamps, acc_values[:, axis], bounds_error=False, 
                                         fill_value=(acc_values[0, axis], acc_values[-1, axis]), kind='linear')
                    aligned_acc[:, axis] = interp_func(common_times)
                except Exception as e:
                    logger.debug(f"Interpolation error for acc axis {axis}: {e}")
                    idx = np.argmin(np.abs(acc_timestamps[:, np.newaxis] - common_times), axis=0)
                    aligned_acc[:, axis] = acc_values[idx, axis]
        
        for axis in range(3):
            if axis < gyro_values.shape[1]:
                try:
                    interp_func = interp1d(gyro_timestamps, gyro_values[:, axis], bounds_error=False, 
                                         fill_value=(gyro_values[0, axis], gyro_values[-1, axis]), kind='linear')
                    aligned_gyro[:, axis] = interp_func(common_times)
                except Exception as e:
                    logger.debug(f"Interpolation error for gyro axis {axis}: {e}")
                    idx = np.argmin(np.abs(gyro_timestamps[:, np.newaxis] - common_times), axis=0)
                    aligned_gyro[:, axis] = gyro_values[idx, axis]
        
        # Apply bandpass filtering with more lenient parameters
        try:
            nyq = 0.5 * self.target_sample_rate
            low, high = 0.01 / nyq, 15.0 / nyq  # More lenient filtering
            b, a = butter(2, [low, high], btype='band')
            
            for axis in range(3):
                aligned_acc[:, axis] = filtfilt(b, a, aligned_acc[:, axis])
            
            low, high = 0.01 / nyq, 12.0 / nyq
            b, a = butter(2, [low, high], btype='band')
            
            for axis in range(3):
                aligned_gyro[:, axis] = filtfilt(b, a, aligned_gyro[:, axis])
        except Exception as e:
            logger.debug(f"Filtering error: {e}")
        
        return aligned_acc, aligned_gyro, common_times
    
    def create_sliding_windows(self, data, window_size=None, stride=32):
        if window_size is None:
            window_size = self.window_size
            
        if data is None or len(data) < window_size // 2:
            return []
            
        # Use simple sliding windows with stride 32
        windows = []
        for start in range(0, max(1, len(data) - window_size + 1), stride):
            if start + window_size <= len(data):
                windows.append(data[start:start + window_size])
            
        return windows if windows else []

    def load_trial_data(self, trial):
        if not ('accelerometer' in trial.files and 'gyroscope' in trial.files):
            return None
            
        # Check if files exist
        if not os.path.exists(trial.files['accelerometer']) or not os.path.exists(trial.files['gyroscope']):
            return None
            
        trial_id = f"S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}"
        logger.debug(f"Processing trial {trial_id}")
            
        aligned_acc, aligned_gyro, aligned_times = self.align_sensor_data(
            trial.files['accelerometer'],
            trial.files['gyroscope']
        )
        
        if aligned_acc is None or aligned_gyro is None or aligned_times is None:
            logger.debug(f"Failed to align data for trial {trial_id}")
            return None
            
        trial_data = {
            'accelerometer': aligned_acc,
            'gyroscope': aligned_gyro,
            'timestamps': aligned_times
        }
        
        # Only process skeleton if available
        if 'skeleton' in trial.files and os.path.exists(trial.files['skeleton']):
            try:
                skeleton_timestamps, skeleton_values = self.parse_csv_file(trial.files['skeleton'])
                
                if skeleton_timestamps is not None and skeleton_values is not None and len(skeleton_values) > 0:
                    num_frames = skeleton_values.shape[0]
                    
                    if skeleton_values.shape[1] == 96:
                        skeleton_data = skeleton_values.reshape(num_frames, 32, 3)
                        
                        if num_frames > 2:
                            # Adjust skeleton timestamps to match IMU timebase
                            if len(skeleton_timestamps) != num_frames:
                                skeleton_times = np.linspace(0, num_frames/30.0, num_frames) * 1000
                            else:
                                skeleton_times = skeleton_timestamps
                            
                            # Align time bases
                            try:
                                skeleton_times = skeleton_times - skeleton_times[0] + aligned_times[0]
                                
                                resampled_skeleton = np.zeros((len(aligned_times), 32, 3))
                                
                                for joint in range(32):
                                    for coord in range(3):
                                        try:
                                            joint_data = skeleton_data[:, joint, coord]
                                            
                                            interp_func = interp1d(
                                                skeleton_times, joint_data,
                                                bounds_error=False,
                                                fill_value=(joint_data[0], joint_data[-1])
                                            )
                                            
                                            resampled_skeleton[:, joint, coord] = interp_func(aligned_times)
                                        except Exception as e:
                                            logger.debug(f"Error interpolating skeleton joint {joint},{coord}: {e}")
                                            continue
                                
                                trial_data['skeleton'] = resampled_skeleton
                            except Exception as e:
                                logger.debug(f"Error aligning skeleton data for {trial_id}: {e}")
            except Exception as e:
                logger.debug(f"Error processing skeleton data for {trial_id}: {e}")
        
        return trial_data

    def process_trial(self, trial, label, filter_instance=None):
        trial_id = f"S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}"
        trial_data = self.load_trial_data(trial)
        
        if trial_data is None:
            logger.debug(f"No data for trial {trial_id}")
            return None
            
        windows = {
            'accelerometer': [],
            'gyroscope': [],
            'labels': []
        }
        
        if 'skeleton' in trial_data:
            windows['skeleton'] = []
            
        # Create sliding windows for accelerometer data
        acc_windows = self.create_sliding_windows(trial_data['accelerometer'], self.window_size, stride=32)
        
        if not acc_windows:
            logger.debug(f"No windows generated for trial {trial_id}")
            return None
        
        # For each accelerometer window, find corresponding gyroscope and skeleton windows
        for i, acc_window in enumerate(acc_windows):
            window_start = i * 32  # Matching the stride
            window_end = window_start + self.window_size
            
            # Ensure we don't go out of bounds
            if window_end > len(trial_data['accelerometer']):
                window_start = len(trial_data['accelerometer']) - self.window_size
                window_end = len(trial_data['accelerometer'])
            
            # Add accelerometer window
            windows['accelerometer'].append(acc_window)
            
            # Add corresponding gyroscope window
            if window_end <= len(trial_data['gyroscope']):
                gyro_window = trial_data['gyroscope'][window_start:window_end]
                windows['gyroscope'].append(gyro_window)
            else:
                # This should rarely happen due to alignment, but just in case
                logger.debug(f"Gyro window bounds issue in trial {trial_id}, window {i}")
                continue
            
            # Add label for this window
            windows['labels'].append(label)
            
            # Add skeleton window if available
            if 'skeleton' in trial_data:
                if window_end <= len(trial_data['skeleton']):
                    skeleton_window = trial_data['skeleton'][window_start:window_end]
                    windows['skeleton'].append(skeleton_window)
                else:
                    logger.debug(f"Skeleton window bounds issue in trial {trial_id}, window {i}")
                    continue
        
        # Convert lists to numpy arrays
        for key in windows:
            if windows[key]:
                try:
                    windows[key] = np.array(windows[key])
                except Exception as e:
                    logger.debug(f"Error converting {key} to array for trial {trial_id}: {e}")
                    windows[key] = np.array([])
            else:
                windows[key] = np.array([])
        
        # Final validation
        if len(windows['accelerometer']) == 0 or len(windows['gyroscope']) == 0 or len(windows['labels']) == 0:
            logger.debug(f"Empty windows for trial {trial_id}")
            return None
        
        return windows

class DatasetBuilder:
    def __init__(self, dataset, mode='window', max_length=128, task='fd', fusion_options=None, **kwargs):
        self.dataset = dataset
        self.data = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fusion_options = fusion_options or {}
        self.target_sample_rate = fusion_options.get('target_sample_rate', 30.0) if fusion_options else 30.0
        self.window_size = fusion_options.get('window_size', 128) if fusion_options else 128
        self.window_overlap = fusion_options.get('window_overlap', 0.5) if fusion_options else 0.5
        self.load_errors = []
        self.filter_instances = {}
    
    def make_dataset(self, subjects, fuse=True):
        self.data = {
            'accelerometer': [],
            'gyroscope': [],
            'labels': []
        }
        
        valid_trials = 0
        skipped_trials = 0
        filter_type = self.fusion_options.get('filter_type', 'none') if self.fusion_options else 'none'
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue
                
            if self.task == 'fd':
                label = int(trial.action_id > 9)
            elif self.task == 'age':
                label = int(trial.subject_id < 29 or trial.subject_id > 46)
            else:
                label = trial.action_id - 1
                
            try:
                trial_id = f"S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}"
                
                # Get or create filter instance for this trial
                filter_key = f"{trial.subject_id}_{trial.action_id}_{filter_type}"
                filter_instance = self.filter_instances.get(filter_key)
                
                trial_data = self.dataset.process_trial(trial, label, filter_instance)
                
                if trial_data is None:
                    skipped_trials += 1
                    continue
                
                if ('accelerometer' not in trial_data or 
                    'gyroscope' not in trial_data or 
                    'labels' not in trial_data or
                    len(trial_data['accelerometer']) == 0 or 
                    len(trial_data['gyroscope']) == 0 or 
                    len(trial_data['labels']) == 0):
                    
                    skipped_trials += 1
                    continue
                    
                for key in trial_data:
                    if key not in self.data:
                        self.data[key] = []
                        
                    self.data[key].append(trial_data[key])
                    
                valid_trials += 1
                
            except Exception as e:
                logger.debug(f"Error processing trial {trial.subject_id}_{trial.action_id}: {e}")
                skipped_trials += 1
                continue
                
        logger.info(f"Processed {valid_trials} valid trials, skipped {skipped_trials} trials")
                
        for key in self.data:
            if self.data[key]:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logger.debug(f"Error concatenating {key} data: {e}")
                    if key in self.data:
                        del self.data[key]
                    
        return self.data
        
    def normalization(self):
        from sklearn.preprocessing import StandardScaler
        
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion'] and len(value.shape) >= 2:
                        num_samples, seq_length = value.shape[:2]
                        feature_dims = np.prod(value.shape[2:]) if len(value.shape) > 2 else 1
                        
                        reshaped_data = value.reshape(num_samples * seq_length, feature_dims)
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                        
                        self.data[key] = norm_data.reshape(value.shape)
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        self.data[key] = StandardScaler().fit_transform(value)
                except Exception as e:
                    logger.debug(f"Error normalizing {key} data: {e}")
        
        return self.data

def prepare_smartfallmm(arg):
    fusion_options = arg.dataset_args.get('fusion_options', {})
    
    if 'target_sample_rate' not in fusion_options:
        fusion_options['target_sample_rate'] = 30.0
        
    if 'window_size' not in fusion_options:
        fusion_options['window_size'] = 128
        
    if 'window_overlap' not in fusion_options:
        fusion_options['window_overlap'] = 0.5
    
    sm_dataset = SmartFallMM(
        root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'),
        fusion_options=fusion_options
    )
    
    sm_dataset.pipe_line(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    
    builder = DatasetBuilder(
        sm_dataset, 
        mode=arg.dataset_args['mode'],
        max_length=arg.dataset_args['max_length'],
        task=arg.dataset_args['task'],
        fusion_options=fusion_options
    )
    
    return builder

def split_by_subjects(builder, subjects, fuse=True):
    data = builder.make_dataset(subjects, fuse)
    norm_data = builder.normalization()
    
    for key in norm_data:
        if key != 'labels' and key in norm_data and len(norm_data[key]) > 0:
            if len(norm_data[key].shape) == 2:
                samples, features = norm_data[key].shape
                norm_data[key] = norm_data[key].reshape(samples, 1, features)
    
    return norm_data
