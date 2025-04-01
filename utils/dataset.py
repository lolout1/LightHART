from typing import List, Dict, Tuple, Union, Optional, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import logging
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

logger = logging.getLogger("dataset")

class ModalityFile: 
    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None: 
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

class Modality:
    def __init__(self, name: str) -> None:
        self.name = name 
        self.files: List[ModalityFile] = []
    
    def add_file(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None: 
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)

class MatchedTrial: 
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path

class SmartFallMM:
    def __init__(self, root_dir: str, fusion_options: Optional[Dict] = None) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {
            "old": {},
            "young": {}
        }
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, str] = {}
        self.fusion_options = fusion_options or {}
        self.target_sample_rate = fusion_options.get('target_sample_rate', 30.0) if fusion_options else 30.0
        self.window_size = fusion_options.get('window_size', 128) if fusion_options else 128
        self.window_overlap = fusion_options.get('window_overlap', 0.5) if fusion_options else 0.5

    def add_modality(self, age_group: str, modality_name: str) -> None:
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}. Expected 'old' or 'young'.")
        self.age_groups[age_group][modality_name] = Modality(modality_name)

    def select_sensor(self, modality_name: str, sensor_name: str = None) -> None:
        if modality_name == "skeleton":
            self.selected_sensors[modality_name] = None
        else:
            if sensor_name is None:
                raise ValueError(f"Sensor must be specified for modality '{modality_name}'")
            self.selected_sensors[modality_name] = sensor_name

    def load_files(self) -> None:
        """Load all data files from the dataset directory structure"""
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
                            logger.error(f"Error processing file {file}: {e}")

    def match_trials(self) -> None:
        """Match modalities to create unified trial data"""
        trial_dict = {}
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    key = (modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][modality_name] = modality_file.file_path

        required_modalities = list(self.age_groups['young'].keys())
        
        # We need at least accelerometer for all trials
        for key, files_dict in trial_dict.items():
            if 'accelerometer' in files_dict and (
                'gyroscope' in files_dict or 'gyroscope' not in required_modalities):
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)
                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)
                self.matched_trials.append(matched_trial)

    def pipe_line(self, age_group: List[str], modalities: List[str], sensors: List[str]):
        """Setup and initialize the data pipeline"""
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
        
    def extract_timestamps_and_values(self, data_frame):
        """Extract timestamps and sensor values from dataframe"""
        if data_frame.shape[1] > 4:  # Meta sensors format
            timestamps = data_frame.iloc[:, 0].values  # Epoch time in ms
            values = data_frame.iloc[:, 3:6].values    # X, Y, Z values
        else:  # Phone/watch format
            timestamps = pd.to_datetime(data_frame.iloc[:, 0]).values.astype(np.int64) / 1e9  # Convert to seconds
            values = data_frame.iloc[:, 1:4].values
            
        return timestamps, values
        
    def resample_to_fixed_rate(self, timestamps, values, target_rate=None):
        """Resample variable-rate sensor data to fixed rate"""
        if target_rate is None:
            target_rate = self.target_sample_rate
            
        if len(timestamps) <= 1 or values.shape[0] <= 1:
            return values, np.array([0.0])
            
        # Convert timestamps to seconds if they're not already
        if np.mean(np.diff(timestamps)) > 1000:  # Likely in milliseconds
            timestamps = timestamps / 1000.0
            
        # Create time points at exactly 1/target_rate intervals
        start_time = timestamps[0]
        end_time = timestamps[-1]
        
        # Ensure we have at least 1 second of data
        if end_time - start_time < 1.0:
            end_time = start_time + 1.0
            
        desired_times = np.arange(start_time, end_time, 1.0/target_rate)
        
        if len(desired_times) < 1:
            return values[:1], np.array([timestamps[0]])
            
        # Create interpolation function for each axis
        resampled_data = np.zeros((len(desired_times), values.shape[1]))
        
        for axis in range(values.shape[1]):
            interp_func = interp1d(
                timestamps, values[:, axis], 
                bounds_error=False, 
                fill_value=(values[0, axis], values[-1, axis])
            )
            resampled_data[:, axis] = interp_func(desired_times)
        
        return resampled_data, desired_times

    def align_sensor_data(self, acc_data, acc_timestamps, gyro_data, gyro_timestamps):
        """Align and synchronize accelerometer and gyroscope data to common timebase"""
        # Resample both to target rate first
        resampled_acc, acc_times = self.resample_to_fixed_rate(acc_timestamps, acc_data)
        resampled_gyro, gyro_times = self.resample_to_fixed_rate(gyro_timestamps, gyro_data)
        
        # Find overlapping time period
        start_time = max(acc_times[0], gyro_times[0])
        end_time = min(acc_times[-1], gyro_times[-1])
        
        # If no overlap, return empty arrays
        if start_time >= end_time:
            logger.warning("No time overlap between accelerometer and gyroscope data")
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        # Create common time base at target rate
        common_times = np.arange(start_time, end_time, 1.0/self.target_sample_rate)
        
        if len(common_times) == 0:
            logger.warning("No common timepoints found after alignment")
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        
        # Resample to common timebase
        aligned_acc = np.zeros((len(common_times), acc_data.shape[1]))
        aligned_gyro = np.zeros((len(common_times), gyro_data.shape[1]))
        
        for axis in range(acc_data.shape[1]):
            acc_interp = interp1d(acc_times, resampled_acc[:, axis], 
                                bounds_error=False, 
                                fill_value=(resampled_acc[0, axis], resampled_acc[-1, axis]))
            aligned_acc[:, axis] = acc_interp(common_times)
        
        for axis in range(gyro_data.shape[1]):
            gyro_interp = interp1d(gyro_times, resampled_gyro[:, axis],
                                  bounds_error=False,
                                  fill_value=(resampled_gyro[0, axis], resampled_gyro[-1, axis]))
            aligned_gyro[:, axis] = gyro_interp(common_times)
        
        return aligned_acc, aligned_gyro, common_times
        
    def create_fixed_windows(self, data, window_size=None, overlap=None, min_windows=1):
        """Creates fixed-size windows with overlap"""
        if window_size is None:
            window_size = self.window_size
            
        if overlap is None:
            overlap = self.window_overlap
            
        if data.shape[0] < window_size:
            # If data is too short, pad with zeros
            padded = np.zeros((window_size, data.shape[1]))
            padded[:data.shape[0]] = data
            return [padded]
        
        stride = int(window_size * (1 - overlap))
        starts = list(range(0, data.shape[0] - window_size + 1, stride))
        
        # Ensure at least min_windows are created
        if len(starts) < min_windows:
            # Create evenly spaced starting points
            starts = np.linspace(0, max(0, data.shape[0] - window_size), min_windows).astype(int)
        
        windows = []
        for start in starts:
            end = start + window_size
            windows.append(data[start:end])
        
        return windows
        
    def load_trial_data(self, trial):
        """Load data for a specific trial with proper alignment"""
        trial_data = {}
        
        # Only proceed if we have both accelerometer and gyroscope
        if not ('accelerometer' in trial.files and 'gyroscope' in trial.files):
            logger.warning(f"Trial S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d} missing required modalities")
            return None
            
        try:
            # Load accelerometer data
            acc_file = trial.files['accelerometer']
            acc_df = pd.read_csv(acc_file, header=None)
            acc_timestamps, acc_values = self.extract_timestamps_and_values(acc_df)
            
            # Load gyroscope data
            gyro_file = trial.files['gyroscope']
            gyro_df = pd.read_csv(gyro_file, header=None)
            gyro_timestamps, gyro_values = self.extract_timestamps_and_values(gyro_df)
            
            # Align the sensor data
            aligned_acc, aligned_gyro, aligned_times = self.align_sensor_data(
                acc_values, acc_timestamps, gyro_values, gyro_timestamps
            )
            
            if aligned_acc.shape[0] == 0 or aligned_gyro.shape[0] == 0:
                logger.warning(f"Failed to align data for trial S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}")
                return None
                
            trial_data['accelerometer'] = aligned_acc
            trial_data['gyroscope'] = aligned_gyro
            trial_data['timestamps'] = aligned_times
            
            # Load skeleton data if available
            if 'skeleton' in trial.files:
                try:
                    skeleton_file = trial.files['skeleton']
                    skeleton_df = pd.read_csv(skeleton_file, header=None)
                    
                    # Skeleton data has 96 columns (32 joints × 3 coordinates)
                    skeleton_values = skeleton_df.values
                    
                    # Reshape to [frames, joints, coordinates]
                    num_frames = skeleton_values.shape[0]
                    skeleton_data = skeleton_values.reshape(num_frames, 32, 3)
                    
                    # Resample skeleton data to match aligned timestamps
                    # Since skeleton is fixed 30fps, we create evenly spaced timestamps
                    skeleton_times = np.linspace(0, num_frames/30.0, num_frames)
                    
                    # Only include skeleton if it has enough frames
                    if num_frames > 2:
                        # Create resampled skeleton data matching aligned timestamps
                        resampled_skeleton = np.zeros((len(aligned_times), 32, 3))
                        
                        # Interpolate each joint coordinate
                        for joint in range(32):
                            for coord in range(3):
                                joint_data = skeleton_data[:, joint, coord]
                                
                                # Create interpolation function
                                interp_func = interp1d(
                                    skeleton_times, joint_data,
                                    bounds_error=False,
                                    fill_value=(joint_data[0], joint_data[-1])
                                )
                                
                                # Apply interpolation to aligned timestamps
                                # Shift time values to match skeleton's time range
                                shifted_times = aligned_times - aligned_times[0]
                                resampled_skeleton[:, joint, coord] = interp_func(shifted_times)
                                
                        trial_data['skeleton'] = resampled_skeleton
                except Exception as e:
                    logger.error(f"Error processing skeleton data: {e}")
            
            return trial_data
            
        except Exception as e:
            logger.error(f"Error loading trial data: {e}")
            return None

    def create_windowed_samples(self, trial_data, label):
        """Create windowed samples from aligned trial data"""
        windows = {
            'accelerometer': [],
            'gyroscope': [],
            'labels': []
        }
        
        if 'skeleton' in trial_data:
            windows['skeleton'] = []
            
        # Create windows from aligned accelerometer and gyroscope data
        acc_windows = self.create_fixed_windows(trial_data['accelerometer'])
        gyro_windows = self.create_fixed_windows(trial_data['gyroscope'])
        
        # Ensure we have matching window counts
        if len(acc_windows) != len(gyro_windows):
            logger.warning(f"Mismatched window counts: {len(acc_windows)} accelerometer vs {len(gyro_windows)} gyroscope")
            min_windows = min(len(acc_windows), len(gyro_windows))
            acc_windows = acc_windows[:min_windows]
            gyro_windows = gyro_windows[:min_windows]
        
        # Add windows to dataset
        for i in range(len(acc_windows)):
            windows['accelerometer'].append(acc_windows[i])
            windows['gyroscope'].append(gyro_windows[i])
            windows['labels'].append(label)
            
            # Add skeleton windows if available
            if 'skeleton' in trial_data:
                skel_windows = self.create_fixed_windows(trial_data['skeleton'])
                if i < len(skel_windows):
                    windows['skeleton'].append(skel_windows[i])
                else:
                    # If we don't have enough skeleton windows, create an empty one
                    empty_skel = np.zeros((self.window_size, 32, 3))
                    windows['skeleton'].append(empty_skel)
        
        # Convert lists to numpy arrays
        for key in windows:
            if windows[key]:
                windows[key] = np.array(windows[key])
            else:
                # Create empty arrays with appropriate shapes if no data
                if key == 'accelerometer' or key == 'gyroscope':
                    windows[key] = np.zeros((0, self.window_size, 3))
                elif key == 'skeleton':
                    windows[key] = np.zeros((0, self.window_size, 32, 3))
                elif key == 'labels':
                    windows[key] = np.zeros(0)
                    
        return windows
                
    def process_trial(self, trial, label):
        """Process a single trial and return windowed data"""
        # Load and align sensor data
        trial_data = self.load_trial_data(trial)
        
        if trial_data is None:
            return None
            
        # Create windowed samples
        windowed_data = self.create_windowed_samples(trial_data, label)
        
        return windowed_data

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
    
    def make_dataset(self, subjects: List[int], fuse: bool = True):
        """Create dataset from selected subjects"""
        self.data = {
            'accelerometer': [],
            'gyroscope': [],
            'labels': []
        }
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue
                
            # Determine label based on task
            if self.task == 'fd':
                # Fall detection: label 1 for falls (activities 10-14), 0 for ADLs
                label = int(trial.action_id > 9)
            elif self.task == 'age':
                # Age classification: 1 for young (ID < 29 or > 46), 0 for old
                label = int(trial.subject_id < 29 or trial.subject_id > 46)
            else:
                # Activity recognition: use action_id - 1 as label
                label = trial.action_id - 1
                
            try:
                # Process the trial data
                trial_data = self.dataset.process_trial(trial, label)
                
                if trial_data is None:
                    continue
                    
                # Add data to dataset
                for key in trial_data:
                    if key not in self.data:
                        self.data[key] = []
                        
                    self.data[key].append(trial_data[key])
            except Exception as e:
                logger.error(f"Error processing trial S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: {e}")
                continue
                
        # Concatenate all data
        for key in self.data:
            if self.data[key]:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logger.error(f"Error concatenating {key} data: {e}")
                    
        return self.data
        
    def normalization(self) -> Dict[str, np.ndarray]:
        """Normalize the sensor data for better training performance"""
        from sklearn.preprocessing import StandardScaler
        
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion'] and len(value.shape) >= 2:
                        # Reshape to 2D for scaling
                        num_samples, seq_length = value.shape[:2]
                        feature_dims = np.prod(value.shape[2:]) if len(value.shape) > 2 else 1
                        
                        reshaped_data = value.reshape(num_samples * seq_length, feature_dims)
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                        
                        # Reshape back to original shape
                        self.data[key] = norm_data.reshape(value.shape)
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        self.data[key] = StandardScaler().fit_transform(value)
                except Exception as e:
                    logger.error(f"Error normalizing {key} data: {e}")
        
        return self.data

def prepare_smartfallmm(arg) -> DatasetBuilder:
    """Create a dataset builder for SmartFallMM dataset"""
    fusion_options = arg.dataset_args.get('fusion_options', {})
    
    # Set default sample rate and window size if not provided
    if 'target_sample_rate' not in fusion_options:
        fusion_options['target_sample_rate'] = 30.0  # 30Hz is standard for human motion
        
    if 'window_size' not in fusion_options:
        fusion_options['window_size'] = 128  # Use 128 samples for fixed window size
        
    if 'window_overlap' not in fusion_options:
        fusion_options['window_overlap'] = 0.5  # 50% overlap between windows
    
    # Create dataset
    sm_dataset = SmartFallMM(
        root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'),
        fusion_options=fusion_options
    )
    
    # Initialize dataset with specified modalities
    sm_dataset.pipe_line(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    
    # Create dataset builder
    builder = DatasetBuilder(
        sm_dataset, 
        mode=arg.dataset_args['mode'],
        max_length=arg.dataset_args['max_length'],
        task=arg.dataset_args['task'],
        fusion_options=fusion_options
    )
    
    return builder

def split_by_subjects(builder, subjects, fuse=True) -> Dict[str, np.ndarray]:
    """Create dataset split by subject IDs"""
    # Make dataset with selected subjects
    data = builder.make_dataset(subjects, fuse)
    
    # Apply normalization
    norm_data = builder.normalization()
    
    # Add validation to ensure data is properly shaped
    for key in norm_data:
        if key != 'labels' and len(norm_data[key]) > 0:
            # Ensure data is 3D for sequences
            if len(norm_data[key].shape) == 2:
                # Add sequence dimension if missing
                logger.warning(f"Adding missing sequence dimension to {key} data")
                samples, features = norm_data[key].shape
                norm_data[key] = norm_data[key].reshape(samples, 1, features)
    
    return norm_data
