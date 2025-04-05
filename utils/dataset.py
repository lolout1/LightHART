from typing import List, Dict, Tuple, Union, Optional, Any
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, butter, filtfilt
import traceback

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
        self.age_groups = {"old": {}, "young": {}}
        self.matched_trials = []
        self.selected_sensors = {}
        self.fusion_options = fusion_options or {}
        self.target_sample_rate = fusion_options.get('target_sample_rate', 30.0) if fusion_options else 30.0
        self.window_size = fusion_options.get('window_size', 128) if fusion_options else 128
        self.window_overlap = fusion_options.get('window_overlap', 0.5) if fusion_options else 0.5
        self.invalid_files = []

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
                            self.invalid_files.append(os.path.join(root, file))

    def match_trials(self) -> None:
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

    def pipe_line(self, age_group: List[str], modalities: List[str], sensors: List[str]):
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
        
    def extract_timestamps_and_values(self, file_path: str):
        try:
            for sep in [',', ';', '\t']:
                try:
                    data_frame = pd.read_csv(file_path, sep=sep, header=None)
                    break
                except:
                    continue
            
            if 'data_frame' not in locals():
                logger.error(f"Failed to parse {file_path}")
                return None, None
                
            data_frame = data_frame.dropna().bfill()
            first_row = data_frame.iloc[0].astype(str)
            header_indicators = ["timestamp", "time", "date", "unnamed", "index"]
            if any(any(indicator in val.lower() for indicator in header_indicators) for val in first_row):
                data_frame = data_frame.iloc[1:].reset_index(drop=True)
                
            if 'skeleton' in file_path:
                if data_frame.shape[1] < 96:
                    logger.warning(f"Invalid skeleton data dimensions in {file_path}")
                    return None, None
                values = data_frame.iloc[:, :96].values.astype(np.float32)
                timestamps = np.arange(len(values))
                return timestamps, values
            
            if data_frame.shape[1] > 4:
                try:
                    timestamps = data_frame.iloc[:, 0].values.astype(np.float64)
                    values = data_frame.iloc[:, 3:6].values.astype(np.float32)
                except:
                    return None, None
            else:
                try:
                    try:
                        timestamps = pd.to_datetime(data_frame.iloc[:, 0], errors='coerce')
                        if pd.isna(timestamps).all():
                            timestamps = pd.to_numeric(data_frame.iloc[:, 0], errors='coerce')
                            if pd.isna(timestamps).any():
                                return None, None
                        else:
                            timestamps = timestamps.values.astype(np.int64) / 1e9
                    except:
                        return None, None
                    values = data_frame.iloc[:, 1:4].values.astype(np.float32)
                except:
                    return None, None
                
            if np.isnan(values).any() or np.isinf(values).any():
                return None, None
                
            if len(values) < 3:
                return None, None
                
            return timestamps, values
            
        except Exception as e:
            logger.error(f"Error extracting data from {file_path}: {e}")
            return None, None
        
    def resample_to_fixed_rate(self, timestamps, values, target_rate=None):
        if target_rate is None:
            target_rate = self.target_sample_rate
            
        if timestamps is None or values is None or len(timestamps) <= 1 or values.shape[0] <= 1:
            return None, None
            
        try:
            if not np.all(np.diff(timestamps) > 0):
                valid_indices = np.concatenate(([0], np.where(np.diff(timestamps) > 0)[0] + 1))
                timestamps = timestamps[valid_indices]
                values = values[valid_indices]
                
                if len(timestamps) <= 1:
                    return None, None
                    
            time_range = timestamps[-1] - timestamps[0]
            if time_range > 1000:
                timestamps = timestamps / 1000.0
            elif time_range > 1e9:
                timestamps = timestamps / 1e9
                
            start_time = timestamps[0]
            end_time = timestamps[-1]
            
            if end_time - start_time < 0.5:
                return None, None
                
            desired_times = np.arange(start_time, end_time, 1.0/target_rate)
            
            if len(desired_times) < 3:
                return None, None
                
            resampled_data = np.zeros((len(desired_times), values.shape[1]))
            
            for axis in range(values.shape[1]):
                try:
                    interp_func = interp1d(timestamps, values[:, axis], 
                                         bounds_error=False, 
                                         fill_value=(values[0, axis], values[-1, axis]))
                    resampled_data[:, axis] = interp_func(desired_times)
                except:
                    idx = np.argmin(np.abs(timestamps[:, np.newaxis] - desired_times), axis=0)
                    resampled_data[:, axis] = values[idx, axis]
            
            return resampled_data, desired_times
            
        except:
            return None, None

    def apply_lowpass_filter(self, data, cutoff=5.0, fs=30.0, order=2):
        if data is None or len(data) <= 3:
            return data
            
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        filtered_data = np.zeros_like(data)
        for axis in range(data.shape[1]):
            try:
                filtered_data[:, axis] = filtfilt(b, a, data[:, axis])
            except:
                filtered_data[:, axis] = data[:, axis]
        
        return filtered_data

    def align_sensor_data(self, acc_file: str, gyro_file: str):
        try:
            acc_timestamps, acc_values = self.extract_timestamps_and_values(acc_file)
            gyro_timestamps, gyro_values = self.extract_timestamps_and_values(gyro_file)
            
            if acc_timestamps is None or gyro_timestamps is None or acc_values is None or gyro_values is None:
                return None, None, None
                
            resampled_acc, acc_times = self.resample_to_fixed_rate(acc_timestamps, acc_values)
            resampled_gyro, gyro_times = self.resample_to_fixed_rate(gyro_timestamps, gyro_values)
            
            if resampled_acc is None or resampled_gyro is None:
                return None, None, None
            
            start_time = max(acc_times[0], gyro_times[0])
            end_time = min(acc_times[-1], gyro_times[-1])
            
            if start_time >= end_time:
                return None, None, None
            
            common_times = np.arange(start_time, end_time, 1.0/self.target_sample_rate)
            
            if len(common_times) < 3:
                return None, None, None
            
            aligned_acc = np.zeros((len(common_times), 3))
            aligned_gyro = np.zeros((len(common_times), 3))
            
            for axis in range(min(3, acc_values.shape[1])):
                try:
                    acc_interp = interp1d(acc_times, resampled_acc[:, axis], 
                                        bounds_error=False, 
                                        fill_value=(resampled_acc[0, axis], resampled_acc[-1, axis]))
                    aligned_acc[:, axis] = acc_interp(common_times)
                except:
                    idx = np.argmin(np.abs(acc_times[:, np.newaxis] - common_times), axis=0)
                    aligned_acc[:, axis] = resampled_acc[idx, axis]
            
            for axis in range(min(3, gyro_values.shape[1])):
                try:
                    gyro_interp = interp1d(gyro_times, resampled_gyro[:, axis],
                                          bounds_error=False,
                                          fill_value=(resampled_gyro[0, axis], resampled_gyro[-1, axis]))
                    aligned_gyro[:, axis] = gyro_interp(common_times)
                except:
                    idx = np.argmin(np.abs(gyro_times[:, np.newaxis] - common_times), axis=0)
                    aligned_gyro[:, axis] = resampled_gyro[idx, axis]
            
            aligned_acc = self.apply_lowpass_filter(aligned_acc, cutoff=10.0, fs=self.target_sample_rate)
            aligned_gyro = self.apply_lowpass_filter(aligned_gyro, cutoff=10.0, fs=self.target_sample_rate)
            
            return aligned_acc, aligned_gyro, common_times
        
        except:
            return None, None, None
    
    def fixed_size_windows(self, data, window_size=None, stride=10):
        if window_size is None:
            window_size = self.window_size
            
        if data is None or len(data) < window_size // 2:
            return None
            
        windows = []
        for start in range(0, len(data) - window_size + 1, stride):
            windows.append(data[start:start + window_size])
            
        if not windows and len(data) > window_size // 2:
            padded = np.zeros((window_size, data.shape[1]))
            padded[:len(data)] = data
            windows = [padded]
            
        return windows

    def selective_sliding_window(self, data, window_size=None, stride=10):
        if window_size is None:
            window_size = self.window_size
            
        if data is None or len(data) < window_size // 2:
            return None
            
        try:
            if data.shape[1] >= 3:
                acc_magnitude = np.sqrt(np.sum(data[:, :3]**2, axis=1))
                height = max(1.5, np.mean(acc_magnitude) + 1.5 * np.std(acc_magnitude))
                distance = max(10, window_size // 8)
                peaks, _ = find_peaks(acc_magnitude, height=height, distance=distance)
                
                if len(peaks) == 0:
                    peaks = [np.argmax(acc_magnitude)]
                
                windows = []
                for peak in peaks:
                    half_window = window_size // 2
                    start = max(0, peak - half_window)
                    end = min(len(data), start + window_size)
                    
                    if end - start < window_size:
                        if start == 0:
                            end = min(len(data), window_size)
                        else:
                            start = max(0, end - window_size)
                    
                    if end - start == window_size:
                        windows.append(data[start:end])
                
                if not windows:
                    return self.fixed_size_windows(data, window_size, stride)
                
                return windows
            else:
                return self.fixed_size_windows(data, window_size, stride)
                
        except:
            return self.fixed_size_windows(data, window_size, stride)

    def load_trial_data(self, trial):
        try:
            trial_id = f"S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}"
            
            if not ('accelerometer' in trial.files and 'gyroscope' in trial.files):
                return None
                
            aligned_acc, aligned_gyro, aligned_times = self.align_sensor_data(
                trial.files['accelerometer'],
                trial.files['gyroscope']
            )
            
            if aligned_acc is None or aligned_gyro is None or aligned_times is None:
                return None
                
            if len(aligned_acc) < self.window_size // 2:
                return None
                
            trial_data = {
                'accelerometer': aligned_acc,
                'gyroscope': aligned_gyro,
                'timestamps': aligned_times
            }
            
            if 'skeleton' in trial.files:
                try:
                    skeleton_timestamps, skeleton_values = self.extract_timestamps_and_values(trial.files['skeleton'])
                    
                    if skeleton_timestamps is not None and skeleton_values is not None:
                        num_frames = skeleton_values.shape[0]
                        
                        if skeleton_values.shape[1] == 96:
                            skeleton_data = skeleton_values.reshape(num_frames, 32, 3)
                            
                            if num_frames > 2:
                                if skeleton_timestamps is None or len(skeleton_timestamps) != num_frames:
                                    skeleton_times = np.linspace(0, num_frames/30.0, num_frames)
                                else:
                                    skeleton_times = skeleton_timestamps
                                
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
                                        except:
                                            idx = np.argmin(np.abs(skeleton_times[:, np.newaxis] - aligned_times), axis=0)
                                            resampled_skeleton[:, joint, coord] = skeleton_data[idx, joint, coord]
                                
                                trial_data['skeleton'] = resampled_skeleton
                except:
                    pass
            
            return trial_data
            
        except:
            return None

    def create_windowed_samples(self, trial_data, label):
        if trial_data is None:
            return None
            
        is_fall = label > 9
        
        windows = {
            'accelerometer': [],
            'gyroscope': [],
            'labels': []
        }
        
        if 'skeleton' in trial_data:
            windows['skeleton'] = []
            
        if is_fall:
            acc_windows = self.selective_sliding_window(trial_data['accelerometer'])
        else:
            acc_windows = self.fixed_size_windows(trial_data['accelerometer'])
        
        if acc_windows is None or len(acc_windows) == 0:
            return windows
        
        for acc_window in acc_windows:
            best_match_start = 0
            best_match_score = float('inf')
            
            for i in range(0, len(trial_data['accelerometer']) - len(acc_window) + 1, 10):
                score = np.sum(np.abs(trial_data['accelerometer'][i:i+5, :] - acc_window[:5, :]))
                if score < best_match_score:
                    best_match_score = score
                    best_match_start = i
            
            windows['accelerometer'].append(acc_window)
            
            gyro_start = best_match_start
            gyro_end = gyro_start + self.window_size
            if gyro_end <= len(trial_data['gyroscope']):
                gyro_window = trial_data['gyroscope'][gyro_start:gyro_end]
            else:
                gyro_window = np.zeros((self.window_size, trial_data['gyroscope'].shape[1]))
                gyro_window[:len(trial_data['gyroscope']) - gyro_start] = trial_data['gyroscope'][gyro_start:]
            
            windows['gyroscope'].append(gyro_window)
            windows['labels'].append(label)
            
            if 'skeleton' in trial_data:
                skel_start = best_match_start
                skel_end = skel_start + self.window_size
                if skel_end <= len(trial_data['skeleton']):
                    skel_window = trial_data['skeleton'][skel_start:skel_end]
                else:
                    skel_window = np.zeros((self.window_size, *trial_data['skeleton'].shape[1:]))
                    skel_window[:len(trial_data['skeleton']) - skel_start] = trial_data['skeleton'][skel_start:]
                
                windows['skeleton'].append(skel_window)
        
        for key in windows:
            if windows[key]:
                windows[key] = np.array(windows[key])
            else:
                if key == 'accelerometer' or key == 'gyroscope':
                    windows[key] = np.array([])
                elif key == 'skeleton':
                    windows[key] = np.array([])
                elif key == 'labels':
                    windows[key] = np.array([])
                    
        return windows
                
    def process_trial(self, trial, label):
        trial_data = self.load_trial_data(trial)
        
        if trial_data is None:
            return None
            
        windowed_data = self.create_windowed_samples(trial_data, label)
        
        if windowed_data is None or ('accelerometer' not in windowed_data) or len(windowed_data['accelerometer']) == 0:
            return None
        
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
        self.load_errors = []
    
    def make_dataset(self, subjects: List[int], fuse: bool = True):
        self.data = {
            'accelerometer': [],
            'gyroscope': [],
            'labels': []
        }
        
        valid_trials = 0
        skipped_trials = 0
        
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
                trial_data = self.dataset.process_trial(trial, label)
                
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
                trial_id = f"S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}"
                error_msg = f"Error processing trial {trial_id}: {str(e)}"
                logger.error(error_msg)
                self.load_errors.append(error_msg)
                skipped_trials += 1
                continue
                
        logger.info(f"Processed {valid_trials} valid trials, skipped {skipped_trials} trials")
                
        for key in self.data:
            if self.data[key]:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logger.error(f"Error concatenating {key} data: {e}")
                    if key in self.data:
                        del self.data[key]
                    
        return self.data
        
    def normalization(self) -> Dict[str, np.ndarray]:
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
                    logger.error(f"Error normalizing {key} data: {e}")
        
        return self.data

def prepare_smartfallmm(arg) -> DatasetBuilder:
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

def split_by_subjects(builder, subjects, fuse=True) -> Dict[str, np.ndarray]:
    data = builder.make_dataset(subjects, fuse)
    
    norm_data = builder.normalization()
    
    for key in norm_data:
        if key != 'labels' and key in norm_data and len(norm_data[key]) > 0:
            if len(norm_data[key].shape) == 2:
                logger.warning(f"Adding missing sequence dimension to {key} data")
                samples, features = norm_data[key].shape
                norm_data[key] = norm_data[key].reshape(samples, 1, features)
    
    return norm_data
