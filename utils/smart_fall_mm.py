import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

logger = logging.getLogger("smart_fall_mm")

class ModalityFile:
    def __init__(self, subject_id, action_id, sequence_number, file_path):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

class MatchedTrial:
    def __init__(self, subject_id, action_id, sequence_number):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files = {}
    
    def add_file(self, modality_name, file_path):
        self.files[modality_name] = file_path

class SmartFallMM:
    def __init__(self, root_dir="data/smartfallmm"):
        self.root_dir = root_dir
        self.age_groups = {"old": {}, "young": {}}
        self.matched_trials = []
        self.selected_sensors = {}

    def add_modality(self, age_group, modality_name):
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}. Expected 'old' or 'young'.")
        if modality_name not in self.age_groups[age_group]:
            self.age_groups[age_group][modality_name] = []

    def select_sensor(self, modality_name, sensor_name=None):
        self.selected_sensors[modality_name] = sensor_name

    def load_files(self):
        for age_group, modalities in self.age_groups.items():
            for modality_name in modalities:
                if modality_name in self.selected_sensors:
                    sensor_name = self.selected_sensors[modality_name]
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                    if not os.path.exists(modality_dir):
                        logger.warning(f"Directory does not exist: {modality_dir}")
                        continue
                    logger.info(f"Loading files from {modality_dir}")
                    file_counter = 0
                    for root, _, files in os.walk(modality_dir):
                        for file in files:
                            try:
                                if file.endswith('.csv'):
                                    subject_id = int(file[1:3])
                                    action_id = int(file[4:6])
                                    sequence_number = int(file[7:9])
                                    file_path = os.path.join(root, file)
                                    if os.path.getsize(file_path) > 0:
                                        trial = ModalityFile(subject_id, action_id, sequence_number, file_path)
                                        self.age_groups[age_group][modality_name].append(trial)
                                        file_counter += 1
                            except (ValueError, IndexError):
                                pass
                    logger.info(f"Loaded {file_counter} files from {modality_dir}")

    def match_trials(self):
        trial_dict = {}
        for age_group, modalities in self.age_groups.items():
            for modality_name, files in modalities.items():
                if modality_name == 'accelerometer':
                    for file in files:
                        key = (file.subject_id, file.action_id, file.sequence_number)
                        if key not in trial_dict:
                            trial_dict[key] = {}
                        trial_dict[key][modality_name] = file.file_path
        for key, files_dict in trial_dict.items():
            subject_id, action_id, sequence_number = key
            if 'accelerometer' in files_dict:
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)
                matched_trial.add_file('accelerometer', files_dict['accelerometer'])
                self.matched_trials.append(matched_trial)
        logger.info(f"Created {len(self.matched_trials)} matched trials")

    def run_pipeline(self, age_group, modalities, sensors):
        logger.info(f"Running pipeline with age_groups={age_group}, modalities={modalities}, sensors={sensors}")
        filtered_modalities = ['accelerometer']
        for age in age_group:
            for modality in filtered_modalities:
                self.add_modality(age, modality)
                for sensor in sensors:
                    if sensor == 'watch':
                        self.select_sensor(modality, sensor)
        self.load_files()
        self.match_trials()
        logger.info(f"Pipeline complete. Found {len(self.matched_trials)} matched trials")

def butterworth_filter(data, cutoff=7.5, fs=25, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

class SmartFallMMBuilder:
    def __init__(self, root="data/smartfallmm", age_group=None, modalities=None, sensors=None, mode="sliding_window", max_length=128, task="fd"):
        self.root = root
        self.age_group = age_group or ["young", "old"]
        self.modalities = modalities or ["accelerometer"]
        self.sensors = sensors or ["watch"]
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.dataset = None
        self.data_builder = None
    
    def run_pipeline(self):
        self.dataset = SmartFallMM(self.root)
        self.dataset.run_pipeline(self.age_group, self.modalities, self.sensors)
        self.data_builder = DatasetBuilder(self.dataset, self.mode, self.max_length, self.task)
    
    def make_subject_folds(self):
        val_subjects = [38, 46]
        always_train_subjects = [45, 36, 29]
        eligible_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
        folds = []
        for test_subject in eligible_subjects:
            test_subjects = [test_subject]
            train_subjects = always_train_subjects + [s for s in eligible_subjects if s != test_subject]
            folds.append({"train": train_subjects, "val": val_subjects, "test": test_subjects})
        return folds
    
    def build_dataset(self, subjects, fuse=False):
        if not self.data_builder:
            raise ValueError("Pipeline not initialized. Call run_pipeline() first.")
        self.data_builder.make_dataset(subjects, fuse)
        return self.data_builder.normalization()

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task="fd"):
        self.dataset = dataset
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.data = defaultdict(list)
    
    def load_file(self, file_path):
        try:
            df = pd.read_csv(file_path, header=None)
            if df.empty:
                return np.array([])
            
            start_row = 0
            for i in range(min(10, len(df))):
                try:
                    df.iloc[i, 1:4].astype(float)
                    start_row = i
                    break
                except (ValueError, TypeError):
                    continue
            
            try:
                acc_data = df.iloc[start_row:, 1:4].astype(float).values
                if len(acc_data) < self.max_length:
                    return np.array([])
                
                smv = np.sqrt(np.sum(acc_data**2, axis=1, keepdims=True))
                return np.hstack((acc_data, smv))
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return np.array([])
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return np.array([])
    
    def process_windows(self, data, window_size=128, stride=10, is_fall=False):
        if len(data) < window_size:
            return []
        
        windows = []
        if is_fall:
            acc_mag = np.sqrt(np.sum(data[:, :3]**2, axis=1))
            threshold = np.mean(acc_mag) + 1.5 * np.std(acc_mag)
            
            peaks = []
            for i in range(window_size//2, len(acc_mag) - window_size//2):
                if acc_mag[i] > threshold and acc_mag[i] == np.max(acc_mag[max(0, i-10):min(len(acc_mag), i+10)]):
                    peaks.append(i)
            
            if not peaks:
                peaks = [len(data)//2]
            
            for peak in peaks:
                start = max(0, peak - window_size//2)
                end = start + window_size
                
                if end > len(data):
                    continue
                
                windows.append(data[start:end])
        else:
            for start in range(0, len(data) - window_size + 1, stride):
                windows.append(data[start:start + window_size])
        
        return np.array(windows) if windows else np.array([])
    
    def process_trial(self, trial, label):
        if 'accelerometer' not in trial.files:
            return None
        
        acc_data = self.load_file(trial.files['accelerometer'])
        if len(acc_data) == 0 or acc_data.shape[1] != 4:
            return None
        
        try:
            acc_data = butterworth_filter(acc_data)
        except Exception as e:
            logger.warning(f"Filtering failed: {e}")
            return None
        
        windows = self.process_windows(acc_data, self.max_length, 10, label == 1)
        
        if len(windows) == 0:
            return None
        
        return {
            'accelerometer': windows,
            'labels': np.repeat(label, len(windows)),
            'subjects': np.repeat(trial.subject_id, len(windows))
        }
    
    def make_dataset(self, subjects, fuse=False):
        self.data = defaultdict(list)
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                
                trial_data = self.process_trial(trial, label)
                
                if trial_data is not None:
                    for key, value in trial_data.items():
                        self.data[key].append(value)
        
        for key in self.data:
            if self.data[key]:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except:
                    self.data[key] = np.array([])
            else:
                self.data[key] = np.array([])
        
        return self.data
    
    def normalization(self):
        for key, value in self.data.items():
            if key not in ['labels', 'subjects'] and len(value) > 0:
                try:
                    num_samples, seq_length, num_features = value.shape
                    flat_data = value.reshape(num_samples * seq_length, num_features)
                    scaler = StandardScaler()
                    norm_data = scaler.fit_transform(flat_data)
                    self.data[key] = norm_data.reshape(num_samples, seq_length, num_features)
                except Exception as e:
                    logger.error(f"Error normalizing {key}: {e}")
        
        return self.data
