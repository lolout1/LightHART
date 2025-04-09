from typing import List, Dict, Tuple, Union, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import logging
from dtaidistance import dtw

logger = logging.getLogger("dataset")

class ModalityFile: 
    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None: 
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

    def __repr__(self) -> str: 
        return f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number})"

class Modality:
    def __init__(self, name: str) -> None:
        self.name = name 
        self.files: List[ModalityFile] = []
    
    def add_file(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None: 
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)
    
    def __repr__(self) -> str:
        return f"Modality(name='{self.name}', files_count={len(self.files)})"

class MatchedTrial: 
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path
    
    def __repr__(self) -> str:
        return f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number})"

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
                            if file.endswith(('.csv', '.mat')):
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                        except Exception as e:
                            logger.error(f"Error processing file {file}: {e}")
    
    def match_trials(self) -> None:
        trial_dict = {}
        
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    key = (modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)
                    
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    
                    trial_dict[key][modality_name] = modality_file.file_path
        
        required_modalities = list(self.selected_sensors.keys())
        
        for key, files_dict in trial_dict.items():
            if all(modality in files_dict for modality in required_modalities):
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
        
        print(f"Loaded {len(self.matched_trials)} matched trials")
        print(f"Modalities: {modalities}")
        print(f"Sensors: {sensors}")
        print(f"Age groups: {age_group}")
        
        if hasattr(self, 'fusion_options') and self.fusion_options.get('filter_type'):
            print(f"Using fusion with filter type: {self.fusion_options['filter_type']}")

def filter_data_by_ids(data: np.ndarray, ids: List[int]) -> np.ndarray:
    if len(ids) == 0:
        return np.array([])
    return data[ids, :]

def filter_repeated_ids(path: List[Tuple[int, int]]) -> Tuple[set, set]:
    seen_first = set()
    seen_second = set()
    
    for (first, second) in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)
    
    return seen_first, seen_second

def prepare_smartfallmm(arg) -> 'DatasetBuilder': 
    from utils.loader import DatasetBuilder
    
    fusion_options = arg.dataset_args.get('fusion_options', {})
    
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
        arg.dataset_args['mode'], 
        arg.dataset_args['max_length'],
        arg.dataset_args['task'],
        fusion_options=fusion_options
    )
    
    return builder

def split_by_subjects(builder, subjects, fuse) -> Dict[str, np.ndarray]:
    fusion_options = getattr(builder, 'fusion_options', {})
    
    if fusion_options and fusion_options.get('enabled', False):
        filter_type = fusion_options.get('filter_type', 'madgwick')
        visualize = fusion_options.get('visualize', False)
        
        print(f"Applying IMU fusion with filter type: {filter_type}")
        print(f"Visualization enabled: {visualize}")
        
        builder.make_dataset(subjects, True, filter_type=filter_type, visualize=visualize)
    else:
        builder.make_dataset(subjects, fuse)
    
    # Ensure subjects are properly tracked
    if hasattr(builder, 'data') and 'labels' in builder.data and len(builder.data['labels']) > 0:
        num_samples = len(builder.data['labels'])
        
        # Check if subjects array exists
        if 'subjects' not in builder.data or len(builder.data['subjects']) != num_samples:
            logger.info("Generating subjects array from trial information")
            
            # Create subject array matching the dataset size
            builder.data['subjects'] = np.zeros(num_samples, dtype=np.int32)
            
            # Extract subject IDs from trial_to_samples mapping
            if hasattr(builder, 'trial_to_samples') and builder.trial_to_samples:
                for trial_id, sample_indices in builder.trial_to_samples.items():
                    try:
                        subject_id = int(trial_id.split('A')[0][1:])  # Extract subject ID from trial ID (e.g., S01A10T05 -> 01)
                        for idx in sample_indices:
                            if idx < num_samples:
                                builder.data['subjects'][idx] = subject_id
                    except (ValueError, IndexError):
                        logger.error(f"Failed to extract subject ID from trial ID: {trial_id}")
            else:
                # Fallback approach: distribute subjects evenly if no trial mapping exists
                logger.warning("No trial-to-sample mapping found. Using fallback subject assignment.")
                subject_list = list(subjects)
                samples_per_subject = max(1, num_samples // len(subject_list))
                
                for i, subject in enumerate(subject_list):
                    start_idx = i * samples_per_subject
                    end_idx = min((i + 1) * samples_per_subject, num_samples)
                    builder.data['subjects'][start_idx:end_idx] = subject
        
        logger.info(f"Subject array shape: {builder.data['subjects'].shape}")
    
    return builder.normalization()

def distribution_viz(labels: np.array, work_dir: str, mode: str) -> None:
    values, count = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(values, count)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title(f'{mode.capitalize()} Label Distribution')
    plt.savefig(os.path.join(work_dir, f'{mode}_label_distribution.png'))
    plt.close()

def subject_distribution_viz(subjects: np.array, labels: np.array, work_dir: str, mode: str) -> None:
    unique_subjects = np.unique(subjects)
    subject_counts = []
    
    for subject in unique_subjects:
        # Count samples per subject
        samples = np.sum(subjects == subject)
        subject_counts.append(samples)
        
        # Count class distribution per subject
        class_counts = []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            class_count = np.sum((subjects == subject) & (labels == label))
            class_counts.append(class_count)
        
        # Optional: plot class distribution for each subject
        plt.figure(figsize=(8, 5))
        plt.bar(unique_labels, class_counts)
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title(f'Subject {subject} Class Distribution')
        plt.savefig(os.path.join(work_dir, f'{mode}_subject_{subject}_distribution.png'))
        plt.close()
    
    # Plot overall subject distribution
    plt.figure(figsize=(12, 6))
    plt.bar(unique_subjects, subject_counts)
    plt.xlabel('Subject ID')
    plt.ylabel('Number of Samples')
    plt.title(f'{mode.capitalize()} Subject Distribution')
    plt.savefig(os.path.join(work_dir, f'{mode}_subject_distribution.png'))
    plt.close()

def analyze_imbalance(data: Dict[str, np.ndarray], work_dir: str, mode: str) -> None:
    """Analyze dataset imbalance and generate statistics."""
    if 'labels' not in data or 'subjects' not in data:
        logger.warning("Cannot analyze imbalance: missing labels or subjects")
        return
    
    labels = data['labels']
    subjects = data['subjects']
    
    # Overall class distribution
    distribution_viz(labels, work_dir, mode)
    
    # Subject distribution
    subject_distribution_viz(subjects, labels, work_dir, mode)
    
    # Calculate class imbalance metrics
    class_counts = np.bincount(labels.astype(int))
    total_samples = len(labels)
    class_ratios = class_counts / total_samples
    
    # Calculate subject imbalance metrics
    subject_counts = np.bincount(subjects.astype(int), minlength=np.max(subjects)+1)
    subject_counts = subject_counts[subject_counts > 0]  # Remove zero entries
    subject_ratios = subject_counts / total_samples
    
    # Save statistics
    stats = {
        'total_samples': total_samples,
        'num_classes': len(class_counts),
        'class_counts': class_counts.tolist(),
        'class_ratios': class_ratios.tolist(),
        'num_subjects': len(subject_counts),
        'subject_counts': subject_counts.tolist(),
        'subject_ratios': subject_ratios.tolist(),
        'max_imbalance_ratio': float(np.max(class_counts) / np.min(class_counts)) if len(class_counts) > 1 else 1.0
    }
    
    # Print summary
    print(f"\n{mode.upper()} Dataset Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"Class imbalance ratio: {stats['max_imbalance_ratio']:.2f}")
    print(f"Number of subjects: {len(subject_counts)}")
    
    # Save statistics to file
    with open(os.path.join(work_dir, f"{mode}_dataset_stats.txt"), "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
