import os
import numpy as np
import logging
from utils.data_utils import process_accelerometer_file

logger = logging.getLogger(__name__)

def scan_directory_for_files(base_dir, sensor_type='watch'):
    """
    Scan directory for accelerometer files for a specific sensor
    
    Args:
        base_dir: Base directory path
        sensor_type: Sensor type (watch, phone, etc.)
        
    Returns:
        Dictionary mapping subject_id to lists of file paths
    """
    acc_dir = os.path.join(base_dir, 'accelerometer', sensor_type)
    if not os.path.exists(acc_dir):
        logger.warning(f"Directory does not exist: {acc_dir}")
        return {}
    
    files_by_subject = {}
    
    # Scan directory
    for root, _, files in os.walk(acc_dir):
        for file in files:
            if file.endswith('.csv'):
                try:
                    subject_id = int(file[1:3])
                    file_path = os.path.join(root, file)
                    
                    if subject_id not in files_by_subject:
                        files_by_subject[subject_id] = []
                    
                    files_by_subject[subject_id].append(file_path)
                except (ValueError, IndexError):
                    logger.warning(f"Skipping invalid filename: {file}")
    
    return files_by_subject

def split_by_subjects(builder, subjects, fuse=False):
    """
    Split dataset by subjects
    
    Args:
        builder: SmartFallMMBuilder object or similar
        subjects: List of subject IDs to include
        fuse: Whether to fuse data from multiple sensors
        
    Returns:
        Dictionary with data split by subjects
    """
    if not hasattr(builder, 'dataset'):
        logger.error("Builder has no dataset attribute")
        return {
            'accelerometer': np.array([]),
            'labels': np.array([]),
            'subjects': np.array([])
        }
    
    # Get matched trials from builder
    trials = builder.dataset.matched_trials if hasattr(builder.dataset, 'matched_trials') else []
    
    # Filter trials by subjects
    filtered_trials = [t for t in trials if t.subject_id in subjects]
    
    # Initialize data containers
    all_windows = []
    all_labels = []
    all_subjects = []
    
    # Process each trial
    for trial in filtered_trials:
        if 'accelerometer' not in trial.files:
            continue
            
        file_path = trial.files['accelerometer']
        
        # Process trial
        try:
            result = process_accelerometer_file(file_path)
            
            if result is None:
                continue
                
            # Determine label based on activity
            if builder.task == 'fd':
                label = 1 if result['is_fall'] else 0
            elif builder.task == 'har':
                label = result['activity_id'] - 1  # Convert to 0-based
            else:
                label = 0  # Default
                
            # Add windows to dataset
            windows = result['windows']
            num_windows = len(windows)
            
            all_windows.append(windows)
            all_labels.extend([label] * num_windows)
            all_subjects.extend([trial.subject_id] * num_windows)
            
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue
    
    # Combine all data
    if all_windows:
        accelerometer = np.concatenate(all_windows, axis=0)
        labels = np.array(all_labels)
        subjects = np.array(all_subjects)
        
        logger.info(f"Created dataset with {len(accelerometer)} samples from {len(filtered_trials)} trials")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return {
            'accelerometer': accelerometer,
            'labels': labels,
            'subjects': subjects
        }
    else:
        logger.warning("No data found for the specified subjects")
        return {
            'accelerometer': np.array([]),
            'labels': np.array([]),
            'subjects': np.array([])
        }

class SmartFallMMBuilder:
    """Builder class for SmartFallMM dataset"""
    def __init__(self, root="data/smartfallmm", age_group=None, modalities=None, 
                 sensors=None, mode="sliding_window", max_length=128, task="fd"):
        self.root = root
        self.age_group = age_group or ["young", "old"]
        self.modalities = modalities or ["accelerometer"]
        self.sensors = sensors or ["watch"]
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.dataset = SmartFallMM(self.root)
        
    def run_pipeline(self):
        """Run data loading pipeline"""
        self.dataset.run_pipeline(self.age_group, self.modalities, self.sensors)
        return self
    
    def build_dataset(self, subjects, fuse=False):
        """Build dataset for specific subjects"""
        return split_by_subjects(self, subjects, fuse)

class SmartFallMM:
    """Class representing the SmartFallMM dataset"""
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
                                        class ModalityFile:
                                            def __init__(self, sid, aid, seq, path):
                                                self.subject_id = sid
                                                self.action_id = aid
                                                self.sequence_number = seq
                                                self.file_path = path
                                                
                                        trial = ModalityFile(subject_id, action_id, sequence_number, file_path)
                                        self.age_groups[age_group][modality_name].append(trial)
                                        file_counter += 1
                            except (ValueError, IndexError) as e:
                                logger.debug(f"Skipping invalid file {file}: {e}")
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
                class MatchedTrial:
                    def __init__(self, sid, aid, seq):
                        self.subject_id = sid
                        self.action_id = aid
                        self.sequence_number = seq
                        self.files = {}
                    
                    def add_file(self, modality_name, file_path):
                        self.files[modality_name] = file_path
                
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

def prepare_smartfallmm(args):
    """Prepare SmartFallMM dataset from args"""
    logger.info("Preparing SmartFallMM dataset")
    
    if not hasattr(args, 'dataset_args') or not args.dataset_args:
        args.dataset_args = {
            'age_group': ['young', 'old'],
            'modalities': ['accelerometer'],
            'sensors': ['watch'],
            'mode': 'sliding_window',
            'max_length': 128,
            'task': 'fd'
        }
    
    # Handle various path formats - try to normalize the path
    root_dir = getattr(args, 'data_dir', 'data/smartfallmm')
    if not os.path.exists(root_dir) and not os.path.isabs(root_dir):
        # Try some common relative paths
        alt_paths = [
            f"../{root_dir}",
            root_dir.replace('/data/', 'data/'),
            root_dir.lstrip('/'),
            "../" + root_dir.lstrip('/')
        ]
        for path in alt_paths:
            if os.path.exists(path):
                root_dir = path
                logger.info(f"Using alternative data path: {root_dir}")
                break
    
    logger.info(f"Using data path: {root_dir}")
    
    builder = SmartFallMMBuilder(
        root=root_dir,
        age_group=args.dataset_args['age_group'],
        modalities=args.dataset_args['modalities'],
        sensors=args.dataset_args['sensors'],
        mode=args.dataset_args['mode'],
        max_length=args.dataset_args['max_length'],
        task=args.dataset_args['task']
    )
    
    builder.run_pipeline()
    return builder
