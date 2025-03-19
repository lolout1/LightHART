# utils/dataset.py
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from utils.loader import DatasetBuilder
import logging

# Configure logging
logger = logging.getLogger("dataset")

class ModalityFile: 
    '''
    Represents an individual file in a modality, containing the subject ID, action ID, sequence number, and file path

    Attributes: 
        subject_id (int): ID of the subject performing the action
        action_id (int): ID of the action being performed
        sequence_number (int): Sequence/trial number
        file_path (str): Path to the data file
    '''

    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None: 
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

    def __repr__(self) -> str:
        return (
            f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, "
            f"sequence_number={self.sequence_number}, file_path='{self.file_path}')"
        )


class Modality:
    '''
    Represents a modality (e.g., accelerometer, gyroscope) containing a list of ModalityFile objects.

    Attributes:
        name (str): Name of the modality.
        files (List[ModalityFile]): List of files belonging to this modality.
    '''

    def __init__(self, name: str) -> None:
        self.name = name 
        self.files: List[ModalityFile] = []
    
    def add_file(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None: 
        '''
        Adds a file to the modality

        Args: 
            subject_id (int): ID of the subject.
            action_id (int): ID of the action.
            sequence_number (int): Sequence number of the trial.
            file_path (str): Path to the file.
        '''
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)
    
    def __repr__(self) -> str:
        return f"Modality(name='{self.name}', files={len(self.files)} files)"
    

class MatchedTrial: 
    """
    Represents a matched trial containing files from different modalities for the same trial.

    Attributes:
        subject_id (int): ID of the subject.
        action_id (int): ID of the action.
        sequence_number (int): Sequence number of the trial.
        files (Dict[str, str]): Dictionary mapping modality names to file paths.
    """
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        '''
        Adds a file to the matched trial for a specific modality

        Args:
            modality_name (str): Name of the modality
            file_path(str): Path to the file
        '''
        self.files[modality_name] = file_path
    
    def __repr__(self) -> str:
        return (
            f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, "
            f"sequence_number={self.sequence_number}, modalities={list(self.files.keys())})"
        )


class SmartFallMM:
    """
    Represents the SmartFallMM dataset, managing the loading of files and matching of trials across modalities.

    Attributes:
        root_dir (str): Root directory of the SmartFallMM dataset.
        age_groups (Dict[str, Dict[str, Modality]]): Dictionary containing 'old' and 'young' groups.
        matched_trials (List[MatchedTrial]): List of matched trials across modalities.
        selected_sensors (Dict[str, str]): Dictionary storing selected sensors for modalities.
        fusion_options (Dict): Optional configuration for IMU fusion (filter type, etc.)
    """
    def __init__(self, root_dir: str, fusion_options: Optional[Dict] = None) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {
            "old": {},
            "young": {}
        }
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, str] = {}  # Stores the selected sensor for each modality
        self.fusion_options = fusion_options or {}  # Store fusion configuration

    def add_modality(self, age_group: str, modality_name: str) -> None:
        """
        Adds a modality to the dataset for a specific age group.
        
        Args:
            age_group (str): Either 'old' or 'young'.
            modality_name (str): Name of the modality (e.g., accelerometer, gyroscope, skeleton).
        """
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}. Expected 'old' or 'young'.")
        
        self.age_groups[age_group][modality_name] = Modality(modality_name)
        logger.info(f"Added modality '{modality_name}' for age group '{age_group}'")

    def select_sensor(self, modality_name: str, sensor_name: str = None) -> None:
        """
        Selects a specific sensor for a given modality.
        
        Args:
            modality_name (str): Name of the modality (e.g., accelerometer, gyroscope).
            sensor_name (str): Name of the sensor (e.g., phone, watch, meta_wrist).
        """
        if modality_name == "skeleton":
            # Skeleton modality doesn't have sensor-specific data
            self.selected_sensors[modality_name] = None
            logger.info(f"Selected skeleton modality (no sensor required)")
        else:
            if sensor_name is None:
                raise ValueError(f"Sensor must be specified for modality '{modality_name}'")
            self.selected_sensors[modality_name] = sensor_name
            logger.info(f"Selected sensor '{sensor_name}' for modality '{modality_name}'")

    def load_files(self) -> None:
        """
        Loads files from the dataset based on selected sensors and age groups.
        Handles skeleton data and sensor-specific data appropriately.
        """
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                # Handle skeleton data (no sensor required)
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    # Only load data from the selected sensor if it exists
                    if modality_name in self.selected_sensors:
                        sensor_name = self.selected_sensors[modality_name]
                        modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                    else:
                        logger.warning(f"No sensor selected for modality '{modality_name}', skipping")
                        continue

                # Check if directory exists
                if not os.path.exists(modality_dir):
                    logger.warning(f"Directory not found: {modality_dir}, skipping")
                    continue

                # Load the files
                file_count = 0
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        try:
                            if file.endswith(('.csv')):
                                # Extract information based on the filename (e.g., S01A10T05)
                                if len(file) >= 9 and file[0] == 'S' and file[3] == 'A' and file[6] == 'T':
                                    subject_id = int(file[1:3])
                                    action_id = int(file[4:6])
                                    sequence_number = int(file[7:9])
                                    file_path = os.path.join(root, file)
                                    modality.add_file(subject_id, action_id, sequence_number, file_path)
                                    file_count += 1
                        except Exception as e:
                            logger.error(f"Error processing file {file}: {e}")
                
                logger.info(f"Loaded {file_count} files for {age_group}/{modality_name}")

    def match_trials(self) -> None:
        """
        Matches files from different modalities based on subject/action/sequence.
        Creates a list of matched trials with consistent modalities.
        """
        trial_dict = {}

        # Step 1: Group files by (subject_id, action_id, sequence_number)
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for modality_file in modality.files:
                    key = (modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)

                    if key not in trial_dict:
                        trial_dict[key] = {}

                    # Add the file under its modality name
                    trial_dict[key][modality_name] = modality_file.file_path

        # Step 2: Filter out incomplete trials
        required_modalities = list(self.age_groups['young'].keys())
        
        matched_count = 0
        incomplete_count = 0
        
        for key, files_dict in trial_dict.items():
            # Check if all required modalities are present for this trial
            if all(modality in files_dict for modality in required_modalities):
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)

                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)

                self.matched_trials.append(matched_trial)
                matched_count += 1
            else:
                incomplete_count += 1

        logger.info(f"Matched {matched_count} complete trials")
        logger.info(f"Skipped {incomplete_count} incomplete trials")

    def pipe_line(self, age_group: List[str], modalities: List[str], sensors: List[str]):
        '''
        A pipeline to load the data and prepare matched trials.
        
        Args:
            age_group: List of age groups ('young', 'old')
            modalities: List of modalities ('accelerometer', 'gyroscope', 'skeleton')
            sensors: List of sensors ('phone', 'watch', 'meta_wrist', 'meta_hip')
        '''
        # Add modalities for each age group
        for age in age_group: 
            for modality in modalities:
                self.add_modality(age, modality)
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else: 
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)

        # Load files for the selected sensors and skeleton data
        self.load_files()

        # Match trials across the modalities
        self.match_trials()
        
        logger.info(f"Pipeline complete: {len(self.matched_trials)} matched trials")
        logger.info(f"Modalities: {modalities}")
        logger.info(f"Sensors: {sensors}")
        logger.info(f"Age groups: {age_group}")
        
        # Log fusion information if available
        if hasattr(self, 'fusion_options') and self.fusion_options:
            filter_type = self.fusion_options.get('filter_type', 'None')
            enabled = self.fusion_options.get('enabled', False)
            logger.info(f"Fusion settings: enabled={enabled}, filter_type={filter_type}")


def prepare_smartfallmm(arg) -> DatasetBuilder: 
    '''
    Function for dataset preparation
    
    Args:
        arg: Configuration object with dataset_args
        
    Returns:
        DatasetBuilder object ready for creating datasets
    '''
    # Check if fusion options are present in the configuration
    fusion_options = arg.dataset_args.get('fusion_options', {})
    
    # Create dataset object with optional fusion configuration
    sm_dataset = SmartFallMM(
        root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'),
        fusion_options=fusion_options
    )
    
    # Setup the dataset pipeline
    sm_dataset.pipe_line(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    
    # Create the dataset builder
    builder = DatasetBuilder(
        sm_dataset, 
        arg.dataset_args['mode'], 
        arg.dataset_args['max_length'],
        arg.dataset_args['task'],
        fusion_options=fusion_options
    )
    
    return builder


def split_by_subjects(builder, subjects, fuse) -> Dict[str, np.ndarray]:
    '''
    Function to filter data by expected subjects and apply fusion if needed
    
    Args:
        builder: DatasetBuilder object
        subjects: List of subject IDs to include
        fuse: Basic flag for fusion (True/False)
        
    Returns:
        Dictionary of normalized dataset components
    '''
    # Get fusion options if available
    fusion_options = getattr(builder, 'fusion_options', {})
    
    # Check if detailed fusion configuration is available
    if fusion_options and fusion_options.get('enabled', False):
        # Use the specified filter type (or default to madgwick)
        filter_type = fusion_options.get('filter_type', 'madgwick')
        visualize = fusion_options.get('visualize', False)
        save_aligned = fusion_options.get('save_aligned', False)
        
        logger.info(f"Applying IMU fusion with filter type: {filter_type}")
        logger.info(f"Visualization enabled: {visualize}")
        logger.info(f"Save aligned data: {save_aligned}")
        
        # Create dataset with enhanced fusion options
        builder.make_dataset(subjects, True, filter_type=filter_type, 
                           visualize=visualize, save_aligned=save_aligned)
    else:
        # Use the basic fusion flag
        builder.make_dataset(subjects, fuse)
    
    # Apply normalization
    norm_data = builder.normalization()
    
    # CRITICAL: Remove 'aligned_timestamps' from the normalized data
    # This prevents the KeyError in the DataLoader
    if 'aligned_timestamps' in norm_data:
        del norm_data['aligned_timestamps']
    
    # Log dataset statistics
    for key, value in norm_data.items():
        if key == 'labels':
            # Log label distribution
            unique_labels, counts = np.unique(value, return_counts=True)
            distribution = {int(label): int(count) for label, count in zip(unique_labels, counts)}
            logger.info(f"Label distribution: {distribution}")
        else:
            # Log shape information for data arrays
            logger.info(f"{key} shape: {value.shape}")
    
    return norm_data


def distribution_viz(labels: np.array, work_dir: str, mode: str) -> None:
    '''
    Visualizes the distribution of labels in the dataset
    
    Args:
        labels: Array of class labels
        work_dir: Directory to save the visualization
        mode: Mode identifier (train, val, test)
    '''
    try:
        # Create target directory if it doesn't exist
        os.makedirs(work_dir, exist_ok=True)
        
        # Get unique labels and their counts
        values, count = np.unique(labels, return_counts=True)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.bar(values, count)
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title(f'{mode.capitalize()} Label Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for i, (v, c) in enumerate(zip(values, count)):
            plt.text(v, c + 0.1, str(c), ha='center')
        
        # Save the figure
        plt.savefig(os.path.join(work_dir, f'{mode}_label_distribution.png'))
        plt.close()
        
        # Log distribution statistics
        total = np.sum(count)
        percentages = (count / total) * 100
        logger.info(f"Class distribution in {mode} set:")
        for i, (val, cnt, pct) in enumerate(zip(values, count, percentages)):
            logger.info(f"  Class {val}: {cnt} samples ({pct:.1f}%)")
    except Exception as e:
        logger.error(f"Error creating distribution visualization: {str(e)}")
