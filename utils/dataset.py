# utils/dataset.py
from typing import List, Dict, Tuple, Union, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime
import logging

# Configure logging
log_dir = "debug_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "dataset.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("dataset")

# Also print to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Import from utils.loader (handle circular imports carefully)
from utils.loader import DatasetBuilder
from utils.imu_fusion import (
    align_sensor_data, 
    process_imu_data, 
    extract_features_from_window,
    hybrid_interpolate,
    save_aligned_sensor_data  # New function to save aligned data
)


class ModalityFile: 
    '''
    Represents an individual file in a modality, containing the subject ID, action ID, sequence number, and file path

    Attributes: 
    subject_id (int) : ID of the subject performing the action
    action_id (int) : ID of the action being performed
    sequence_number (int) : Sequence/trial number
    file_path (str) : Path to the data file
    '''

    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None: 
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

    def __repr__(self) -> str : 
        return (
            f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, file_path='{self.file_path}')"
        )


class Modality:
    '''
    Represents a modality (e.g., RGB, Depth) containing a list of ModalityFile objects.

    Attributes:
        name (str): Name of the modality.
        files (List[ModalityFile]): List of files belonging to this modality.
    '''

    def __init__(self, name : str) -> None:
        self.name = name 
        self.files : List[ModalityFile] = []
    
    def add_file(self, subject_id: int , action_id: int, sequence_number: int, file_path: str) -> None: 
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
        return f"Modality(name='{self.name}', files={self.files})"
    

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
        self.files: Dict[str, List[str, ]] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        '''
        Adds a file to the matched trial for a specific modality

        Args:
            modality_name (str) : Name of the modality
            file_path(str) : Path to the file
        '''
        self.files[modality_name] = file_path
    
    def __repr__(self) -> str:
        return f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, files={self.files})"


class SmartFallMM:
    """
    Represents the SmartFallMM dataset, managing the loading of files and matching 
    of trials across modalities and specific sensors.

    Attributes:
        root_dir (str): Root directory of the SmartFallMM dataset.
        age_groups (Dict[str, Dict[str, Modality]]): Dictionary containing 'old' and 'young' groups, 
                                                     each having a dictionary of modality names to Modality objects.
        matched_trials (List[MatchedTrial]): List of matched trials containing files from different modalities.
        selected_sensors (Dict[str, str]): Dictionary storing selected sensors for modalities 
                                          like 'accelerometer' and 'gyroscope'. 
        fusion_options (Dict): Optional configuration for IMU fusion (filter type, etc.)
    """

    def __init__(self, root_dir: str, fusion_options: Optional[Dict] = None) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {
            "old": {},
            "young": {}
        }
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, str] = {}  # Stores the selected sensor for each modality (e.g., accelerometer)
        self.fusion_options = fusion_options or {}  # Store fusion configuration
        
        # Create directories for aligned data
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        for dir_name in ["accelerometer", "gyroscope", "skeleton"]:
            os.makedirs(os.path.join(self.aligned_data_dir, dir_name), exist_ok=True)

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

    def select_sensor(self, modality_name: str, sensor_name: str = None) -> None:
        """
        Selects a specific sensor for a given modality if applicable. Only files from this 
        sensor will be loaded for modalities like 'accelerometer' or 'gyroscope'.
        For modalities like 'skeleton', no sensor is needed.

        Args:
            modality_name (str): Name of the modality (e.g., accelerometer, gyroscope, skeleton).
            sensor_name (str): Name of the sensor (e.g., phone, watch, meta_wrist, meta_hip). None for 'skeleton'.
        """
        if modality_name == "skeleton":
            # Skeleton modality doesn't have sensor-specific data
            self.selected_sensors[modality_name] = None
        else:
            if sensor_name is None:
                raise ValueError(f"Sensor must be specified for modality '{modality_name}'")
            self.selected_sensors[modality_name] = sensor_name

    def load_files(self) -> None:
        """
        Loads files from the dataset based on selected sensors and age groups.
        Skeleton data is loaded without sensor selection.
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
                        continue

                # Load the files
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        try:
                            if file.endswith(('.csv')):
                                # Extract information based on the filename
                                subject_id = int(file[1:3])  # Assuming S001 format for subject
                                action_id = int(file[4:6])  # Assuming A001 format for action
                                sequence_number = int(file[7:9])  # Assuming T001 format for trial
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                        except Exception as e:
                            logger.error(f"Error processing file {file}: {e}")

    def match_trials(self) -> None:
        """
        Matches files from different modalities based on subject ID, action ID, and sequence number.
        Only trials that have matching files in all modalities will be kept in matched_trials.
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
        required_modalities = list(self.age_groups['young'].keys())  # Assuming all age groups have the same modalities

        for key, files_dict in trial_dict.items():
            # Check if all required modalities are present for this trial
            if all(modality in files_dict for modality in required_modalities):
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)

                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)

                self.matched_trials.append(matched_trial)

    def pipe_line(self, age_group: List[str], modalities: List[str], sensors: List[str]):
        '''
        A pipeline to load the data 
        
        Args:
            age_group: List of age groups ('young', 'old')
            modalities: List of modalities ('accelerometer', 'gyroscope', 'skeleton')
            sensors: List of sensors ('phone', 'watch', 'meta_wrist', 'meta_hip')
        '''
        for age in age_group: 
            for modality in modalities:
                self.add_modality(age, modality)
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else: 
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)

        # Load files for the selected sensors and skeleton data)
        self.load_files()

        # Match trials across the modalities
        self.match_trials()
        
        logger.info(f"Loaded {len(self.matched_trials)} matched trials")
        logger.info(f"Modalities: {modalities}")
        logger.info(f"Sensors: {sensors}")
        logger.info(f"Age groups: {age_group}")
        
        if hasattr(self, 'fusion_options') and self.fusion_options.get('filter_type'):
            logger.info(f"Using fusion with filter type: {self.fusion_options['filter_type']}")


def save_aligned_data(aligned_acc, aligned_gyro, aligned_skl, timestamps, subject_id, action_id, trial_id):
    """
    Save aligned sensor data to designated directories
    
    Args:
        aligned_acc: Aligned accelerometer data
        aligned_gyro: Aligned gyroscope data
        aligned_skl: Aligned skeleton data
        timestamps: Aligned timestamps
        subject_id: Subject identifier
        action_id: Action identifier
        trial_id: Trial identifier
        
    Returns:
        Dictionary of paths where data was saved
    """
    base_dir = os.path.join(os.getcwd(), "data/aligned")
    
    # Create directories if they don't exist
    for subdir in ["accelerometer", "gyroscope", "skeleton"]:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    # Generate filename
    filename = f"S{subject_id:02d}A{action_id:02d}T{trial_id:02d}.csv"
    paths = {}
    
    try:
        # Save accelerometer data (always linear acceleration)
        if aligned_acc is not None and len(aligned_acc) > 0:
            acc_path = os.path.join(base_dir, "accelerometer", filename)
            acc_df = pd.DataFrame(aligned_acc, columns=["x", "y", "z"])
            # Add timestamps if available
            if timestamps is not None and len(timestamps) == len(aligned_acc):
                acc_df.insert(0, "timestamp", timestamps)
            acc_df.to_csv(acc_path, index=False)
            paths["accelerometer"] = acc_path
        
        # Save gyroscope data
        if aligned_gyro is not None and len(aligned_gyro) > 0:
            gyro_path = os.path.join(base_dir, "gyroscope", filename)
            gyro_df = pd.DataFrame(aligned_gyro, columns=["x", "y", "z"])
            # Add timestamps if available
            if timestamps is not None and len(timestamps) == len(aligned_gyro):
                gyro_df.insert(0, "timestamp", timestamps)
            gyro_df.to_csv(gyro_path, index=False)
            paths["gyroscope"] = gyro_path
        
        # Save skeleton data
        if aligned_skl is not None and len(aligned_skl) > 0:
            skl_path = os.path.join(base_dir, "skeleton", filename)
            
            # Process skeleton data based on its shape
            if len(aligned_skl.shape) == 3:  # (frames, joints, 3)
                # Flatten to 2D for saving
                frames, joints, coords = aligned_skl.shape
                skl_data = aligned_skl.reshape(frames, joints * coords)
            else:
                skl_data = aligned_skl
                
            # Save without headers (consistent with original format)
            pd.DataFrame(skl_data).to_csv(skl_path, index=False, header=False)
            paths["skeleton"] = skl_path
            
        logger.info(f"Saved aligned data for S{subject_id:02d}A{action_id:02d}T{trial_id:02d}")
        return paths
        
    except Exception as e:
        logger.error(f"Error saving aligned data: {str(e)}")
        return {}


def align_and_save(data, subject_id, action_id, trial_id, aligned_dir="data/aligned"):
    """
    Align multimodal sensor data, save it to files, and return the aligned data
    
    Args:
        data: Dictionary of modality data (accelerometer, gyroscope, skeleton)
        subject_id: Subject identifier
        action_id: Action identifier
        trial_id: Trial identifier
        aligned_dir: Directory for saving aligned data
        
    Returns:
        Dictionary of aligned data
    """
    # Check if we have the necessary modalities
    required_modalities = ["accelerometer", "gyroscope"]
    if not all(modality in data for modality in required_modalities):
        missing = [m for m in required_modalities if m not in data]
        logger.warning(f"Missing required modalities for alignment: {missing}")
        return data  # Return original data if missing required modalities
    
    try:
        # Extract timestamps if they exist in the raw data
        acc_times = None
        gyro_times = None
        
        # Check if accelerometer data has timestamps (first column)
        if isinstance(data["accelerometer"], pd.DataFrame) and data["accelerometer"].shape[1] >= 4:
            acc_times = data["accelerometer"].iloc[:, 0].values
        
        # Check if gyroscope data has timestamps (first column)
        if isinstance(data["gyroscope"], pd.DataFrame) and data["gyroscope"].shape[1] >= 4:
            gyro_times = data["gyroscope"].iloc[:, 0].values
        
        # Handle numpy arrays vs DataFrames for sensor values
        acc_data = data["accelerometer"].iloc[:, 1:4].values if isinstance(data["accelerometer"], pd.DataFrame) else data["accelerometer"]
        gyro_data = data["gyroscope"].iloc[:, 1:4].values if isinstance(data["gyroscope"], pd.DataFrame) else data["gyroscope"]
        
        # Extract skeleton data if available
        skl_data = data.get("skeleton", None)
        
        # Align accelerometer and gyroscope data
        aligned_acc, aligned_gyro, aligned_times = align_sensor_data(
            pd.DataFrame(np.column_stack([acc_times, acc_data]) if acc_times is not None else acc_data),
            pd.DataFrame(np.column_stack([gyro_times, gyro_data]) if gyro_times is not None else gyro_data)
        )
        
        # Align skeleton data with inertial data if available
        aligned_skl = None
        if skl_data is not None:
            # Placeholder for skeleton alignment logic (modify as needed)
            # For simplicity, we'll just use the skeleton data as is for now
            aligned_skl = skl_data
        
        # Save the aligned data
        saved_paths = save_aligned_data(
            aligned_acc,
            aligned_gyro,
            aligned_skl,
            aligned_times,
            subject_id,
            action_id,
            trial_id
        )
        
        # Return dictionary of aligned data
        aligned_data = {
            "accelerometer": aligned_acc,
            "gyroscope": aligned_gyro,
            "aligned_timestamps": aligned_times
        }
        
        if aligned_skl is not None:
            aligned_data["skeleton"] = aligned_skl
            
        return aligned_data
        
    except Exception as e:
        logger.error(f"Error during alignment: {str(e)}")
        # Return original data in case of error
        return data


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
    
    # Create dataset object with fusion configuration
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


def split_by_subjects(builder, subjects, fuse, filter_type='madgwick', visualize=False) -> Dict[str, np.ndarray]:
    '''
    Function to filter data by expected subjects and apply fusion if needed
    
    Args:
        builder: DatasetBuilder object
        subjects: List of subject IDs to include
        fuse: Basic flag for fusion (True/False)
        filter_type: Type of filter to use ('madgwick', 'comp', 'kalman', 'ekf', 'ukf')
        visualize: Whether to generate visualizations
        
    Returns:
        Dictionary of normalized dataset components
    '''
    # Get fusion options if available
    fusion_options = getattr(builder, 'fusion_options', {})
    
    # Check if detailed fusion configuration is available
    if fusion_options and fusion_options.get('enabled', False):
        # Use the specified filter type
        filter_type_config = fusion_options.get('filter_type', 'madgwick')
        visualize_config = fusion_options.get('visualize', False)
        
        # Override with supplied parameters if provided
        filter_type = filter_type or filter_type_config
        visualize = visualize or visualize_config
        
        logger.info(f"Applying IMU fusion with filter type: {filter_type}")
        logger.info(f"Visualization enabled: {visualize}")
        
        # Create dataset with enhanced fusion options
        builder.make_dataset(subjects, True, filter_type=filter_type, visualize=visualize)
    else:
        # Use the basic fusion flag
        builder.make_dataset(subjects, fuse, filter_type=filter_type, visualize=visualize)
    
    # Apply normalization
    norm_data = builder.normalization()
    
    return norm_data


def distribution_viz(labels: np.array, work_dir: str, mode: str) -> None:
    '''
    Visualizes the distribution of labels in the dataset
    
    Args:
        labels: Array of class labels
        work_dir: Directory to save the visualization
        mode: Mode identifier (train, val, test)
    '''
    values, count = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(values, count)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title(f'{mode.capitalize()} Label Distribution')
    plt.savefig(os.path.join(work_dir, f'{mode}_label_distribution.png'))
    plt.close()
