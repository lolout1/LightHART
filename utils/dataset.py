from typing import List, Dict, Optional
import os
import numpy as np
from utils.loader import DatasetBuilder


class ModalityFile: 
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


class MatchedTrial:
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}  # modality_sensor -> file_path
    
    def add_file(self, modality_sensor_key: str, file_path: str) -> None:
        """
        Adds a file to the matched trial

        Args:
            modality_sensor_key (str): Combined key of modality_sensor (e.g. 'accelerometer_phone')
            file_path (str): Path to the data file
        """
        self.files[modality_sensor_key] = file_path
    
    def __repr__(self) -> str:
        return f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, files={self.files})"


class SmartFallMM:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.matched_trials: List[MatchedTrial] = []
        self.modality_sensors: Dict[str, List[str]] = {}
        # Added sampling rate parameter
        self.sampling_rate = 31.25  # 32ms interval

    def add_modality(self, age_group: str, modality_name: str, sensors: Optional[List[str]] = None) -> None:
        """
        Add a modality with its associated sensors to the dataset.
        
        Args:
            age_group (str): The age group ('young' or 'old')
            modality_name (str): Name of the modality (e.g., 'accelerometer')
            sensors (Optional[List[str]]): List of sensors for this modality. None for skeleton.
        """
        modality_key = f"{age_group}_{modality_name}"
        self.modality_sensors[modality_key] = sensors if sensors else [None]
        print(f"Added modality {modality_key} with sensors {sensors}")

    def load_files(self) -> None:
        """Loads files from the directory structure into matched trials"""
        for modality_key, sensors in self.modality_sensors.items():
            age_group, modality_name = modality_key.split('_', 1)
            
            for sensor in sensors:
                # Construct path
                if sensor:
                    data_path = os.path.join(
                        self.root_dir,
                        age_group,
                        modality_name,
                        sensor
                    )
                else:
                    data_path = os.path.join(
                        self.root_dir,
                        age_group,
                        modality_name
                    )

                print(f"Looking for files in: {data_path}")
                if not os.path.exists(data_path):
                    print(f"Warning: Path does not exist: {data_path}")
                    # Print parent directory contents
                    parent_dir = os.path.dirname(data_path)
                    if os.path.exists(parent_dir):
                        print(f"Contents of {parent_dir}:")
                        print(os.listdir(parent_dir))
                    continue

                files_found = 0
                for file in os.listdir(data_path):
                    if file.endswith('.csv'):
                        files_found += 1
                        try:
                            subject_id = int(file[1:3])
                            action_id = int(file[4:6])
                            sequence_number = int(file[7:9])
                            file_path = os.path.join(data_path, file)
                            
                            # Find or create matching trial
                            trial = self._find_or_create_matched_trial(subject_id, action_id, sequence_number)
                            modality_sensor_key = f"{modality_name}_{sensor}" if sensor else modality_name
                            trial.add_file(modality_sensor_key, file_path)

                        except (ValueError, IndexError) as e:
                            print(f"Error processing file {file}: {str(e)}")
                print(f"Found {files_found} CSV files in {data_path}")

    def match_trials(self) -> None:
        """Enhanced trial matching with quality checks for fall detection"""
        print(f"Before matching: {len(self.matched_trials)} trials")
        complete_trials = []
        
        required_keys = set()
        for modality_key, sensors in self.modality_sensors.items():
            _, modality_name = modality_key.split('_', 1)
            if sensors[0] is None:
                required_keys.add(modality_name)
            else:
                for sensor in sensors:
                    required_keys.add(f"{modality_name}_{sensor}")
        
        # Enhanced quality checking for fall detection
        for trial in self.matched_trials:
            if all(key in trial.files for key in required_keys):
                # Additional quality checks for fall detection
                has_valid_data = True
                for key, file_path in trial.files.items():
                    try:
                        # Quick validation of file contents
                        with open(file_path, 'r') as f:
                            first_line = f.readline()
                            if not first_line:
                                has_valid_data = False
                                break
                    except Exception:
                        has_valid_data = False
                        break
                
                if has_valid_data:
                    complete_trials.append(trial)
            
        print(f"After matching and validation: {len(complete_trials)} complete trials")
        self.matched_trials = complete_trials

    def _find_or_create_matched_trial(self, subject_id: int, action_id: int, sequence_number: int) -> MatchedTrial:
        """Find or create a trial matching the given identifiers"""
        for trial in self.matched_trials:
            if (trial.subject_id == subject_id and 
                trial.action_id == action_id and 
                trial.sequence_number == sequence_number):
                return trial
        new_trial = MatchedTrial(subject_id, action_id, sequence_number)
        self.matched_trials.append(new_trial)
        return new_trial

    def pipe_line(self, age_groups: List[str], modalities: List[str], sensors: Dict[str, List[str]]) -> None:
        """Pipeline to load and match data"""
        print(f"Pipeline input: age_groups={age_groups}, modalities={modalities}, sensors={sensors}")
        for age_group in age_groups:
            for modality in modalities:
                if modality == 'skeleton':
                    self.add_modality(age_group, modality)
                else:
                    sensor_list = sensors.get(modality, [])
                    self.add_modality(age_group, modality, sensor_list)

        print("Configured modalities:", self.modality_sensors)
        self.load_files()
        self.match_trials()

    def select_sensor(self, modality: str, sensor: Optional[str] = None) -> None:
        """
        Select a specific sensor for a modality.
        
        Args:
            modality (str): The modality to select sensor for (e.g., 'accelerometer')
            sensor (Optional[str]): The sensor to select (e.g., 'phone'). None for modalities without sensors.
        """
        # Update modality_sensors to only include the selected sensor
        updated_sensors = {}
        for key, sensors in self.modality_sensors.items():
            _, mod = key.split('_', 1)
            if mod == modality:
                if sensor is None:
                    updated_sensors[key] = [None]
                else:
                    updated_sensors[key] = [sensor]
            else:
                updated_sensors[key] = sensors
        
        self.modality_sensors = updated_sensors
        print(f"Selected sensor {sensor} for modality {modality}")


def prepare_smartfallmm(arg) -> DatasetBuilder:
    """Function for dataset preparation"""
    print("\nDataset preparation starting...")
    print(f"Current working directory: {os.getcwd()}")
    root_dir = os.path.join(os.getcwd(), arg.dataset_args['root_dir'])
    print(f"Full data path: {root_dir}")
    print(f"Path exists: {os.path.exists(root_dir)}")
    
    # Try to list contents of data directory
    try:
        print(f"\nContents of {os.path.dirname(root_dir)}:")
        print(os.listdir(os.path.dirname(root_dir)))
    except Exception as e:
        print(f"Could not list directory contents: {e}")
    
    sm_dataset = SmartFallMM(root_dir=root_dir)
    sm_dataset.pipe_line(
        age_groups=arg.dataset_args['age_groups'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    builder = DatasetBuilder(
        sm_dataset,
        arg.dataset_args['mode'],
        arg.dataset_args['max_length'],
        arg.dataset_args['task']
    )
    return builder


def filter_subjects(builder: DatasetBuilder, subjects: List[int]) -> Dict[str, np.ndarray]:
    builder.make_dataset(subjects)
    norm_data = builder.normalization()
    return norm_data

if __name__ == "__main__":
    dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'))

    # Add modalities for 'young' age group
    dataset.add_modality("young", "accelerometer")
    dataset.add_modality("young", "skeleton")

    # Add modalities for 'old' age group
    dataset.add_modality("old", "accelerometer")
    dataset.add_modality("old", "skeleton")

    # Select the sensor type for accelerometer and gyroscope
    dataset.select_sensor("accelerometer", "phone")

    # For skeleton, no sensor needs to be selected
    dataset.select_sensor("skeleton")

    # Load files for the selected sensors and skeleton data
    dataset.load_files()

    # Match trials across the modalities
    dataset.match_trials()
