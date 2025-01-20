from typing import List, Dict, Optional
import os
import numpy as np
from utils.loader import DatasetBuilder

class ModalityFile:
    """
    Represents a single data file containing sensor readings for one trial.
    """
    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

    def __repr__(self) -> str:
        return (
            f"ModalityFile(subject_id={self.subject_id}, "
            f"action_id={self.action_id}, "
            f"sequence_number={self.sequence_number}, "
            f"file_path='{self.file_path}')"
        )

class Modality:
    """
    Groups all files for a particular sensor type (e.g., accelerometer, skeleton).
    """
    def __init__(self, name: str) -> None:
        self.name = name
        self.files: List[ModalityFile] = []

    def add_file(self,
                 subject_id: int,
                 action_id: int,
                 sequence_number: int,
                 file_path: str) -> None:
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)

class MatchedTrial:
    """
    Groups synchronized data from multiple modalities for a single trial.
    """
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}

    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path

class SmartFallMM:
    """
    Main dataset class for the SmartFallMM dataset, handling data organization and loading.
    """
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {
            "old": {},
            "young": {}
        }
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, Optional[str]] = {}

    def add_modality(self, age_group: str, modality_name: str) -> None:
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}. Expected 'old' or 'young'")
        self.age_groups[age_group][modality_name] = Modality(modality_name)

    def select_sensor(self, modality_name: str, sensor_name: Optional[str] = None) -> None:
        if modality_name == "skeleton":
            self.selected_sensors[modality_name] = None
        else:
            if sensor_name is None:
                raise ValueError(f"Sensor must be specified for modality '{modality_name}'")
            self.selected_sensors[modality_name] = sensor_name

    def load_files(self) -> None:
        loaded_files = 0
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    if modality_name not in self.selected_sensors:
                        continue
                    sensor_name = self.selected_sensors[modality_name]
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)

                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            try:
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                                loaded_files += 1
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Skipping malformed filename: {file} ({str(e)})")
                                continue
        print(f"Successfully loaded {loaded_files} files")

    def match_trials(self) -> None:
        trial_dict = {}
        file_counts = {
            modality: 0
            for age_group in self.age_groups
            for modality in self.age_groups[age_group]
        }

        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                file_counts[modality_name] += len(modality.files)
                for modality_file in modality.files:
                    key = (modality_file.subject_id,
                           modality_file.action_id,
                           modality_file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][modality_name] = modality_file.file_path

        complete_trials = 0
        fall_trials = 0
        non_fall_trials = 0
        required_modalities = list(self.age_groups['young'].keys())

        for key, files_dict in trial_dict.items():
            if all(modality in files_dict for modality in required_modalities):
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)
                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)
                self.matched_trials.append(matched_trial)
                complete_trials += 1
                if action_id > 9:
                    fall_trials += 1
                else:
                    non_fall_trials += 1

        print("\nDataset Statistics:")
        print("-" * 50)
        print("Raw File Counts:")
        for modality, count in file_counts.items():
            print(f"{modality}: {count} files")
        print("-" * 50)
        print(f"Complete Matched Trials: {complete_trials}")
        print(f"Fall Trials: {fall_trials}")
        print(f"Non-Fall Trials: {non_fall_trials}")
        if non_fall_trials > 0:
            print(f"Fall/Non-Fall Ratio: {fall_trials/non_fall_trials:.2f}")
        print("-" * 50)

    def pipe_line(self, age_group: List[str],
                  modalities: List[str],
                  sensors: List[str]) -> None:
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

def prepare_smartfallmm(arg) -> DatasetBuilder:
    sm_dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'))
    sm_dataset.pipe_line(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    builder = DatasetBuilder(
        dataset=sm_dataset,
        mode=arg.dataset_args['mode'],
        max_length=arg.dataset_args['max_length'],
        window_size=arg.dataset_args.get('window_size', 64),
        stride_size=arg.dataset_args.get('stride_size', 10),
        task=arg.dataset_args['task']
    )
    return builder

def filter_subjects(builder: DatasetBuilder, subjects: List[int]) -> Dict[str, np.ndarray]:
    """
    Process data for specific subjects.
    """
    builder.make_dataset(subjects)
    return builder.normalization()
