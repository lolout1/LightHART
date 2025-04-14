from typing import List, Dict
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
        return f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, file_path='{self.file_path}')"

class Modality:
    def __init__(self, name: str) -> None:
        self.name = name
        self.files: List[ModalityFile] = []
    
    def add_file(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        modality_file = ModalityFile(subject_id, action_id, sequence_number, file_path)
        self.files.append(modality_file)
    
    def __repr__(self) -> str:
        return f"Modality(name='{self.name}', files={self.files})"

class MatchedTrial:
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}
    
    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path
    
    def __repr__(self) -> str:
        return f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, sequence_number={self.sequence_number}, files={self.files})"

class SmartFallMM:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {"old": {}, "young": {}}
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, str] = {}

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
                            if file.endswith(('.csv')):
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                        except:
                            pass

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
            if all(modality in files_dict for modality in required_modalities):
                subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number)
                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)
                self.matched_trials.append(matched_trial)

    def _find_or_create_matched_trial(self, subject_id: int, action_id: int, sequence_number: int) -> MatchedTrial:
        for trial in self.matched_trials:
            if (trial.subject_id == subject_id and trial.action_id == action_id
                    and trial.sequence_number == sequence_number):
                return trial
        new_trial = MatchedTrial(subject_id, action_id, sequence_number)
        self.matched_trials.append(new_trial)
        return new_trial

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

def prepare_smartfallmm(arg) -> DatasetBuilder:
    sm_dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data/smartfallmm'))
    sm_dataset.pipe_line(age_group=arg.dataset_args['age_group'], 
                        modalities=arg.dataset_args['modalities'], 
                        sensors=arg.dataset_args['sensors'])
    builder = DatasetBuilder(sm_dataset, arg.dataset_args['mode'], arg.dataset_args['max_length'],
                             arg.dataset_args['task'])
    return builder

def split_by_subjects(builder, subjects, fuse) -> Dict[str, np.ndarray]:
    builder.make_dataset(subjects, fuse)
    norm_data = builder.normalization()
    return norm_data
