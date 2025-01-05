"""
File: dataset.py

Description:
    - Contains classes for ModalityFile, MatchedTrial, and SmartFallMM
      tailored to a specific sensor-based fall detection dataset.
    - Provides a pipeline to configure, load, and match data files.
    - Ties into the DatasetBuilder from loader.py for actual processing.
"""

import os
from typing import List, Dict, Optional
import numpy as np

from utils.loader import DatasetBuilder


# ---------------------- DATA CLASSES ---------------------- #

class ModalityFile: 
    """
    Simple container: subject_id, action_id, sequence_number, file_path.
    """
    def __init__(self, subject_id: int, action_id: int, sequence_number: int, file_path: str) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

    def __repr__(self) -> str:
        return (f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, "
                f"sequence_number={self.sequence_number}, file_path='{self.file_path}')")


class MatchedTrial:
    """
    Groups files from multiple modalities/sensors into a single trial.
    """
    def __init__(self, subject_id: int, action_id: int, sequence_number: int) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}  # key = 'modality_sensor', val = file_path
    
    def add_file(self, modality_sensor_key: str, file_path: str) -> None:
        """
        Adds a file for this subject/action/trial.
        Example: 'accelerometer_phone' -> path/to/data.csv
        """
        self.files[modality_sensor_key] = file_path
    
    def __repr__(self) -> str:
        return (f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, "
                f"sequence_number={self.sequence_number}, files={self.files})")


class SmartFallMM:
    """
    Main class for managing the SmartFallMM dataset:
    - Add modalities per age group
    - Load files from disk
    - Match trials
    - Integrates with DatasetBuilder for final usage
    """
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.matched_trials: List[MatchedTrial] = []
        self.modality_sensors: Dict[str, List[str]] = {}
        self.sampling_rate = 31.25  # Default ~32 Hz

    def add_modality(self, age_group: str, modality_name: str, sensors: Optional[List[str]] = None) -> None:
        """
        Add a modality under a given age group, with optional sensors.
        e.g. ("young", "accelerometer", ["phone","watch"])
        """
        modality_key = f"{age_group}_{modality_name}"
        self.modality_sensors[modality_key] = sensors if sensors else [None]
        print(f"Added modality {modality_key} with sensors {sensors}")

    def load_files(self) -> None:
        """
        Traverse directories, parse filenames, and populate matched trials.
        Directory structure example:
          root_dir/young/accelerometer/phone/*.csv
        """
        needs_cleaning_dir = os.path.join(self.root_dir, 'needs_cleaning')
        os.makedirs(needs_cleaning_dir, exist_ok=True)
        print(f"Using needs_cleaning directory: {needs_cleaning_dir}")

        for modality_key, sensors in self.modality_sensors.items():
            age_group, modality_name = modality_key.split('_', 1)

            for sensor in sensors:
                if sensor is None:
                    data_path = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    data_path = os.path.join(self.root_dir, age_group, modality_name, sensor)

                print(f"Looking for files in: {data_path}")
                if not os.path.exists(data_path):
                    print(f"Warning: Path does not exist: {data_path}")
                    continue

                files_found = 0
                files_with_errors = 0
                for file in os.listdir(data_path):
                    if file.endswith('.csv'):
                        files_found += 1
                        file_path = os.path.join(data_path, file)
                        try:
                            # Attempt to parse: SxxAyyTzz.csv
                            # e.g. S01A02T03.csv
                            subject_id = int(file[1:3])   # '01'
                            action_id = int(file[4:6])    # '02'
                            sequence_number = int(file[7:9])  # '03'

                            trial = self._find_or_create_matched_trial(subject_id, action_id, sequence_number)
                            key_ = f"{modality_name}_{sensor}" if sensor else modality_name
                            trial.add_file(key_, file_path)
                        except (ValueError, IndexError) as e:
                            files_with_errors += 1
                            print(f"Error processing file {file}: {str(e)}")
                            self._handle_needs_cleaning(file, data_path, needs_cleaning_dir)
                print(f"Found {files_found} CSV files in {data_path}, {files_with_errors} errors -> cleaning dir")

    def _handle_needs_cleaning(self, file: str, src_dir: str, cleaning_dir: str) -> None:
        """
        Copies problematic file to 'needs_cleaning' subfolder
        to preserve directory structure.
        """
        import shutil
        relative_path = os.path.relpath(src_dir, self.root_dir)
        target_dir = os.path.join(cleaning_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)
        
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(target_dir, file)
        try:
            shutil.copy2(src_file, dest_file)
            print(f"Successfully copied {file} to {dest_file} for cleaning")
        except Exception as err:
            print(f"Error copying file {file}: {err}")

    def match_trials(self) -> None:
        """
        Ensures each trial has all required modality_sensor keys. 
        If not, it's discarded.
        """
        required_keys = set()
        for mk, sensors in self.modality_sensors.items():
            _, modality_name = mk.split('_', 1)
            if sensors[0] is None:
                required_keys.add(modality_name)
            else:
                for sensor in sensors:
                    required_keys.add(f"{modality_name}_{sensor}")

        print(f"Before matching: {len(self.matched_trials)} trials")
        complete_trials = []
        for trial in self.matched_trials:
            if all(k in trial.files for k in required_keys):
                complete_trials.append(trial)
        print(f"After matching: {len(complete_trials)} trials (complete)")
        self.matched_trials = complete_trials

    def _find_or_create_matched_trial(self, subject_id: int, action_id: int, sequence_number: int) -> MatchedTrial:
        for trial in self.matched_trials:
            if (trial.subject_id == subject_id and
                trial.action_id == action_id and
                trial.sequence_number == sequence_number):
                return trial
        new_t = MatchedTrial(subject_id, action_id, sequence_number)
        self.matched_trials.append(new_t)
        return new_t

    def pipe_line(self, age_groups: List[str], modalities: List[str], sensors: Dict[str, List[str]]) -> None:
        """
        High-level pipeline to add modality_sensors, then load and match.
        """
        print(f"Pipeline input: age_groups={age_groups}, modalities={modalities}, sensors={sensors}")
        for ag in age_groups:
            for mod in modalities:
                if mod == 'skeleton':
                    self.add_modality(ag, mod)
                else:
                    sensor_list = sensors.get(mod, [])
                    self.add_modality(ag, mod, sensor_list)

        print("Configured modalities:", self.modality_sensors)
        self.load_files()
        self.match_trials()

    def select_sensor(self, modality: str, sensor: Optional[str] = None) -> None:
        """
        Adjust existing sensor lists for a given modality, restricting
        to just one sensor if provided (or None).
        """
        updated = {}
        for key, sensor_list in self.modality_sensors.items():
            _, mod = key.split('_', 1)
            if mod == modality:
                if sensor is None:
                    updated[key] = [None]
                else:
                    updated[key] = [sensor]
            else:
                updated[key] = sensor_list
        self.modality_sensors = updated
        print(f"Selected sensor {sensor} for modality {modality}")


# -------------- PREPARE & FILTER FUNCTIONS -------------- #

def prepare_smartfallmm(arg) -> DatasetBuilder:
    """
    Generic function for dataset preparation.
    Expects 'arg.dataset_args' to contain:
        - 'root_dir'
        - 'age_groups'
        - 'modalities'
        - 'sensors'
        - 'mode'
        - 'max_length'
        - 'task'
    """
    print("\nDataset preparation starting...")
    print(f"Current working directory: {os.getcwd()}")
    root_dir = os.path.join(os.getcwd(), arg.dataset_args['root_dir'])
    print(f"Full data path: {root_dir}")
    print(f"Path exists: {os.path.exists(root_dir)}")

    # Attempt to list contents
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
    """
    Filter out specific subjects, build dataset, then normalize.
    """
    builder.make_dataset(subjects)
    norm_data = builder.normalization()
    return norm_data


# -------------- STANDALONE USAGE EXAMPLE -------------- #

if __name__ == "__main__":
    dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data', 'smartfallmm'))

    # Add modalities for 'young' / 'old'
    dataset.add_modality("young", "accelerometer")
    dataset.add_modality("young", "skeleton")
    dataset.add_modality("old", "accelerometer")
    dataset.add_modality("old", "skeleton")

    # Select the sensor type for accelerometer
    dataset.select_sensor("accelerometer", "phone")
    # For skeleton, no sensor
    dataset.select_sensor("skeleton")

    # Load and match
    dataset.load_files()
    dataset.match_trials()

    # Build dataset
    builder = DatasetBuilder(dataset, mode="avg_pool", max_length=256, task='fd')
    builder.make_dataset(subjects=[1, 2])
    normalized = builder.normalization()

    # Optional visualization
    builder.visualize_trial(trial_index=0)
