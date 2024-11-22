import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
import re

# Modality classes

class ModalityFile:
    """
    Represents an individual file in a modality, containing the subject ID, action ID, sequence number, age group, and file path.
    """

    def __init__(self, subject_id: int, action_id: int, sequence_number: int, age_group: str, file_path: str) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.age_group = age_group
        self.file_path = file_path

    def __repr__(self) -> str:
        return (
            f"ModalityFile(subject_id={self.subject_id}, action_id={self.action_id}, "
            f"sequence_number={self.sequence_number}, age_group='{self.age_group}', file_path='{self.file_path}')"
        )

class Modality:
    """
    Represents a modality containing a list of ModalityFile objects.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.files: List[ModalityFile] = []

    def add_file(self, subject_id: int, action_id: int, sequence_number: int, age_group: str, file_path: str) -> None:
        modality_file = ModalityFile(subject_id, action_id, sequence_number, age_group, file_path)
        self.files.append(modality_file)

    def __repr__(self) -> str:
        return f"Modality(name='{self.name}', files={self.files})"

class MatchedTrial:
    """
    Represents a matched trial containing files from different modalities for the same trial.
    """

    def __init__(self, subject_id: int, action_id: int, sequence_number: int, age_group: str) -> None:
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.age_group = age_group
        self.files: Dict[str, str] = {}

    def add_file(self, modality_name: str, file_path: str) -> None:
        self.files[modality_name] = file_path

    def __repr__(self) -> str:
        return (
            f"MatchedTrial(subject_id={self.subject_id}, action_id={self.action_id}, "
            f"sequence_number={self.sequence_number}, age_group='{self.age_group}', files={self.files})"
        )

# SmartFallMM dataset class

class SmartFallMM:
    """
    Manages the loading of files and matching of trials across modalities and specific sensors.
    """

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.age_groups: Dict[str, Dict[str, Modality]] = {
            "old": {},
            "young": {}
        }
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, List[str]] = {}  # Support multiple sensors per modality

    def add_modality(self, age_group: str, modality_name: str) -> None:
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}. Expected 'old' or 'young'.")
        if modality_name not in self.age_groups[age_group]:
            self.age_groups[age_group][modality_name] = Modality(modality_name)

    def select_sensor(self, modality_name: str, sensor_name: str = None) -> None:
        if modality_name == "skeleton":
            self.selected_sensors[modality_name] = [None]
        else:
            if sensor_name is None:
                raise ValueError(f"Sensor must be specified for modality '{modality_name}'")
            if modality_name not in self.selected_sensors:
                self.selected_sensors[modality_name] = []
            self.selected_sensors[modality_name].append(sensor_name)

    def load_files(self) -> None:
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                    self._load_modality_files(modality_dir, modality, age_group)
                else:
                    if modality_name in self.selected_sensors:
                        for sensor_name in self.selected_sensors[modality_name]:
                            modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                            self._load_modality_files(modality_dir, modality, age_group, sensor_name)
                    else:
                        print(f"No sensors selected for modality '{modality_name}'. Skipping.")

    def _load_modality_files(self, modality_dir: str, modality: Modality, age_group: str, sensor_name: str = None) -> None:
        if not os.path.exists(modality_dir):
            print(f"Directory not found: {modality_dir}")
            return
        for root, _, files in os.walk(modality_dir):
            for file in files:
                if file.endswith('.csv'):
                    try:
                        # Use regex to parse filenames like 'S40A06T03.csv'
                        pattern = r'^S(\d+)A(\d+)T(\d+)\.csv$'
                        match = re.match(pattern, file)
                        if match:
                            subject_id = int(match.group(1))
                            action_id = int(match.group(2))
                            sequence_number = int(match.group(3))
                            file_path = os.path.join(root, file)
                            modality.add_file(subject_id, action_id, sequence_number, age_group, file_path)
                        else:
                            print(f"Invalid file name format: {file}. Skipping.")
                    except Exception as e:
                        print(f"Error parsing file name '{file}': {e}. Skipping.")

    def match_trials(self) -> None:
        trial_dict = {}
        required_modalities = set()

        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                required_modalities.add(modality_name)
                for modality_file in modality.files:
                    key = (age_group, modality_file.subject_id, modality_file.action_id, modality_file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    modality_key = modality_name
                    trial_dict[key][modality_key] = modality_file.file_path

        for key, files_dict in trial_dict.items():
            if required_modalities.issubset(files_dict.keys()):
                age_group, subject_id, action_id, sequence_number = key
                matched_trial = MatchedTrial(subject_id, action_id, sequence_number, age_group)
                for modality_name, file_path in files_dict.items():
                    matched_trial.add_file(modality_name, file_path)
                self.matched_trials.append(matched_trial)

# Data processing functions

    def csvloader(file_path: str, modality_name: str, **kwargs):
        try:
            file_data = pd.read_csv(file_path, index_col=False, header=0)
            if modality_name == 'accelerometer':
                data = file_data.iloc[:, -3:]  # Select last 3 columns for x, y, z
            else:
                data = file_data
            # Convert to numeric and fill NaNs
            data = data.apply(pd.to_numeric, errors='coerce')
            if data.isnull().values.all():
                print(f"All data is NaN in file '{file_path}'. Skipping.")
                return None
            data = data.fillna(method='ffill').fillna(method='bfill')
            return data.to_numpy(dtype=np.float32)
        except Exception as e:
            print(f"Error loading file '{file_path}': {e}")
            return None

    def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
        original_length = sequence.shape[0]
        if original_length == target_length:
            return sequence
        elif original_length < target_length:
            # Pad the sequence
            padding = np.zeros((target_length - original_length, sequence.shape[1]))
            return np.vstack((sequence, padding))
        else:
            # Truncate or resample
            indices = np.linspace(0, original_length - 1, num=target_length)
            resampled_sequence = np.zeros((target_length, sequence.shape[1]))
            for i in range(sequence.shape[1]):
                resampled_sequence[:, i] = np.interp(indices, np.arange(original_length), sequence[:, i])
            return resampled_sequence


class DatasetBuilder:
    """
    Builds a dataset by processing and normalizing data files.
    """

    def __init__(self, dataset, mode, max_length, task='fd', **kwargs):
        self.dataset = dataset
        self.data: Dict[str, List[np.ndarray]] = {}
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.kwargs = kwargs

    def make_dataset(self, subjects: List[int]) -> None:
        self.data = {'labels': [], 'subject_ids': []}
        total_trials = 0
        skipped_trials = 0
        for trial in self.dataset.matched_trials:
            total_trials += 1
            if trial.subject_id in subjects:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                else:
                    label = trial.action_id - 1

                valid_data = True
                trial_data = {}

                for modality, file_path in trial.files.items():
                    data = csvloader(file_path, modality)
                    if data is None:
                        print(f"Data is None for modality '{modality}' in trial {trial}. Skipping this trial.")
                        valid_data = False
                        break
                    data = resample_sequence(data, self.max_length)
                    if data is None:
                        print(f"Resampled data is None for modality '{modality}' in trial {trial}. Skipping this trial.")
                        valid_data = False
                        break
                    trial_data[modality] = data

                if valid_data:
                    self.data['labels'].append(label)
                    self.data['subject_ids'].append(trial.subject_id)
                    for modality in trial_data:
                        if modality not in self.data:
                            self.data[modality] = []
                        self.data[modality].append(trial_data[modality])
                else:
                    skipped_trials += 1
                    print(f"Total trials: {total_trials}, Skipped trials: {skipped_trials}, Processed trials: {total_trials - skipped_trials}")
        # Convert lists to arrays and check if they are empty
        for key in self.data:
            if key not in ['labels', 'subject_ids']:
                if len(self.data[key]) > 0:
                    # Ensure all arrays have the same shape before stacking
                    shapes = [arr.shape for arr in self.data[key]]
                    if len(set(shapes)) > 1:
                        print(f"Inconsistent shapes in data for modality '{key}': {set(shapes)}")
                    self.data[key] = np.stack(self.data[key], axis=0)
                else:
                    print(f"No data available for modality '{key}'.")
                    self.data[key] = np.empty((0,))
            else:
                self.data[key] = np.array(self.data[key])

        # Corrected check for empty labels array
        if self.data['labels'].size == 0:
            print("No data found for the specified subjects.")
            return

    def normalization(self) -> Dict[str, np.ndarray]:
        """
        Normalizes the data.
        """
        for key, value in self.data.items():
            if key not in ['labels', 'subject_ids']:
                if value.size == 0:
                    print(f"No data to normalize for modality '{key}'.")
                    continue
                num_samples, length = value.shape[:2]
                # Flatten for scaling
                reshaped_data = value.reshape(num_samples * length, -1)
                scaler = StandardScaler().fit(reshaped_data)
                norm_data = scaler.transform(reshaped_data)
                self.data[key] = norm_data.reshape(num_samples, length, -1)
        return self.data

# Dataset class

class UTD_mm(Dataset):
    """
    PyTorch Dataset class for UTD multimodal data.
    """

    def __init__(self, dataset):
        self.modalities = [key for key in dataset.keys() if key not in ['labels', 'subject_ids']]
        self.data = dataset
        self.labels = dataset['labels']
        self.num_samples = len(self.labels)

        # Ensure consistency across modalities
        for modality in self.modalities:
            if len(self.data[modality]) != self.num_samples:
                raise ValueError(f"Inconsistent sample count in modality '{modality}' and labels.")

    def __len__(self):
        return self.num_samples

    @staticmethod
    def calculate_smv(sample: torch.Tensor) -> torch.Tensor:
        """
        Calculate Signal Magnitude Vector (SMV) from accelerometer data.
        """
        return torch.sqrt(torch.sum(sample ** 2, dim=-1, keepdim=True))

    def __getitem__(self, index):
        """
        Get a single sample by index.
        """
        data = {}
        for modality in self.modalities:
            modality_data = torch.tensor(self.data[modality][index], dtype=torch.float32)
            if modality in ['watch', 'phone', 'accelerometer']:
                smv = self.calculate_smv(modality_data)
                modality_data = torch.cat((smv, modality_data), dim=-1)
            data[modality] = modality_data
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return data, label, index

# Prepare dataset functions

def prepare_smartfallmm(arg):
    """
    Function for dataset preparation.
    """
    # Initialize the SmartFallMM dataset
    sm_dataset = SmartFallMM(root_dir=arg.dataset_args['root_dir'])
    for age_group in arg.dataset_args['age_group']:
        for modality in arg.dataset_args['modalities']:
            sm_dataset.add_modality(age_group, modality)
            if modality == 'skeleton':
                sm_dataset.select_sensor(modality)
            else:
                for sensor in arg.dataset_args['sensors'][modality]:
                    sm_dataset.select_sensor(modality, sensor)
    # Load files and match trials
    sm_dataset.load_files()
    sm_dataset.match_trials()
    print(f"Number of matched trials: {len(sm_dataset.matched_trials)}")
    matched_subjects = set(trial.subject_id for trial in sm_dataset.matched_trials)
    print(f"Subjects in matched trials: {matched_subjects}")

    # Build the dataset
    builder = DatasetBuilder(sm_dataset, arg.dataset_args['mode'], arg.dataset_args['max_length'],
                             arg.dataset_args['task'])
    subjects = arg.subjects
    builder.make_dataset(subjects)
    # Check if any data was loaded
    if not builder.data.get('labels'):
        print("No data was loaded. Exiting dataset preparation.")
        return None
    norm_data = builder.normalization()
    return norm_data

def prepare_dataset(arg):
    """
    Prepares the dataset and splits it into train, val, and test sets.
    """
    # Prepare the data
    dataset = prepare_smartfallmm(arg)
    if dataset is None:
        return None

    # Now split the data into train, val, test based on subject IDs
    labels = np.array(dataset['labels'])
    subject_ids = np.array(dataset['subject_ids'])
    data = {key: np.array(dataset[key]) for key in dataset if key not in ['labels', 'subject_ids']}

    # Shuffle subject IDs before splitting
    unique_subject_ids = np.unique(subject_ids)
    np.random.shuffle(unique_subject_ids)

    # Split data based on shuffled unique subject IDs
    num_subjects = len(unique_subject_ids)
    if num_subjects == 0:
        print("No subjects found in the data.")
        return None
    train_subjects = unique_subject_ids[:int(0.7 * num_subjects)]
    val_subjects = unique_subject_ids[int(0.7 * num_subjects):int(0.85 * num_subjects)]
    test_subjects = unique_subject_ids[int(0.85 * num_subjects):]

    train_indices = np.isin(subject_ids, train_subjects)
    val_indices = np.isin(subject_ids, val_subjects)
    test_indices = np.isin(subject_ids, test_subjects)

    # Optional: Check class distribution
    def check_class_distribution(labels, indices):
        unique, counts = np.unique(labels[indices], return_counts=True)
        distribution = dict(zip(unique, counts))
        return distribution

    print("Training set class distribution:", check_class_distribution(labels, train_indices))
    print("Validation set class distribution:", check_class_distribution(labels, val_indices))
    print("Test set class distribution:", check_class_distribution(labels, test_indices))

    dataset_dict = {
        'train': {
            **{modality: data[modality][train_indices] for modality in data},
            'labels': labels[train_indices]
        },
        'val': {
            **{modality: data[modality][val_indices] for modality in data},
            'labels': labels[val_indices]
        },
        'test': {
            **{modality: data[modality][test_indices] for modality in data},
            'labels': labels[test_indices]
        }
    }

    return dataset_dict

# If you want to run this module independently
if __name__ == "__main__":
    import argparse
    import yaml

    # Create an argument parser
    parser = argparse.ArgumentParser(description='SmartFallMM Data Loader')
    parser.add_argument('--config', type=str, help='Path to the config file')
    args_cmd = parser.parse_args()

    # Load config
    with open(args_cmd.config, 'r') as f:
        config = yaml.safe_load(f)
    arg = argparse.Namespace(**config)

    print("Preparing dataset...")
    dataset_dict = prepare_dataset(arg)
    if dataset_dict is None:
        print("Dataset preparation failed.")
    else:
        print("Dataset prepared successfully!")

        # Create the PyTorch datasets
        train_dataset = UTD_mm(dataset_dict['train'])
        val_dataset = UTD_mm(dataset_dict['val'])
        test_dataset = UTD_mm(dataset_dict['test'])

        # You can now use train_dataset, val_dataset, test_dataset with DataLoader for training your model
