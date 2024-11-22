from typing import List, Dict

class MatchedTrial:
    """Represents a matched trial containing files from different modalities for the same trial."""
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
    """Represents the SmartFallMM dataset, managing file loading and trial matching."""
    def __init__(self, root_dir: str, age_groups: List[str], modalities: List[str], sensors: Dict[str, List[str]]) -> None:
        self.root_dir = root_dir
        self.age_groups = age_groups
        self.modalities = modalities
        self.sensors = sensors
        self.matched_trials: List[MatchedTrial] = []

    def load_files(self) -> None:
        """Loads files from the dataset and matches trials."""
        trial_dict = {}
        for age_group in self.age_groups:
            for modality in self.modalities:
                sensor_list = self.sensors.get(modality, [None])
                for sensor in sensor_list:
                    modality_dir = f"{self.root_dir}/{age_group}/{modality}/{sensor}" if sensor else f"{self.root_dir}/{age_group}/{modality}"
                    self._load_modality_files(modality, sensor, modality_dir, trial_dict)
        self._finalize_trials(trial_dict)

    def _load_modality_files(self, modality, sensor, modality_dir, trial_dict):
        """Helper function to load files for a given modality."""
        import os
        for root, _, files in os.walk(modality_dir):
            for file in files:
                if file.endswith('.csv'):
                    subject_id, action_id, sequence_number = self._parse_filename(file)
                    key = (subject_id, action_id, sequence_number)
                    trial_dict.setdefault(key, {})
                    trial_dict[key][f"{modality}_{sensor}" if sensor else modality] = f"{root}/{file}"

    def _finalize_trials(self, trial_dict):
        """Converts trial_dict into a list of MatchedTrial objects."""
        for key, files in trial_dict.items():
            trial = MatchedTrial(*key)
            trial.files.update(files)
            self.matched_trials.append(trial)

    @staticmethod
    def _parse_filename(filename):
        """Parses subject, action, and sequence from a file name."""
        subject_id = int(filename[1:3])
        action_id = int(filename[4:6])
        sequence_number = int(filename[7:9])
        return subject_id, action_id, sequence_number
