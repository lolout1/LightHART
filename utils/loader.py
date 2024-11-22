# utils/loader.py

from typing import List, Dict
import os
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

from .shared import SmartFallMM  # Import SmartFallMM from shared.py

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """Applies a Butterworth filter to the data."""
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0)


class DatasetBuilder:
    """Builds datasets for training and evaluation."""

    def __init__(self, dataset: SmartFallMM, mode: str, max_length: int, task: str = 'fd') -> None:
        self.dataset = dataset
        self.data: Dict[str, List[np.array]] = {}
        self.mode = mode
        self.max_length = max_length
        self.task = task

    def make_dataset(self, subjects: List[int]):
        '''
        Reads all the files and makes a numpy array with all data
        '''
        # Re-initialize self.data at the start of the method
        self.data = {}
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                label = self._get_label(trial)
                # Ensure self.data['labels'] is a list before appending
                if 'labels' not in self.data or not isinstance(self.data['labels'], list):
                    self.data['labels'] = []
                self.data['labels'].append(label)

                for modality, file_path in trial.files.items():
                    try:
                        unimodal_data = self._process_file(file_path, modality)
                        self.data.setdefault(modality, []).append(unimodal_data)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        os.remove(file_path)
        # After processing, convert lists to NumPy arrays
        for key in self.data:
            self.data[key] = np.stack(self.data[key], axis=0)

    def _get_label(self, trial):
        """Generates a label for the trial based on the task."""
        if self.task == 'fd':
            # For fall detection, label falls as 1 and non-falls as 0
            return int(trial.action_id > 9)
        elif self.task == 'age':
            # For age classification, label young subjects as 0 and old subjects as 1
            return int(trial.subject_id < 29 or trial.subject_id > 46)
        else:
            # Default behavior: use action ID as label
            return trial.action_id - 1

    def _process_file(self, file_path, modality):
        """Processes a single file and returns the data."""
        # Read the CSV file; adjust delimiter and header as needed
        data = np.loadtxt(file_path, delimiter=',')
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # Trim or pad the data to match max_length
        if data.shape[0] > self.max_length:
            data = data[:self.max_length, :]
        elif data.shape[0] < self.max_length:
            padding = np.zeros((self.max_length - data.shape[0], data.shape[1]))
            data = np.vstack((data, padding))
        return data

    def _finalize_data(self):
        """Finalizes the dataset structure."""
        for key in self.data:
            if key != 'labels':
                # Stack the data for each modality into a single numpy array
                self.data[key] = np.stack(self.data[key], axis=0)
            else:
                # Convert labels list to numpy array
                self.data[key] = np.array(self.data[key])

    def normalization(self) -> Dict[str, np.ndarray]:
        """Normalizes the data using StandardScaler."""
        for key, value in self.data.items():
            if key != 'labels':
                # Reshape data for normalization
                num_samples, length, channels = value.shape
                reshaped_data = value.reshape(num_samples * length, channels)
                # Apply normalization
                scaler = StandardScaler()
                norm_data = scaler.fit_transform(reshaped_data)
                # Reshape back to original dimensions
                self.data[key] = norm_data.reshape(num_samples, length, channels)
        return self.data
