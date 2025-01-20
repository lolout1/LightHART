"""
Dataset Builder
"""
import os
from typing import List, Dict
import numpy as np

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

from utils.processor.base import Processor

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """Function to filter noise."""
    if data is None or len(data) == 0:
        return data

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    required_padlen = 3 * max(len(a), len(b))
    if data.shape[0] <= required_padlen:
        padlen = data.shape[0] - 1
        if padlen < 1:
            return data
        return filtfilt(b, a, data, axis=0, padlen=padlen)
    else:
        return filtfilt(b, a, data, axis=0)

class DatasetBuilder:
    """
    Builds a numpy file for the data and labels.
    Each sliding window is now [128, channels].
    """
    def __init__(self, dataset: object, mode: str, max_length: int, task='fd', **kwargs) -> None:
        assert mode in ['avg_pool', 'sliding_window'], f'Unsupported processing method {mode}'
        self.dataset = dataset
        self.data: Dict[str, List[np.array]] = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task

    def make_dataset(self, subjects: List[int]):
        """
        Reads all the files and makes a numpy array with all data.
        Each window is an independent sample of shape [128, channels].
        """
        self.data = {}
        windowed_data: Dict[str, List[np.ndarray]] = {}
        windowed_labels: List[int] = []

        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1

                for modality, file_path in trial.files.items():
                    keys = self.kwargs.get('keys', None)
                    key = None
                    if keys:
                        key = keys.get(modality.lower(), None)

                    processor = Processor(
                        file_path,
                        self.mode,
                        self.max_length,
                        window_size=128,   # Window size is 128
                        stride_size=32,    # Same stride
                        key=key
                    )
                    try:
                        processed_data = processor.process()
                        if processed_data is None:
                            continue

                        # Apply Butterworth filter
                        filtered_data = butterworth_filter(processed_data, cutoff=1.0, fs=20)

                        # shape => either [n_windows, 128, feats] or [128, feats]
                        if modality not in windowed_data:
                            windowed_data[modality] = []

                        if len(filtered_data.shape) == 3:
                            # multiple windows
                            n_windows = filtered_data.shape[0]
                            for i in range(n_windows):
                                windowed_data[modality].append(filtered_data[i])
                                windowed_labels.append(label)
                        else:
                            # single sample [128, feats]
                            windowed_data[modality].append(filtered_data)
                            windowed_labels.append(label)

                    except Exception as e:
                        print(e)
                        continue

        for modality in windowed_data:
            self.data[modality] = np.array(windowed_data[modality], dtype=np.float32)
        self.data['labels'] = np.array(windowed_labels, dtype=np.int64) if windowed_labels else np.array([], dtype=int)

    def normalization(self) -> Dict[str, np.ndarray]:
        """
        Function to normalize data across all windows.
        """
        for key, value in self.data.items():
            if key != 'labels' and value.size > 0:
                # shape => [N, 128, channels]
                n_samples, length, channels = value.shape
                flat = value.reshape(n_samples * length, channels)
                norm_data = StandardScaler().fit_transform(flat)
                self.data[key] = norm_data.reshape(n_samples, length, channels)
        return self.data
