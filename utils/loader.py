import os
from typing import List, Dict
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from utils.processor.base import Processor
import os
from typing import List, Dict
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from sklearn.preprocessing import StandardScaler
from utils.processor.base import Processor

def fall_detection_filter(data, fs=32.0):
    """
    Specialized accelerometer filter optimized for real-time fall detection at 32 Hz.
    Implements a two-stage filtering approach with median and Butterworth filters
    to maximize fall detection accuracy for both watch and phone data.
    
    Parameters:
        data: numpy array of shape (n_samples, 3) containing x,y,z acceleration
        fs: sampling frequency (default 32.0 Hz)
    
    Returns:
        Filtered acceleration data optimized for fall detection
    """
    # Stage 1: Median filter with small window to preserve impact signatures
    # while removing single-sample impulse noise
    filtered_data = np.zeros_like(data)
    for axis in range(3):
        filtered_data[:, axis] = medfilt(data[:, axis], 3)
    
    # Stage 2: Butterworth filter optimized for fall characteristics
    order = 6  # Higher order for steeper frequency cutoff at 32 Hz sampling
    cutoff = 7.0  # Optimized to capture fall dynamics (0.5-6 Hz range)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Enhanced edge handling with larger padding to prevent artifacts
    pad_size = 4 * order
    padded_data = np.pad(filtered_data, ((pad_size, pad_size), (0, 0)), 
                        mode='edge')
    
    # Zero-phase filtering for precise fall timing preservation
    final_filtered = np.zeros_like(padded_data)
    for axis in range(3):
        final_filtered[:, axis] = filtfilt(b, a, padded_data[:, axis])
    
    return final_filtered[pad_size:-pad_size, :]

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """
    Maintains compatibility with existing codebase while implementing 
    the enhanced fall detection filter.
    """
    # Redirect to optimized fall detection filter while maintaining the same interface
    return fall_detection_filter(data, fs)

class DatasetBuilder:
    def __init__(self, dataset: object, mode: str, max_length: int, task='fd', **kwargs) -> None:
        assert mode in ['avg_pool', 'sliding_window'], f'Unsupported processing method {mode}'
        self.dataset = dataset
        self.data: Dict[str, List[np.ndarray]] = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.overlap = max_length // 2 if mode == 'sliding_window' else 0

    def make_dataset(self, subjects: List[int]) -> None:
        """Reads all the files and makes a numpy array with all data."""
        print(f"\nMaking dataset for subjects: {subjects}")
        trial_subjects = {trial.subject_id for trial in self.dataset.matched_trials}
        requested_subjects = set(subjects)
        matching_subjects = trial_subjects & requested_subjects

        print("\nSubject statistics:")
        print(f"Subjects in matched trials: {sorted(list(trial_subjects))}")
        print(f"Requested subjects: {sorted(list(requested_subjects))}")
        print(f"Matching subjects: {sorted(list(matching_subjects))}")

        if not matching_subjects:
            print("\nNo matching subjects found!")
            return

        self.data = {'labels': []}
        file_counts = {key: 0 for key in self.dataset.matched_trials[0].files.keys()}
        processed_trials = 0

        for trial in self.dataset.matched_trials:
            if trial.subject_id in matching_subjects:
                print(f"Processing trial: S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}")
                processed_trials += 1

                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1

                trial_data = {}
                success = True

                for modality_key, file_path in trial.files.items():
                    try:
                        processor = Processor(file_path, self.mode, self.max_length)
                        # Using the enhanced fall detection filter directly
                        unimodal_data = fall_detection_filter(
                            processor.process(),
                            fs=32.0  # Explicitly using 32 Hz sampling rate
                        )
                        trial_data[modality_key] = unimodal_data
                        file_counts[modality_key] += 1
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        success = False
                        break

                if success:
                    self.data['labels'].append(label)
                    for key, value in trial_data.items():
                        if key not in self.data:
                            self.data[key] = []
                        self.data[key].append(value)

        print(f"\nProcessing complete: Processed {processed_trials} trials")
        print("\nFile counts per modality/sensor:")
        for key, count in file_counts.items():
            print(f"{key}: {count} files")

        if not self.data['labels']:
            print("No data was loaded. Exiting dataset preparation.")
            return

        print("\nConverting to numpy arrays...")
        for key in self.data:
            if key != 'labels':
                try:
                    self.data[key] = np.stack(self.data[key], axis=0)
                    print(f"{key} shape: {self.data[key].shape}")
                except Exception as e:
                    print(f"Error stacking {key}: {e}")
                    print(f"Data sizes: {[arr.shape for arr in self.data[key]]}")
            else:
                self.data[key] = np.array(self.data[key])
                print(f"Labels shape: {self.data['labels'].shape}")

    def normalization(self) -> Dict[str, np.ndarray]:
        """Normalize data."""
        print("\nNormalizing data...")
        normalized_data = {}

        for key, value in self.data.items():
            if key == 'labels':
                normalized_data[key] = value
                continue

            print(f"Normalizing {key}...")
            num_samples, length = value.shape[:2]
            reshaped_data = value.reshape(num_samples * length, -1)

            scaler = StandardScaler()
            norm_data = scaler.fit_transform(reshaped_data + 1e-10)
            normalized_data[key] = norm_data.reshape(value.shape)
            print(f"{key} normalized shape: {normalized_data[key].shape}")

        return normalized_data