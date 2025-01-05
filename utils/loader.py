"""
File: loader.py

Description:
    - Contains filter definitions (median + Butterworth).
    - Contains DatasetBuilder with minimal changes to preserve
      the existing structure but adding placeholders for
      visualization code as requested.
"""

import os
from typing import List, Dict
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from sklearn.preprocessing import StandardScaler

from utils.processor.base import Processor
import matplotlib.pyplot as plt


# ------------------- FILTER FUNCTIONS ------------------- #

def fall_detection_filter(data, fs=31.125):
    """
    Specialized accelerometer filter optimized for real-time fall detection at 32 Hz.
    Implements a two-stage filtering approach with median and Butterworth filters
    to maximize fall detection accuracy for both watch and phone data.
    
    Parameters:
        data (np.ndarray): shape (n_samples, 3) containing x,y,z acceleration
        fs (float): sampling frequency (default 32.0 Hz)
    
    Returns:
        np.ndarray: Filtered acceleration data optimized for fall detection
    """
    # Stage 1: Median filter with small window
    filtered_data = np.zeros_like(data)
    for axis in range(3):
        filtered_data[:, axis] = medfilt(data[:, axis], 3)
    
    # Stage 2: Butterworth filter
    order = 6  # Higher order for sharper cutoff
    cutoff = 7.0  # around 0.5-6 Hz range for typical falls
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Edge handling with padding
    pad_size = 4 * order
    padded_data = np.pad(filtered_data, ((pad_size, pad_size), (0, 0)), mode='edge')
    
    # Zero-phase filtering for precise timing
    final_filtered = np.zeros_like(padded_data)
    for axis in range(3):
        final_filtered[:, axis] = filtfilt(b, a, padded_data[:, axis])
    
    return final_filtered[pad_size:-pad_size, :]


def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """
    Maintains compatibility with existing codebase while implementing
    the enhanced fall detection filter (fall_detection_filter).
    """
    # We simply redirect to fall_detection_filter in this example
    return fall_detection_filter(data, fs)


# ------------------- DATASET BUILDER ------------------- #

class DatasetBuilder:
    """
    Builds the final dataset from a 'dataset' object, applying processing.
    Includes placeholders for visualization of raw/filtered/normalized data.
    """
    def __init__(self, dataset: object, mode: str, max_length: int, task='fd', **kwargs) -> None:
        assert mode in ['avg_pool', 'sliding_window'], f"Unsupported processing method {mode}"
        self.dataset = dataset
        self.data: Dict[str, List[np.ndarray]] = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.overlap = max_length // 2 if mode == 'sliding_window' else 0

        # Optional placeholders for raw, filtered, normalized data
        self._raw_data: Dict[str, List[np.ndarray]] = {}
        self._filtered_data: Dict[str, List[np.ndarray]] = {}
        self._normalized_data: Dict[str, np.ndarray] = {}

    def make_dataset(self, subjects: List[int]) -> None:
        """
        Reads all the files for the chosen subjects and accumulates them as numpy arrays.
        """
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

        # Create needs_cleaning directory if it doesn't exist
        needs_cleaning_dir = os.path.join(self.dataset.root_dir, 'needs_cleaning')
        os.makedirs(needs_cleaning_dir, exist_ok=True)
        print(f"Using needs_cleaning directory: {needs_cleaning_dir}")

        self.data = {'labels': []}
        # Initialize counters
        file_counts = {key: 0 for key in self.dataset.matched_trials[0].files.keys()}
        error_counts = {key: 0 for key in self.dataset.matched_trials[0].files.keys()}
        processed_trials = 0

        # Clear raw/filtered in case of repeated calls
        self._raw_data.clear()
        self._filtered_data.clear()

        for trial in self.dataset.matched_trials:
            if trial.subject_id in matching_subjects:
                print(f"Processing trial: S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}")
                processed_trials += 1

                # Label logic: 'fd' => fall detection, etc.
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
                        # Process raw data using your custom Processor
                        processor = Processor(file_path, self.mode, self.max_length)
                        raw_array = processor.process()

                        # Store raw data for visualization
                        if modality_key not in self._raw_data:
                            self._raw_data[modality_key] = []
                        self._raw_data[modality_key].append(raw_array)

                        # Filter
                        unimodal_data = fall_detection_filter(raw_array, fs=31.125)
                        if modality_key not in self._filtered_data:
                            self._filtered_data[modality_key] = []
                        self._filtered_data[modality_key].append(unimodal_data)

                        trial_data[modality_key] = unimodal_data
                        file_counts[modality_key] += 1
                    except Exception as e:
                        error_counts[modality_key] += 1
                        print(f"Error processing file {file_path}: {e}")

                        # Attempt to copy to needs_cleaning
                        try:
                            relative_path = os.path.relpath(os.path.dirname(file_path), self.dataset.root_dir)
                            target_dir = os.path.join(needs_cleaning_dir, relative_path)
                            os.makedirs(target_dir, exist_ok=True)

                            import shutil
                            target_file = os.path.join(target_dir, os.path.basename(file_path))
                            shutil.copy2(file_path, target_file)
                            print(f"Successfully copied {os.path.basename(file_path)} to {target_file} for cleaning")
                        except Exception as copy_error:
                            print(f"Error copying file {file_path}: {str(copy_error)}")

                        success = False
                        break

                if success:
                    self.data['labels'].append(label)
                    for key, value in trial_data.items():
                        if key not in self.data:
                            self.data[key] = []
                        self.data[key].append(value)

        print(f"\nProcessing complete: Processed {processed_trials} trials")
        print("\nFile statistics per modality/sensor:")
        for key in file_counts.keys():
            print(f"{key}: {file_counts[key]} successful, {error_counts[key]} errors")

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
        """
        Normalize data across all modalities/sensors.
        """
        print("\nNormalizing data...")
        normalized_data = {}
        self._normalized_data.clear()  # reset

        for key, value in self.data.items():
            if key == 'labels':
                normalized_data[key] = value
                continue

            print(f"Normalizing {key}...")
            # value shape => (num_trials, length, channels=3?) or similar
            num_samples, length = value.shape[:2]
            reshaped_data = value.reshape(num_samples * length, -1)

            scaler = StandardScaler()
            norm_data = scaler.fit_transform(reshaped_data + 1e-10)
            normalized_data[key] = norm_data.reshape(value.shape)
            print(f"{key} normalized shape: {normalized_data[key].shape}")

            # Store for optional visualization
            self._normalized_data[key] = normalized_data[key]

        return normalized_data

    # ------------------- VISUALIZATION ------------------- #
    def visualize_trial(self, trial_index: int = 0, modality_key: str = None):
        """
        Visualize a single trial's raw, filtered, and normalized data (if available).
        This is just a placeholder. Adjust indexing and shapes if your data differs.
        """
        if not self._raw_data:
            print("No raw data found. Please run make_dataset first.")
            return
        if not self._filtered_data:
            print("No filtered data found. Please run make_dataset first.")
            return

        # If user doesn't specify a modality key, pick the first
        if modality_key is None:
            modality_key = list(self._raw_data.keys())[0]

        if modality_key not in self._raw_data:
            print(f"Modality key '{modality_key}' not found in raw data.")
            return
        if modality_key not in self._filtered_data:
            print(f"Modality key '{modality_key}' not found in filtered data.")
            return

        raw_array_list = self._raw_data[modality_key]
        filtered_array_list = self._filtered_data[modality_key]
        if trial_index >= len(raw_array_list):
            print(f"Trial index {trial_index} out of range for modality '{modality_key}'")
            return

        raw = raw_array_list[trial_index]
        filt = filtered_array_list[trial_index]
        norm = None
        if modality_key in self._normalized_data:
            norm = self._normalized_data[modality_key][trial_index]

        # Now we can plot them side-by-side
        time_axis = np.arange(raw.shape[0])
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True)
        axes = axes.flatten()

        axis_labels = ['X', 'Y', 'Z']
        data_titles = ['Raw', 'Filtered', 'Normalized']

        for i in range(3):
            # Row i => X/Y/Z axis
            # col 0 => raw, col 1 => filtered, col 2 => normalized
            # raw
            axes[i*3 + 0].plot(time_axis, raw[:, i], color='blue')
            axes[i*3 + 0].set_title(f"{axis_labels[i]} - {data_titles[0]}")
            axes[i*3 + 0].grid(True)

            # filtered
            axes[i*3 + 1].plot(time_axis, filt[:, i], color='green')
            axes[i*3 + 1].set_title(f"{axis_labels[i]} - {data_titles[1]}")
            axes[i*3 + 1].grid(True)

            # normalized
            if norm is not None:
                axes[i*3 + 2].plot(time_axis, norm[:, i], color='red')
                axes[i*3 + 2].set_title(f"{axis_labels[i]} - {data_titles[2]}")
                axes[i*3 + 2].grid(True)
            else:
                axes[i*3 + 2].set_title(f"{axis_labels[i]} - No Normalized Data")

        plt.suptitle(f"Trial Index={trial_index}, Modality Key='{modality_key}'")
        plt.tight_layout()
        plt.show()
