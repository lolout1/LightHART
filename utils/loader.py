import os
from typing import List, Dict
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from utils.processor.base import Processor

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0)

class DatasetBuilder:
    def __init__(self, dataset: object, mode: str, max_length: int, task='fd', **kwargs) -> None:
        assert mode in ['avg_pool', 'sliding_window'], f'Unsupported processing method {mode}'
        self.dataset = dataset
        self.data: Dict[str, List[np.ndarray]] = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task

    def make_dataset(self, subjects: List[int]) -> None:
        """Reads all the files and makes a numpy array with all data"""
        print(f"\nMaking dataset for subjects: {subjects}")
        
        # Print subject statistics
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
        
        # Count trials per subject
        trials_per_subject = {}
        for trial in self.dataset.matched_trials:
            if trial.subject_id in matching_subjects:
                trials_per_subject[trial.subject_id] = trials_per_subject.get(trial.subject_id, 0) + 1
        
        print("\nTrials per matching subject:")
        for subject_id, count in sorted(trials_per_subject.items()):
            print(f"Subject {subject_id}: {count} trials")
        
        # Process matched trials
        self.data = {'labels': []}
        file_counts = {key: 0 for key in self.dataset.matched_trials[0].files.keys()}
        processed_trials = 0
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id in matching_subjects:
                print(f"Processing trial: S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}")
                processed_trials += 1
                
                # Process label
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                
                # Process each modality and sensor
                trial_data = {}
                success = True
                for modality_key, file_path in trial.files.items():
                    try:
                        processor = Processor(file_path, self.mode, self.max_length)
                        unimodal_data = butterworth_filter(processor.process(), cutoff=1.0, fs=20)
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

        print(f"\nProcessing complete:")
        print(f"Processed {processed_trials} trials")
        print("\nFile counts per modality/sensor:")
        for key, count in file_counts.items():
            print(f"{key}: {count} files")
            
        if not self.data['labels']:
            print("No data was loaded. Exiting dataset preparation.")
            return
            
        # Convert lists to numpy arrays
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
        """Normalize the data using StandardScaler"""
        print("\nNormalizing data...")
        normalized_data = {}
        
        for key, value in self.data.items():
            if key == 'labels':
                normalized_data[key] = value
                continue
                
            print(f"Normalizing {key}...")
            num_samples, length = value.shape[:2]
            reshaped_data = value.reshape(num_samples * length, -1)
            norm_data = StandardScaler().fit_transform(reshaped_data)
            normalized_data[key] = norm_data.reshape(value.shape)
            print(f"{key} normalized shape: {normalized_data[key].shape}")
            
        return normalized_data