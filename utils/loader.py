import os
import numpy as np
import pandas as pd
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time
import sys
from tqdm import tqdm
from utils.imu_fusion import (
    align_sensor_data, save_aligned_data, process_window_with_filter, 
    process_sequential_windows, create_filter_id, clear_filters, get_filter,
    align_sequence
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loader")
MAX_THREADS = 30
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(MAX_THREADS)
dataset_locks = defaultdict(threading.Lock)

def csvloader(file_path, **kwargs):
    try:
        try: file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        except: file_data = pd.read_csv(file_path, index_col=False, header=None, sep=';').dropna().bfill()
        
        if 'skeleton' in file_path: cols = 96
        else:
            if file_data.shape[1] > 4:
                cols = file_data.shape[1] - 3
                file_data = file_data.iloc[:, 3:]
            else: cols = 3
        
        if file_data.shape[1] < cols:
            missing_cols = cols - file_data.shape[1]
            for i in range(missing_cols): file_data[f'missing_{i}'] = 0
        
        if file_data.shape[0] > 2: activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        else: activity_data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
        
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise

def matloader(file_path, **kwargs):
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']: raise ValueError(f'Unsupported {key} for matlab file')
    from scipy.io import loadmat
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def pad_sequence_numpy(sequence, max_sequence_length, input_shape):
    import torch
    import torch.nn.functional as F
    shape = list(input_shape)
    shape[0] = max_sequence_length
    if len(sequence) > max_sequence_length:
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
        pooled = F.avg_pool1d(sequence_tensor, kernel_size=len(sequence)//max_sequence_length, stride=len(sequence)//max_sequence_length)
        sequence = pooled.permute(0, 2, 1).squeeze(0).numpy()
    result = np.zeros(shape, sequence.dtype)
    result[:len(sequence)] = sequence[:max_sequence_length]
    return result

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', fusion_options=None, **kwargs):
        if mode not in ['avg_pool', 'sliding_window']: raise ValueError(f"Unsupported processing method {mode}")
        self.dataset = dataset
        self.data = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.fusion_options = fusion_options or {}
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        os.makedirs(os.path.join(self.aligned_data_dir, "accelerometer"), exist_ok=True)
        os.makedirs(os.path.join(self.aligned_data_dir, "gyroscope"), exist_ok=True)
        
        # Explicit caching setting
        self.use_cache = False
        
        if fusion_options:
            self.fusion_enabled = fusion_options.get('enabled', False)
            self.filter_type = fusion_options.get('filter_type', 'madgwick')
            # Force stateful processing
            self.stateful = True
            self.window_stride = fusion_options.get('window_stride', 32)
            self.filter_params = fusion_options
            self.is_linear_acc = fusion_options.get('is_linear_acc', True)
        else:
            self.fusion_enabled = False
            self.filter_type = 'madgwick'
            self.stateful = True
            self.window_stride = 32
            self.filter_params = {}
            self.is_linear_acc = True
    
    def load_file(self, file_path):
        try:
            file_type = file_path.split('.')[-1]
            if file_type not in ['csv', 'mat']: raise ValueError(f"Unsupported file type {file_type}")
            loader = LOADER_MAP[file_type]
            data = loader(file_path, **self.kwargs)
            return data
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def process_single_trial(self, trial_id, trial_data, label, fuse, filter_type, filter_params, base_filter_id):
        """Process a single trial with sequential window filtering"""
        print(f"Processing trial S{trial_id[0]}A{trial_id[1]}T{trial_id[2]} with filter_id={base_filter_id}")
        
        # Process sequential windows with filter state preservation
        processed_data = process_sequential_windows(
            data=trial_data,
            window_size=self.max_length,
            stride=self.window_stride,
            label=label,
            filter_type=filter_type,
            filter_params=filter_params,
            base_filter_id=base_filter_id,
            stateful=True,  # Always use stateful processing
            is_linear_acc=self.is_linear_acc
        )
        
        return processed_data

    def make_dataset(self, subjects, fuse=False, filter_type=None, visualize=False, save_aligned=False):
        print(f"Starting dataset processing for {len(subjects)} subjects")
        start_time = time.time()
        self.data = {}
        self.fuse = fuse
        if filter_type is None and hasattr(self, 'filter_type'): filter_type = self.filter_type
        else: filter_type = 'madgwick'
        if hasattr(self, 'fusion_options'): save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)
        
        # Clear all filters
        clear_filters()
        
        # Count trials and organize by subject-action pair
        subject_action_groups = {}
        eligible_trials = 0
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects: continue
            eligible_trials += 1
            key = (trial.subject_id, trial.action_id)
            if key not in subject_action_groups: subject_action_groups[key] = []
            subject_action_groups[key].append(trial)
        
        # Setup progress tracking
        total_trials = eligible_trials
        processed_count = 0
        successful_count = 0
        skipped_count = 0
        failed_count = 0
        
        logger.info(f"Processing {total_trials} trials with filter_type={filter_type}, fuse={fuse}")
        print(f"Processing {total_trials} trials with filter type={filter_type}")
        
        # Process trials with progress bar
        with tqdm(total=total_trials, desc="Processing trials", file=sys.stdout) as progress_bar:
            for (subject_id, action_id), trials in subject_action_groups.items():
                # Create a base filter ID for this subject-action pair
                base_filter_id = create_filter_id(subject_id, action_id, filter_type=filter_type)
                print(f"Processing subject {subject_id}, action {action_id} with base_filter_id={base_filter_id}")
                
                # Reset filter state for this subject-action group
                get_filter(base_filter_id, filter_type, params=self.filter_params, reset=True)
                
                # Process all trials for this subject-action pair
                for trial in trials:
                    if self.task == 'fd': label = int(trial.action_id > 9)
                    elif self.task == 'age': label = int(trial.subject_id < 29 or trial.subject_id > 46)
                    else: label = trial.action_id - 1
                    
                    trial_id = (trial.subject_id, trial.action_id, trial.sequence_number)
                    trial_data = {}
                    
                    # Load data files
                    try:
                        for modality_name, file_path in trial.files.items():
                            try:
                                unimodal_data = self.load_file(file_path)
                                trial_data[modality_name] = unimodal_data
                            except Exception as e:
                                print(f"Failed to load {modality_name} for S{trial_id[0]}A{trial_id[1]}T{trial_id[2]}: {e}")
                                continue
                        
                        # Skip if either accelerometer or gyroscope is missing
                        if 'accelerometer' not in trial_data or 'gyroscope' not in trial_data:
                            print(f"Skipping S{trial_id[0]}A{trial_id[1]}T{trial_id[2]}: Missing required sensors")
                            skipped_count += 1
                            progress_bar.update(1)
                            processed_count += 1
                            continue
                        
                        # Skip if data is too short
                        if len(trial_data['accelerometer']) < 5 or len(trial_data['gyroscope']) < 5:
                            print(f"Skipping S{trial_id[0]}A{trial_id[1]}T{trial_id[2]}: Sensor data too short")
                            skipped_count += 1
                            progress_bar.update(1)
                            processed_count += 1
                            continue
                        
                        # Align accelerometer and gyroscope data
                        print(f"Aligning data for S{trial_id[0]}A{trial_id[1]}T{trial_id[2]}")
                        trial_data = align_sequence(trial_data)
                        
                        # Skip if alignment failed
                        if 'accelerometer' not in trial_data or 'gyroscope' not in trial_data:
                            print(f"Skipping S{trial_id[0]}A{trial_id[1]}T{trial_id[2]}: Failed alignment")
                            skipped_count += 1
                            progress_bar.update(1)
                            processed_count += 1
                            continue
                        
                        # Skip if aligned data is too short
                        if len(trial_data['accelerometer']) < 5 or len(trial_data['gyroscope']) < 5:
                            print(f"Skipping S{trial_id[0]}A{trial_id[1]}T{trial_id[2]}: Aligned data too short")
                            skipped_count += 1
                            progress_bar.update(1)
                            processed_count += 1
                            continue
                        
                        # Save aligned data if requested
                        if save_aligned:
                            aligned_acc = trial_data.get('accelerometer')
                            aligned_gyro = trial_data.get('gyroscope')
                            aligned_timestamps = trial_data.get('aligned_timestamps')
                            if aligned_acc is not None and aligned_gyro is not None and len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                                save_aligned_data(
                                    trial.subject_id,
                                    trial.action_id,
                                    trial.sequence_number,
                                    aligned_acc,
                                    aligned_gyro,
                                    aligned_timestamps if aligned_timestamps is not None else np.arange(len(aligned_acc))
                                )
                        
                        # Process trial data using sequential window filtering
                        processed_data = self.process_single_trial(
                            trial_id, 
                            trial_data, 
                            label, 
                            fuse, 
                            filter_type, 
                            self.filter_params,
                            base_filter_id
                        )
                        
                        if processed_data is not None and 'labels' in processed_data and len(processed_data['labels']) > 0:
                            window_count = len(processed_data['labels'])
                            print(f"Trial S{trial_id[0]}A{trial_id[1]}T{trial_id[2]}: Generated {window_count} windows")
                            self._add_trial_data(processed_data)
                            successful_count += 1
                        else:
                            print(f"Trial S{trial_id[0]}A{trial_id[1]}T{trial_id[2]}: No valid windows")
                            failed_count += 1
                    except Exception as e:
                        print(f"Error processing S{trial_id[0]}A{trial_id[1]}T{trial_id[2]}: {e}")
                        failed_count += 1
                    
                    # Update progress
                    processed_count += 1
                    progress_bar.update(1)
                    
                    # Periodic status report
                    if processed_count % 5 == 0 or processed_count == total_trials:
                        elapsed = time.time() - start_time
                        remaining = (elapsed / processed_count) * (total_trials - processed_count) if processed_count > 0 else 0
                        print(f"\nProgress: {processed_count}/{total_trials} trials ({processed_count/total_trials*100:.1f}%)")
                        print(f"Status: {successful_count} successful, {skipped_count} skipped, {failed_count} failed")
                        print(f"Time: {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
                        progress_bar.set_postfix({
                            'success': successful_count,
                            'skipped': skipped_count,
                            'failed': failed_count
                        })
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\nProcessing complete: {total_time:.1f}s total ({total_time/60:.1f}min)")
        print(f"Results: {successful_count} successful, {skipped_count} skipped, {failed_count} failed")
        
        # Concatenate data from all trials
        print("Concatenating data...")
        for key in self.data:
            if len(self.data[key]) > 0:
                try: 
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                    print(f"{key} shape: {self.data[key].shape}")
                except Exception as e:
                    print(f"Error concatenating {key}: {str(e)}")
                    del self.data[key]
        
        # Create default quaternion if missing
        if 'quaternion' not in self.data and 'accelerometer' in self.data:
            acc_shape = self.data['accelerometer'].shape
            self.data['quaternion'] = np.zeros((acc_shape[0], acc_shape[1], 4))
            self.data['quaternion'][..., 0] = 1.0
            print(f"Created default quaternion data: {self.data['quaternion'].shape}")
        
        # Clear filters at the end
        clear_filters()
        return self.data

    def _add_trial_data(self, trial_data):
        for modality, modality_data in trial_data.items():
            if modality not in self.data: self.data[modality] = []
            if isinstance(modality_data, np.ndarray) and modality_data.size > 0:
                self.data[modality].append(modality_data)

    def normalization(self):
        from sklearn.preprocessing import StandardScaler
        print("Normalizing data...")
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion'] and len(value.shape) >= 2:
                        num_samples, length = value.shape[:2]
                        reshaped_data = value.reshape(num_samples * length, -1)
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                        self.data[key] = norm_data.reshape(value.shape)
                        print(f"Normalized {key}: {self.data[key].shape}")
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        self.data[key] = StandardScaler().fit_transform(value)
                        print(f"Normalized {key}: {self.data[key].shape}")
                except Exception as e: 
                    print(f"Error normalizing {key}: {str(e)}")
        return self.data
