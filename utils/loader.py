import os
import numpy as np
import pandas as pd
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import traceback
import time
from tqdm import tqdm

from utils.imu_fusion import (
    align_sensor_data, save_aligned_data, process_window_with_filter, 
    process_sequential_windows, create_filter_id, clear_filters
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loader")

MAX_THREADS = 30
thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
file_semaphore = threading.Semaphore(MAX_THREADS)
dataset_locks = defaultdict(threading.Lock)

def csvloader(file_path, **kwargs):
    try:
        try:
            file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        except:
            try:
                file_data = pd.read_csv(file_path, index_col=False, header=None, sep=';').dropna().bfill()
            except:
                file_data = pd.read_csv(file_path).dropna().bfill()
        
        if 'skeleton' in file_path:
            cols = 96
        else:
            if file_data.shape[1] > 4:
                cols = file_data.shape[1] - 3
                file_data = file_data.iloc[:, 3:]
            else:
                cols = 3
        
        if file_data.shape[1] < cols:
            logger.warning(f"File has fewer columns than expected: {file_data.shape[1]} < {cols}")
            missing_cols = cols - file_data.shape[1]
            for i in range(missing_cols):
                file_data[f'missing_{i}'] = 0
        
        if file_data.shape[0] > 2:
            activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        else:
            activity_data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
        
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise

def matloader(file_path, **kwargs):
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        raise ValueError(f'Unsupported {key} for matlab file')
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
    result[:min(len(sequence), max_sequence_length)] = sequence[:max_sequence_length]
    
    return result

def align_sequence(data):
    acc_data = data.get('accelerometer')
    gyro_data = data.get('gyroscope')
    
    if acc_data is None or gyro_data is None or len(acc_data) == 0 or len(gyro_data) == 0:
        return data
    
    acc_timestamps = None
    gyro_timestamps = None
    
    # Extract timestamps if available
    if acc_data.shape[1] > 3:
        acc_timestamps = acc_data[:, 0]
        acc_values = acc_data[:, 1:4]
    else:
        acc_values = acc_data
        acc_timestamps = np.arange(len(acc_data))
    
    if gyro_data.shape[1] > 3:
        gyro_timestamps = gyro_data[:, 0]
        gyro_values = gyro_data[:, 1:4]
    else:
        gyro_values = gyro_data
        gyro_timestamps = np.arange(len(gyro_data))
    
    try:
        aligned_acc, aligned_gyro, common_times = align_sensor_data(
            acc_values, gyro_values, 
            timestamps_acc=acc_timestamps, 
            timestamps_gyro=gyro_timestamps
        )
        
        if len(aligned_acc) == 0 or len(aligned_gyro) == 0:
            return data
        
        aligned_data = {
            'accelerometer': aligned_acc,
            'gyroscope': aligned_gyro,
            'aligned_timestamps': common_times
        }
        
        # Include skeleton data if available
        skeleton_data = data.get('skeleton')
        if skeleton_data is not None and len(skeleton_data) > 0:
            skeleton_times = np.linspace(0, len(skeleton_data)/30.0, len(skeleton_data))
            aligned_skeleton = np.zeros((len(common_times), skeleton_data.shape[1]))
            
            for joint in range(skeleton_data.shape[1]):
                aligned_skeleton[:, joint] = np.interp(common_times, skeleton_times, skeleton_data[:, joint])
            
            aligned_data['skeleton'] = aligned_skeleton
        
        return aligned_data
    
    except Exception as e:
        logger.error(f"Error in alignment: {str(e)}")
        logger.error(traceback.format_exc())
        return data

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', fusion_options=None, **kwargs):
        if mode not in ['avg_pool', 'sliding_window']:
            raise ValueError(f"Unsupported processing method {mode}")
        
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
        
        if fusion_options:
            self.fusion_enabled = fusion_options.get('enabled', False)
            self.filter_type = fusion_options.get('filter_type', 'madgwick')
            self.stateful = not fusion_options.get('process_per_window', False)
            self.window_stride = fusion_options.get('window_stride', 32)
            self.filter_params = fusion_options
            self.is_linear_acc = fusion_options.get('is_linear_acc', True)
            
            logger.info(f"Fusion options: enabled={self.fusion_enabled}, filter_type={self.filter_type}, stateful={self.stateful}")
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
            if file_type not in ['csv', 'mat']:
                raise ValueError(f"Unsupported file type {file_type}")
            
            loader = LOADER_MAP[file_type]
            data = loader(file_path, **self.kwargs)
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def process(self, data, label, fuse=False, filter_type='madgwick', filter_params=None, trial_id=None):
        if self.mode == 'avg_pool':
            processed_data = {}
            
            for modality, modality_data in data.items():
                if modality != 'labels':
                    processed_data[modality] = pad_sequence_numpy(
                        sequence=modality_data,
                        max_sequence_length=self.max_length,
                        input_shape=modality_data.shape
                    )
            
            processed_data['labels'] = np.array([label])
            
            if fuse and 'accelerometer' in processed_data and 'gyroscope' in processed_data:
                try:
                    timestamps = processed_data.get('aligned_timestamps', None)
                    filter_id = create_filter_id(
                        subject_id=trial_id[0] if trial_id else 0,
                        action_id=trial_id[1] if trial_id else 0,
                        trial_id=trial_id[2] if trial_id and len(trial_id) > 2 else None,
                        filter_type=filter_type
                    )
                    
                    result = process_window_with_filter(
                        processed_data['accelerometer'],
                        processed_data['gyroscope'],
                        timestamps=timestamps,
                        filter_type=filter_type,
                        filter_id=filter_id,
                        reset_filter=not self.stateful,
                        is_linear_acc=self.is_linear_acc,
                        filter_params=filter_params
                    )
                    
                    if 'quaternion' in result:
                        processed_data['quaternion'] = result['quaternion']
                except Exception as e:
                    logger.error(f"Fusion processing failed: {str(e)}")
                    processed_data['quaternion'] = np.zeros((self.max_length, 4))
                    processed_data['quaternion'][:, 0] = 1.0
            
            if 'quaternion' not in processed_data:
                processed_data['quaternion'] = np.zeros((self.max_length, 4))
                processed_data['quaternion'][:, 0] = 1.0
        else:
            # Sliding window mode - process sequentially with state preservation
            filter_id = create_filter_id(
                subject_id=trial_id[0] if trial_id else 0,
                action_id=trial_id[1] if trial_id else 0,
                trial_id=trial_id[2] if trial_id and len(trial_id) > 2 else None,
                filter_type=filter_type
            )
            
            processed_data = process_sequential_windows(
                data=data,
                window_size=self.max_length,
                stride=self.window_stride,
                label=label,
                filter_type=filter_type,
                filter_params=filter_params,
                base_filter_id=filter_id,
                stateful=self.stateful,
                is_linear_acc=self.is_linear_acc
            )
        
        return processed_data

    def make_dataset(self, subjects, fuse=False, filter_type=None, visualize=False, save_aligned=False):
        self.data = {}
        self.fuse = fuse
        
        if filter_type is None and hasattr(self, 'filter_type'):
            filter_type = self.filter_type
        else:
            filter_type = 'madgwick'
            
        if hasattr(self, 'fusion_options'): 
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)
        
        # Clear filter cache
        clear_filters()
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        # Group trials by subject and action for preserving filter state across trials
        subject_action_groups = {}
        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue
            
            key = (trial.subject_id, trial.action_id)
            if key not in subject_action_groups:
                subject_action_groups[key] = []
            
            subject_action_groups[key].append(trial)
        
        # Count total trials
        total_trials = sum(len(trials) for trials in subject_action_groups.values())
        logger.info(f"Processing {total_trials} trials with filter_type={filter_type}, fuse={fuse}")
        
        # Create progress bar
        progress = tqdm(total=total_trials, desc="Processing trials")
        # Store start time for elapsed time calculation
        start_time = time.time()
        
        progress.set_postfix({
            "success": processed_count,
            "skipped": skipped_count, 
            "failed": failed_count
        })
        
        # Process each subject-action group
        for (subject_id, action_id), trials in subject_action_groups.items():
            logger.info(f"Processing subject {subject_id}, action {action_id} with base_filter_id=S{subject_id}_A{action_id}_{filter_type}")
            
            # Create filter ID for this subject-action pair
            base_filter_id = create_filter_id(subject_id, action_id, filter_type=filter_type)
            
            for trial in trials:
                # Determine label based on task
                if self.task == 'fd':
                    # Fall detection (binary classification)
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    # Age classification
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    # Action classification
                    label = trial.action_id - 1
                
                # Create trial ID
                trial_id = (trial.subject_id, trial.action_id, trial.sequence_number)
                trial_data = {}
                
                # Load data for each modality
                try:
                    # Load accelerometer data
                    if 'accelerometer' in trial.files:
                        try:
                            acc_data = self.load_file(trial.files['accelerometer'])
                            trial_data['accelerometer'] = acc_data
                        except Exception as e:
                            logger.error(f"Failed to load accelerometer for S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: {str(e)}")
                    
                    # Load gyroscope data
                    if 'gyroscope' in trial.files:
                        try:
                            gyro_data = self.load_file(trial.files['gyroscope'])
                            trial_data['gyroscope'] = gyro_data
                        except Exception as e:
                            logger.error(f"Failed to load gyroscope for S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: {str(e)}")
                    
                    # Load skeleton data if available
                    if 'skeleton' in trial.files:
                        try:
                            skeleton_data = self.load_file(trial.files['skeleton'])
                            trial_data['skeleton'] = skeleton_data
                        except Exception as e:
                            logger.warning(f"Failed to load skeleton for S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: {str(e)}")
                    
                    # Check if required sensors are available
                    if 'accelerometer' not in trial_data or 'gyroscope' not in trial_data:
                        logger.warning(f"Skipping S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: Missing required sensors")
                        skipped_count += 1
                        progress.set_postfix({
                            "success": processed_count,
                            "skipped": skipped_count, 
                            "failed": failed_count
                        })
                        progress.update(1)
                        continue
                    
                    # Check data length
                    if len(trial_data['accelerometer']) < 5 or len(trial_data['gyroscope']) < 5:
                        logger.warning(f"Skipping S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: Data too short")
                        skipped_count += 1
                        progress.set_postfix({
                            "success": processed_count,
                            "skipped": skipped_count, 
                            "failed": failed_count
                        })
                        progress.update(1)
                        continue
                    
                    # Align sensor data
                    aligned_data = align_sequence(trial_data)
                    
                    # Check if alignment succeeded
                    if 'accelerometer' not in aligned_data or 'gyroscope' not in aligned_data:
                        logger.warning(f"Skipping S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: Alignment failed")
                        skipped_count += 1
                        progress.set_postfix({
                            "success": processed_count,
                            "skipped": skipped_count, 
                            "failed": failed_count
                        })
                        progress.update(1)
                        continue
                    
                    # Check aligned data length
                    if len(aligned_data['accelerometer']) < 5 or len(aligned_data['gyroscope']) < 5:
                        logger.warning(f"Skipping S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: Aligned data too short")
                        skipped_count += 1
                        progress.set_postfix({
                            "success": processed_count,
                            "skipped": skipped_count, 
                            "failed": failed_count
                        })
                        progress.update(1)
                        continue
                    
                    # Save aligned data if requested
                    if save_aligned:
                        try:
                            aligned_acc = aligned_data['accelerometer']
                            aligned_gyro = aligned_data['gyroscope']
                            aligned_timestamps = aligned_data.get('aligned_timestamps')
                            
                            if aligned_acc is not None and aligned_gyro is not None and len(aligned_acc) > 0 and len(aligned_gyro) > 0:
                                save_aligned_data(
                                    trial.subject_id,
                                    trial.action_id,
                                    trial.sequence_number,
                                    aligned_acc,
                                    aligned_gyro,
                                    aligned_timestamps
                                )
                        except Exception as e:
                            logger.warning(f"Failed to save aligned data: {str(e)}")
                    
                    # Process data
                    try:
                        processed_data = self.process(
                            aligned_data, 
                            label, 
                            fuse, 
                            filter_type, 
                            self.filter_params,
                            trial_id
                        )
                        
                        if processed_data is not None and 'labels' in processed_data and len(processed_data['labels']) > 0:
                            self._add_trial_data(processed_data)
                            processed_count += 1
                        else:
                            logger.warning(f"No valid data after processing for S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}")
                            failed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: {str(e)}")
                        logger.error(traceback.format_exc())
                        failed_count += 1
                
                except Exception as e:
                    logger.error(f"Error processing S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}: {str(e)}")
                    logger.error(traceback.format_exc())
                    failed_count += 1
                
                # Update progress
                progress.set_postfix({
                    "success": processed_count,
                    "skipped": skipped_count, 
                    "failed": failed_count
                })
                progress.update(1)
        
        # Calculate elapsed time correctly
        elapsed_time = time.time() - start_time
        
        # Print summary
        progress.close()
        print(f"\nProcessing complete: {elapsed_time:.1f}s total ({elapsed_time/60:.1f}min)")
        print(f"Results: {processed_count} successful, {skipped_count} skipped, {failed_count} failed")
        
        # Concatenate data
        print("Concatenating data...")
        for key in self.data:
            if len(self.data[key]) > 0:
                try:
                    self.data[key] = np.concatenate(self.data[key], axis=0)
                except Exception as e:
                    logger.error(f"Error concatenating {key}: {str(e)}")
                    del self.data[key]
        
        # Add default quaternion data if missing
        if 'quaternion' not in self.data and 'accelerometer' in self.data:
            acc_shape = self.data['accelerometer'].shape
            if len(acc_shape) >= 2:  # Check for at least 2 dimensions
                self.data['quaternion'] = np.zeros((acc_shape[0], acc_shape[1], 4))
                self.data['quaternion'][..., 0] = 1.0
        
        # Clean up
        clear_filters()
        return self.data

    def _add_trial_data(self, trial_data):
        for modality, modality_data in trial_data.items():
            if modality not in self.data:
                self.data[modality] = []
            
            if isinstance(modality_data, np.ndarray) and modality_data.size > 0:
                self.data[modality].append(modality_data)

    def normalization(self):
        from sklearn.preprocessing import StandardScaler
        
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion'] and len(value.shape) >= 2:
                        num_samples, length = value.shape[:2]
                        reshaped_data = value.reshape(num_samples * length, -1)
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                        self.data[key] = norm_data.reshape(value.shape)
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        self.data[key] = StandardScaler().fit_transform(value)
                except Exception as e:
                    logger.error(f"Error normalizing {key}: {str(e)}")
        
        return self.data
