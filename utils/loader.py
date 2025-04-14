# utils/loader.py
from collections import defaultdict
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import torch
from utils.accelerometer_processor import AccelerometerProcessor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataLoader')

def csvloader(file_path, **kwargs):
    import pandas as pd
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        # Try to load CSV file with different options if needed
        try:
            file_data = pd.read_csv(file_path, index_col=False, header=None)
        except Exception as e:
            logger.warning(f"Initial CSV load failed: {str(e)}. Trying with error handling options.")
            file_data = pd.read_csv(file_path, index_col=False, header=None, 
                                   error_bad_lines=False, warn_bad_lines=True)
        
        # Check data shape
        if file_data.shape[0] <= 3 or file_data.shape[1] < 3:
            logger.error(f"Insufficient data in file: {file_path}, shape={file_data.shape}")
            return None
            
        # Clean data
        file_data = file_data.replace([np.inf, -np.inf], np.nan).dropna().bfill()
        
        # Determine number of columns to use
        if 'skeleton' in file_path: 
            cols = 96
        else: 
            cols = 3
            
        # Extract activity data, handling possible indexing errors
        try:
            if file_data.shape[1] >= cols:
                activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
            else:
                logger.error(f"Not enough columns in data: expected at least {cols}, got {file_data.shape[1]}")
                return None
                
            # Verify we have usable data
            if activity_data.shape[0] < 5 or np.isnan(activity_data).any():
                logger.error(f"Invalid values in data from {file_path}")
                return None
                
            return activity_data
        except Exception as e:
            logger.error(f"Error extracting data from CSV: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {str(e)}")
        return None

def matloader(file_path, **kwargs):
    from scipy.io import loadmat
    try:
        key = kwargs.get('key', None)
        if key not in ['d_iner', 'd_skel']:
            logger.error(f"Unsupported key {key} for matlab file")
            return None
            
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        try:
            data = loadmat(file_path)[key]
            return data
        except Exception as e:
            logger.error(f"Error loading MAT file {file_path}: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error in MAT loader: {str(e)}")
        return None

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', **kwargs):
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.processor = AccelerometerProcessor(target_freq=25, window_size=max_length, window_overlap=0.5)
        self.stats = {
            'total_files': 0, 
            'processed_files': 0, 
            'errors': 0, 
            'falls': 0, 
            'non_falls': 0, 
            'activities': defaultdict(int)
        }
        
    def load_file(self, file_path):
        """Load a file using the appropriate loader"""
        try:
            loader = self._import_loader(file_path)
            data = loader(file_path, **self.kwargs)
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None
    
    def _import_loader(self, file_path):
        """Determine the appropriate loader based on file extension"""
        file_type = file_path.split('.')[-1]
        if file_type not in ['csv', 'mat']:
            logger.error(f"Unsupported file type {file_type}")
            raise ValueError(f"Unsupported file type {file_type}")
        return LOADER_MAP[file_type]
    
    def process_accelerometer_data(self, file_path, label, action_id):
        """Process accelerometer data with robust error handling"""
        self.stats['total_files'] += 1
        self.stats['activities'][action_id] += 1
        
        try:
            # Use the improved accelerometer processor
            acc_data = self.processor.load_and_preprocess(file_path)
            
            # Generate windows from the processed data
            windows = self.processor.segment_windows(acc_data)
            
            # Add magnitude feature
            enhanced_windows = self.processor.enhance_features(windows)
            
            # Create labels for each window
            labels = np.full(len(enhanced_windows), label)
            
            # Update statistics
            self.stats['processed_files'] += 1
            if label == 1:
                self.stats['falls'] += len(enhanced_windows)
            else:
                self.stats['non_falls'] += len(enhanced_windows)
                
            return enhanced_windows, labels
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None, None
    
    def make_dataset(self, subjects, fuse):
        """Create the dataset from files for specified subjects"""
        self.data = defaultdict(list)
        self.fuse = fuse
        
        acc_data_all = []
        labels_all = []
        fall_count = 0
        non_fall_count = 0
        
        # Process each trial in the dataset
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                # Determine label based on task
                if self.task == 'fd':
                    label = int(trial.action_id > 9)  # Fall detection: label=1 for falls (action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1  # Activity recognition
                
                # Process accelerometer data if available
                if 'accelerometer' in trial.files:
                    windows, window_labels = self.process_accelerometer_data(
                        trial.files['accelerometer'], label, trial.action_id
                    )
                    
                    if windows is not None and len(windows) > 0:
                        acc_data_all.append(windows)
                        labels_all.append(window_labels)
                        if label == 1:
                            fall_count += len(windows)
                        else:
                            non_fall_count += len(windows)
        
        # Ensure we have at least one sample of each class
        if fall_count == 0 or non_fall_count == 0:
            # Create synthetic samples for missing class
            if fall_count == 0:
                logger.warning("No fall samples found - adding synthetic fall data")
                synthetic_fall = np.random.randn(5, self.max_length, 4) * 0.1
                acc_data_all.append(synthetic_fall)
                labels_all.append(np.ones(5))
                fall_count = 5
            
            if non_fall_count == 0:
                logger.warning("No non-fall samples found - adding synthetic non-fall data")
                synthetic_nonfall = np.random.randn(5, self.max_length, 4) * 0.01
                acc_data_all.append(synthetic_nonfall)
                labels_all.append(np.zeros(5))
                non_fall_count = 5
        
        # Combine all the data
        if acc_data_all:
            try:
                self.data['accelerometer'] = np.concatenate(acc_data_all, axis=0)
                self.data['labels'] = np.concatenate(labels_all, axis=0)
                
                # Create dummy skeleton data if needed
                dummy_skl = np.zeros((self.data['accelerometer'].shape[0], 
                                      self.data['accelerometer'].shape[1], 32, 3))
                self.data['skeleton'] = dummy_skl
            except Exception as e:
                logger.error(f"Error concatenating data: {str(e)}. Creating fallback dataset.")
                self.data['accelerometer'] = np.zeros((10, self.max_length, 4))
                self.data['labels'] = np.array([0] * 5 + [1] * 5)
                self.data['skeleton'] = np.zeros((10, self.max_length, 32, 3))
        else:
            logger.warning("No data could be processed. Creating fallback dataset.")
            self.data['accelerometer'] = np.zeros((10, self.max_length, 4))
            self.data['labels'] = np.array([0] * 5 + [1] * 5)
            self.data['skeleton'] = np.zeros((10, self.max_length, 32, 3))
        
        # Log dataset statistics
        logger.info(f"Dataset Statistics for subjects {subjects}:")
        logger.info(f"Total files: {self.stats['total_files']}")
        logger.info(f"Successfully processed: {self.stats['processed_files']} ({self.stats['processed_files']/max(1, self.stats['total_files'])*100:.1f}%)")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Falls: {fall_count} samples")
        logger.info(f"Non-falls: {non_fall_count} samples")
        logger.info("Activities distribution:")
        for act_id, count in sorted(self.stats['activities'].items()):
            is_fall = 'Fall' if act_id > 9 else 'ADL'
            logger.info(f"  Activity {act_id} ({is_fall}): {count} files")
        
        return self.data
    
    def normalization(self):
        """Normalize the data using StandardScaler"""
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    num_samples, length = value.shape[:2]
                    # Reshape for normalization
                    flattened = value.reshape(num_samples*length, -1)
                    # Check for NaN values
                    if np.isnan(flattened).any():
                        logger.warning(f"NaN values found in {key} data. Replacing with zeros.")
                        flattened = np.nan_to_num(flattened)
                    # Apply normalization
                    norm_data = StandardScaler().fit_transform(flattened)
                    # Reshape back
                    self.data[key] = norm_data.reshape(num_samples, length, -1)
                except Exception as e:
                    logger.error(f"Error normalizing {key} data: {str(e)}. Keeping original values.")
        return self.data
