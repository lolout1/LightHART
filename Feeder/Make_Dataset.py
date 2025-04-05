import pandas as pd
import numpy as np
import math
import torch
import random
import torch.nn.functional as F
import scipy.stats as s
from einops import rearrange
from typing import Dict, Tuple, List, Union, Optional
import logging
import time
import os
import traceback

logger = logging.getLogger("make_dataset")

class Utd_Dataset(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        dataset = np.load(npz_file)
        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.num_samples = self.dataset.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        data = self.dataset[index, :, : , :]
        data = torch.tensor(data)
        label = self.labels[index]
        label = label - 1
        label = torch.tensor(label).long()
        return data, label

class Berkley_mhad(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        dataset = np.load(npz_file)
        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.num_samples = self.dataset.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        data = self.dataset[index, :, :]
        data = torch.tensor(data)
        label = self.labels[index]
        label = label - 1
        label = torch.tensor(label).long()
        return data, label

class Bmhad_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size, transform = None):
        self.acc_data = dataset['acc_data']
        self.skl_data = dataset['skl_data']
        self.labels = dataset['labels']
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        self.batch_size = batch_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        skl_data = torch.tensor(self.skl_data[index, :, :, :])
        acc_data = torch.tensor(self.acc_data[index, : , :])
        label = self.labels[index]
        label = torch.tensor(label).long()
        return (acc_data, skl_data), label, index

class UTD_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size, drop_last=False):
        start_time = time.time()
        self.available_modalities = []
        self.valid_indices = []
        
        if 'labels' not in dataset or dataset['labels'] is None:
            logger.error("No labels found in dataset")
            self.labels = np.zeros(1)
            self.total_samples = 1
        else:
            self.labels = dataset['labels']
            self.total_samples = len(self.labels)
        
        if 'accelerometer' not in dataset or dataset['accelerometer'] is None:
            logger.error("Accelerometer data is required but not provided")
            self.acc_data = np.zeros((self.total_samples, 64, 3))
        else:
            self.acc_data = dataset['accelerometer']
            self.available_modalities.append('accelerometer')
        
        if 'gyroscope' in dataset and dataset['gyroscope'] is not None and len(dataset['gyroscope']) > 0:
            self.gyro_data = dataset['gyroscope']
            self.available_modalities.append('gyroscope')
        else:
            self.gyro_data = np.zeros((self.total_samples, 64, 3))
            logger.info("Gyroscope data not available, using zeros")
            
        if 'quaternion' in dataset and dataset['quaternion'] is not None and len(dataset['quaternion']) > 0:
            self.quat_data = dataset['quaternion']
            self.available_modalities.append('quaternion')
        else:
            self.quat_data = np.zeros((self.total_samples, 64, 4))
            logger.info("Quaternion data not available, using zeros")
            
        if 'skeleton' in dataset and dataset['skeleton'] is not None and len(dataset['skeleton']) > 0:
            self.skl_data = dataset['skeleton']
            try:
                if len(self.skl_data.shape) == 3:
                    self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                    if self.skl_features % 3 == 0:
                        n_joints = self.skl_features // 3
                        self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, n_joints, 3)
                self.available_modalities.append('skeleton')
            except Exception as e:
                logger.error(f"Error reshaping skeleton data: {str(e)}")
                self.skl_data = None
        else:
            self.skl_data = None
        
        self.acc_seq = self.acc_data.shape[1] if hasattr(self.acc_data, 'shape') and len(self.acc_data.shape) > 1 else 64
        self.channels = self.acc_data.shape[2] if hasattr(self.acc_data, 'shape') and len(self.acc_data.shape) > 2 else 3
        self.batch_size = batch_size
        self.transform = None
        self.crop_size = 64
        
        for i in range(self.total_samples):
            try:
                valid = True
                if i >= len(self.acc_data) or len(self.acc_data[i]) == 0:
                    logger.warning(f"Sample {i}: Missing accelerometer data")
                    valid = False
                    continue
                
                if valid:
                    self.valid_indices.append(i)
            except Exception as e:
                logger.error(f"Error validating sample {i}: {str(e)}")
        
        self.num_samples = len(self.valid_indices)
        if self.num_samples == 0:
            logger.warning("No valid samples found - creating fallback sample")
            self.valid_indices = [0]
            self.num_samples = 1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        try:
            valid_index = self.valid_indices[index % len(self.valid_indices)]
            
            acc = torch.tensor(self.acc_data[valid_index], dtype=torch.float32)
            gyro = torch.tensor(self.gyro_data[valid_index], dtype=torch.float32)
            
            if hasattr(self, 'quat_data'):
                quat = torch.tensor(self.quat_data[valid_index], dtype=torch.float32)
            else:
                quat = torch.zeros((self.acc_seq, 4), dtype=torch.float32)
            
            label = self.labels[valid_index]
            return (acc, gyro, quat), label, valid_index
            
        except Exception as e:
            logger.error(f"Error fetching data at index {index}: {str(e)}")
            logger.error(traceback.format_exc())
            return self._create_fallback_sample()
    
    def _create_fallback_sample(self):
        emergency_acc = torch.zeros((self.acc_seq, self.channels), dtype=torch.float32)
        emergency_gyro = torch.zeros((self.acc_seq, self.channels), dtype=torch.float32)
        emergency_quat = torch.zeros((self.acc_seq, 4), dtype=torch.float32)
        emergency_quat[:, 0] = 1.0  # Identity quaternion
        return (emergency_acc, emergency_gyro, emergency_quat), 0, 0
    
    @staticmethod
    def custom_collate_fn(batch):
        try:
            data_tuples = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            indices = [item[2] for item in batch]
            
            # Each data_tuple now contains (acc, gyro, quat)
            acc_tensors = []
            gyro_tensors = []
            quat_tensors = []
            
            for data_tuple in data_tuples:
                if len(data_tuple) >= 3:
                    acc, gyro, quat = data_tuple
                    acc_tensors.append(acc)
                    gyro_tensors.append(gyro)
                    quat_tensors.append(quat)
                elif len(data_tuple) == 2:
                    acc, gyro = data_tuple
                    acc_tensors.append(acc)
                    gyro_tensors.append(gyro)
                    # Create dummy quaternion
                    quat = torch.zeros((acc.shape[0], 4), dtype=torch.float32)
                    quat[:, 0] = 1.0  # Identity quaternion
                    quat_tensors.append(quat)
                else:
                    # Handle unexpected tuple length
                    acc = data_tuple[0] if len(data_tuple) > 0 else torch.zeros((64, 3))
                    acc_tensors.append(acc)
                    gyro = torch.zeros_like(acc)
                    gyro_tensors.append(gyro)
                    quat = torch.zeros((acc.shape[0], 4), dtype=torch.float32)
                    quat[:, 0] = 1.0  # Identity quaternion
                    quat_tensors.append(quat)
            
            try:
                acc_batch = torch.stack(acc_tensors)
                gyro_batch = torch.stack(gyro_tensors)
                quat_batch = torch.stack(quat_tensors)
            except Exception as e:
                logger.error(f"Error stacking tensors: {e}")
                logger.error(traceback.format_exc())
                # Create fallback tensors
                batch_size = len(data_tuples)
                acc_batch = torch.zeros((batch_size, 64, 3))
                gyro_batch = torch.zeros((batch_size, 64, 3))
                quat_batch = torch.zeros((batch_size, 64, 4))
                quat_batch[:, :, 0] = 1.0  # Identity quaternion
            
            labels_tensor = torch.tensor(labels)
            indices_tensor = torch.tensor(indices)
            
            return {'accelerometer': acc_batch, 'gyroscope': gyro_batch, 'quaternion': quat_batch}, labels_tensor, indices_tensor
        
        except Exception as e:
            logger.error(f"Error in collate function: {str(e)}")
            logger.error(traceback.format_exc())
            batch_size = len(batch)
            return {'accelerometer': torch.zeros((batch_size, 64, 3)), 
                   'gyroscope': torch.zeros((batch_size, 64, 3)), 
                   'quaternion': torch.zeros((batch_size, 64, 4))}, torch.zeros(batch_size).long(), torch.zeros(batch_size).long()
