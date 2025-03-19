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
        data = dict()
        skl_data = torch.tensor(self.skl_data[index, :, :, :])
        acc_data = torch.tensor(self.acc_data[index, : , :])
        data['skl_data'] = skl_data
        data['acc_data'] =  acc_data
        label = self.labels[index]
        label = torch.tensor(label).long()
        return data, label, index

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
            self.gyro_data = None
            logger.info("Gyroscope data not available")
            
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
            
        if 'quaternion' in dataset and dataset['quaternion'] is not None and len(dataset['quaternion']) > 0:
            self.quaternion_data = dataset['quaternion']
            self.available_modalities.append('quaternion')
        else:
            self.quaternion_data = None
            
        if 'linear_acceleration' in dataset and dataset['linear_acceleration'] is not None:
            self.linear_acceleration_data = dataset['linear_acceleration']
            self.available_modalities.append('linear_acceleration')
        else:
            self.linear_acceleration_data = None
            
        if 'fusion_features' in dataset and dataset['fusion_features'] is not None:
            self.fusion_features_data = dataset['fusion_features']
            self.available_modalities.append('fusion_features')
        else:
            self.fusion_features_data = None
            
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
                
                if 'gyroscope' in self.available_modalities and (i >= len(self.gyro_data) or len(self.gyro_data[i]) == 0):
                    logger.warning(f"Sample {i}: Missing gyroscope data")
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
        
        elapsed_time = time.time() - start_time
        logger.info(f"Initialized UTD_mm dataset with {self.num_samples} valid samples from {self.total_samples} total")
        logger.info(f"Available modalities: {self.available_modalities}")
    
    def random_crop(self, data: torch.Tensor) -> torch.Tensor:
        length = data.shape[0]
        if length <= self.crop_size:
            return data
        start_idx = np.random.randint(0, length - self.crop_size)
        return data[start_idx : start_idx + self.crop_size, :]
    
    def cal_smv(self, sample: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(sample, dim=-2, keepdim=True)
        zero_mean = sample - mean
        sum_squared = torch.sum(torch.square(zero_mean), dim=-1, keepdim=True)
        smv = torch.sqrt(sum_squared)
        return smv
    
    def calculate_weight(self, data):
        return torch.sqrt(torch.sum(data**2, dim=-1, keepdim=True))
    
    def calculate_pitch(self, data):
        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]
        return torch.atan2(ay, torch.sqrt(ax**2 + az**2))
    
    def calculate_roll(self, data):
        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]
        return torch.atan2(ax, torch.sqrt(ay**2 + az**2))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        try:
            valid_index = self.valid_indices[index % len(self.valid_indices)]
            data = {}
            
            data['accelerometer'] = torch.tensor(self.acc_data[valid_index], dtype=torch.float32)
            
            if 'gyroscope' in self.available_modalities:
                data['gyroscope'] = torch.tensor(self.gyro_data[valid_index], dtype=torch.float32)
            
            if 'skeleton' in self.available_modalities:
                data['skeleton'] = torch.tensor(self.skl_data[valid_index], dtype=torch.float32)
            
            if 'quaternion' in self.available_modalities:
                data['quaternion'] = torch.tensor(self.quaternion_data[valid_index], dtype=torch.float32)
            
            if 'linear_acceleration' in self.available_modalities:
                data['linear_acceleration'] = torch.tensor(self.linear_acceleration_data[valid_index], dtype=torch.float32)
            
            if 'fusion_features' in self.available_modalities:
                data['fusion_features'] = torch.tensor(self.fusion_features_data[valid_index], dtype=torch.float32)
            
            label = self.labels[valid_index]
            return data, label, valid_index
            
        except Exception as e:
            logger.error(f"Error fetching data at index {index}: {str(e)}")
            return self._create_fallback_sample()
    
    def _create_fallback_sample(self):
        emergency_data = {'accelerometer': torch.zeros((self.acc_seq, self.channels), dtype=torch.float32)}
        
        if 'gyroscope' in self.available_modalities:
            emergency_data['gyroscope'] = torch.zeros((self.acc_seq, self.channels), dtype=torch.float32)
        
        if 'quaternion' in self.available_modalities:
            quat = torch.zeros((self.acc_seq, 4), dtype=torch.float32)
            quat[:, 0] = 1.0
            emergency_data['quaternion'] = quat
        
        return emergency_data, 0, 0
    
    @staticmethod
    def custom_collate_fn(batch):
        try:
            datas = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            indices = [item[2] for item in batch]
            
            all_keys = set()
            for data in datas:
                all_keys.update(data.keys())
            
            batch_dict = {}
            for key in all_keys:
                valid_tensors = []
                valid_shapes = []
                first_shape = None
                
                for data in datas:
                    if key in data:
                        tensor = data[key]
                        if first_shape is None:
                            first_shape = tensor.shape
                            valid_shapes.append(first_shape)
                        if tensor.shape == first_shape:
                            valid_tensors.append(tensor)
                
                if len(valid_tensors) == len(datas):
                    try:
                        batch_dict[key] = torch.stack(valid_tensors)
                    except Exception as e:
                        logger.error(f"Error stacking tensors for {key}: {e}")
            
            labels_tensor = torch.tensor(labels)
            indices_tensor = torch.tensor(indices)
            
            return batch_dict, labels_tensor, indices_tensor
        
        except Exception as e:
            logger.error(f"Error in collate function: {str(e)}")
            fallback_dict = {'accelerometer': torch.zeros((1, 64, 3))}
            return fallback_dict, torch.tensor([0]), torch.tensor([0])
