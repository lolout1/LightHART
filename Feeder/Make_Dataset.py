import pandas as pd
import numpy as np
import math
import torch
import random
import torch.nn.functional as F
import scipy.stats as s
from einops import rearrange
from typing import Dict, Tuple, List, Union, Optional
import quaternion  # numpy-quaternion package
from utils.imu_fusion import process_imu_data, extract_features_from_window
from scipy.spatial.transform import Rotation
import logging

logger = logging.getLogger("make_dataset")

#################### MAIN #####################
# CREATE PYTORCH DATASET
'''
Input Args:
data = ncrc or ntu
num_frames = mocap and nturgb+d frame count!
acc_frames = frames from acc sensor per action
'''

class Utd_Dataset(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        # Load data and labels from npz file
        dataset = np.load(npz_file)
        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.num_samples = self.dataset.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # Get the batch containing the requested index
        data = self.dataset[index, :, : , :]
        data = torch.tensor(data)
        label = self.labels[index]
        label = label - 1
        label = torch.tensor(label)
        label = label.long()
        return data, label

class Berkley_mhad(torch.utils.data.Dataset):
    def __init__(self, npz_file):
        # Load data and labels from npz file
        dataset = np.load(npz_file)
        self.dataset = dataset['data']
        self.labels = dataset['labels']
        self.num_samples = self.dataset.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # Get the batch containing the requested index
        data = self.dataset[index, :, :]
        data = torch.tensor(data)
        label = self.labels[index]
        label = label - 1
        label = torch.tensor(label)
        label = label.long()
        return data, label

class Bmhad_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size, transform = None):
        # Load data and labels from npz file
        self.acc_data = dataset['acc_data']
        self.skl_data = dataset['skl_data']
        self.labels = dataset['labels']
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        self.batch_size = batch_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # Get the batch containing the requested index
        data = dict()
        skl_data = torch.tensor(self.skl_data[index, :, :, :])
        acc_data = torch.tensor(self.acc_data[index, : , :])
        data['skl_data'] = skl_data
        data['acc_data'] =  acc_data
        label = self.labels[index]
        label = torch.tensor(label)
        label = label.long()
        return data, label, index

class UTD_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size):
        """
        Initialize the dataset for multimodal SmartFall data.
        Args:
            dataset: Dictionary containing sensor data and labels
            batch_size: Batch size for the dataloader
        """
        # Initialize data tracking
        self.available_modalities = []
        
        # Load labels
        self.labels = dataset['labels']
        
        # Load accelerometer data (required)
        if 'accelerometer' not in dataset or dataset['accelerometer'] is None:
            raise ValueError("Accelerometer data is required but not provided")
        self.acc_data = dataset['accelerometer'] 
        self.available_modalities.append('accelerometer')
        
        # Load gyroscope data if available
        if 'gyroscope' in dataset and dataset['gyroscope'] is not None:
            self.gyro_data = dataset['gyroscope']
            self.available_modalities.append('gyroscope')
        else:
            self.gyro_data = None
            
        # Load skeleton data if available
        if 'skeleton' in dataset and dataset['skeleton'] is not None:
            self.skl_data = dataset['skeleton']
            self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
            self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, -1, 3)
            self.available_modalities.append('skeleton')
        else:
            self.skl_data = None
            
        # Load quaternion data if available (produced by IMU fusion)
        if 'quaternion' in dataset and dataset['quaternion'] is not None:
            self.quaternion_data = dataset['quaternion']
            self.available_modalities.append('quaternion')
        else:
            self.quaternion_data = None
            
        # Load linear acceleration data if available
        if 'linear_acceleration' in dataset and dataset['linear_acceleration'] is not None:
            self.linear_acceleration_data = dataset['linear_acceleration']
            self.available_modalities.append('linear_acceleration')
        else:
            self.linear_acceleration_data = None
            
        # Load fusion features if available
        if 'fusion_features' in dataset and dataset['fusion_features'] is not None:
            self.fusion_features_data = dataset['fusion_features']
            self.available_modalities.append('fusion_features')
        else:
            self.fusion_features_data = None
            
        # Store dataset dimensions
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        self.channels = self.acc_data.shape[2]
        self.batch_size = batch_size
        self.transform = None
        self.crop_size = 64
        
        logger.info(f"Initialized UTD_mm dataset with {self.num_samples} samples and modalities: {self.available_modalities}")
    
    def random_crop(self, data: torch.Tensor) -> torch.Tensor:
        '''
        Function to add random cropping to the data
        Arg:
            data:
        Output:
            crop_data: will return croped data
        '''
        length = data.shape[0]
        start_idx = np.random.randint(0, length-self.crop_size-1)
        return data[start_idx : start_idx+self.crop_size, :]
    
    def cal_smv(self, sample: torch.Tensor) -> torch.Tensor:
        '''
        Function to calculate Signal Magnitude Vector (SMV)
        '''
        mean = torch.mean(sample, dim=-2, keepdim=True)
        zero_mean = sample - mean
        sum_squared = torch.sum(torch.square(zero_mean), dim=-1, keepdim=True)
        smv = torch.sqrt(sum_squared)
        return smv
    
    def calculate_weight(self, data):
        """
        Calculate the magnitude (weight) of accelerometer data.
        Parameters:
        - data: A PyTorch tensor of shape (128, 3) where each row is [ax, ay, az].
        Returns:
        - A 1D PyTorch tensor of shape (128,) containing the magnitude for each row.
        """
        return torch.sqrt(torch.sum(data**2, dim=-1, keepdim=True))
    
    def calculate_pitch(self, data):
        """
        Calculate the pitch from accelerometer data.
        Parameters:
        - data: A PyTorch tensor of shape (128, 3) where each row is [ax, ay, az].
        Returns:
        - A 1D PyTorch tensor of shape (128,) containing the pitch angle for each row in radians.
        """
        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]
        return torch.atan2(ay, torch.sqrt(ax**2 + az**2))
    
    def calculate_roll(self, data):
        """
        Calculate the roll from accelerometer data.
        Parameters:
        - data: A PyTorch tensor of shape (128, 3) where each row is [ax, ay, az].
        Returns:
        - A 1D PyTorch tensor of shape (128,) containing the roll angle for each row in radians.
        """
        ax = data[:, 0]
        ay = data[:, 1]
        az = data[:, 2]
        return torch.atan2(ax, torch.sqrt(ay**2 + az**2))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        """
        Get a data item by index with robust bounds checking
        
        Args:
            index: Index of the data point
            
        Returns:
            data: Dictionary containing modality data
            label: Class label
            index: Original index
        """
        # Handle out-of-bounds indices with modulo arithmetic
        valid_index = index % self.num_samples
        
        # Create dictionary to hold data
        data = {}
        
        # Add accelerometer data (always required)
        if valid_index < len(self.acc_data):
            data['accelerometer'] = torch.tensor(self.acc_data[valid_index], dtype=torch.float32)
        else:
            # This should never happen, but just in case
            data['accelerometer'] = torch.zeros((self.acc_seq, self.channels), dtype=torch.float32)
        
        # Add gyroscope data if available
        if self.gyro_data is not None and valid_index < len(self.gyro_data):
            data['gyroscope'] = torch.tensor(self.gyro_data[valid_index], dtype=torch.float32)
        
        # Add skeleton data if available
        if self.skl_data is not None and valid_index < len(self.skl_data):
            data['skeleton'] = torch.tensor(self.skl_data[valid_index], dtype=torch.float32)
        
        # Add quaternion data if available
        if self.quaternion_data is not None and valid_index < len(self.quaternion_data):
            data['quaternion'] = torch.tensor(self.quaternion_data[valid_index], dtype=torch.float32)
        
        # Add linear acceleration data if available
        if self.linear_acceleration_data is not None and valid_index < len(self.linear_acceleration_data):
            data['linear_acceleration'] = torch.tensor(self.linear_acceleration_data[valid_index], dtype=torch.float32)
        
        # Add fusion features if available
        if self.fusion_features_data is not None and valid_index < len(self.fusion_features_data):
            data['fusion_features'] = torch.tensor(self.fusion_features_data[valid_index], dtype=torch.float32)
        
        # Get label
        label = self.labels[valid_index]
        
        return data, label, valid_index
    
    @staticmethod
    def custom_collate_fn(batch):
        """
        Custom collate function to handle missing keys across samples.
        
        Args:
            batch: List of (data, label, index) tuples
            
        Returns:
            Collated batch with consistent keys
        """
        # Extract data dictionaries, labels, and indices
        datas = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        indices = [item[2] for item in batch]
        
        # Determine all keys present in any sample
        all_keys = set()
        for data in datas:
            all_keys.update(data.keys())
        
        # Create consistent batch with all keys
        batch_dict = {}
        
        # Process each key across all samples
        for key in all_keys:
            # Check if key exists in all samples
            key_valid = True
            tensors = []
            
            for data in datas:
                if key in data and data[key] is not None:
                    tensors.append(data[key])
                else:
                    key_valid = False
                    break
            
            # If key is valid in all samples, stack the tensors
            if key_valid and tensors:
                try:
                    batch_dict[key] = torch.stack(tensors)
                except Exception as e:
                    logger.warning(f"Error stacking {key} tensors: {e}. Skipping key.")
                    continue
            else:
                # Handle missing keys by creating zeros
                # First, find a sample that has this key
                sample_tensor = None
                for data in datas:
                    if key in data and data[key] is not None:
                        sample_tensor = data[key]
                        break
                
                if sample_tensor is not None:
                    # Create compatible tensors for all samples
                    shape = list(sample_tensor.shape)
                    dtype = sample_tensor.dtype
                    device = sample_tensor.device
                    
                    tensors = []
                    for data in datas:
                        if key in data and data[key] is not None:
                            tensors.append(data[key])
                        else:
                            tensors.append(torch.zeros(shape, dtype=dtype, device=device))
                    
                    try:
                        batch_dict[key] = torch.stack(tensors)
                    except Exception as e:
                        logger.warning(f"Error stacking {key} tensors with zeros: {e}. Skipping key.")
                        continue
        
        # Convert labels and indices to tensors
        labels_tensor = torch.tensor(labels)
        indices_tensor = torch.tensor(indices)
        
        return batch_dict, labels_tensor, indices_tensor


def cal_smv(data):
    """
    Calculate Signal Magnitude Vector (SMV) for a batch of inertial data.
    Args:
        data: Tensor of shape [batch, sequence, channels]
    Returns:
        SMV tensor of shape [batch, sequence, 1]
    """
    mean = torch.mean(data, dim=-2, keepdim=True)
    zero_mean = data - mean
    sum_squared = torch.sum(torch.square(zero_mean), dim=-1, keepdim=True)
    smv = torch.sqrt(sum_squared)
    return smv

if __name__ == "__main__":
    data = torch.randn((8, 128, 3))
    smv = cal_smv(data)
