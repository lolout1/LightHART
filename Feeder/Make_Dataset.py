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
        # Identify the inertial modality if present
        self.inertial_modality = next((modality for modality in dataset if modality in ['accelerometer', 'gyroscope']), None)
        
        # Always load accelerometer data and labels
        self.acc_data = dataset['accelerometer']
        self.labels = dataset['labels']
        
        # Check if fusion data is available
        self.has_fusion = 'quaternion' in dataset or 'linear_acceleration' in dataset or 'fusion_features' in dataset
        
        # Initialize fusion-related attributes to None by default
        self.quaternion_data = None
        self.linear_acceleration_data = None
        self.fusion_features_data = None
        self.aligned_timestamps_data = None
        
        if self.has_fusion:
            # Load fusion-related data if available
            if 'quaternion' in dataset:
                self.quaternion_data = dataset['quaternion']
                print(f"Quaternion data shape: {self.quaternion_data.shape}")
                
            if 'linear_acceleration' in dataset:
                self.linear_acceleration_data = dataset['linear_acceleration']
                print(f"Linear acceleration data shape: {self.linear_acceleration_data.shape}")
                
            if 'fusion_features' in dataset:
                self.fusion_features_data = dataset['fusion_features']
                print(f"Fusion features shape: {self.fusion_features_data.shape}")
                
            if 'aligned_timestamps' in dataset:
                self.aligned_timestamps_data = dataset['aligned_timestamps']
        
        # Load gyroscope data if available
        self.gyro_data = dataset.get('gyroscope', None)
        
        # Load skeleton data if available
        if 'skeleton' in dataset:
            self.skl_data = dataset['skeleton']
            self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
            self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, -1, 3)
        else:
            self.skl_data = None
            
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        self.channels = self.acc_data.shape[2]
        self.batch_size = batch_size
        self.transform = None
        self.crop_size = 64
    
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
        
        # Add modalities safely with bounds checking
        if valid_index < len(self.acc_data):
            data['accelerometer'] = torch.tensor(self.acc_data[valid_index, :, :])
        
        if self.gyro_data is not None and valid_index < len(self.gyro_data):
            data['gyroscope'] = torch.tensor(self.gyro_data[valid_index, :, :])
        
        # Add other modalities if available
        if self.quaternion_data is not None and valid_index < len(self.quaternion_data):
            data['quaternion'] = torch.tensor(self.quaternion_data[valid_index, :, :])
        
        if self.linear_acceleration_data is not None and valid_index < len(self.linear_acceleration_data):
            data['linear_acceleration'] = torch.tensor(self.linear_acceleration_data[valid_index, :, :])
        
        if self.fusion_features_data is not None and valid_index < len(self.fusion_features_data):
            data['fusion_features'] = torch.tensor(self.fusion_features_data[valid_index, :])
        
        if self.aligned_timestamps_data is not None and valid_index < len(self.aligned_timestamps_data):
            data['aligned_timestamps'] = torch.tensor(self.aligned_timestamps_data[valid_index, :])
        
        if self.skl_data is not None and valid_index < len(self.skl_data):
            data['skeleton'] = torch.tensor(self.skl_data[valid_index, :, :, :])
        
        # Get label
        label = self.labels[valid_index]
        
        return data, label, valid_index


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
