"""
PyTorch dataset and collate function for quaternion-enhanced IMU data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset

class MultimodalQuatFeeder(Dataset):
    """
    Dataset for multimodal data with quaternion-enhanced IMU features.
    
    Handles both fused IMU and skeleton data for teacher model,
    or just IMU data for student model.
    """
    
    def __init__(self, data_dict):
        """
        Initialize dataset with processed data dictionary.
        
        Args:
            data_dict: Dictionary with keys:
                - 'fused_imu' or 'accelerometer': List of IMU windows
                - 'skeleton' (optional): List of skeleton windows
                - 'labels': List of labels
        """
        self.labels = data_dict['labels']
        
        # Determine which IMU key to use
        if 'fused_imu' in data_dict:
            self.imu_key = 'fused_imu'
        else:
            self.imu_key = 'accelerometer'
            
        self.imu_data = data_dict.get(self.imu_key, [])
        self.skel_data = data_dict.get('skeleton', [])
        
        # Validate data
        assert len(self.imu_data) == len(self.labels), \
            f"IMU data length mismatch: {len(self.imu_data)} vs labels {len(self.labels)}"
            
        if len(self.skel_data) > 0:
            assert len(self.skel_data) == len(self.labels), \
                f"Skeleton data length mismatch: {len(self.skel_data)} vs labels {len(self.labels)}"
    
    def __len__(self):
        """Get dataset length."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get data item.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (imu_data, skeleton_data, label)
        """
        imu_arr = self.imu_data[idx]
        lab = self.labels[idx]
        
        # Return skeleton data if available, otherwise empty array
        if len(self.skel_data) > 0:
            sk_arr = self.skel_data[idx]
        else:
            sk_arr = np.zeros((0, 0), dtype=np.float32)
            
        return imu_arr.astype(np.float32), sk_arr.astype(np.float32), lab

def multimodal_quat_collate_fn(batch):
    """
    Collate function for variable-length sequences.
    
    Args:
        batch: List of (imu, skeleton, label) tuples
        
    Returns:
        Tuple of (imu_list, skeleton_list, labels_tensor)
    """
    # Separate components
    imu_list, sk_list, lab_list = [], [], []
    
    for (imu, sk, lab) in batch:
        imu_list.append(imu)
        sk_list.append(sk)
        lab_list.append(lab)
    
    # Convert labels to tensor
    labels = torch.tensor(lab_list, dtype=torch.long)
    
    return imu_list, sk_list, labels

def pad_collate_fn(batch, fixed_imu_len=128, fixed_skel_len=None):
    """
    Alternative collate function that pads sequences to fixed length.
    
    Args:
        batch: List of (imu, skeleton, label) tuples
        fixed_imu_len: Fixed length for IMU sequences
        fixed_skel_len: Fixed length for skeleton sequences (or None for variable)
        
    Returns:
        Tuple of (imu_tensor, imu_mask, skeleton_tensor, skeleton_mask, labels_tensor)
    """
    # Separate components
    imu_list, sk_list, lab_list = [], [], []
    
    for (imu, sk, lab) in batch:
        imu_list.append(imu)
        sk_list.append(sk)
        lab_list.append(lab)
    
    # Get batch size
    batch_size = len(imu_list)
    
    # Determine feature dimensions
    if batch_size > 0:
        # Skip time column
        imu_feat_dim = imu_list[0].shape[1] - 1
        
        # Check if skeleton data exists
        has_skeleton = (len(sk_list[0]) > 0)
        if has_skeleton:
            skel_feat_dim = sk_list[0].shape[1] - 1  # Skip time column
        else:
            skel_feat_dim = 96  # Default
    else:
        imu_feat_dim = 16  # Default
        skel_feat_dim = 96  # Default
        has_skeleton = False
    
    # Create padded tensors for IMU
    imu_tensor = torch.zeros((batch_size, fixed_imu_len, imu_feat_dim), dtype=torch.float32)
    imu_mask = torch.ones((batch_size, fixed_imu_len), dtype=torch.bool)
    
    # Fill IMU tensor and create mask
    for i in range(batch_size):
        imu = imu_list[i]
        seq_len = min(imu.shape[0], fixed_imu_len)
        
        # Copy data without time column
        imu_tensor[i, :seq_len, :] = torch.from_numpy(imu[:seq_len, 1:])
        
        # Set mask (False = valid data, True = padding)
        imu_mask[i, :seq_len] = False
    
    # Handle skeleton if present
    if has_skeleton:
        if fixed_skel_len is None:
            # Use maximum length in batch
            max_skel_len = max(sk.shape[0] for sk in sk_list)
            fixed_skel_len = max_skel_len
        
        skel_tensor = torch.zeros((batch_size, fixed_skel_len, skel_feat_dim), dtype=torch.float32)
        skel_mask = torch.ones((batch_size, fixed_skel_len), dtype=torch.bool)
        
        # Fill skeleton tensor and create mask
        for i in range(batch_size):
            sk = sk_list[i]
            if sk.shape[0] > 0:
                seq_len = min(sk.shape[0], fixed_skel_len)
                
                # Copy data without time column
                skel_tensor[i, :seq_len, :] = torch.from_numpy(sk[:seq_len, 1:1+skel_feat_dim])
                
                # Set mask
                skel_mask[i, :seq_len] = False
    else:
        # Empty tensors if no skeleton data
        skel_tensor = torch.zeros((batch_size, 1, skel_feat_dim), dtype=torch.float32)
        skel_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    
    # Convert labels to tensor
    labels = torch.tensor(lab_list, dtype=torch.long)
    
    return imu_tensor, imu_mask, skel_tensor, skel_mask, labels
