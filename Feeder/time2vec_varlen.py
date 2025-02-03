# Feeder/time2vec_varlen.py

import numpy as np
import torch
from torch.utils.data import Dataset

class Time2VecVarLenFeeder(Dataset):
    """
    Feeder that handles variable-length windows, e.g. from watch accelerometer
    using the 'variable_time' mode.
    
    Expects processed data (from your DatasetBuilder) with the following keys:
       - 'accelerometer': a list of NumPy arrays, each of shape (n_i, 3 + time2vec_dim)
       - 'labels': a list of labels (one per window) of the same length.
    """
    def __init__(self, dataset: dict, **kwargs):
        self.acc_data = dataset.get('accelerometer', [])
        self.labels   = dataset.get('labels', [])
        # Optionally, you could also process additional modalities such as 'skeleton'
        assert len(self.acc_data) == len(self.labels), \
            f"acc_data length {len(self.acc_data)} != labels length {len(self.labels)}"

    def __len__(self):
        return len(self.acc_data)

    def __getitem__(self, idx):
        # Each item is expected to be an array with shape (n_i, 3 + time2vec_dim)
        data_ = self.acc_data[idx]
        label_ = self.labels[idx]
        return data_.astype(np.float32), label_, idx

def time2vec_varlen_collate_fn(batch):
    """
    Collate function to pad variable-length sequences.
    
    Each element in batch is a tuple (data_, label_, idx) where:
      - data_ has shape (n_i, feat_dim)
    
    This function returns:
      - data_tensor: a tensor of shape (B, max_len, feat_dim)
      - mask_tensor: a boolean tensor of shape (B, max_len) with True for padded positions
      - labels_tensor: a tensor of shape (B,)
    """
    max_len = max(item[0].shape[0] for item in batch)
    feat_dim = batch[0][0].shape[1]
    B = len(batch)
    
    data_tensor = torch.zeros((B, max_len, feat_dim), dtype=torch.float)
    mask_tensor = torch.ones((B, max_len), dtype=torch.bool)  # True indicates padding
    labels = []
    
    for i, (arr, lbl, idx) in enumerate(batch):
        length = arr.shape[0]
        data_tensor[i, :length, :] = torch.from_numpy(arr)
        mask_tensor[i, :length] = False  # mark actual data positions as False (not padded)
        labels.append(lbl)
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels_tensor, mask_tensor

