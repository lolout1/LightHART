import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional



class FallDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size=32):
        """Fall detection dataset with 4-channel output (SMV + x,y,z)"""
        self.acc_data = dataset['accelerometer']
        self.labels = dataset['labels']
        self.num_samples = self.acc_data.shape[0]
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get accelerometer data as tensor
        acc_data = torch.tensor(self.acc_data[index], dtype=torch.float32)

        # Always ensure 4 channels (SMV + x,y,z)
        if acc_data.shape[-1] == 3:
            # Calculate SMV
            x, y, z = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]
            smv = torch.sqrt(x**2 + y**2 + z**2).unsqueeze(-1)
            acc_data = torch.cat([smv, acc_data], dim=-1)

        # Get label
        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return acc_data, label, index
