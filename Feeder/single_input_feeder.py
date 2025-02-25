import torch
from torch.utils.data import Dataset
import numpy as np

class SingleInputFeeder(Dataset):
    """
    For a dictionary:
       {
         'accelerometer': [np.array(128,4), np.array(128,4), ...],
         'labels': [0,1,1,0,...]
       }
    This feeder yields (arr, label) per item => arr.shape=(T,4).
    """
    def __init__(self, data_dict):
        self.acc_data = data_dict.get('accelerometer', [])
        self.labels   = data_dict.get('labels', [])
        assert len(self.acc_data) == len(self.labels), "Mismatch between data and labels"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = self.acc_data[idx]  # shape (T,4)
        lab = self.labels[idx]
        return arr.astype(np.float32), int(lab)

def single_input_collate_fn(batch):
    """
    Collate => pad variable lengths if needed. Usually T=128 if you always have full windows.
    Returns => data: (B, maxT, 4), labels: (B,), mask: (B, maxT).
    """
    if len(batch) == 0:
        return torch.zeros(0,0), torch.zeros(0), torch.ones(0,0)

    # find max length among all items
    max_len = max(item[0].shape[0] for item in batch)
    B = len(batch)

    data_tensor = torch.zeros((B, max_len, 4), dtype=torch.float32)
    mask_tensor = torch.ones((B, max_len), dtype=torch.bool)
    labels = []

    for i,(arr, lab) in enumerate(batch):
        length = arr.shape[0]
        data_tensor[i, :length, :] = torch.from_numpy(arr)
        mask_tensor[i, :length] = False
        labels.append(lab)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels_tensor, mask_tensor
