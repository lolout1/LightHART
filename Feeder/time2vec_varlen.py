# Feeder/time2vec_varlen.py

import numpy as np
import torch
from torch.utils.data import Dataset

class Time2VecVarLenFeeder(Dataset):
    """
    Expects e.g. dataset['accelerometer'], dataset['labels'].
    """
    def __init__(self, dataset:dict, transform=None):
        self.transform = transform
        self.acc_data = dataset.get('accelerometer', [])
        self.labels   = dataset.get('labels', [])
        self.skel_data= dataset.get('skeleton', [])  # optional

        print(f"[DEBUG:Feeder] #acc windows={len(self.acc_data)}, #labels={len(self.labels)}, #skel={len(self.skel_data)}")

        if len(self.acc_data)!=len(self.labels):
            raise ValueError(f"Mismatch #acc_data={len(self.acc_data)} vs #labels={len(self.labels)}")

    def __len__(self):
        return len(self.acc_data)

    def __getitem__(self, idx):
        arr = self.acc_data[idx]
        lab = self.labels[idx]
        if self.transform is not None:
            arr = self.transform(arr)
        arr = arr.astype(np.float32)
        return arr, lab, idx


def time2vec_varlen_collate_fn(batch):
    """
    Collate function to pad variable-length sequences.
    Returns:
     - data_tensor: shape (B, max_len, feat_dim)
     - labels_tensor: shape (B,)
     - mask_tensor: shape (B, max_len), bool
    """
    if len(batch)==0:
        # Rare edge case => return empty
        return torch.zeros(0), torch.zeros(0), torch.ones(0)

    max_len = max(item[0].shape[0] for item in batch)
    feat_dim = batch[0][0].shape[1]
    B = len(batch)
    data_tensor = torch.zeros((B, max_len, feat_dim), dtype=torch.float)
    mask_tensor = torch.ones((B, max_len), dtype=torch.bool)
    labels = []

    for i,(arr, lab, idx) in enumerate(batch):
        length = arr.shape[0]
        data_tensor[i, :length, :] = torch.from_numpy(arr)
        mask_tensor[i, :length] = False
        labels.append(lab)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels_tensor, mask_tensor

