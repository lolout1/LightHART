import numpy as np
import torch
from torch.utils.data import Dataset

class MultiModalVarLenFeeder(Dataset):
    """
    Feeder for multi-modal variable-length data for teacher training.
    Expects processed data (from your DatasetBuilder) with the following keys:
       - 'accelerometer': list of NumPy arrays, each of shape (n_i, accel_feat_dim)
       - 'skeleton': list of NumPy arrays, each of shape (m_i, skel_feat_dim)
       - 'accel_time': list of NumPy arrays, each of shape (n_i,) with accelerometer timestamps
       - 'labels': list of labels (one per window)
    """
    def __init__(self, dataset: dict, **kwargs):
        self.acc_data = dataset.get('accelerometer', [])
        self.skel_data = dataset.get('skeleton', [])
        self.accel_time = dataset.get('accel_time', [])
        self.labels = dataset.get('labels', [])
        # Check that lengths match:
        assert len(self.acc_data) == len(self.labels), "Length mismatch: accelerometer data and labels"
        assert len(self.skel_data) == len(self.labels), "Length mismatch: skeleton data and labels"
        assert len(self.accel_time) == len(self.labels), "Length mismatch: accel_time and labels"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a tuple: (skeleton_data, accelerometer_data, accel_time, mask, label)
        accel_sample = self.acc_data[idx]
        skel_sample = self.skel_data[idx]
        accel_time_sample = self.accel_time[idx]
        label_sample = self.labels[idx]
        # Create a mask for the accelerometer data (assumes no padding yet, so mask is all False)
        mask = np.zeros(accel_sample.shape[0], dtype=bool)
        return (skel_sample.astype(np.float32), 
                accel_sample.astype(np.float32), 
                accel_time_sample.astype(np.float32),
                mask, 
                label_sample)

def multimodal_varlen_collate_fn(batch):
    """
    Collate function for multi-modal variable-length sequences.
    Each element in batch is a tuple: (skel_data, accel_data, accel_time, mask, label).
    We pad the accelerometer data and accel_time to the max length in the batch.
    The skeleton data are left as a list (or padded separately if desired).
    """
    # Pad accelerometer data and accel_time.
    max_len = max(item[1].shape[0] for item in batch)
    accel_feat_dim = batch[1][1].shape[1]  # feature dimension of accelerometer data
    B = len(batch)
    accel_tensor = torch.zeros((B, max_len, accel_feat_dim), dtype=torch.float)
    mask_tensor = torch.ones((B, max_len), dtype=torch.bool)
    accel_time_tensor = torch.zeros((B, max_len), dtype=torch.float)
    skel_list = []
    labels = []
    for i, (skel, accel, accel_time, mask, label) in enumerate(batch):
        L = accel.shape[0]
        accel_tensor[i, :L, :] = torch.from_numpy(accel)
        accel_time_tensor[i, :L] = torch.from_numpy(accel_time)
        mask_tensor[i, :L] = False
        skel_list.append(torch.from_numpy(skel))
        labels.append(label)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return accel_tensor, skel_list, accel_time_tensor, mask_tensor, labels_tensor

