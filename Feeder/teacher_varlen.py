# Feeder/teacher_varlen.py

import torch
from torch.utils.data import Dataset
import numpy as np

class TeacherVarLenFeeder(Dataset):
    """
    data_dict => 
      skeleton: list of windows => each is shape (Ts, 1+3J) or (Ts,3J)
      accelerometer: list => each is shape (Ta,4) => [time,x,y,z]
      labels => list => same length
    We produce (skel_feats, accel_xyz, accel_time, label).
    """

    def __init__(self, data_dict: dict, num_joints=32):
        self.skel_data = data_dict.get('skeleton', [])
        self.acc_data  = data_dict.get('accelerometer', [])
        self.labels    = data_dict.get('labels', [])
        self.num_joints= num_joints

        assert len(self.skel_data) == len(self.acc_data) == len(self.labels), \
            f"TeacherVarLenFeeder mismatch: skel={len(self.skel_data)}, acc={len(self.acc_data)}, lbl={len(self.labels)}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # skeleton => might be shape (Ts,1+3J) => remove first col if it exists:
        skel_arr = self.skel_data[idx]
        if skel_arr.shape[1] == 1 + 3*self.num_joints:
            skel_feats = skel_arr[:,1:]  # => (Ts, 3J)
        else:
            skel_feats = skel_arr  # => (Ts, 3J) if time was never added
        skel_feats = skel_feats.astype(np.float32)

        # accelerometer => shape => (Ta,4) => [time,x,y,z]
        accel_arr  = self.acc_data[idx]
        accel_time = accel_arr[:,0].astype(np.float32)
        accel_xyz  = accel_arr[:,1:].astype(np.float32)  # (Ta,3)

        label = self.labels[idx]
        return skel_feats, accel_xyz, accel_time, label


def teacher_varlen_collate_fn(batch):
    """
    Each item => (skel_feats, accel_xyz, accel_time, label)
    We'll produce => (skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels)
    Where:
      - skel_pad => (B, Ts_max, skel_dim=3*J)
      - accel_pad=> (B, Ta_max, 3)
      - time_pad => (B, Ta_max) => the 'time' col
      - skel_mask=> (B, Ts_max) bool
      - accel_mask=>(B, Ta_max) bool
      - labels => (B,)
    """
    B = len(batch)

    max_skel_len = max(item[0].shape[0] for item in batch) # item[0] => skel_feats
    max_acc_len  = max(item[1].shape[0] for item in batch) # item[1] => accel_xyz

    skel_dim = batch[0][0].shape[1]  # 3*J
    skel_pad = torch.zeros((B, max_skel_len, skel_dim), dtype=torch.float32)
    skel_mask= torch.ones((B, max_skel_len), dtype=torch.bool)

    accel_pad= torch.zeros((B, max_acc_len, 3), dtype=torch.float32)
    accel_mask= torch.ones((B, max_acc_len), dtype=torch.bool)

    time_pad = torch.zeros((B, max_acc_len), dtype=torch.float32)

    labels_list = []

    for i,(skel_feats, accel_xyz, accel_time, label) in enumerate(batch):
        Ts = skel_feats.shape[0]
        Ta = accel_xyz.shape[0]

        skel_pad[i, :Ts, :] = torch.from_numpy(skel_feats)
        skel_mask[i, :Ts]   = False

        accel_pad[i, :Ta, :] = torch.from_numpy(accel_xyz)
        accel_mask[i, :Ta]   = False

        time_pad[i, :Ta]     = torch.from_numpy(accel_time)

        labels_list.append(label)

    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    return skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels_tensor

