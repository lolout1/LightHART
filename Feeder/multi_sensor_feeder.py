# Feeder/multi_sensor_feeder.py

import torch
import numpy as np
from torch.utils.data import Dataset

class MultiSensorFeeder(Dataset):
    """
    data_dict => 
      "teacher_inertial" => var-len arrays
      "student_inertial" => fixed-len arrays if 'fixed_count' is used
      "skeleton" => var-len arrays
      "labels" => list of ints
    We'll produce items => (teacher_inert, student_inert, skeleton, label).
    """
    def __init__(self, data_dict):
        self.t_inert = data_dict["teacher_inertial"]
        self.s_inert = data_dict["student_inertial"]
        self.skel    = data_dict["skeleton"]
        self.labels  = data_dict["labels"]
        N = len(self.labels)
        assert (len(self.t_inert)==N) and (len(self.s_inert)==N) and (len(self.skel)==N)
        print(f"DEBUG: MultiSensorFeeder init => #samples={N}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t_i = self.t_inert[idx].astype(np.float32)
        s_i = self.s_inert[idx].astype(np.float32)
        sk  = self.skel[idx].astype(np.float32)
        lab = self.labels[idx]
        return (t_i, s_i, sk, lab)

def multi_sensor_collate_fn(batch):
    """
    batch => list of (teacher_inert, student_inert, skeleton, label)
    teacher_inert => var-len => produce mask
    student_inert => fixed-len => no mask
    skeleton => var-len => produce mask
    """
    B = len(batch)
    t_list, s_list, sk_list, lab_list = [], [], [], []
    for i in range(B):
        t_list.append(batch[i][0])
        s_list.append(batch[i][1])
        sk_list.append(batch[i][2])
        lab_list.append(batch[i][3])

    # teacher => var-len
    t_lens = [arr.shape[0] for arr in t_list]
    t_feat = t_list[0].shape[1] if B>0 else 0
    max_t  = max(t_lens) if t_lens else 0

    # skeleton => var-len
    sk_lens= [arr.shape[0] for arr in sk_list]
    sk_feat= sk_list[0].shape[1] if B>0 else 0
    max_sk = max(sk_lens) if sk_lens else 0

    # student => fixed-len => e.g. shape(128, feat_dim), no mask
    # we assume the first item => shape(128, s_feat), etc.
    if B>0:
        s_feat = s_list[0].shape[1]
        s_len  = s_list[0].shape[0] # e.g. 128
    else:
        s_feat, s_len = 0, 0

    t_tensor = torch.zeros((B, max_t, t_feat), dtype=torch.float32)
    t_mask   = torch.ones((B, max_t), dtype=torch.bool)
    sk_tensor= torch.zeros((B, max_sk, sk_feat), dtype=torch.float32)
    sk_mask  = torch.ones((B, max_sk), dtype=torch.bool)

    s_tensor = torch.zeros((B, s_len, s_feat), dtype=torch.float32)  # no mask for student

    labels   = torch.zeros((B,), dtype=torch.long)

    for i in range(B):
        lt = t_lens[i]
        lsk= sk_lens[i]
        t_tensor[i,:lt,:] = torch.from_numpy(t_list[i])
        t_mask[i,:lt]     = False

        sk_tensor[i,:lsk,:] = torch.from_numpy(sk_list[i])
        sk_mask[i,:lsk]     = False

        s_tensor[i,:,:]     = torch.from_numpy(s_list[i])
        labels[i]           = lab_list[i]

    return t_tensor, t_mask, s_tensor, sk_tensor, sk_mask, labels

