import torch
import numpy as np

class UTD_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size):
        self.acc_data = dataset['accelerometer']
        self.labels = dataset['labels']
        self.skl_data = dataset['skeleton']
        self.num_samples = self.acc_data.shape[0]
        self.acc_seq = self.acc_data.shape[1]
        self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
        self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, -1, 3)
        self.channels = self.acc_data.shape[2]
        self.batch_size = batch_size
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        skl_data = torch.tensor(self.skl_data[index, :, :, :])
        acc_data = torch.tensor(self.acc_data[index, :, :])
        data = dict()
        data['accelerometer'] = acc_data
        data['skeleton'] = skl_data
        label = self.labels[index]
        label = torch.tensor(label)
        label = label.long()
        return data, label, index
