import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from utils.dataset import split_by_subjects, prepare_smartfallmm

logger = logging.getLogger("feeder")

class FallDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.accelerometer = data.get('accelerometer', None)
        self.gyroscope = data.get('gyroscope', None)
        self.quaternion = data.get('quaternion', None)
        self.linear_acceleration = data.get('linear_acceleration', None)
        self.fusion_features = data.get('fusion_features', None)
        self.labels = data.get('labels', None)
        self.subjects = data.get('subjects', None)
        self.shape = self.accelerometer.shape if self.accelerometer is not None else None
        logger.info(f"Dataset initialized with {len(self)} samples")
        
    def __len__(self):
        return len(self.labels) if self.labels is not None else 0
        
    def __getitem__(self, idx):
        data_dict = {}
        if self.accelerometer is not None:
            data_dict['accelerometer'] = torch.from_numpy(self.accelerometer[idx]).float()
        if self.gyroscope is not None:
            data_dict['gyroscope'] = torch.from_numpy(self.gyroscope[idx]).float()
        if self.quaternion is not None:
            data_dict['quaternion'] = torch.from_numpy(self.quaternion[idx]).float()
        if self.linear_acceleration is not None:
            data_dict['linear_acceleration'] = torch.from_numpy(self.linear_acceleration[idx]).float()
        if self.fusion_features is not None:
            data_dict['fusion_features'] = torch.from_numpy(self.fusion_features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        subject = torch.tensor(self.subjects[idx], dtype=torch.long) if self.subjects is not None else torch.tensor(0)
        return data_dict, label, subject

class Feeder:
    def __init__(self, args):
        self.args = args
        self.subjects = args.subjects
        self.fuse = args.fusion
        
    def load_data(self):
        logger.info(f"Loading data for fold {self.args.fold}, subjects {self.subjects}")
        try:
            data = split_by_subjects(prepare_smartfallmm(self.args), self.subjects, self.fuse)
            if 'subjects' not in data or len(data['subjects']) == 0:
                logger.warning("Subject data not found, creating dummy subject IDs")
                if 'labels' in data and len(data['labels']) > 0:
                    data['subjects'] = np.zeros(len(data['labels']), dtype=np.int32)
            dataset = FallDataset(data)
            logger.info(f"Dataset loaded with {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return FallDataset({
                'accelerometer': np.array([]),
                'gyroscope': np.array([]),
                'labels': np.array([]),
                'subjects': np.array([])
            })
