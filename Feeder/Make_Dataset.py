import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("feeder")

class FallDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.accelerometer = data.get('accelerometer', None)
        self.labels = data.get('labels', None)
        self.subjects = data.get('subjects', None)
        
        if self.accelerometer is not None:
            self.shape = self.accelerometer.shape
        else:
            self.shape = None
            
        logger.info(f"Dataset initialized with {len(self)} samples")
        if self.accelerometer is not None:
            logger.info(f"Accelerometer shape: {self.accelerometer.shape}")
        
    def __len__(self):
        return len(self.labels) if self.labels is not None else 0
        
    def __getitem__(self, idx):
        if self.accelerometer is None or len(self.accelerometer) <= idx:
            raise IndexError(f"Dataset index {idx} out of range")
            
        acc_data = torch.from_numpy(self.accelerometer[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        subject = torch.tensor(self.subjects[idx], dtype=torch.long) if self.subjects is not None else torch.tensor(0)
        
        return acc_data, label, subject

class Feeder:
    def __init__(self, args):
        self.args = args
        self.subjects = args.subjects
        self.fuse = False
        
    def load_data(self):
        logger.info(f"Loading data for subjects {self.subjects}")
        
        try:
            from utils.dataset import prepare_smartfallmm, split_by_subjects
            
            builder = prepare_smartfallmm(self.args)
            
            data = split_by_subjects(builder, self.subjects, self.fuse)
            
            if 'subjects' not in data or len(data['subjects']) == 0:
                logger.warning("Subject data not found, creating dummy subject IDs")
                if 'labels' in data and len(data['labels']) > 0:
                    data['subjects'] = np.zeros(len(data['labels']), dtype=np.int32)
            
            dataset = FallDataset(data)
            
            if len(dataset) > 0:
                logger.info(f"Dataset loaded with {len(dataset)} samples")
                if dataset.accelerometer is not None:
                    logger.info(f"Accelerometer shape: {dataset.accelerometer.shape}")
                
                if dataset.labels is not None:
                    unique, counts = np.unique(dataset.labels, return_counts=True)
                    logger.info(f"Label distribution: {dict(zip(unique, counts))}")
            else:
                logger.warning("Dataset is empty!")
                
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return FallDataset({
                'accelerometer': np.array([]),
                'labels': np.array([]),
                'subjects': np.array([])
            })
