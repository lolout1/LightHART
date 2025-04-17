import torch
import numpy as np
from torch.utils.data import Dataset
import logging
import os
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

def butterworth_filter(data, cutoff=7.5, fs=25, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

class FallDetectionDataset(Dataset):
    def __init__(self, data):
        self.acc_data = data.get('accelerometer', np.array([]))
        self.labels = data.get('labels', np.array([]))
        self.subjects = data.get('subjects', np.array([]))
        
        if len(self.acc_data) > 0:
            logger.info(f"Dataset created with {len(self.acc_data)} samples, shape: {self.acc_data.shape}")
        else:
            logger.warning("Empty dataset created")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        acc_data = torch.tensor(self.acc_data[index], dtype=torch.float32)
        
        # Always ensure 4 channels (SMV + x,y,z)
        if acc_data.shape[-1] == 3:
            # Calculate SMV
            x, y, z = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]
            smv = torch.sqrt(x**2 + y**2 + z**2).unsqueeze(-1)
            acc_data = torch.cat([smv, acc_data], dim=-1)
        
        # Get label
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        
        return acc_data, label

def split_by_subjects(builder, subjects, fuse=False):
    logger.info(f"Building dataset for subjects: {subjects}")
    try:
        data = builder.build_dataset(subjects, fuse)
        if len(data.get('accelerometer', [])) == 0:
            logger.warning(f"No data found for subjects: {subjects}")
        return data
    except Exception as e:
        logger.error(f"Error building dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'accelerometer': np.array([]),
            'labels': np.array([]),
            'subjects': np.array([])
        }

def prepare_smartfallmm(args):
    logger.info("Preparing SmartFallMM dataset")
    
    if not hasattr(args, 'dataset_args') or not args.dataset_args:
        args.dataset_args = {
            'age_group': ['young', 'old'],
            'modalities': ['accelerometer'],
            'sensors': ['watch'],
            'mode': 'sliding_window',
            'max_length': 128,
            'task': 'fd'
        }
    
    # Handle various path formats - try to normalize the path
    root_dir = getattr(args, 'data_dir', '../data/smartfallmm')
    if not os.path.exists(root_dir) and not os.path.isabs(root_dir):
        # Try some common relative paths
        alt_paths = [
            f"../{root_dir}",
            root_dir.replace('/data/', 'data/'),
            root_dir.lstrip('/'),
            "../" + root_dir.lstrip('/')
        ]
        for path in alt_paths:
            if os.path.exists(path):
                root_dir = path
                logger.info(f"Using alternative data path: {root_dir}")
                break
    
    from utils.smart_fall_mm import SmartFallMMBuilder
    
    builder = SmartFallMMBuilder(
        root=root_dir,
        age_group=args.dataset_args['age_group'],
        modalities=args.dataset_args['modalities'],
        sensors=args.dataset_args['sensors'],
        mode=args.dataset_args['mode'],
        max_length=args.dataset_args['max_length'],
        task=args.dataset_args['task']
    )
    
    builder.run_pipeline()
    return builder
