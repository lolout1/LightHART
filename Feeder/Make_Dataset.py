import numpy as np
import os
import logging
from utils.dataset import prepare_smartfallmm, split_by_subjects

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feeder")

class UTD_mm:
    def __init__(self, dataset=None, data_path=None, fold=None, subjects=None, fuse=False, **kwargs):
        self.data_path = data_path
        self.fold = fold
        self.subjects = subjects  # <-- Use the subject list passed from main.py
        self.fuse = fuse
        self.fusion_options = kwargs.get('fusion_options', {})
        if dataset is not None:
            self.prepared_data = dataset
        elif data_path and fold and subjects:
            self.prepared_data = self._load_data()
        else:
            self.prepared_data = self._create_empty_dataset()
        self._truncate_data()
        if isinstance(self.prepared_data, dict) and 'labels' in self.prepared_data:
            logger.info(f"Dataset loaded with {len(self.prepared_data['labels'])} samples")

    def _create_empty_dataset(self):
        return {
            'accelerometer': np.zeros((0, 64, 3)),
            'gyroscope': np.zeros((0, 64, 3)),
            'quaternion': np.zeros((0, 64, 4)),
            'linear_acceleration': np.zeros((0, 64, 3)),
            'fusion_features': np.zeros((0, 43)),
            'labels': np.zeros(0, dtype=np.int64),
            'subject': np.zeros(0, dtype=np.int32)
        }

    def _load_data(self):
        logger.info(f"Loading data for fold {self.fold}, subjects {self.subjects}")
        try:
            fusion_options = self.fusion_options  # pass fusion options along
            class Args:
                def __init__(self, fusion_options):
                    self.dataset_args = {
                        'age_group': ['young'],
                        'modalities': ['accelerometer', 'gyroscope'],
                        'sensors': ['watch'],
                        'mode': 'sliding_window',
                        'max_length': 64,
                        'task': 'fd',
                        'fusion_options': fusion_options
                    }
            args_obj = Args(fusion_options)
            data = split_by_subjects(prepare_smartfallmm(args_obj), self.subjects, self.fuse)
            if data is None or 'labels' not in data or len(data['labels']) == 0:
                return self._create_empty_dataset()
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_empty_dataset()

    def _truncate_data(self):
        # Ensure all arrays have the same length
        if self.prepared_data and 'labels' in self.prepared_data:
            lengths = []
            for k, v in self.prepared_data.items():
                if isinstance(v, np.ndarray) and v.ndim > 0:
                    lengths.append(v.shape[0])
            if lengths:
                min_length = min(lengths)
                for k, v in self.prepared_data.items():
                    if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] > min_length:
                        self.prepared_data[k] = v[:min_length]

    def __getitem__(self, index):
        if not self.prepared_data or 'labels' not in self.prepared_data or index >= len(self.prepared_data['labels']):
            raise ValueError(f"Dataset not loaded or index {index} out of range")
        data = {k: self.prepared_data[k][index] for k in self.prepared_data if k != 'labels'}
        return data, self.prepared_data['labels'][index]

    def __len__(self):
        return len(self.prepared_data['labels']) if self.prepared_data and 'labels' in self.prepared_data else 0

class SmartFallMM(UTD_mm):
    def __init__(self, dataset=None, data_path=None, fold=None, subjects=None, fuse=False, **kwargs):
        super().__init__(dataset=dataset, data_path=data_path, fold=fold, subjects=subjects, fuse=fuse, **kwargs)

