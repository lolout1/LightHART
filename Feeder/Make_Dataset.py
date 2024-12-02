import torch
import numpy as np
from typing import Dict, Tuple, Optional

class UTD_mm(torch.utils.data.Dataset):
    def __init__(self, dataset: Dict[str, np.ndarray], batch_size: int,
                 use_smv: bool = False, window_size: Optional[int] = None):
        self.batch_size = batch_size
        self.use_smv = use_smv
        self.window_size = window_size
        
        if 'labels' not in dataset:
            raise ValueError("Dataset must contain 'labels' key")
        self.labels = torch.tensor(dataset['labels'], dtype=torch.float)
        self.num_samples = len(self.labels)
        
        self.modalities = {}
        for key, data in dataset.items():
            if key == 'labels':
                continue
                
            if not isinstance(data, np.ndarray):
                raise TypeError(f"Data for {key} must be numpy array")
            
            data = self._fix_data_shape(data)
            self.modalities[key] = torch.from_numpy(data.copy()).float()

    def _fix_data_shape(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 3:
            raise ValueError("Data must be 3-dimensional")
            
        if data.shape[1] == self.num_samples:
            data = np.transpose(data, (1, 0, 2))
            
        if data.shape[0] != self.num_samples:
            raise ValueError(f"Number of samples mismatch: {data.shape[0]} vs {self.num_samples}")
            
        return data

    def calculate_smv(self, accelerometer_data: torch.Tensor) -> torch.Tensor:
        squared_sum = torch.sum(accelerometer_data ** 2, dim=-1, keepdim=True)
        return torch.sqrt(squared_sum)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
        if index >= self.num_samples:
            raise IndexError(f"Index {index} out of bounds")
            
        sample_data = {}
        for key, modality_data in self.modalities.items():
            sensor_data = modality_data[index]
            
            if self.use_smv:
                smv = self.calculate_smv(sensor_data)
                sensor_data = torch.cat([sensor_data, smv], dim=-1)
                
            sample_data[key] = sensor_data
            
        return sample_data, self.labels[index], index

    def get_sample_info(self, index: int) -> Dict[str, torch.Size]:
        sample_data, _, _ = self[index]
        return {key: tensor.shape for key, tensor in sample_data.items()}