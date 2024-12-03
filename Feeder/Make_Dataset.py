import torch
import numpy as np
from typing import Dict, Tuple

class UTD_mm(torch.utils.data.Dataset):
    def __init__(self, dataset: Dict[str, np.ndarray], batch_size: int):
        """
        Initializes the UTD_mm dataset.

        Args:
            dataset (Dict[str, np.ndarray]): A dictionary containing modality data and labels.
            batch_size (int): The batch size for data loading.
        """
        print("\nInitializing UTD_mm dataset")
        
        # Convert labels to torch.long
        self.labels = torch.tensor(dataset['labels'], dtype=torch.long)
        self.num_samples = len(self.labels)
        print(f"Number of samples: {self.num_samples}")
        
        # Process each modality and sensor, converting to torch.float32
        self.modalities = {}
        for key, data in dataset.items():
            if key != 'labels':
                # Convert numpy array to torch tensor with dtype float32
                self.modalities[key] = torch.from_numpy(data).float()
                print(f"{key} shape: {self.modalities[key].shape}")
        
        self.batch_size = batch_size
        print("Dataset initialization complete")

    def cal_smv(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Smoothed Magnitude Velocity (SMV) for a given sensor sample.

        Args:
            sample (torch.Tensor): Tensor of shape [T, C] where T is the time steps and C is the coordinates.

        Returns:
            torch.Tensor: SMV tensor of shape [T, 1].
        """
        # Compute SMV across the last dimension (C)
        smv = torch.sqrt(torch.sum(sample ** 2, dim=-1, keepdim=True))
        return smv

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
        """
        Retrieves the data and label for a given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, int]: A tuple containing:
                - data (Dict[str, torch.Tensor]): Dictionary with modality data.
                - label (torch.Tensor): The label tensor.
                - index (int): The index of the sample.
        """
 
        data = {}
        
        # Load each modality's data
        for key, modality_data in self.modalities.items():
            if key == 'labels':
                continue
                
            sensor_data = modality_data[index]  # Shape: [T, C] or [T, J, C]
            
            if key == 'skeleton':
                # Keep skeleton data in its original format [T, J, C]
                data[key] = sensor_data
            else:
                # For IMU data, add SMV
                smv = self.cal_smv(sensor_data)  # Shape: [T, 1]
                sensor_data = torch.cat((sensor_data, smv), dim=-1)  # Shape: [T, C+1]
                data[key] = sensor_data
        
        label = self.labels[index]
        return data, label, index

# Example usage for testing
if __name__ == "__main__":
    # Create sample data
    data = {
        'accelerometer_phone': np.random.randn(128, 32, 3).astype(np.float32),   # [num_samples, T, C]
        'accelerometer_watch': np.random.randn(128, 32, 3).astype(np.float32),   # [num_samples, T, C]
        'skeleton': np.random.randn(128, 32, 25, 3).astype(np.float32),          # [num_samples, T, J, C]
        'labels': np.random.randint(0, 2, 128)                                  # [num_samples]
    }
    
    # Initialize dataset
    dataset = UTD_mm(data, batch_size=16)
    
    # Retrieve a sample
    sample_data, sample_label, sample_idx = dataset[0]
    print("\nSample data keys:", sample_data.keys())
    for key in sample_data:
        print(f"{key} data shape: {sample_data[key].shape}, dtype: {sample_data[key].dtype}")
    print(f"Sample label: {sample_label}, dtype: {sample_label.dtype}")
    print(f"Sample index: {sample_idx}")
    
    # If you want to test cal_smv separately, create an instance and call it
    test_sample = torch.randn((8, 3))  # Example tensor [T, C]
    smv = dataset.cal_smv(test_sample)
    print(f"\nCalculated SMV shape: {smv.shape}, dtype: {smv.dtype}")
