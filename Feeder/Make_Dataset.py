import torch
import numpy as np
import matplotlib.pyplot as plt
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

    def calculate_smv(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Signal Magnitude Vector (SMV) for a given sensor sample.

        Args:
            sample (torch.Tensor): Tensor of shape [T, C] where T is the time steps and C is the coordinates.

        Returns:
            torch.Tensor: SMV tensor of shape [T, 1].
        """
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
        for key, modality_data in self.modalities.items():
            if key == 'labels':
                continue

            sensor_data = modality_data[index]
            if key == 'accelerometer_watch' or key == 'accelerometer_phone':
                smv = self.calculate_smv(sensor_data)
                sensor_data = torch.cat((sensor_data, smv), dim=-1)
            data[key] = sensor_data

        label = self.labels[index]
        return data, label, index

    def plot_data(self, index: int):
        """
        Visualizes data for a given index.

        Args:
            index (int): Index of the sample to visualize.
        """
        sample_data, label, _ = self.__getitem__(index)

        fig, axes = plt.subplots(len(sample_data), 1, figsize=(12, 8 * len(sample_data)))

        for idx, (modality, data) in enumerate(sample_data.items()):
            ax = axes[idx] if len(sample_data) > 1 else axes
            time = np.arange(data.shape[0]) / 31.25  # Assuming 31.25 Hz sampling rate

            if 'smv' in modality:
                ax.plot(time, data[:, -1], label="SMV", color='red')
            for i in range(data.shape[1] - 1):
                ax.plot(time, data[:, i], label=f"Axis {i + 1}")

            ax.set_title(f"{modality.capitalize()} Data (Label: {label})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_distribution(self):
        """
        Visualizes the distribution of labels in the dataset.
        """
        unique, counts = torch.unique(self.labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(unique.numpy(), counts.numpy(), color='blue', alpha=0.7)
        plt.title("Label Distribution")
        plt.xlabel("Labels")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()


# Example Usage in Notebook:
# dataset = {
#     'accelerometer_watch': np.random.randn(1000, 3),
#     'accelerometer_phone': np.random.randn(1000, 3),
#     'labels': np.random.randint(0, 10, size=1000),
# }
# utd = UTD_mm(dataset=dataset, batch_size=32)
# utd.plot_data(0)
# utd.plot_distribution()
