import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple

class VariableLengthFeeder(Dataset):
    def __init__(self, dataset, **kwargs):
        """
        Dataset for handling variable length sequences
        Args:
            dataset: Dictionary containing modality data and labels
        """
        self.data = dataset.data
        self.modalities = ['accelerometer']  # Focus on accelerometer data
        self.labels = torch.tensor(self.data['labels'], dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int]:
        sample = {}
        for modality in self.modalities:
            if modality in self.data:
                # Convert numpy array to tensor
                sample[modality] = torch.from_numpy(self.data[modality][idx]).float()
        return sample, self.labels[idx], idx

def collate_variable_length(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int]]:
    """
    Custom collate function for variable length sequences
    Args:
        batch: List of (sample, label, idx) tuples
    Returns:
        Tuple of (batched_samples, labels, indices)
    """
    # Separate samples, labels and indices
    samples, labels, indices = zip(*batch)
    
    # Get all modalities
    modalities = samples[0].keys()
    
    batched_samples = {}
    for modality in modalities:
        # Get sequences for this modality
        sequences = [sample[modality] for sample in samples]
        
        # Get lengths of sequences
        lengths = torch.tensor([seq.size(0) for seq in sequences])
        
        # Pack the sequences
        packed_sequences = torch.nn.utils.rnn.pack_sequence(
            sequences, enforce_sorted=False
        )
        
        batched_samples[modality] = packed_sequences
    
    # Stack labels
    labels = torch.stack(labels)
    
    return batched_samples, labels, list(indices)

def create_data_loader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader that handles variable length sequences
    Args:
        dataset: Dataset instance with processed data
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
    Returns:
        DataLoader instance
    """
    dataset = VariableLengthFeeder(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True
    )
