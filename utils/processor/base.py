from typing import Any, List
from abc import ABC, abstractmethod
from scipy.io import loadmat
import pandas as pd
import numpy as np 
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger("processor")

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    """
    Load a CSV file and extract the sensor data.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional keyword arguments
        
    Returns:
        Numpy array with the sensor data
    """
    try:
        try:
            file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        except:
            file_data = pd.read_csv(file_path, index_col=False, header=None, sep=';').dropna().bfill()
        
        # Determine the number of columns based on the file type
        if 'skeleton' in file_path:
            cols = 96  # Fixed size for skeleton data (32 joints * 3 coordinates)
        else:
            if file_data.shape[1] > 4:
                # Format with timestamps and extra columns
                cols = file_data.shape[1] - 3
                file_data = file_data.iloc[:, 3:]
            else:
                # Simple format with 3 columns (x, y, z)
                cols = 3
        
        # Handle missing columns
        if file_data.shape[1] < cols:
            logger.warning(f"File has fewer columns than expected: {file_data.shape[1]} < {cols}")
            missing_cols = cols - file_data.shape[1]
            for i in range(missing_cols):
                file_data[f'missing_{i}'] = 0
        
        # Skip header rows if present
        if file_data.shape[0] > 2:
            activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        else:
            activity_data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
        
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise

def matloader(file_path: str, **kwargs):
    """
    Load a MAT file and extract the sensor data.
    
    Args:
        file_path: Path to the MAT file
        **kwargs: Additional keyword arguments
        
    Returns:
        Numpy array with the sensor data
    """
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        raise ValueError(f'Unsupported {key} for matlab file')
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {
    'csv': csvloader, 
    'mat': matloader
}

def ensure_3d_vector(v, default_value=0.0):
    """
    Ensure that a vector has 3 components.
    
    Args:
        v: Input vector
        default_value: Default value for missing components
        
    Returns:
        3D vector
    """
    if v is None:
        return np.array([default_value, default_value, default_value])
    
    v_array = np.asarray(v)
    if v_array.size == 0:
        return np.array([default_value, default_value, default_value])
    
    if v_array.shape[-1] == 3:
        return v_array
    
    if v_array.ndim == 1:
        if v_array.size == 1:
            return np.array([v_array[0], default_value, default_value])
        if v_array.size == 2:
            return np.array([v_array[0], v_array[1], default_value])
        return v_array[:3]
    
    if v_array.ndim == 2 and v_array.shape[0] == 1:
        if v_array.shape[1] == 1:
            return np.array([v_array[0, 0], default_value, default_value])
        if v_array.shape[1] == 2:
            return np.array([v_array[0, 0], v_array[0, 1], default_value])
        return v_array[0, :3]
    
    return np.array([default_value, default_value, default_value])

def avg_pool(sequence: np.array, window_size: int = 5, stride: int = 1, 
             max_length: int = 128, shape: int = None) -> np.ndarray:
    """
    Apply average pooling to reduce sequence length.
    
    Args:
        sequence: Input sequence
        window_size: Size of the pooling window
        stride: Stride for pooling
        max_length: Maximum length of the output sequence
        shape: Shape of the input sequence (optional)
        
    Returns:
        Pooled sequence
    """
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    stride = ((sequence.shape[2] // max_length) + 1 if max_length < sequence.shape[2] else 1)
    sequence = F.avg_pool1d(sequence, kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1, 0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, input_shape: np.array) -> np.ndarray:
    """
    Pad a sequence to a fixed length.
    
    Args:
        sequence: Input sequence
        max_sequence_length: Target length for the sequence
        input_shape: Shape of the input sequence
        
    Returns:
        Padded sequence
    """
    shape = list(input_shape)
    shape[0] = max_sequence_length
    
    # Use average pooling if sequence is too long
    if len(sequence) > max_sequence_length:
        pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    else:
        pooled_sequence = sequence
    
    # Create new sequence with target length
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def fixed_size_windows(data: np.ndarray, window_size: int = 128, overlap: float = 0.5, min_windows: int = 1) -> List[np.ndarray]:
    """
    Creates fixed-size windows with overlap for consistent processing.
    
    Args:
        data: Input sensor data with shape (n_samples, n_features)
        window_size: Size of each window
        overlap: Overlap ratio between consecutive windows (0.0-1.0)
        min_windows: Minimum number of windows to create
        
    Returns:
        List of windows, each with shape (window_size, n_features)
    """
    if len(data) < window_size:
        # If data is too short, pad with zeros
        padded = np.zeros((window_size, data.shape[1]))
        padded[:len(data)] = data
        return [padded]
    
    stride = int(window_size * (1 - overlap))
    starts = list(range(0, len(data) - window_size + 1, stride))
    
    # Ensure at least min_windows are created
    if len(starts) < min_windows:
        # Create evenly spaced starting points
        if len(data) <= window_size:
            starts = [0]
        else:
            starts = np.linspace(0, len(data) - window_size, min_windows).astype(int).tolist()
    
    windows = []
    for start in starts:
        end = start + window_size
        if end <= len(data):
            windows.append(data[start:end])
    
    return windows

class Processor(ABC):
    """Base class for data processing."""
    
    def __init__(self, file_path: str, mode: str, max_length: int, label: int, **kwargs):
        assert mode in ['fixed_window', 'avg_pool'], f'Processing mode: {mode} is undefined'
        self.label = label 
        self.mode = mode
        self.max_length = max_length
        self.data = []
        self.file_path = file_path
        self.input_shape = []
        self.kwargs = kwargs

    def set_input_shape(self, sequence: np.ndarray) -> List[int]:
        """Set the input shape based on the sequence."""
        self.input_shape = sequence.shape

    def _import_loader(self, file_path: str):
        """Import the appropriate loader for the file type."""
        file_type = file_path.split('.')[-1]
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        return LOADER_MAP[file_type]
    
    def load_file(self, file_path: str):
        """Load a file using the appropriate loader."""
        loader = self._import_loader(file_path)
        data = loader(file_path, **self.kwargs)
        self.set_input_shape(data)
        return data

    def process(self, data):
        """Process the input data."""
        if self.mode == 'avg_pool':
            # Ensure data has 3 columns for accelerometer/gyroscope
            if len(data.shape) == 1 or (len(data.shape) > 1 and data.shape[1] < 3):
                valid_data = []
                for i in range(len(data)):
                    try:
                        valid_data.append(ensure_3d_vector(data[i]))
                    except:
                        valid_data.append(np.zeros(3))
                data = np.array(valid_data)
                self.input_shape = data.shape
            
            # Pad or truncate sequence to fixed length
            data = pad_sequence_numpy(
                sequence=data, 
                max_sequence_length=self.max_length,
                input_shape=self.input_shape
            )
        else:  # fixed_window
            # Create fixed-size windows
            data = fixed_size_windows(
                data,
                window_size=self.max_length,
                overlap=0.5,
                min_windows=1
            )
        
        return data
