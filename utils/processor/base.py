from typing import Any, List, Tuple, Optional, Dict, Union
from abc import ABC, abstractmethod
from scipy.io import loadmat
import pandas as pd
import numpy as np 
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, butter, filtfilt
import logging
import os
import traceback

logger = logging.getLogger("processor")

def csvloader(file_path: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a CSV file and extract the sensor data with robust error handling.
    """
    try:
        # Try different separators to handle various file formats
        for sep in [',', ';', '\t']:
            try:
                # First try with no header
                file_data = pd.read_csv(file_path, sep=sep, header=None, 
                                       error_bad_lines=False, warn_bad_lines=True)
                
                # Check if this worked with a reasonable result
                if file_data.shape[1] >= 3:
                    break
                    
                # If not, try with header
                file_data = pd.read_csv(file_path, sep=sep, header=0,
                                      error_bad_lines=False, warn_bad_lines=True)
                if file_data.shape[1] >= 3:
                    break
            except Exception:
                continue
        
        # If we couldn't parse the file with any separator, raise exception
        if 'file_data' not in locals():
            raise ValueError(f"Failed to parse CSV file with any common separator")
        
        # Drop rows with NaN values and apply backfill
        file_data = file_data.dropna().bfill()
        
        # Detect file type based on column count
        timestamps = None
        
        # Handle files with timestamp column (first column)
        if file_data.shape[1] >= 4:
            # Extract timestamps based on format
            try:
                # Try to parse as datetime
                timestamps = pd.to_datetime(file_data.iloc[:, 0], errors='coerce')
                
                # Convert to float (seconds since epoch)
                timestamps = timestamps.values.astype(np.int64) / 1e9
                
                # Check if we have valid timestamps (not all NaT)
                if pd.isna(timestamps).all():
                    # Try to parse as float directly
                    timestamps = file_data.iloc[:, 0].astype(float).values
                
                # Extract sensor values (columns 1-3 or more)
                if 'skeleton' in file_path:
                    # Skeleton data (all columns after timestamp)
                    values = file_data.iloc[:, 1:].values.astype(float)
                else:
                    # Regular IMU data (3 columns after timestamp)
                    values = file_data.iloc[:, 1:4].values.astype(float)
            except Exception as e:
                # If timestamp parsing fails, treat first column as data
                logger.warning(f"Timestamp parsing failed: {str(e)}")
                if 'skeleton' in file_path:
                    values = file_data.values.astype(float)
                else:
                    # Take the first 3 columns or all if less than 3
                    col_count = min(file_data.shape[1], 3)
                    values = file_data.iloc[:, :col_count].values.astype(float)
                timestamps = np.arange(len(values))
        else:
            # No timestamp column
            if 'skeleton' in file_path:
                values = file_data.values.astype(float)
            else:
                # Take the first 3 columns or all if less than 3
                col_count = min(file_data.shape[1], 3)
                values = file_data.iloc[:, :col_count].values.astype(float)
            timestamps = np.arange(len(values))
        
        # Ensure we have valid sensor data
        if values.size == 0:
            raise ValueError("No valid sensor data found")
            
        # Ensure we have proper 3-axis data for non-skeleton files
        if 'skeleton' not in file_path and values.shape[1] < 3:
            # Pad with zeros to ensure 3 columns
            padded = np.zeros((values.shape[0], 3))
            padded[:, :values.shape[1]] = values
            values = padded
            
        # Skip potential header rows if numeric conversion failed
        try:
            # Test if first row can be converted to float
            float(values[0, 0])
        except (ValueError, TypeError):
            # Skip first few rows
            values = values[2:] if len(values) > 2 else values
            timestamps = timestamps[2:] if len(timestamps) > 2 else timestamps
            
        return values, timestamps
        
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        # Return empty data rather than raising exception
        return np.zeros((0, 3)), None

def matloader(file_path: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a MAT file and extract the sensor data.
    """
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        raise ValueError(f"Unsupported {key} for matlab file")
    try:
        data = loadmat(file_path)[key]
        return data, None
    except Exception as e:
        logger.error(f"Error loading MAT {file_path}: {str(e)}")
        return np.zeros((0, 3)), None

LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

def apply_lowpass_filter(data: np.ndarray, cutoff: float = 5.0, fs: float = 30.0, order: int = 2) -> np.ndarray:
    """
    Apply a low-pass filter to sensor data.
    """
    if len(data) <= 3:  # Not enough data to filter
        return data
        
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered_data = np.zeros_like(data)
    for axis in range(data.shape[1]):
        try:
            filtered_data[:, axis] = filtfilt(b, a, data[:, axis])
        except Exception as e:
            logger.warning(f"Filtering failed for axis {axis}: {str(e)}")
            filtered_data[:, axis] = data[:, axis]
    
    return filtered_data

def ensure_3d_vector(v, default_value=0.0):
    """
    Ensure that a vector has 3 components.
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
    """
    try:
        if sequence.size == 0:
            return sequence
            
        shape = sequence.shape if shape is None else shape
        sequence = sequence.reshape(shape[0], -1)
        sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        stride = ((sequence.shape[2] // max_length) + 1 if max_length < sequence.shape[2] else 1)
        sequence = F.avg_pool1d(sequence, kernel_size=window_size, stride=stride)
        sequence = sequence.squeeze(0).numpy().transpose(1, 0)
        sequence = sequence.reshape(-1, *shape[1:])
        return sequence
    except Exception as e:
        logger.error(f"Error in avg_pool: {str(e)}")
        # Return original sequence if pooling fails
        return sequence[:max_length] if len(sequence) > max_length else sequence

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, input_shape: np.array) -> np.ndarray:
    """
    Pad a sequence to a fixed length.
    """
    try:
        if sequence.size == 0:
            # Create empty sequence of right shape
            shape = list(input_shape)
            shape[0] = max_sequence_length
            return np.zeros(shape, dtype=np.float32)
            
        shape = list(input_shape)
        shape[0] = max_sequence_length
        
        # Use average pooling if sequence is too long
        if len(sequence) > max_sequence_length:
            pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
        else:
            pooled_sequence = sequence
        
        # Create new sequence with target length
        new_sequence = np.zeros(shape, dtype=np.float32)
        new_sequence[:len(pooled_sequence)] = pooled_sequence[:max_sequence_length]
        return new_sequence
    except Exception as e:
        logger.error(f"Error in pad_sequence_numpy: {str(e)}")
        # Return zero array if padding fails
        shape = list(input_shape)
        shape[0] = max_sequence_length
        return np.zeros(shape, dtype=np.float32)

def fixed_size_windows(data: np.ndarray, window_size: int = 128, overlap: float = 0.5, min_windows: int = 1) -> List[np.ndarray]:
    """
    Creates fixed-size windows with overlap for consistent processing.
    """
    try:
        if len(data) < window_size:
            # If data is too short, pad with zeros
            padded = np.zeros((window_size, data.shape[1]))
            padded[:len(data)] = data
            return [padded]
        
        stride = int(window_size * (1 - overlap))
        if stride < 1:
            stride = 1
            
        starts = list(range(0, len(data) - window_size + 1, stride))
        
        # Ensure at least min_windows are created
        if len(starts) < min_windows:
            # Create evenly spaced starting points
            if len(data) <= window_size:
                starts = [0]
            else:
                try:
                    starts = np.linspace(0, len(data) - window_size, min_windows).astype(int).tolist()
                except:
                    # Fallback to simple windows
                    starts = [0]
                    if len(data) > window_size:
                        starts.append(len(data) - window_size)
        
        windows = []
        for start in starts:
            end = start + window_size
            if end <= len(data):
                windows.append(data[start:end])
        
        # If no windows were created, return at least one
        if not windows:
            padded = np.zeros((window_size, data.shape[1]))
            padded[:min(len(data), window_size)] = data[:min(len(data), window_size)]
            windows = [padded]
            
        return windows
    except Exception as e:
        logger.error(f"Error creating windows: {str(e)}")
        # Return an empty window as fallback
        return [np.zeros((window_size, data.shape[1]))]

def selective_sliding_window(data: np.ndarray, window_size: int, height: float = 1.4, 
                           distance: int = 50, max_windows: int = 10) -> List[np.ndarray]:
    """
    Create windows around peaks in the signal.
    """
    try:
        # Calculate magnitude of acceleration vector
        sqrt_sum = np.sqrt(np.sum(data**2, axis=1))
        
        # Find peaks in signal
        peaks, _ = find_peaks(sqrt_sum, height=height, distance=distance)
        
        windows = []
        for peak in peaks:
            # Create window centered on peak
            half_window = window_size // 2
            start = max(0, peak - half_window)
            end = min(len(data), start + window_size)
            
            # Handle edge cases
            if end - start < window_size:
                if start == 0:
                    end = min(len(data), window_size)
                else:
                    start = max(0, end - window_size)
            
            if end - start == window_size:
                windows.append(data[start:end])
                
            # Limit number of windows
            if len(windows) >= max_windows:
                break
        
        # If no windows were created, create at least one
        if not windows and len(data) > 0:
            if len(data) >= window_size:
                # Take the center section
                middle = len(data) // 2
                half_window = window_size // 2
                start = max(0, middle - half_window)
                end = min(len(data), start + window_size)
                windows.append(data[start:end])
            else:
                # Pad the data
                padded = np.zeros((window_size, data.shape[1]))
                padded[:len(data)] = data
                windows.append(padded)
                
        return windows
    except Exception as e:
        logger.error(f"Error in selective_sliding_window: {str(e)}")
        # Return an empty window as fallback
        return [np.zeros((window_size, data.shape[1]))]

class Processor(ABC):
    """Base class for data processing."""
    
    def __init__(self, file_path: str, mode: str, max_length: int, label: int, **kwargs):
        assert mode in ['fixed_window', 'avg_pool', 'sliding_window', 'selective_window'], \
               f'Processing mode: {mode} is undefined'
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
        return self.input_shape

    def _import_loader(self, file_path: str):
        """Import the appropriate loader for the file type."""
        file_type = file_path.split('.')[-1].lower()
        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'
        return LOADER_MAP[file_type]
    
    def load_file(self, file_path: str):
        """Load a file using the appropriate loader."""
        try:
            loader = self._import_loader(file_path)
            data, timestamps = loader(file_path, **self.kwargs)
            
            # Ensure data is valid
            if data is None or data.size == 0:
                logger.warning(f"Empty data loaded from {file_path}")
                data = np.zeros((self.max_length, 3))
            
            # Apply filtering if needed
            if data.shape[0] > 3 and data.ndim == 2 and data.shape[1] <= 3:
                data = apply_lowpass_filter(data)
                
            self.set_input_shape(data)
            return data, timestamps
        
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            fallback_data = np.zeros((self.max_length, 3))
            self.set_input_shape(fallback_data)
            return fallback_data, None

    def process(self, data):
        """Process the input data."""
        try:
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
                    self.set_input_shape(data)
                
                # Pad or truncate sequence to fixed length
                processed_data = pad_sequence_numpy(
                    sequence=data, 
                    max_sequence_length=self.max_length,
                    input_shape=self.input_shape
                )
                return processed_data
                
            elif self.mode == 'fixed_window' or self.mode == 'sliding_window':
                # Create fixed-size windows
                return fixed_size_windows(
                    data,
                    window_size=self.max_length,
                    overlap=0.5,
                    min_windows=1
                )
                
            elif self.mode == 'selective_window':
                # Create windows around peaks
                if self.label == 1:  # Falls
                    return selective_sliding_window(
                        data, 
                        window_size=self.max_length,
                        height=1.2, 
                        distance=128
                    )
                else:  # ADLs
                    return selective_sliding_window(
                        data, 
                        window_size=self.max_length,
                        height=1.2, 
                        distance=128
                    )
            else:
                # Default to fixed windows
                return fixed_size_windows(
                    data,
                    window_size=self.max_length,
                    overlap=0.5,
                    min_windows=1
                )
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            # Return a default window
            return [np.zeros((self.max_length, 3))]
