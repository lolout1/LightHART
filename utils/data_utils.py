import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import logging
from utils.processor.base import csvloader, matloader

logger = logging.getLogger(__name__)

def safe_csv_loader(file_path):
    """
    Safely load CSV file with accelerometer data, handling various formats
    and errors gracefully.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (accelerometer_data, timestamps) or (None, None) if loading fails
    """
    try:
        # Try standard csvloader from base.py first
        try:
            data = csvloader(file_path)
            logger.debug(f"Successfully loaded {file_path} with csvloader")
            return data, None
        except Exception as e:
            logger.debug(f"csvloader failed for {file_path}: {e}, trying pandas")
        
        # If that fails, try with pandas
        try:
            df = pd.read_csv(file_path, header=None)
            
            # Check if any non-numeric data in any row
            for i in range(min(10, len(df))):
                try:
                    # Try to convert the first few rows to float
                    row_data = df.iloc[i, 1:4].astype(float)
                    # If successful, consider this the header row
                    start_row = i
                    break
                except (ValueError, TypeError):
                    continue
            else:
                # If no valid row found in the first 10, try again with default header
                start_row = 0
            
            # Extract accelerometer data and timestamps
            try:
                # Check if first column appears to be a timestamp
                first_col = df.iloc[start_row:, 0]
                has_timestamp = False
                
                # Check if looks like a timestamp (contains ':' or '-')
                sample = str(first_col.iloc[0])
                if ':' in sample or '-' in sample:
                    has_timestamp = True
                    timestamps = pd.to_datetime(first_col, errors='coerce')
                    # Handle NaT values
                    if timestamps.isna().any():
                        logger.warning(f"Found NaT timestamps in {file_path}, using indices instead")
                        timestamps = None
                        has_timestamp = False
                
                # Extract accelerometer data
                if has_timestamp:
                    acc_data = df.iloc[start_row:, 1:4].astype(float).values
                    # Convert timestamps to milliseconds from epoch
                    timestamps = timestamps.astype(np.int64) // 1_000_000
                    timestamps = timestamps.values
                else:
                    acc_data = df.iloc[start_row:, 0:3].astype(float).values
                    timestamps = None
                
                logger.debug(f"Loaded {file_path} with pandas: shape={acc_data.shape}")
                return acc_data, timestamps
                
            except Exception as e:
                logger.warning(f"Error extracting data from {file_path}: {e}")
                return None, None
            
        except Exception as e:
            logger.warning(f"Error loading {file_path} with pandas: {e}")
            return None, None
    
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None, None

def butterworth_filter(data, cutoff=7.5, fs=50, order=4):
    """
    Apply Butterworth low-pass filter to accelerometer data
    
    Args:
        data: Accelerometer data array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered data
    """
    # Check for empty data
    if data is None or len(data) == 0:
        return data
    
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        logger.warning(f"Filtering failed: {e}")
        return data  # Return original data if filtering fails

def compute_smv(data):
    """
    Compute Signal Magnitude Vector (SMV) from x,y,z acceleration
    
    Args:
        data: Numpy array with shape (samples, 3) containing x,y,z acceleration
        
    Returns:
        Array with shape (samples, 4) containing SMV + x,y,z
    """
    if data is None or len(data) == 0:
        return data
    
    try:
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        smv = np.sqrt(x**2 + y**2 + z**2).reshape(-1, 1)
        return np.hstack((smv, data))
    except Exception as e:
        logger.warning(f"SMV computation failed: {e}")
        return data  # Return original data if SMV computation fails

def process_accelerometer_file(file_path, window_size=128, stride=10, filter_data=True):
    """
    Process accelerometer data file into windows
    
    Args:
        file_path: Path to CSV file
        window_size: Size of sliding windows
        stride: Stride between windows
        filter_data: Whether to apply Butterworth filter
        
    Returns:
        Dictionary with 'windows' and 'is_fall' keys
    """
    # Extract subject and activity information from filename
    try:
        filename = os.path.basename(file_path)
        subject_id = int(filename[1:3])
        activity_id = int(filename[4:6])
        is_fall = activity_id > 9  # Activities 10+ are falls
    except:
        logger.warning(f"Could not parse filename {file_path}, assuming not a fall")
        subject_id = 0
        activity_id = 0
        is_fall = False
    
    # Load data
    acc_data, timestamps = safe_csv_loader(file_path)
    
    # Skip if loading failed
    if acc_data is None or len(acc_data) == 0:
        logger.warning(f"Skipping {file_path}: Failed to load data")
        return None
    
    # Filter data if requested
    if filter_data:
        acc_data = butterworth_filter(acc_data)
    
    # Compute SMV
    acc_data = compute_smv(acc_data)
    
    # Skip if data is too short
    if len(acc_data) < window_size:
        logger.warning(f"Skipping {file_path}: Data too short ({len(acc_data)} < {window_size})")
        return None
    
    # Extract windows
    windows = []
    
    if is_fall:
        # For falls, focus on the peak acceleration
        acc_mag = np.sqrt(np.sum(acc_data[:, 1:4]**2, axis=1))
        threshold = np.mean(acc_mag) + 1.5 * np.std(acc_mag)
        
        # Find peaks as local maxima above threshold
        peaks = []
        for i in range(10, len(acc_mag) - 10):
            if (acc_mag[i] > threshold and 
                acc_mag[i] == np.max(acc_mag[max(0, i-10):min(len(acc_mag), i+10)])):
                peaks.append(i)
        
        # If no peaks found, use the middle of the signal
        if not peaks:
            peaks = [len(acc_data)//2]
        
        # Extract windows centered at peaks
        for peak in peaks:
            start = max(0, peak - window_size//2)
            end = start + window_size
            
            if end > len(acc_data):
                continue
            
            windows.append(acc_data[start:end])
    else:
        # For regular activities, use sliding windows
        for start in range(0, len(acc_data) - window_size + 1, stride):
            windows.append(acc_data[start:start + window_size])
    
    if not windows:
        logger.warning(f"No windows extracted from {file_path}")
        return None
    
    return {
        'windows': np.array(windows),
        'subject_id': subject_id,
        'activity_id': activity_id,
        'is_fall': is_fall
    }
