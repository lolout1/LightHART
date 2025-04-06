'''
Dataset Builder for SmartFallMM

This module handles loading, preprocessing, and alignment of multi-modal sensor data
for human activity recognition and fall detection. It implements specialized techniques
for handling variably-sampled sensor data and fusion of accelerometer and gyroscope data
using different orientation filters.

Key features:
- High-performance parallel processing with thread pooling
- Efficient alignment of variable-rate sensor data
- Support for Madgwick, Kalman, and Extended Kalman filters
- Feature extraction optimized for fall detection
'''
import os
from typing import List, Dict, Tuple, Union, Optional, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.io import loadmat
from numpy.linalg import norm
from dtaidistance import dtw
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# Import IMU fusion module
from utils.imu_fusion import (
    process_imu_data, 
    save_aligned_sensor_data,
    hybrid_interpolate,
    extract_features_from_window,
    cleanup_resources,
    update_thread_configuration
)

# Configure logging
log_dir = "debug_logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "loader.log"),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("loader")

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def csvloader(file_path: str, **kwargs) -> np.ndarray:
    '''
    Loads data from CSV files with appropriate handling for different formats.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments

    Returns:
        Numpy array with the loaded data
    '''
    logger.debug(f"Loading CSV file: {file_path}")
    try:
        # Read the file and handle missing values
        file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()

        # Determine number of columns to use based on data type
        if 'skeleton' in file_path:
            cols = 96  # Skeleton data has 32 joints × 3 coordinates
            logger.debug(f"Detected skeleton data with {cols} columns")
        else:
            # Check if this is a meta sensor file or other inertial data
            if file_data.shape[1] > 4:
                # Meta sensor format: epoch, time, elapsed time, x, y, z
                cols = file_data.shape[1] - 3
                file_data = file_data.iloc[:, 3:]  # Skip first 3 columns
            else:
                cols = 3  # Standard inertial data has 3 axes (x, y, z)
            logger.debug(f"Detected inertial sensor data with {cols} columns")

        # Validate data shape
        if file_data.shape[1] < cols:
            logger.warning(f"File has fewer columns than expected: {file_data.shape[1]} < {cols}")
            # Add zero columns if needed
            missing_cols = cols - file_data.shape[1]
            for i in range(missing_cols):
                file_data[f'missing_{i}'] = 0

        # Extract data, skipping header rows if present
        if file_data.shape[0] > 2:
            activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        else:
            activity_data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
            
        logger.debug(f"Extracted data with shape: {activity_data.shape}")

        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise


def matloader(file_path: str, **kwargs) -> np.ndarray:
    '''
    Loads data from MATLAB (.mat) files.

    Args:
        file_path: Path to the MAT file
        **kwargs: Additional arguments including 'key' to specify which data to load

    Returns:
        Numpy array with the loaded data
    '''
    logger.debug(f"Loading MAT file: {file_path}")

    # Check for valid key
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        logger.error(f"Unsupported key for MatLab file: {key}")
        raise ValueError(f"Unsupported {key} for matlab file")

    try:
        # Load data from the specified key
        data = loadmat(file_path)[key]
        logger.debug(f"Loaded MAT data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading MAT {file_path}: {str(e)}")
        raise


# Map file extensions to appropriate loaders
LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}


def avg_pool(sequence: np.ndarray, window_size: int = 5, stride: int = 1,
            max_length: int = 512, shape: Optional[Tuple] = None) -> np.ndarray:
    '''
    Applies average pooling to smooth data and reduce sequence length.

    Args:
        sequence: Input data sequence
        window_size: Size of pooling window
        stride: Stride for pooling
        max_length: Maximum target length
        shape: Shape of the input (used for reshaping)

    Returns:
        Pooled data sequence
    '''
    logger.debug(f"Applying avg_pool with window_size={window_size}, stride={stride}, max_length={max_length}")

    start_time = time.time()

    # Store original shape and reshape for pooling
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)

    # Convert to torch tensor for F.avg_pool1d
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

    # Calculate appropriate stride to achieve target length
    if max_length < sequence_tensor.shape[2]:
        stride = ((sequence_tensor.shape[2] // max_length) + 1)
        logger.debug(f"Adjusted stride to {stride} for max_length={max_length}")
    else:
        stride = 1

    # Apply pooling
    pooled = F.avg_pool1d(sequence_tensor, kernel_size=window_size, stride=stride)

    # Convert back to numpy and reshape
    pooled_np = pooled.squeeze(0).numpy().transpose(1, 0)
    result = pooled_np.reshape(-1, *shape[1:])

    elapsed_time = time.time() - start_time
    logger.debug(f"avg_pool complete: input shape {shape} → output shape {result.shape} in {elapsed_time:.4f}s")

    return result


def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int,
                      input_shape: np.ndarray) -> np.ndarray:
    '''
    Pads or truncates a sequence to a specified length.

    Args:
        sequence: Input data sequence
        max_sequence_length: Target sequence length
        input_shape: Shape of the input (used for reshaping)

    Returns:
        Padded/truncated sequence of uniform length
    '''
    logger.debug(f"Padding sequence to length {max_sequence_length}")

    # Create target shape
    shape = list(input_shape)
    shape[0] = max_sequence_length

    # Apply pooling if needed
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)

    # Create zero-padded array of target shape
    new_sequence = np.zeros(shape, sequence.dtype)

    # Fill with pooled data up to available length
    actual_length = min(len(pooled_sequence), max_sequence_length)
    new_sequence[:actual_length] = pooled_sequence[:actual_length]

    logger.debug(f"Padding complete: shape {input_shape} → {new_sequence.shape}")

    return new_sequence


def create_dataframe_with_timestamps(data, start_time=0, sample_rate=None):
    """
    Creates a DataFrame with synthetic timestamps when actual timestamps are not available.

    Args:
        data: Sensor data array
        start_time: Start time in seconds
        sample_rate: Estimated sample rate (if None, assumes evenly spaced samples)

    Returns:
        DataFrame with timestamp column and sensor data
    """
    num_samples = data.shape[0]

    if sample_rate is None:
        # Create evenly spaced timestamps (assuming 30Hz if not specified)
        timestamps = np.linspace(start_time, start_time + num_samples/30, num_samples)
    else:
        # Create timestamps based on sample rate
        timestamps = np.arange(num_samples) / sample_rate + start_time

    # Create a DataFrame with timestamp and data columns
    df = pd.DataFrame()
    df['timestamp'] = timestamps

    # Add the data columns
    for i in range(data.shape[1]):
        df[f'axis_{i}'] = data[:, i]

    return df


def filter_data_by_ids(data: np.ndarray, ids: List[int]) -> np.ndarray:
    '''
    Selects specific indices from a data array.

    Args:
        data: Input data array
        ids: List of indices to select

    Returns:
        Filtered data array containing only the selected indices
    '''
    logger.debug(f"Filtering data with {len(ids)} indices")

    if len(ids) == 0:
        logger.warning("Empty ID list for filtering")
        return np.array([])

    result = data[ids, :]
    logger.debug(f"Filtered data shape: {result.shape}")

    return result


def filter_repeated_ids(path: List[Tuple[int, int]]) -> Tuple[set, set]:
    '''
    Filters DTW path to create one-to-one mapping between indices.

    Args:
        path: List of index pairs from DTW warping path

    Returns:
        Two sets containing unique indices for each sequence
    '''
    logger.debug(f"Filtering repeated IDs from path with {len(path)} points")

    seen_first = set()
    seen_second = set()

    for (first, second) in path:
        if first not in seen_first and second not in seen_second:
            seen_first.add(first)
            seen_second.add(second)

    logger.debug(f"Filtered to {len(seen_first)} unique first indices and {len(seen_second)} unique second indices")

    return seen_first, seen_second


def align_sequence(data: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
    '''
    Aligns data from multiple sensor modalities using hybrid interpolation and DTW.

    Args:
        data: Dictionary containing sensor data for different modalities
        **kwargs: Additional arguments including filter_type and is_linear_acc

    Returns:
        Dictionary with aligned sensor data
    '''
    logger.info("Aligning sensor data")

    start_time = time.time()
    filter_type = kwargs.get('filter_type', 'ekf')
    is_linear_acc = kwargs.get('is_linear_acc', True)  # Default: data is linear acceleration

    # If we have both accelerometer and gyroscope, align them
    if 'accelerometer' in data and 'gyroscope' in data:
        try:
            # Create DataFrames with timestamps
            acc_df = create_dataframe_with_timestamps(data['accelerometer'])
            gyro_df = create_dataframe_with_timestamps(data['gyroscope'])

            # Align using sensor alignment function
            aligned_acc, aligned_gyro, aligned_times = align_sensor_data(
                acc_df, gyro_df, target_freq=30.0
            )

            if len(aligned_acc) > 0:
                # Update data with interpolated values
                data['accelerometer'] = aligned_acc
                data['gyroscope'] = aligned_gyro
                data['aligned_timestamps'] = aligned_times
                logger.info(f"Aligned inertial sensors: {len(aligned_acc)} samples")
                
                # Apply orientation filtering
                if filter_type:
                    logger.info(f"Applying {filter_type} filter to aligned data")
                    fusion_results = process_imu_data(
                        acc_data=aligned_acc,
                        gyro_data=aligned_gyro,
                        timestamps=aligned_times,
                        filter_type=filter_type,
                        return_features=True,
                        is_linear_acc=is_linear_acc
                    )
                    
                    # Add fusion results to data
                    data.update(fusion_results)
                    logger.info(f"IMU fusion completed with {filter_type} filter")
            else:
                logger.warning("Sensor alignment failed - insufficient overlap")
        except Exception as e:
            logger.error(f"Sensor alignment error: {str(e)}")

    # If we also have skeleton data, align it with inertial data using DTW
    if 'skeleton' in data and 'accelerometer' in data:
        try:
            # Get reference skeleton joint data (typically left wrist)
            joint_id = 9  # Left wrist joint
            skeleton_joint_data = data['skeleton'][:, (joint_id - 1) * 3 : joint_id * 3]
            logger.debug(f"Using joint ID {joint_id} (left wrist) with shape {skeleton_joint_data.shape}")

            # Get accelerometer data
            inertial_data = data['accelerometer']
            
            # Calculate Frobenius norm for both data types
            skeleton_frob_norm = np.linalg.norm(skeleton_joint_data, axis=1)
            inertial_frob_norm = np.linalg.norm(inertial_data, axis=1)

            # Perform DTW alignment
            path = dtw.warping_path(skeleton_frob_norm, inertial_frob_norm)
            logger.debug(f"DTW path found with {len(path)} points")

            # Filter to unique indices
            skeleton_idx, inertial_ids = filter_repeated_ids(path)

            # Apply filtering
            data['skeleton'] = filter_data_by_ids(data['skeleton'], list(skeleton_idx))
            for key in ['accelerometer', 'gyroscope', 'quaternion', 'linear_acceleration']:
                if key in data and data[key] is not None and len(data[key]) > 0:
                    data[key] = filter_data_by_ids(data[key], list(inertial_ids))
            
            if 'aligned_timestamps' in data:
                data['aligned_timestamps'] = filter_data_by_ids(
                    data['aligned_timestamps'].reshape(-1, 1), list(inertial_ids)
                ).flatten()

            logger.info(f"DTW alignment complete: {len(skeleton_idx)} skeleton frames, {len(inertial_ids)} inertial frames")
        except Exception as e:
            logger.error(f"DTW alignment error: {str(e)}")

    elapsed_time = time.time() - start_time
    logger.debug(f"Alignment completed in {elapsed_time:.2f}s")

    return data


def sliding_window(data: np.ndarray, is_fall: bool = False, window_size: int = 64, stride: int = 32) -> List[np.ndarray]:
    """
    Extract sliding windows from time series data.
    
    Args:
        data: Input data array
        is_fall: Whether this is fall data (for peak detection)
        window_size: Size of each window
        stride: Stride between consecutive windows
    
    Returns:
        List of extracted windows
    """
    if len(data) < window_size:
        logger.warning(f"Data length ({len(data)}) smaller than window size ({window_size})")
        return []

    windows = []
    
    if is_fall:
        # For fall data, detect peaks in the acceleration magnitude
        acc_magnitude = np.sqrt(np.sum(data**2, axis=1))
        mean_mag, std_mag = np.mean(acc_magnitude), np.std(acc_magnitude)
        
        # Adaptive threshold based on the data statistics
        threshold = max(1.4, mean_mag + 1.5 * std_mag)
        
        # Find peaks with prominence
        peaks, _ = find_peaks(acc_magnitude, height=threshold, distance=window_size//4, prominence=0.5)
        
        if len(peaks) == 0:
            # If no peaks found, use the maximum point
            peaks = [np.argmax(acc_magnitude)]
        
        # Create windows centered around each peak
        for peak in peaks:
            start = max(0, peak - window_size // 2)
            end = min(len(data), start + window_size)
            
            # Handle edge cases
            if end - start < window_size:
                if start == 0:
                    end = min(len(data), window_size)
                else:
                    start = max(0, end - window_size)
            
            if end - start == window_size:
                windows.append(data[start:end])
    else:
        # For non-fall data, use regular sliding windows
        for start in range(0, len(data) - window_size + 1, stride):
            windows.append(data[start:start + window_size])
    
    return windows


def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int,
                            label: int, fuse: bool, filter_type: str = 'ekf',
                            is_linear_acc: bool = True) -> Dict[str, np.ndarray]:
    """
    Creates windows with optional fusion processing.
    
    Args:
        data: Dictionary of sensor data arrays
        window_size: Size of each window
        label: Label for this activity
        fuse: Whether to apply sensor fusion
        filter_type: Type of fusion filter to use
        is_linear_acc: Whether accelerometer data is linear acceleration
        
    Returns:
        Dictionary of windowed data arrays
    """
    start_time = time.time()
    logger.info(f"Creating windows with fusion={fuse}, filter_type={filter_type}")

    windowed_data = defaultdict(list)

    # Check for required modalities
    has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and len(data['gyroscope']) > 0
    if fuse and not has_gyro:
        logger.warning("Fusion requested but gyroscope data not available")
        fuse = False

    # Determine if this is fall data for adaptive windowing
    is_fall = label == 1
    
    # Create windows for acceleration data
    acc_windows = sliding_window(
        data['accelerometer'], 
        is_fall=is_fall, 
        window_size=window_size,
        stride=10 if is_fall else 32
    )
    
    # Early exit if no windows could be created
    if not acc_windows:
        logger.warning("No windows could be created")
        return windowed_data
    
    # Create windows for all other modalities
    for modality, modality_data in data.items():
        if modality != 'labels' and modality_data is not None and len(modality_data) > 0:
            # Skip 'aligned_timestamps' to be processed separately
            if modality == 'aligned_timestamps':
                continue
                
            # For accelerometer, use windows we already created
            if modality == 'accelerometer':
                windowed_data[modality] = np.array(acc_windows)
                continue
            
            # For other modalities, create matching windows
            try:
                modality_windows = []
                
                # Ensure matching lengths
                min_len = min(len(modality_data), len(data['accelerometer']))
                if min_len < window_size:
                    logger.warning(f"Modality {modality} has insufficient length: {min_len} < {window_size}")
                    continue
                
                # Use same window indices as accelerometer
                for acc_window in acc_windows:
                    # Find index in original accelerometer data
                    matched = False
                    for i in range(len(data['accelerometer']) - window_size + 1):
                        if np.array_equal(acc_window, data['accelerometer'][i:i+window_size]):
                            # Found match - extract window from this modality
                            if i + window_size <= len(modality_data):
                                modality_windows.append(modality_data[i:i+window_size])
                                matched = True
                            break
                    
                    if not matched:
                        # Fallback for no match: use zero padding
                        logger.warning(f"No match found for window in {modality}, using zero padding")
                        modality_windows.append(np.zeros((window_size, modality_data.shape[1]), dtype=modality_data.dtype))
                
                # Convert to array and store
                windowed_data[modality] = np.array(modality_windows)
                
            except Exception as e:
                logger.error(f"Error creating windows for {modality}: {str(e)}")
    
    # Handle timestamps if available
    if 'aligned_timestamps' in data:
        try:
            timestamps_windows = []
            for acc_window in acc_windows:
                # Find index in original accelerometer data
                for i in range(len(data['accelerometer']) - window_size + 1):
                    if np.array_equal(acc_window, data['accelerometer'][i:i+window_size]):
                        if i + window_size <= len(data['aligned_timestamps']):
                            timestamps_windows.append(data['aligned_timestamps'][i:i+window_size])
                        break
            
            # Convert to array and store if we have any windows
            if timestamps_windows:
                windowed_data['aligned_timestamps'] = np.array(timestamps_windows)
        except Exception as e:
            logger.error(f"Error creating windows for timestamps: {str(e)}")
    
    # Apply fusion if requested and we have both accelerometer and gyroscope
    if fuse and 'accelerometer' in windowed_data and 'gyroscope' in windowed_data:
        try:
            # Process each window with fusion
            quaternions = []
            linear_accelerations = []
            fusion_features = []
            
            # Create ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                # Submit tasks
                futures = []
                for i in range(len(windowed_data['accelerometer'])):
                    acc_window = windowed_data['accelerometer'][i]
                    gyro_window = windowed_data['gyroscope'][i]
                    
                    # Get timestamps if available
                    timestamps = None
                    if 'aligned_timestamps' in windowed_data:
                        timestamps = windowed_data['aligned_timestamps'][i]
                    
                    # Submit task to process window
                    futures.append(executor.submit(
                        process_imu_data,
                        acc_data=acc_window,
                        gyro_data=gyro_window,
                        timestamps=timestamps,
                        filter_type=filter_type,
                        return_features=True,
                        is_linear_acc=is_linear_acc
                    ))
                
                # Collect results
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {filter_type} fusion"):
                    result = future.result()
                    quaternions.append(result['quaternion'])
                    linear_accelerations.append(result['linear_acceleration'])
                    if 'fusion_features' in result:
                        fusion_features.append(result['fusion_features'])
            
            # Add results to windowed data
            windowed_data['quaternion'] = np.array(quaternions)
            windowed_data['linear_acceleration'] = np.array(linear_accelerations)
            if fusion_features:
                windowed_data['fusion_features'] = np.array(fusion_features)
            
            logger.info(f"Applied {filter_type} fusion to {len(quaternions)} windows")
            
        except Exception as e:
            logger.error(f"Error in fusion processing: {str(e)}")

    # Add labels
    windowed_data['labels'] = np.repeat(label, len(acc_windows))

    elapsed_time = time.time() - start_time
    logger.info(f"Created {len(acc_windows)} windows in {elapsed_time:.2f}s")

    return windowed_data


class DatasetBuilder:
    '''
    Builds a dataset from sensor data files for machine learning.

    This class handles loading, pre-processing, alignment, and normalization of
    multi-modal sensor data for human activity recognition and fall detection tasks.

    Args:
        dataset: Dataset object containing matched trials
        mode: Processing mode ('avg_pool' or 'sliding_window')
        max_length: Maximum sequence length
        task: Task type ('fd' for fall detection, 'har' for activity recognition, 'age' for age detection)
        fusion_options: Configuration options for sensor fusion
        **kwargs: Additional arguments
    '''
    def __init__(self, dataset: object, mode: str, max_length: int, task='fd', 
                 fusion_options=None, **kwargs) -> None:
        logger.info(f"Initializing DatasetBuilder with mode={mode}, task={task}")

        if mode not in ['avg_pool', 'sliding_window']:
            logger.error(f"Unsupported processing method: {mode}")
            raise ValueError(f"Unsupported processing method {mode}")

        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.fusion_options = fusion_options or {}

        # Create directory for aligned data
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        for dir_name in ["accelerometer", "gyroscope", "skeleton", "quaternion"]:
            os.makedirs(os.path.join(self.aligned_data_dir, dir_name), exist_ok=True)

        # Log fusion options if present
        if fusion_options:
            fusion_enabled = fusion_options.get('enabled', False)
            filter_type = fusion_options.get('filter_type', 'madgwick')
            logger.info(f"Fusion options: enabled={fusion_enabled}, filter_type={filter_type}")

    def load_file(self, file_path):
        '''
        Loads sensor data from a file.

        Args:
            file_path: Path to the data file

        Returns:
            Numpy array containing the loaded data
        '''
        logger.debug(f"Loading file: {file_path}")

        try:
            loader = self._import_loader(file_path)
            data = loader(file_path, **self.kwargs)
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def _import_loader(self, file_path: str):
        '''
        Determines appropriate loader based on file extension.

        Args:
            file_path: Path to the data file

        Returns:
            Loader function for the file type
        '''
        file_type = file_path.split('.')[-1]

        if file_type not in ['csv', 'mat']:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type {file_type}")

        return LOADER_MAP[file_type]

    def process(self, data, label, fuse=False, filter_type='madgwick', visualize=False, is_linear_acc=True):
        '''
        Processes data using either average pooling or sliding windows.

        Args:
            data: Dictionary of sensor data
            label: Activity label
            fuse: Whether to apply sensor fusion
            filter_type: Type of filter to use
            visualize: Whether to generate visualizations
            is_linear_acc: Whether accelerometer data is linear acceleration

        Returns:
            Dictionary of processed data
        '''
        logger.info(f"Processing data for label {label} with mode={self.mode}, fusion={fuse}, filter={filter_type}")

        if self.mode == 'avg_pool':
            # Use average pooling to create fixed-length data
            logger.debug("Applying average pooling")
            processed_data = {}

            for modality, modality_data in data.items():
                if modality != 'labels':
                    processed_data[modality] = pad_sequence_numpy(
                        sequence=modality_data,
                        max_sequence_length=self.max_length,
                        input_shape=modality_data.shape
                    )

            # Add label
            processed_data['labels'] = np.array([label])

            # Apply fusion if requested
            if fuse and 'accelerometer' in processed_data and 'gyroscope' in processed_data:
                try:
                    logger.debug(f"Applying sensor fusion with {filter_type} filter")
                    
                    # Extract timestamps if available
                    timestamps = processed_data.get('aligned_timestamps', None)
                    
                    # Process with IMU fusion
                    fusion_result = process_imu_data(
                        acc_data=processed_data['accelerometer'],
                        gyro_data=processed_data['gyroscope'],
                        timestamps=timestamps,
                        filter_type=filter_type,
                        return_features=True,
                        is_linear_acc=is_linear_acc
                    )
                    
                    # Add fusion results to processed data
                    processed_data.update(fusion_result)
                    
                except Exception as e:
                    logger.error(f"Fusion processing failed: {str(e)}")

            return processed_data
        else:
            # Use sliding window approach
            logger.debug("Using sliding window approach")
            
            # Process the aligned data using selective sliding window
            processed_data = selective_sliding_window(
                data=data,
                window_size=self.max_length,
                label=label,
                fuse=fuse,
                filter_type=filter_type,
                is_linear_acc=is_linear_acc
            )

            return processed_data

    def _add_trial_data(self, trial_data):
        '''
        Adds processed trial data to the dataset.

        Args:
            trial_data: Dictionary of processed sensor data for a trial
        '''
        logger.debug("Adding trial data to dataset")

        for modality, modality_data in trial_data.items():
            if len(modality_data) > 0:  # Only add non-empty data
                self.data[modality].append(modality_data)
                logger.debug(f"Added {modality} data with shape {modality_data.shape if hasattr(modality_data, 'shape') else len(modality_data)}")

    def _len_check(self, d):
        '''
        Checks if data dictionary has sufficient length in each modality.

        Args:
            d: Dictionary of data arrays

        Returns:
            Boolean indicating if all modalities have sufficient data
        '''
        return all(len(v) > 1 for v in d.values())

    def _process_trial(self, trial, label, fuse, filter_type, visualize, save_aligned=False, is_linear_acc=True):
        """
        Process a single trial with robust error handling.

        Args:
            trial: Trial object containing modality file paths
            label: Class label for this trial
            fuse: Whether to use sensor fusion
            filter_type: Type of filter to use
            visualize: Whether to generate visualizations
            save_aligned: Whether to save aligned data to files
            is_linear_acc: Whether accelerometer data is linear acceleration

        Returns:
            Processed trial data or None if processing failed
        """
        try:
            # Create dictionary to hold trial data
            trial_data = {}
            
            # Load data from each modality
            for modality, file_path in trial.files.items():
                try:
                    unimodal_data = self.load_file(file_path)
                    trial_data[modality] = unimodal_data
                except Exception as e:
                    logger.error(f"Error loading {modality} from {file_path}: {str(e)}")
                    return None
            
            # Align sensor data and apply orientation filter
            trial_data = align_sequence(
                trial_data, 
                filter_type=filter_type, 
                is_linear_acc=is_linear_acc
            )
            
            # Save aligned data if requested
            if save_aligned:
                aligned_acc = trial_data.get('accelerometer')
                aligned_gyro = trial_data.get('gyroscope')
                aligned_quat = trial_data.get('quaternion')
                aligned_timestamps = trial_data.get('aligned_timestamps')
                
                if aligned_acc is not None and aligned_gyro is not None:
                    save_aligned_sensor_data(
                        trial.subject_id, 
                        trial.action_id, 
                        trial.sequence_number,
                        aligned_acc,
                        aligned_gyro,
                        aligned_quat,
                        aligned_timestamps if aligned_timestamps is not None else np.arange(len(aligned_acc))
                    )
            
            # Process the aligned data
            processed_data = self.process(
                trial_data, 
                label, 
                fuse, 
                filter_type, 
                visualize,
                is_linear_acc
            )
            
            return processed_data
        
        except Exception as e:
            logger.error(f"Trial processing failed: {str(e)}")
            return None

    def make_dataset(self, subjects: List[int], fuse: bool, filter_type: str = 'madgwick', 
                    visualize: bool = False, save_aligned: bool = False, is_linear_acc: bool = True):
        '''
        Creates a dataset from the sensor data files for the specified subjects.

        Args:
            subjects: List of subject IDs to include
            fuse: Whether to apply sensor fusion
            filter_type: Type of fusion filter to use ('madgwick', 'kalman', 'ekf')
            visualize: Whether to generate visualizations
            save_aligned: Whether to save aligned data to files
            is_linear_acc: Whether accelerometer data is linear acceleration
        '''
        logger.info(f"Making dataset for subjects={subjects}, fuse={fuse}, filter_type={filter_type}")

        start_time = time.time()
        self.data = defaultdict(list)
        self.fuse = fuse
        
        # Check if save_aligned is specified in fusion options
        if hasattr(self, 'fusion_options'):
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(8, len(self.dataset.matched_trials))) as executor:
            # Create a dictionary to track futures for each trial
            future_to_trial = {}
            
            # Submit tasks for processing each trial
            for trial in self.dataset.matched_trials:
                if trial.subject_id not in subjects:
                    continue
                
                # Determine label based on task
                if self.task == 'fd':  # Fall detection
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:  # Activity recognition
                    label = trial.action_id - 1
                
                future = executor.submit(
                    self._process_trial, 
                    trial, 
                    label, 
                    fuse, 
                    filter_type, 
                    visualize,
                    save_aligned,
                    is_linear_acc
                )
                future_to_trial[future] = trial
            
            # Collect results with progress tracking
            count = 0
            processed_count = 0
            skipped_count = 0
            
            for future in tqdm(as_completed(future_to_trial), total=len(future_to_trial), desc="Processing trials"):
                trial = future_to_trial[future]
                count += 1
                
                try:
                    trial_data = future.result()
                    if trial_data is not None and self._len_check(trial_data):
                        self._add_trial_data(trial_data)
                        processed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error processing trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number}: {str(e)}")
                    skipped_count += 1

        # Concatenate data from all trials
        for key in self.data:
            values = self.data[key]
            if all(isinstance(x, np.ndarray) for x in values):
                try:
                    self.data[key] = np.concatenate(values, axis=0)
                    logger.info(f"Concatenated {key} data with shape {self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error concatenating {key} data: {str(e)}")
            else:
                logger.warning(f"Cannot concatenate {key} data - mixed types")

        elapsed_time = time.time() - start_time
        logger.info(f"Dataset creation complete: processed {processed_count}/{count} trials, skipped {skipped_count} in {elapsed_time:.2f}s")

    def normalization(self) -> Dict[str, np.ndarray]:
        '''
        Normalizes each modality in the dataset.

        Returns:
            Dictionary with normalized data for each modality
        '''
        logger.info("Normalizing dataset")

        start_time = time.time()

        # Normalize each modality separately (except labels)
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    # Check if this is a feature that needs normalization
                    if key in ['accelerometer', 'gyroscope', 'quaternion', 'linear_acceleration'] and len(value.shape) >= 2:
                        # Reshape for standardization
                        num_samples, length = value.shape[:2]
                        orig_shape = value.shape

                        # StandardScaler works on 2D data, so reshape
                        reshaped_data = value.reshape(num_samples * length, -1)

                        # Standardize data
                        norm_data = StandardScaler().fit_transform(reshaped_data)

                        # Reshape back to original shape
                        self.data[key] = norm_data.reshape(orig_shape)

                        logger.debug(f"Normalized {key} data: shape={self.data[key].shape}")
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        # These are already extracted features, normalize them directly
                        self.data[key] = StandardScaler().fit_transform(value)
                        logger.debug(f"Normalized {key} features: shape={self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error normalizing {key} data: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(f"Normalization complete in {elapsed_time:.2f}s")

        return self.data

    def cleanup(self):
        """
        Properly clean up resources when done with the dataset builder
        """
        # Clean up IMU fusion thread resources
        cleanup_resources()
