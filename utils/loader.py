import os
import time
import traceback
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import find_peaks, savgol_filter
import threading

from utils.imu_fusion import (
    process_imu_data,
    save_aligned_sensor_data,
    align_sensor_data,
    MAX_THREADS,
    thread_pool,
    file_semaphore
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loader")

def csvloader(file_path: str, **kwargs) -> np.ndarray:
    logger.debug(f"Loading CSV file: {file_path}")
    try:
        try:
            file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        except:
            file_data = pd.read_csv(file_path, index_col=False, header=None, sep=';').dropna().bfill()
        if 'skeleton' in file_path:
            cols = 96
        else:
            if file_data.shape[1] > 4:
                cols = file_data.shape[1] - 3
                file_data = file_data.iloc[:, 3:]
            else:
                cols = 3
        if file_data.shape[1] < cols:
            logger.warning(f"File has fewer columns than expected: {file_data.shape[1]} < {cols}")
            missing_cols = cols - file_data.shape[1]
            for i in range(missing_cols):
                file_data[f'missing_{i}'] = 0
        if file_data.shape[0] > 2:
            activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        else:
            activity_data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        raise

def matloader(file_path: str, **kwargs) -> np.ndarray:
    logger.debug(f"Loading MAT file: {file_path}")
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        logger.error(f"Unsupported key for MatLab file: {key}")
        raise ValueError(f"Unsupported {key} for matlab file")
    try:
        from scipy.io import loadmat
        data = loadmat(file_path)[key]
        logger.debug(f"Loaded MAT data with shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading MAT {file_path}: {str(e)}")
        raise

LOADER_MAP = {
    'csv': csvloader,
    'mat': matloader
}

def avg_pool(sequence: np.ndarray, window_size: int = 5, stride: int = 1,
            max_length: int = 512, shape: Optional[Tuple] = None) -> np.ndarray:
    logger.debug(f"Applying avg_pool with window_size={window_size}, stride={stride}, max_length={max_length}")
    start_time = time.time()
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    import torch.nn.functional as F
    import torch
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
    if max_length < sequence_tensor.shape[2]:
        stride = ((sequence_tensor.shape[2] // max_length) + 1)
        logger.debug(f"Adjusted stride to {stride} for max_length={max_length}")
    else:
        stride = 1
    pooled = F.avg_pool1d(sequence_tensor, kernel_size=window_size, stride=stride)
    pooled_np = pooled.squeeze(0).numpy().transpose(1, 0)
    result = pooled_np.reshape(-1, *shape[1:])
    elapsed_time = time.time() - start_time
    logger.debug(f"avg_pool complete: input shape {shape} → output shape {result.shape} in {elapsed_time:.4f}s")
    return result

def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int,
                      input_shape: np.ndarray) -> np.ndarray:
    logger.debug(f"Padding sequence to length {max_sequence_length}")
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    actual_length = min(len(pooled_sequence), max_sequence_length)
    new_sequence[:actual_length] = pooled_sequence[:actual_length]
    logger.debug(f"Padding complete: shape {input_shape} → {new_sequence.shape}")
    return new_sequence

def sliding_window(data: np.ndarray, clearing_time_index: int, max_time: int,
                  sub_window_size: int, stride_size: int) -> np.ndarray:
    logger.debug(f"Creating sliding windows with window_size={sub_window_size}, stride={stride_size}")
    if clearing_time_index < sub_window_size - 1:
        logger.error(f"Invalid clearing_time_index: {clearing_time_index} < {sub_window_size - 1}")
        raise AssertionError("Clearing value needs to be greater or equal to (window size - 1)")
    start = clearing_time_index - sub_window_size + 1
    if max_time >= data.shape[0] - sub_window_size:
        max_time = max_time - sub_window_size + 1
        logger.debug(f"Adjusted max_time to {max_time}")
    sub_windows = (
        start +
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time, step=stride_size), 0).T
    )
    result = data[sub_windows]
    logger.debug(f"Created {result.shape[0]} windows from data with shape {data.shape}")
    return result

def hybrid_interpolate(x, y, x_new, threshold=2.0, window_size=5):
    if len(x) < 2 or len(y) < 2:
        logger.warning("Not enough points for interpolation")
        return np.full_like(x_new, y[0] if len(y) > 0 else 0.0)
    try:
        from scipy.signal import savgol_filter
        dy = np.diff(y)
        dx = np.diff(x)
        rates = np.abs(dy / np.maximum(dx, 1e-10))
        if len(rates) >= window_size:
            rates = savgol_filter(rates, window_size, 2)
        rapid_changes = rates > threshold
        if not np.any(rapid_changes):
            try:
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(x, y)
                return cs(x_new)
            except Exception as e:
                logger.warning(f"Cubic spline failed: {e}, falling back to linear")
                from scipy.interpolate import interp1d
                linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
                return linear_interp(x_new)
        if np.all(rapid_changes):
            from scipy.interpolate import interp1d
            linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
            return linear_interp(x_new)
        from scipy.interpolate import interp1d, CubicSpline
        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        try:
            spline_interp = CubicSpline(x, y)
        except Exception as e:
            logger.warning(f"Cubic spline failed: {e}, using linear for all points")
            return linear_interp(x_new)
        y_interp = np.zeros_like(x_new, dtype=float)
        segments = []
        segment_start = None
        for i in range(len(rapid_changes)):
            if rapid_changes[i] and segment_start is None:
                segment_start = i
            elif not rapid_changes[i] and segment_start is not None:
                segments.append((segment_start, i))
                segment_start = None
        if segment_start is not None:
            segments.append((segment_start, len(rapid_changes)))
        linear_mask = np.zeros_like(x_new, dtype=bool)
        buffer = 0.05
        for start_idx, end_idx in segments:
            t_start = max(x[start_idx] - buffer, x[0])
            t_end = min(x[min(end_idx, len(x)-1)] + buffer, x[-1])
            linear_mask |= (x_new >= t_start) & (x_new <= t_end)
        if np.any(linear_mask):
            y_interp[linear_mask] = linear_interp(x_new[linear_mask])
        if np.any(~linear_mask):
            y_interp[~linear_mask] = spline_interp(x_new[~linear_mask])
        return y_interp
    except Exception as e:
        logger.error(f"Hybrid interpolation failed: {e}")
        from scipy.interpolate import interp1d
        linear_interp = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
        return linear_interp(x_new)

def align_sequence(data):
    try:
        acc_data = data.get('accelerometer')
        gyro_data = data.get('gyroscope')
        skeleton_data = data.get('skeleton')
        if acc_data is None or gyro_data is None:
            return data
        from utils.imu_fusion import align_sensor_data
        aligned_acc, aligned_gyro, common_times = align_sensor_data(acc_data, gyro_data)
        if len(aligned_acc) == 0 or len(aligned_gyro) == 0:
            logger.warning("Sensor alignment failed, using original data")
            return data
        aligned_data = {
            'accelerometer': aligned_acc,
            'gyroscope': aligned_gyro,
            'aligned_timestamps': common_times
        }
        if skeleton_data is not None:
            skeleton_times = np.linspace(0, len(skeleton_data)/30.0, len(skeleton_data))
            aligned_skeleton = np.zeros((len(common_times), skeleton_data.shape[1], skeleton_data.shape[2]))
            for joint in range(skeleton_data.shape[1]):
                for coord in range(skeleton_data.shape[2]):
                    aligned_skeleton[:, joint, coord] = np.interp(
                        common_times,
                        skeleton_times,
                        skeleton_data[:, joint, coord]
                    )
            aligned_data['skeleton'] = aligned_skeleton
        return aligned_data
    except Exception as e:
        logger.error(f"Error in sequence alignment: {str(e)}")
        logger.error(traceback.format_exc())
        return data

def _extract_window(data, start, end, window_size, fuse, filter_type='madgwick'):
    window_data = {}
    for modality, modality_data in data.items():
        if modality != 'labels' and modality_data is not None and len(modality_data) > 0:
            try:
                if modality == 'aligned_timestamps':
                    if len(modality_data.shape) == 1:
                        window_data_array = modality_data[start:min(end, len(modality_data))]
                        if len(window_data_array) < window_size:
                            padded = np.zeros(window_size, dtype=window_data_array.dtype)
                            padded[:len(window_data_array)] = window_data_array
                            window_data_array = padded
                    else:
                        window_data_array = modality_data[start:min(end, len(modality_data)), :]
                        if window_data_array.shape[0] < window_size:
                            padded = np.zeros((window_size, window_data_array.shape[1]), dtype=window_data_array.dtype)
                            padded[:window_data_array.shape[0]] = window_data_array
                            window_data_array = padded
                else:
                    window_data_array = modality_data[start:min(end, len(modality_data)), :]
                    if window_data_array.shape[0] < window_size:
                        padded = np.zeros((window_size, window_data_array.shape[1]), dtype=window_data_array.dtype)
                        padded[:window_data_array.shape[0]] = window_data_array
                        window_data_array = padded
                window_data[modality] = window_data_array
            except Exception as e:
                logger.error(f"Error extracting {modality} window: {str(e)}")
                if modality == 'accelerometer':
                    window_data[modality] = np.zeros((window_size, 3))
                elif modality == 'gyroscope':
                    window_data[modality] = np.zeros((window_size, 3))
                elif modality == 'quaternion':
                    window_data[modality] = np.zeros((window_size, 4))
                else:
                    window_data[modality] = None
    if fuse and 'accelerometer' in window_data and 'gyroscope' in window_data:
        try:
            acc_window = window_data['accelerometer']
            gyro_window = window_data['gyroscope']
            timestamps = None
            if 'aligned_timestamps' in window_data:
                timestamps = window_data['aligned_timestamps']
                if len(timestamps.shape) > 1:
                    timestamps = timestamps[:, 0] if timestamps.shape[1] > 0 else None
            from utils.imu_fusion import process_imu_data
            fusion_results = process_imu_data(
                acc_data=acc_window,
                gyro_data=gyro_window,
                timestamps=timestamps,
                filter_type=filter_type,
                return_features=False
            )
            window_data['quaternion'] = fusion_results.get('quaternion', np.zeros((window_size, 4)))
            window_data['linear_acceleration'] = fusion_results.get('linear_acceleration', acc_window)
            logger.debug(f"Added fusion data to window using {filter_type} filter")
        except Exception as e:
            logger.error(f"Error in fusion processing: {str(e)}")
            window_data['quaternion'] = np.zeros((window_size, 4))
    else:
        window_data['quaternion'] = np.zeros((window_size, 4))
    if 'quaternion' not in window_data or window_data['quaternion'] is None:
        window_data['quaternion'] = np.zeros((window_size, 4))
    elif window_data['quaternion'].shape[0] != window_size:
        temp = np.zeros((window_size, 4))
        if window_data['quaternion'].shape[0] < window_size:
            temp[:window_data['quaternion'].shape[0]] = window_data['quaternion']
        else:
            temp = window_data['quaternion'][:window_size]
        window_data['quaternion'] = temp
    return window_data

def selective_sliding_window(data: Dict[str, np.ndarray], window_size: int, peaks: Union[List[int], np.ndarray],
                           label: int, fuse: bool, filter_type: str = 'madgwick') -> Dict[str, np.ndarray]:
    start_time = time.time()
    logger.info(f"Creating {len(peaks)} selective windows with {filter_type} fusion")
    from collections import defaultdict
    windowed_data = defaultdict(list)
    has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and len(data['gyroscope']) > 0
    has_acc = 'accelerometer' in data and data['accelerometer'] is not None and len(data['accelerometer']) > 0
    if not has_acc:
        logger.error("Missing accelerometer data - required for processing")
        return {'labels': np.array([label])}
    if fuse and not has_gyro:
        logger.warning("Fusion requested but gyroscope data not available")
        fuse = False
    max_workers = min(8, len(peaks)) if len(peaks) > 0 else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for peak_idx, peak in enumerate(peaks):
            start = max(0, peak - window_size // 2)
            end = min(len(data['accelerometer']), start + window_size)
            if end - start < window_size // 2:
                logger.debug(f"Skipping window at peak {peak}: too small ({end-start} < {window_size//2})")
                continue
            futures.append((
                peak_idx,
                peak,
                executor.submit(
                    _extract_window,
                    data,
                    start,
                    end,
                    window_size,
                    fuse,
                    filter_type
                )
            ))
        from tqdm import tqdm
        windows_created = 0
        collection_iterator = tqdm(futures, desc="Processing windows") if len(futures) > 5 else futures
        for peak_idx, peak, future in collection_iterator:
            try:
                window_data = future.result()
                if fuse and ('quaternion' not in window_data or window_data['quaternion'] is None):
                    logger.warning(f"Window at peak {peak} missing quaternion data, adding zeros")
                    window_data['quaternion'] = np.zeros((window_size, 4))
                for modality, modality_window in window_data.items():
                    if modality_window is not None:
                        windowed_data[modality].append(modality_window)
                windows_created += 1
            except Exception as e:
                logger.error(f"Error processing window at peak {peak}: {str(e)}")
    for modality in windowed_data:
        if modality != 'labels' and len(windowed_data[modality]) > 0:
            try:
                windowed_data[modality] = np.array(windowed_data[modality])
                logger.debug(f"Converted {modality} windows to array with shape {windowed_data[modality].shape}")
            except Exception as e:
                logger.error(f"Error converting {modality} windows to array: {str(e)}")
                if modality == 'quaternion':
                    windowed_data[modality] = np.zeros((windows_created, window_size, 4))
    windowed_data['labels'] = np.repeat(label, windows_created)
    if fuse and ('quaternion' not in windowed_data or len(windowed_data['quaternion']) == 0):
        logger.warning("No quaternion data in final windows, adding zeros")
        if 'accelerometer' in windowed_data and len(windowed_data['accelerometer']) > 0:
            num_windows = len(windowed_data['accelerometer'])
            windowed_data['quaternion'] = np.zeros((num_windows, window_size, 4))
    elapsed_time = time.time() - start_time
    logger.info(f"Created {windows_created} windows in {elapsed_time:.2f}s")
    return windowed_data

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', fusion_options=None, **kwargs):
        logger.info(f"Initializing DatasetBuilder with mode={mode}, task={task}")
        if mode not in ['avg_pool', 'sliding_window']:
            logger.error(f"Unsupported processing method: {mode}")
            raise ValueError(f"Unsupported processing method {mode}")
        self.dataset = dataset
        self.data = {}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.fusion_options = fusion_options or {}
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        for dir_name in ["accelerometer", "gyroscope", "skeleton"]:
            os.makedirs(os.path.join(self.aligned_data_dir, dir_name), exist_ok=True)
        if fusion_options:
            fusion_enabled = fusion_options.get('enabled', False)
            filter_type = fusion_options.get('filter_type', 'madgwick')
            logger.info(f"Fusion options: enabled={fusion_enabled}, filter_type={filter_type}")

    def load_file(self, file_path):
        logger.debug(f"Loading file: {file_path}")
        try:
            file_type = file_path.split('.')[-1]
            if file_type not in ['csv', 'mat']:
                logger.error(f"Unsupported file type: {file_type}")
                raise ValueError(f"Unsupported file type {file_type}")
            loader = LOADER_MAP[file_type]
            data = loader(file_path, **self.kwargs)
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise

    def process(self, data, label, fuse=False, filter_type='madgwick', visualize=False):
        logger.info(f"Processing data for label {label} with mode={self.mode}, fusion={fuse}")
        if self.mode == 'avg_pool':
            logger.debug("Applying average pooling")
            processed_data = {}
            for modality, modality_data in data.items():
                if modality != 'labels':
                    processed_data[modality] = pad_sequence_numpy(
                        sequence=modality_data,
                        max_sequence_length=self.max_length,
                        input_shape=modality_data.shape
                    )
            processed_data['labels'] = np.array([label])
            if fuse and 'accelerometer' in processed_data and 'gyroscope' in processed_data:
                try:
                    logger.debug(f"Applying sensor fusion with {filter_type} filter")
                    timestamps = processed_data.get('aligned_timestamps', None)
                    fusion_result = process_imu_data(
                        acc_data=processed_data['accelerometer'],
                        gyro_data=processed_data['gyroscope'],
                        timestamps=timestamps,
                        filter_type=filter_type,
                        return_features=False
                    )
                    processed_data.update(fusion_result)
                except Exception as e:
                    logger.error(f"Fusion processing failed: {str(e)}")
                    processed_data['quaternion'] = np.zeros((self.max_length, 4))
            if 'quaternion' not in processed_data:
                processed_data['quaternion'] = np.zeros((self.max_length, 4))
            return processed_data
        else:
            logger.debug("Using peak detection for windowing")
            sqrt_sum = np.sqrt(np.sum(data['accelerometer']**2, axis=1))
            if label == 1:
                peaks, _ = find_peaks(sqrt_sum, height=12, distance=10)
            else:
                peaks, _ = find_peaks(sqrt_sum, height=10, distance=20)
            logger.debug(f"Found {len(peaks)} peaks")
            processed_data = selective_sliding_window(
                data=data,
                window_size=self.max_length,
                peaks=peaks,
                label=label,
                fuse=fuse,
                filter_type=filter_type
            )
            return processed_data

    def _add_trial_data(self, trial_data):
        logger.debug("Adding trial data to dataset")
        for modality, modality_data in trial_data.items():
            if isinstance(modality_data, np.ndarray) and modality_data.size > 0:
                if modality not in self.data:
                    self.data[modality] = []
                self.data[modality].append(modality_data)
                logger.debug(f"Appended {modality} data with shape {modality_data.shape}")

    def _len_check(self, d):
        return all(len(v) > 0 for v in d.values())

    def _process_trial(self, trial, label, fuse, filter_type, visualize, save_aligned=False):
        try:
            trial_data = {}
            for modality_name, file_path in trial.files.items():
                try:
                    unimodal_data = self.load_file(file_path)
                    trial_data[modality_name] = unimodal_data
                except Exception as e:
                    logger.error(f"Error loading {modality_name} from {file_path}: {str(e)}")
                    return None
            from utils.loader import align_sequence
            trial_data = align_sequence(trial_data)
            if save_aligned:
                aligned_acc = trial_data.get('accelerometer')
                aligned_gyro = trial_data.get('gyroscope')
                aligned_skl = trial_data.get('skeleton')
                aligned_timestamps = trial_data.get('aligned_timestamps')
                if aligned_acc is not None and aligned_gyro is not None:
                    save_aligned_sensor_data(
                        trial.subject_id,
                        trial.action_id,
                        trial.sequence_number,
                        aligned_acc,
                        aligned_gyro,
                        aligned_skl,
                        aligned_timestamps if aligned_timestamps is not None else np.arange(len(aligned_acc))
                    )
            processed_data = self.process(trial_data, label, fuse, filter_type, visualize)
            return processed_data
        except Exception as e:
            logger.error(f"Trial processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def make_dataset(self, subjects: List[int], fuse: bool, filter_type: str = 'madgwick',
                    visualize: bool = False, save_aligned: bool = False):
        logger.info(f"Making dataset for subjects={subjects}, fuse={fuse}, filter_type={filter_type}")
        start_time = time.time()
        self.data = {}
        self.fuse = fuse
        if hasattr(self, 'fusion_options'):
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        future_to_trial = {}
        with ThreadPoolExecutor(max_workers=min(8, len(self.dataset.matched_trials))) as executor:
            count = 0
            processed_count = 0
            skipped_count = 0
            for trial in self.dataset.matched_trials:
                if trial.subject_id not in subjects:
                    continue
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                future = executor.submit(
                    self._process_trial,
                    trial,
                    label,
                    fuse,
                    filter_type,
                    False,
                    save_aligned
                )
                future_to_trial[future] = trial
            from tqdm import tqdm
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
        if 'quaternion' not in self.data and 'accelerometer' in self.data:
            logger.warning("Adding empty quaternion data to final dataset")
            acc_shape = self.data['accelerometer'].shape
            self.data['quaternion'] = np.zeros((acc_shape[0], acc_shape[1], 4))
        elapsed_time = time.time() - start_time
        logger.info(f"Dataset creation complete: processed {processed_count}/{count} trials, skipped {skipped_count} in {elapsed_time:.2f}s")

    def normalization(self) -> Dict[str, np.ndarray]:
        logger.info("Normalizing dataset")
        start_time = time.time()
        from sklearn.preprocessing import StandardScaler
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion', 'linear_acceleration'] and len(value.shape) >= 2:
                        num_samples, length = value.shape[:2]
                        reshaped_data = value.reshape(num_samples * length, -1)
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                        self.data[key] = norm_data.reshape(value.shape)
                        logger.debug(f"Normalized {key} data: shape={self.data[key].shape}")
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        self.data[key] = StandardScaler().fit_transform(value)
                        logger.debug(f"Normalized {key} features: shape={self.data[key].shape}")
                except Exception as e:
                    logger.error(f"Error normalizing {key} data: {str(e)}")
        elapsed_time = time.time() - start_time
        logger.info(f"Normalization complete in {elapsed_time:.2f}s")
        return self.data

