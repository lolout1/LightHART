import os, numpy as np, pandas as pd, torch, torch.nn.functional as F
import logging, time
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("loader")

from utils.imu_fusion import (
    process_imu_data,
    save_aligned_sensor_data,
    align_sensor_data,
    hybrid_interpolate,
    extract_features_from_window,
    cleanup_resources
)

def csvloader(file_path):
    try:
        file_data = pd.read_csv(file_path, index_col=False, header=None).dropna().bfill()
        if 'skeleton' in file_path:
            cols = 96
        elif file_data.shape[1] > 4:
            cols = file_data.shape[1] - 3
            file_data = file_data.iloc[:, 3:]
        else:
            cols = 3
        if file_data.shape[0] > 2:
            activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
        else:
            activity_data = file_data.iloc[:, -cols:].to_numpy(dtype=np.float32)
        return activity_data
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        return np.array([])

def matloader(file_path, **kwargs):
    from scipy.io import loadmat
    key = kwargs.get('key', None)
    if key not in ['d_iner', 'd_skel']:
        raise ValueError(f"Unsupported key {key} for matlab file")
    return loadmat(file_path)[key]

LOADER_MAP = {'csv': csvloader, 'mat': matloader}

def avg_pool(sequence, window_size=5, stride=1, max_length=512, shape=None):
    shape = sequence.shape if shape is None else shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis=0).transpose(0, 2, 1)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
    stride = ((sequence_tensor.shape[2] // max_length) + 1) if max_length < sequence_tensor.shape[2] else 1
    pooled = F.avg_pool1d(sequence_tensor, kernel_size=window_size, stride=stride)
    return pooled.squeeze(0).numpy().transpose(1, 0).reshape(-1, *shape[1:])

def pad_sequence_numpy(sequence, max_sequence_length, input_shape):
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length=max_sequence_length, shape=input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    actual_length = min(len(pooled_sequence), max_sequence_length)
    new_sequence[:actual_length] = pooled_sequence[:actual_length]
    return new_sequence

def create_dataframe_with_timestamps(data, start_time=0, sample_rate=30):
    num_samples = data.shape[0]
    timestamps = np.arange(num_samples) / sample_rate + start_time
    df = pd.DataFrame()
    df['timestamp'] = timestamps
    for i in range(data.shape[1]):
        df[f'axis_{i}'] = data[:, i]
    return df

def sliding_window(data, is_fall=False, window_size=64, stride=32):
    if len(data) < window_size:
        return []
    windows = []
    if is_fall:
        acc_magnitude = np.sqrt(np.sum(data**2, axis=1))
        mean_mag, std_mag = np.mean(acc_magnitude), np.std(acc_magnitude)
        threshold = max(1.4, mean_mag + 1.5 * std_mag)
        peaks, _ = find_peaks(acc_magnitude, height=threshold, distance=window_size//4, prominence=0.5)
        if len(peaks) == 0:
            peaks = [np.argmax(acc_magnitude)]
        for peak in peaks:
            start = max(0, peak - window_size // 2)
            end = min(len(data), start + window_size)
            if end - start < window_size:
                if start == 0:
                    end = min(len(data), window_size)
                else:
                    start = max(0, end - window_size)
            if end - start == window_size:
                windows.append(data[start:end])
    else:
        for start in range(0, len(data) - window_size + 1, stride):
            windows.append(data[start:start + window_size])
    if not windows and len(data) >= window_size:
        center = np.argmax(np.sqrt(np.sum(data**2, axis=1)))
        start = max(0, min(len(data) - window_size, center - window_size // 2))
        windows.append(data[start:start + window_size])
    return windows

def selective_sliding_window(data, window_size, label, fuse=False, filter_type='ekf', is_linear_acc=True):
    from collections import defaultdict
    windowed_data = defaultdict(list)
    has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and len(data['gyroscope']) > 0
    if fuse and not has_gyro:
        logger.warning("Fusion requested but gyroscope data not available")
        fuse = False
    is_fall = label == 1
    acc_windows = sliding_window(data['accelerometer'], is_fall=is_fall, window_size=window_size, stride=10 if is_fall else 32)
    if not acc_windows:
        return windowed_data
    for modality, modality_data in data.items():
        if modality != 'labels' and modality_data is not None and len(modality_data) > 0:
            if modality == 'aligned_timestamps':
                continue
            if modality == 'accelerometer':
                windowed_data[modality] = np.array(acc_windows)
                continue
            try:
                min_len = min(len(modality_data), len(data['accelerometer']))
                if min_len < window_size:
                    continue
                modality_windows = []
                for acc_window in acc_windows:
                    matched = False
                    for i in range(len(data['accelerometer']) - window_size + 1):
                        if np.array_equal(acc_window, data['accelerometer'][i:i+window_size]):
                            if i + window_size <= len(modality_data):
                                modality_windows.append(modality_data[i:i+window_size])
                                matched = True
                            break
                    if not matched:
                        modality_windows.append(np.zeros((window_size, modality_data.shape[1]), dtype=modality_data.dtype))
                windowed_data[modality] = np.array(modality_windows)
            except Exception as e:
                logger.error(f"Error creating windows for {modality}: {str(e)}")
    if 'aligned_timestamps' in data:
        try:
            timestamps_windows = []
            for acc_window in acc_windows:
                for i in range(len(data['accelerometer']) - window_size + 1):
                    if np.array_equal(acc_window, data['accelerometer'][i:i+window_size]):
                        if i + window_size <= len(data['aligned_timestamps']):
                            timestamps_windows.append(data['aligned_timestamps'][i:i+window_size])
                        break
            if timestamps_windows:
                windowed_data['aligned_timestamps'] = np.array(timestamps_windows)
        except Exception as e:
            logger.error(f"Error creating windows for timestamps: {str(e)}")
    if fuse and 'accelerometer' in windowed_data and 'gyroscope' in windowed_data:
        try:
            quaternions, linear_accelerations, fusion_features = [], [], []
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                futures = []
                for i in range(len(windowed_data['accelerometer'])):
                    timestamps = (windowed_data['aligned_timestamps'][i]
                                  if 'aligned_timestamps' in windowed_data and len(windowed_data['aligned_timestamps']) > i
                                  else None)
                    futures.append(executor.submit(
                        process_imu_data,
                        acc_data=windowed_data['accelerometer'][i],
                        gyro_data=windowed_data['gyroscope'][i],
                        timestamps=timestamps,
                        filter_type=filter_type,
                        return_features=True,
                        is_linear_acc=is_linear_acc
                    ))
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {filter_type} fusion"):
                    result = future.result()
                    quaternions.append(result['quaternion'])
                    linear_accelerations.append(result['linear_acceleration'])
                    if 'fusion_features' in result:
                        fusion_features.append(result['fusion_features'])
            windowed_data['quaternion'] = np.array(quaternions)
            windowed_data['linear_acceleration'] = np.array(linear_accelerations)
            if fusion_features:
                windowed_data['fusion_features'] = np.array(fusion_features)
        except Exception as e:
            logger.error(f"Error in fusion processing: {str(e)}")
    windowed_data['labels'] = np.repeat(label, len(acc_windows))
    return windowed_data

class DatasetBuilder:
    def __init__(self, dataset, mode, max_length, task='fd', fusion_options=None, **kwargs):
        if mode not in ['avg_pool', 'sliding_window']:
            raise ValueError(f"Unsupported processing method {mode}")
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = None
        self.fusion_options = fusion_options or {}
        self.aligned_data_dir = os.path.join(os.getcwd(), "data/aligned")
        for dir_name in ["accelerometer", "gyroscope", "skeleton", "quaternion"]:
            os.makedirs(os.path.join(self.aligned_data_dir, dir_name), exist_ok=True)
        logger.info(f"Initialized DatasetBuilder: mode={mode}, task={task}, fusion={self.fusion_options}")
    def load_file(self, file_path):
        try:
            file_type = file_path.split('.')[-1]
            if file_type not in ['csv', 'mat']:
                raise ValueError(f"Unsupported file type {file_type}")
            return LOADER_MAP[file_type](file_path, **self.kwargs)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    def process(self, data, label, fuse=False, filter_type='madgwick', visualize=False, is_linear_acc=True):
        if self.mode == 'avg_pool':
            processed_data = {}
            for modality, modality_data in data.items():
                if modality != 'labels' and len(modality_data) > 0:
                    processed_data[modality] = pad_sequence_numpy(
                        sequence=modality_data,
                        max_sequence_length=self.max_length,
                        input_shape=modality_data.shape
                    )
            processed_data['labels'] = np.array([label])
            if fuse and 'accelerometer' in processed_data and 'gyroscope' in processed_data:
                try:
                    timestamps = data.get('aligned_timestamps', None)
                    fusion_result = process_imu_data(
                        acc_data=processed_data['accelerometer'],
                        gyro_data=processed_data['gyroscope'],
                        timestamps=timestamps,
                        filter_type=filter_type,
                        return_features=True,
                        is_linear_acc=is_linear_acc
                    )
                    processed_data.update(fusion_result)
                except Exception as e:
                    logger.error(f"Fusion processing failed: {str(e)}")
            return processed_data
        else:
            return selective_sliding_window(
                data=data,
                window_size=self.max_length,
                label=label,
                fuse=fuse,
                filter_type=filter_type,
                is_linear_acc=is_linear_acc
            )
    def _process_trial(self, trial, label, fuse, filter_type, visualize, save_aligned=False, is_linear_acc=True):
        try:
            trial_data = {}
            for modality, file_path in trial.files.items():
                trial_data[modality] = self.load_file(file_path)
            if 'accelerometer' in trial_data and 'gyroscope' in trial_data:
                if len(trial_data['accelerometer']) < 3 or len(trial_data['gyroscope']) < 3:
                    logger.warning(f"Trial {trial.subject_id}-{trial.action_id}-{trial.sequence_number} has insufficient data")
                    return None
                acc_df = create_dataframe_with_timestamps(trial_data['accelerometer'])
                gyro_df = create_dataframe_with_timestamps(trial_data['gyroscope'])
                aligned_acc, aligned_gyro, aligned_times = align_sensor_data(acc_df, gyro_df, target_freq=30.0)
                if aligned_acc is not None:
                    trial_data['accelerometer'] = aligned_acc
                    trial_data['gyroscope'] = aligned_gyro
                    trial_data['aligned_timestamps'] = aligned_times
                    if save_aligned:
                        save_aligned_sensor_data(
                            trial.subject_id,
                            trial.action_id,
                            trial.sequence_number,
                            aligned_acc,
                            aligned_gyro,
                            None,
                            aligned_times
                        )
            return self.process(trial_data, label, fuse, filter_type, visualize, is_linear_acc)
        except Exception as e:
            logger.error(f"Trial processing failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    def _add_trial_data(self, trial_data):
        for modality, modality_data in trial_data.items():
            if len(modality_data) > 0:
                self.data[modality].append(modality_data)
    def _len_check(self, d):
        return all(len(v) > 1 for v in d.values())
    def make_dataset(self, subjects, fuse=False, filter_type='madgwick', visualize=False, save_aligned=False, is_linear_acc=True):
        logger.info(f"Making dataset for subjects={subjects}, fuse={fuse}, filter_type={filter_type}")
        start_time = time.time()
        self.data = defaultdict(list)
        self.fuse = fuse
        if hasattr(self, 'fusion_options'):
            save_aligned = save_aligned or self.fusion_options.get('save_aligned', False)
        with ThreadPoolExecutor(max_workers=min(8, len(self.dataset.matched_trials))) as executor:
            future_to_trial = {}
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
                    trial, label, fuse, filter_type, visualize, save_aligned, is_linear_acc
                )
                future_to_trial[future] = trial
            count, processed_count, skipped_count = 0, 0, 0
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
        elapsed_time = time.time() - start_time
        logger.info(f"Dataset creation complete: processed {processed_count}/{count} trials, skipped {skipped_count} in {elapsed_time:.2f}s")
    def normalization(self):
        for key, value in self.data.items():
            if key != 'labels' and len(value) > 0:
                try:
                    if key in ['accelerometer', 'gyroscope', 'quaternion', 'linear_acceleration'] and len(value.shape) >= 2:
                        num_samples, length = value.shape[:2]
                        orig_shape = value.shape
                        reshaped_data = value.reshape(num_samples * length, -1)
                        norm_data = StandardScaler().fit_transform(reshaped_data)
                        self.data[key] = norm_data.reshape(orig_shape)
                    elif key == 'fusion_features' and len(value.shape) == 2:
                        self.data[key] = StandardScaler().fit_transform(value)
                except Exception as e:
                    logger.error(f"Error normalizing {key} data: {str(e)}")
        return self.data
    def cleanup(self):
        cleanup_resources()

