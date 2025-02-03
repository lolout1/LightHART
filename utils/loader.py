# utils/loader.py

import os
from typing import List, Dict, Tuple
import numpy as np
from numpy.linalg import norm
from dtaidistance import dtw
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

from utils.processor.base import (
    Processor,
    parse_watch_csv,
    sliding_windows_by_time,
    Time2Vec,
)

###########################################################
#  Overlap function for multiple inertial modalities
###########################################################
def overlap_inertial(data1: np.ndarray, data2: np.ndarray):
    """
    Only keep the overlapping time range for two inertial arrays,
    each shaped (N, 4): [time, x, y, z], sorted by time.
    If there's no overlap, returns empty arrays.
    """
    if data1.shape[0] == 0 or data2.shape[0] == 0:
        return data1, data2
    
    start_t = max(data1[0, 0], data2[0, 0])
    end_t   = min(data1[-1, 0], data2[-1, 0])

    if end_t < start_t:
        # No overlap
        return np.zeros((0, 4), dtype=data1.dtype), np.zeros((0, 4), dtype=data2.dtype)

    mask1 = (data1[:, 0] >= start_t) & (data1[:, 0] <= end_t)
    mask2 = (data2[:, 0] >= start_t) & (data2[:, 0] <= end_t)
    return data1[mask1], data2[mask2]

def filter_data_by_ids(data: np.ndarray, ids: List[int]):
    return data[ids]

def filter_repeated_ids(path: List[Tuple[int, int]]):
    seen_first = set()
    seen_second = set()
    for (f, s) in path:
        if f not in seen_first and s not in seen_second:
            seen_first.add(f)
            seen_second.add(s)
    return seen_first, seen_second

def align_sequence(data: Dict[str, np.ndarray], idx):
    """
    Existing alignment:
      - If skeleton + 1 inertial modality => use DTW to align skeleton vs. that inertial.
      - If multiple inertial modalities => overlap them by time.
      - Skeleton has no timestamps (30 FPS). We do DTW to match length with inertial.
    """
    if len(data) == 0:
        return data

    # List all modalities except skeleton
    dynamic_keys = [k for k in data.keys() if k != "skeleton"]

    # 1) If we have skeleton + exactly one inertial => run DTW alignment
    if 'skeleton' in data and len(dynamic_keys) == 1:
        inertial_name = dynamic_keys[0]
        skeleton_joint_data = data['skeleton'][idx]
        inertial_data       = data[inertial_name][idx]
        if skeleton_joint_data.size == 0 or inertial_data.size == 0:
            return data
        joint_id = 9  # example joint
        skel_slice = skeleton_joint_data[:, (joint_id - 1) * 3: joint_id * 3]
        iner_slice = inertial_data[:, 1:]  # skip time column => (x,y,z)
        skel_norm = norm(skel_slice, axis=1)
        iner_norm = norm(iner_slice, axis=1)
        path = dtw.warping_path(skel_norm, iner_norm)
        s_idx, i_idx = filter_repeated_ids(path)
        s_idx = sorted(list(s_idx))
        i_idx = sorted(list(i_idx))
        data['skeleton'][idx] = skeleton_joint_data[s_idx]
        data[inertial_name][idx] = inertial_data[i_idx]

    # 2) If we have multiple inertial streams => keep only overlapping time
    elif len(dynamic_keys) > 1:
        # Overlap them pairwise (can be extended if you have 2+ inertial streams)
        base_key = dynamic_keys[0]
        for k in dynamic_keys[1:]:
            base_data = data[base_key][idx]
            other_data = data[k][idx]
            overlapped_base, overlapped_other = overlap_inertial(base_data, other_data)
            data[base_key][idx] = overlapped_base
            data[k][idx]        = overlapped_other

    # 3) If skeleton + multiple inertial => you can do time overlap first, then DTW with skeleton
    #     This would be more advanced. For minimal changes, we just do the above rules.
    return data


def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0)

class DatasetBuilder:
    """
    Extended to handle variable-time windows from watch/phone accelerometer/gyroscope
    if mode='variable_time'. Skeleton is 30 FPS and uses DTW-based alignment.
    """
    def __init__(self, dataset: object, mode: str, max_length: int, task='fd', **kwargs) -> None:
        self.dataset = dataset
        self.data: Dict[str, List[np.ndarray]] = {}
        self.processed_data: Dict[str, List[np.ndarray]] = {'labels': []}
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task

        # For variable_time
        self.window_size_sec = kwargs.get('window_size_sec', 4.0)  ### CHANGED default to 4.0
        self.stride_sec = kwargs.get('stride_sec', 1.0)
        self.time2vec_dim = kwargs.get('time2vec_dim', 8)

    def make_dataset(self, subjects: List[int]):
        self.data = {}
        self.processed_data = {'labels': []}
        count = 0

        for trial in self.dataset.matched_trials:
            # Filter for desired subjects
            if trial.subject_id not in subjects:
                continue

            # Label logic for fall detection (fd) or age
            if self.task == 'fd':
                label = int(trial.action_id > 9)
            elif self.task == 'age':
                label = int(trial.subject_id < 29 or trial.subject_id > 46)
            else:
                label = trial.action_id - 1

            # 1) Load raw data for each modality into self.data
            for modality, file_path in trial.files.items():
                proc = Processor(file_path, self.mode, self.max_length,
                                 time2vec_dim=self.time2vec_dim,
                                 window_size_sec=self.window_size_sec,
                                 stride_sec=self.stride_sec,
                                 **self.kwargs)
                try:
                    raw_data = proc.load_file()
                    self.data[modality] = self.data.get(modality, [])
                    self.data[modality].append(raw_data)
                except Exception as e:
                    print("Error loading file:", file_path, e)
                    continue

            # 2) Align the newly loaded data across modalities (skeleton ↔ inertial or inertial ↔ inertial)
            try:
                self.data = align_sequence(self.data, count)
            except Exception as e:
                print("[WARN] Alignment failed:", e)

            # 3) Convert each modality's raw data into windows or keep shape
            for modality, file_path in trial.files.items():
                if modality not in self.data or len(self.data[modality]) <= count:
                    continue

                proc = Processor(file_path, self.mode, self.max_length,
                                 time2vec_dim=self.time2vec_dim,
                                 window_size_sec=self.window_size_sec,
                                 stride_sec=self.stride_sec,
                                 **self.kwargs)
                aligned_data = self.data[modality][count]
                proc.set_input_shape(aligned_data)
                out = proc.process(aligned_data)

                if self.mode == 'variable_time':
                    if len(out) > 0:
                        self.processed_data[modality] = self.processed_data.get(modality, [])
                        self.processed_data[modality].extend(out)
                        self.processed_data['labels'].extend([label] * len(out))
                else:
                    # For 'avg_pool' or 'sliding_window'
                    if out.shape[0] != 0:
                        self.processed_data[modality] = self.processed_data.get(modality, [])
                        self.processed_data[modality].append(out)
                        if out.ndim > 1:
                            n_windows = out.shape[0]
                        else:
                            n_windows = 1
                        self.processed_data['labels'].extend([label] * n_windows)

            count += 1

        # 4) For non-variable_time, unify arrays (concatenate) after processing
        if self.mode != 'variable_time':
            for key in self.processed_data:
                if key == 'labels':
                    self.processed_data[key] = np.array(self.processed_data[key])
                else:
                    self.processed_data[key] = np.concatenate(self.processed_data[key], axis=0)

    def normalization(self):
        # Skip normalization for variable_time to avoid messing with variable window sizes.
        if self.mode == 'variable_time':
            return self.processed_data

        for k, v in self.processed_data.items():
            if k == 'labels':
                continue
            num_samples, length = v.shape[:2]
            norm_data = StandardScaler().fit_transform(v.reshape(num_samples * length, -1))
            self.processed_data[k] = norm_data.reshape(num_samples, length, -1)
        return self.processed_data
