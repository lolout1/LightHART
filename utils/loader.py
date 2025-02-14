# utils/loader.py

import os
import numpy as np
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from dtaidistance import dtw
from numpy.linalg import norm

from utils.processor.base import Processor

def filter_repeated_ids(path):
    sk_list = []
    in_list = []
    for (s, i) in path:
        if s not in sk_list and i not in in_list:
            sk_list.append(s)
            in_list.append(i)
    return sk_list, in_list

def dtw_align_skeleton(sk_array, in_array):
    if sk_array.size == 0 or in_array.size == 0:
        return sk_array
    joint_id = 9
    start_col = 1 + (joint_id-1)*3
    end_col   = 1 + joint_id*3
    sk_slice  = sk_array[:, start_col:end_col]
    in_slice  = in_array[:,1:]
    sk_norm   = norm(sk_slice, axis=1)
    in_norm   = norm(in_slice, axis=1)

    path = dtw.warping_path(sk_norm, in_norm)
    sk_idx, _ = filter_repeated_ids(path)
    if len(sk_idx) == 0:
        return np.zeros_like(sk_array[:0])
    return sk_array[sk_idx]

class DatasetBuilder:
    """
    A multi-threaded approach for building skeleton + accelerometer windows.
    For 'variable_time': we do time-based windowing, then keep min(#windows) across modalities.
    If any modality or file is empty => skip entire trial => ensures labels match.
    """
    def __init__(self, dataset, mode, max_length, task='fd',
                 window_size_sec=4.0, stride_sec=0.5, time2vec_dim=8, **kwargs):
        self.dataset = dataset
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.time2vec_dim = time2vec_dim
        self.kwargs = kwargs

        self.data = {}
        self.processed_data = {'labels': []}

    def _trial_label(self, trial):
        if self.task == 'fd':
            return int(trial.action_id > 9)
        elif self.task == 'age':
            return int(trial.subject_id < 29 or trial.subject_id > 46)
        else:
            return trial.action_id - 1

    def process_trial(self, trial, subjects):
        if trial.subject_id not in subjects:
            return None
        label = self._trial_label(trial)

        # 1) Load raw data for each modality
        trial_raw = {}
        for mod, fp in trial.files.items():
            is_skel = (mod=='skeleton')
            try:
                proc = Processor(file_path=fp,
                                 mode=self.mode,
                                 max_length=self.max_length,
                                 time2vec_dim=self.time2vec_dim,
                                 window_size_sec=self.window_size_sec,
                                 stride_sec=self.stride_sec)
                raw_data = proc.load_file(is_skeleton=is_skel)
                # If empty => skip entire trial
                if raw_data.size == 0:
                    print(f"[INFO] Skipping entire trial => file={fp} returned empty or error.")
                    return None

                trial_raw.setdefault(mod, []).append(raw_data)
            except Exception as e:
                print(f"[ERROR] load {fp} => {e}")
                traceback.print_exc()
                return None

        # 2) If skeleton + 1 inertial => do DTW => reindex skeleton
        if 'skeleton' in trial_raw and len(trial_raw.keys()) == 2:
            inertial_key = [m for m in trial_raw if m != 'skeleton'][0]
            sk_data = trial_raw['skeleton'][0]
            in_data = trial_raw[inertial_key][0]
            if sk_data.size > 0 and in_data.size > 0:
                new_skel = dtw_align_skeleton(sk_data, in_data)
                trial_raw['skeleton'][0] = new_skel

        # 3) Window each modality
        trial_processed = {}
        for mod, fp in trial.files.items():
            if mod not in trial_raw or len(trial_raw[mod]) == 0:
                # missing => skip entire trial
                return None

            arr = trial_raw[mod][0]
            if arr.size == 0:
                return None

            try:
                proc = Processor(fp, self.mode, self.max_length,
                                 time2vec_dim=self.time2vec_dim,
                                 window_size_sec=self.window_size_sec,
                                 stride_sec=self.stride_sec)
                proc.set_input_shape(arr)
                out_wins = proc.process(arr)
                # If no windows => skip entire trial
                if len(out_wins) == 0:
                    return None

                # unify => always a list
                if isinstance(out_wins, list):
                    windows = out_wins
                else:
                    # older modes => shape?
                    if out_wins.ndim == 3:
                        windows = [out_wins[i] for i in range(out_wins.shape[0])]
                    else:
                        windows = [out_wins]

                trial_processed[mod] = windows
            except Exception as e:
                print(f"[ERROR] Windowing {mod} => {e}")
                traceback.print_exc()
                return None

        if len(trial_processed) == 0:
            return None

        # 4) "Minimum windows approach"
        # find global min
        min_len = min(len(v) for v in trial_processed.values())
        if min_len == 0:
            print("[INFO] min_len=0 => skipping entire trial.")
            return None

        for m in trial_processed:
            trial_processed[m] = trial_processed[m][:min_len]

        # produce label list => same min_len
        label_list = [label] * min_len

        return (trial_processed, label_list)

    def make_dataset(self, subjects, max_workers=12):
        self.data.clear()
        self.processed_data.clear()
        self.processed_data['labels'] = []
        tasks = [(trial, subjects) for trial in self.dataset.matched_trials]

        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_trial, t, s) for (t,s) in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    # skip trial
                    continue
                trial_proc, lab_list = res
                # merge
                for m in trial_proc:
                    if m not in self.processed_data:
                        self.processed_data[m] = []
                    self.processed_data[m].extend(trial_proc[m])
                self.processed_data['labels'].extend(lab_list)

        # debug
        for k in self.processed_data:
            print(f"[DEBUG] make_dataset => key={k}, #items={len(self.processed_data[k])}")

    def normalization(self):
        if self.mode == 'variable_time':
            return self.processed_data
        from sklearn.preprocessing import StandardScaler
        for k, v in self.processed_data.items():
            if k == 'labels':
                continue
            big = np.concatenate(v, axis=0)
            shape_ = big.shape
            feat_dim = shape_[-1]
            flat = big.reshape(-1, feat_dim)
            scaler = StandardScaler().fit(flat)
            new_list = []
            for arr in v:
                s = arr.shape
                f = arr.reshape(-1, feat_dim)
                f_ = scaler.transform(f)
                new_list.append(f_.reshape(s))
            self.processed_data[k] = new_list
        return self.processed_data

