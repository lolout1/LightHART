# File: utils/processor/single_input_preprocessing.py
"""
Minimal example of reading CSV => [time, x, y, z],
converting to time-elapsed format, sub-sampling to 128 frames,
and returning (128,4) => [x, y, z, t_elapsed].
Also includes a function to build a dataset from multiple CSVs.
"""

import os
import numpy as np
import pandas as pd

def read_accel_csv_single_window(
    file_path: str,
    required_count=128
):
    """
    Reads a single CSV => columns [time, x, y, z].
    Returns a (required_count, 4) array => [x, y, z, t_elapsed].
    If not enough frames, returns None.
    """
    try:
        df = pd.read_csv(file_path, header=None, engine='python').dropna()
        if df.shape[1] < 4:
            print(f"[WARN] {file_path}: Not enough columns => skipping.")
            return None

        time_vals = df.iloc[:,0].astype(float).values
        x_vals    = df.iloc[:,1].astype(float).values
        y_vals    = df.iloc[:,2].astype(float).values
        z_vals    = df.iloc[:,3].astype(float).values

        N = len(time_vals)
        if N < required_count:
            print(f"[WARN] {file_path}: Only {N} frames, need {required_count} => skipping.")
            return None

        # Convert time => elapsed from first sample
        t0 = time_vals[0]
        t_elapsed = time_vals - t0  # keep as ms or s depending on data
        # shape => (N,4): [x, y, z, t_elapsed]
        data_4 = np.column_stack([x_vals, y_vals, z_vals, t_elapsed])

        # sub-sample to exactly required_count
        idx = np.linspace(0, N-1, required_count).astype(int)
        data_4_sub = data_4[idx]  # => shape (required_count,4)
        return data_4_sub

    except Exception as e:
        print(f"[ERROR] read_accel_csv_single_window => {e}")
        return None


def build_accel_dataset(csv_dir, labels_dict, required_count=128):
    """
    Example function to iterate over a directory of CSVs,
    read each using read_accel_csv_single_window,
    and store them alongside labels.

    Args:
      csv_dir: directory containing CSV files
      labels_dict: a {filename => label} mapping or a function that obtains label from filename
      required_count: #frames per window

    Returns:
      samples_4: list of shape (128,4) arrays
      labels: list of integer labels
    """
    samples_4 = []
    labels = []

    for f in os.listdir(csv_dir):
        if not f.endswith('.csv'):
            continue
        file_path = os.path.join(csv_dir, f)
        arr_4 = read_accel_csv_single_window(file_path, required_count=required_count)
        if arr_4 is not None:
            # obtain label from dictionary or filename
            label = labels_dict.get(f, 0)
            samples_4.append(arr_4)
            labels.append(label)

    return samples_4, labels

