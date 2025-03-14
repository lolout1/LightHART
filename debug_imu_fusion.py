#!/usr/bin/env python3
"""
debug_imu_fusion.py

Demonstration script that:
1) Loads watch (accelerometer, gyroscope) + skeleton CSV data from a specified directory.
2) Uses robust alignment (DTW fallback) from utils/processor/base.py.
3) Applies IMU fusion filters (StandardKalmanIMU, ExtendedKalmanIMU, UnscentedKalmanIMU).
4) Optionally plots results and logs debug info.

Requires:
  - utils/processor/base.py
  - utils/imu_fusion.py
  - utils/enhanced_dataset_builder.py (optional)
  - dtaidistance (if you want DTW)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import the base module for CSV parsing & alignment
from utils.processor.base import (
    parse_watch_csv,
    create_skeleton_timestamps,
    robust_align_modalities,
    sliding_windows_by_time_fixed,
    sliding_windows_by_time
)

# Import your IMU fusion filters
from utils.imu_fusion import (
    StandardKalmanIMU,
    ExtendedKalmanIMU,
    UnscentedKalmanIMU,
)

# Optional advanced dataset builder
# from utils.enhanced_dataset_builder import EnhancedDatasetBuilder
# from utils.loader_quat import DatasetBuilderQuat  # or whichever you prefer


def debug_trial(
    accel_fp: str,
    gyro_fp: str,
    skel_fp: str,
    filter_type: str = 'ekf',
    window_size_sec: float = 4.0,
    stride_sec: float = 1.0,
    wrist_idx: int = 9,
    do_plot: bool = False
):
    """
    Debug a single trial by:
      1) Loading watch accel, watch gyro, skeleton.
      2) Aligning them (DTW fallback).
      3) Applying selected IMU fusion (standard / ekf / ukf).
      4) Creating sliding windows (4s -> 128).
      5) (Optional) plotting or returning debug info.

    Returns a dict with debug results.
    """

    debug_info = {}

    # 1) Parse watch CSV
    print(f"[load_watch_data] Loading {accel_fp} ...")
    try:
        accel_data = parse_watch_csv(accel_fp)  # shape(N,4)
        print(f"[load_watch_data] {accel_fp}: shape={accel_data.shape}")
    except Exception as e:
        print(f"[load_watch_data] error reading {accel_fp}: {e}")
        return None

    # 2) Parse gyro CSV
    has_gyro = False
    gyro_data = np.zeros((0,4), dtype=np.float32)
    if gyro_fp and os.path.exists(gyro_fp):
        try:
            gyro_data = parse_watch_csv(gyro_fp)
            if gyro_data.shape[0] > 0:
                has_gyro = True
            print(f"[load_watch_data] {gyro_fp}: shape={gyro_data.shape}")
        except Exception as e:
            print(f"[load_watch_data] error reading {gyro_fp}: {e}")
            has_gyro = False

    # 3) Parse skeleton CSV
    has_skel = False
    skel_data = np.zeros((0, 97), dtype=np.float32)
    if skel_fp and os.path.exists(skel_fp):
        try:
            df = np.loadtxt(skel_fp, delimiter=",")  # or pd.read_csv
            if df.ndim == 1:
                # single line => shape(96,) => expand
                df = df.reshape((1, -1))
            if df.shape[1] == 96:
                # add timestamps
                df = create_skeleton_timestamps(df, fps=30.0)
            skel_data = df.astype(np.float32)
            if skel_data.shape[0] > 0:
                has_skel = True
            print(f"[load_skeleton_data] {skel_fp} => shape={skel_data.shape}")
        except Exception as e:
            print(f"[load_skeleton_data] error reading {skel_fp}: {e}")
            has_skel = False

    # If empty accel => skip
    if accel_data.shape[0] == 0:
        print(f"[debug_trial] empty accel => skip {accel_fp}")
        return None

    # 4) Align skeleton and IMU if we have skeleton
    aligned_imu = accel_data
    aligned_skel= np.zeros((0, skel_data.shape[1])) if skel_data.shape[0]==0 else skel_data
    aligned_ts  = accel_data[:,0]

    if has_skel:
        # We align the raw IMU values => shape(N,3) or (N,4 minus time col?), so let's skip the first col => time
        ali_imu, ali_skel, ali_ts = robust_align_modalities(
            imu_data= accel_data[:,1:],   # (N,3)
            skel_data= skel_data,         # (M,1+96) or (M,97)
            imu_timestamps= accel_data[:,0], 
            method='dtw',
            min_points=5
        )
        if ali_imu.shape[0] < 5 or ali_skel.shape[0] < 5:
            print(f"[debug_trial] alignment => insufficient => skip")
            # Possibly continue or return
            return None
        # Re-insert time col into aligned_imu
        aligned_imu = np.column_stack([ali_ts, ali_imu])  # shape(*,4)
        aligned_skel= ali_skel  # shape(*,96) or 97
        aligned_ts  = ali_ts

    # 5) IMU fusion
    # pick a filter
    if filter_type=='standard':
        filter_obj = StandardKalmanIMU()
    elif filter_type=='ekf':
        filter_obj = ExtendedKalmanIMU()
    else:
        filter_obj = UnscentedKalmanIMU()

    # Interpolate gyro => aligned_imu timestamps if has_gyro
    final_imu = aligned_imu
    if has_gyro and gyro_data.shape[0]>0:
        from scipy.interpolate import interp1d
        g_ts = gyro_data[:,0]
        g_xyz= gyro_data[:,1:]
        f = interp1d(g_ts, g_xyz, axis=0, fill_value='extrapolate', bounds_error=False)
        new_gyro = f(aligned_ts)
        # run filter
        fused = filter_obj.process_sequence(
            accel_data= aligned_imu[:,1:], # skip time col => shape(*,3)
            gyro_data= new_gyro, 
            timestamps= aligned_ts
        )
        # shape(*,13) => e.g. [acc, gyro, quat, euler]
        # merge with time col
        fused_imu = np.column_stack([aligned_ts, fused])
    else:
        # no gyro => do partial approach => only standard accel
        # might skip or do some dummy approach
        # e.g. we can pass zeros as gyro
        z_gyro = np.zeros_like(aligned_imu[:,1:])
        fused = filter_obj.process_sequence(
            accel_data= aligned_imu[:,1:],
            gyro_data= z_gyro,
            timestamps= aligned_ts
        )
        fused_imu = np.column_stack([aligned_ts, fused])

    debug_info['fused_imu'] = fused_imu
    debug_info['aligned_skel']= aligned_skel

    # 6) Window the fused data
    # Let's do 4s->128 approach
    fused_windows = sliding_windows_by_time_fixed(
        fused_imu,
        window_size_sec= window_size_sec,
        stride_sec= stride_sec,
        fixed_count=128
    )
    # Skeleton => variable-len
    skel_windows = []
    if has_skel and aligned_skel.shape[1]>=2:
        # If we know the first col is time or we just unify?
        # If we used robust_align => the skeleton is shape(*,96) or 97 => we might re-insert time
        # Let's suppose we do not have the time col => quickly do so:
        # or we skip
        # For demonstration, let's skip or assume user
        # (The more correct approach is to ensure aligned_skel also has the time in col0)
        pass

    # 7) (Optional) Plot
    if do_plot:
        # Let's do a simple plot of the fused orientation
        times = fused_imu[:,0]
        euler = fused_imu[:,-3:]  # the last 3 columns are euler angles
        plt.figure(figsize=(10,5))
        plt.plot(times, euler[:,0], label='Roll')
        plt.plot(times, euler[:,1], label='Pitch')
        plt.plot(times, euler[:,2], label='Yaw')
        plt.xlabel('Time (sec)')
        plt.ylabel('Angle (rad)')
        plt.title(f"IMU Fusion ({filter_type}) Debug")
        plt.legend()
        plt.show()

    return debug_info


def main():
    parser = argparse.ArgumentParser("Debug IMU Fusion Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/smartfallmm",
                        help="Base directory for dataset (containing 'young'/'old', etc.)")
    parser.add_argument("--subjects", type=str, default="29,30,31",
                        help="Comma-separated list of subject IDs to test.")
    parser.add_argument("--actions", type=str, default=None,
                        help="Comma-separated list of action IDs to test or None => all.")
    parser.add_argument("--filters", type=str, default="ekf",
                        help="Comma-separated list of filters to test: standard,ekf,ukf")
    parser.add_argument("--max_trials", type=int, default=5,
                        help="Maximum number of trials to debug.")
    parser.add_argument("--window_size", type=float, default=4.0,
                        help="Window size in seconds for sliding windows.")
    parser.add_argument("--stride", type=float, default=1.0,
                        help="Stride in seconds for sliding windows.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max sequence length for fixed windows.")
    parser.add_argument("--wrist_idx", type=int, default=9,
                        help="Index of wrist joint for alignment (if needed).")
    parser.add_argument("--output_dir", type=str, default="debug_output",
                        help="Directory to store debug results.")
    parser.add_argument("--plot", action='store_true', help="If set, do a plot of orientation.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Running IMU fusion debug ===")

    # 1) Gather some "trials" logic. Suppose we have them enumerated or do a quick approach:
    subject_list = [int(x.strip()) for x in args.subjects.split(',')]
    if args.actions is not None:
        action_list = [int(x.strip()) for x in args.actions.split(',')]
    else:
        action_list = None

    # For demonstration, let's pretend we have CSV named e.g. SXXAYYTZZ.csv in 
    #   data/smartfallmm/young/accelerometer/watch/
    #   data/smartfallmm/young/gyroscope/watch/
    #   data/smartfallmm/young/skeleton/
    # We'll just create a small list of plausible filenames
    # In real code, you'd call e.g. a "get_trial_list" function from your dataset.

    all_trials = []
    # This is pseudo code for scanning:
    base_accel = os.path.join(args.data_dir, "young", "accelerometer", "watch")
    base_gyro  = os.path.join(args.data_dir, "young", "gyroscope", "watch")
    base_skel  = os.path.join(args.data_dir, "young", "skeleton")

    # naive scan for files:
    # each file named e.g. S29A06T03.csv => subject=29, action=06, trial=03
    # We'll do a quick approach:
    for root, dirs, files in os.walk(base_accel):
        for f in files:
            if not f.endswith(".csv"):
                continue
            # parse subject,action,trial
            # e.g. "S29A06T03.csv"
            # Need e.g. S, subject(2?), A, action(2?), T, trial(2?)
            # check length
            # your real logic may differ
            try:
                name = os.path.splitext(f)[0]  # S29A06T03
                subj = int(name[1:3])
                act  = int(name[4:6])
                trial= int(name[7:9])
                if subj not in subject_list:
                    continue
                if (action_list is not None) and (act not in action_list):
                    continue
                # we store file paths
                acc_fp = os.path.join(root, f)
                gyro_fp= os.path.join(base_gyro, f)  # same filename
                skel_fp= os.path.join(base_skel, f)  # same filename
                all_trials.append((subj,act,trial,acc_fp,gyro_fp,skel_fp))
            except:
                pass

    # sort or limit to max_trials
    all_trials = sorted(all_trials, key=lambda x: (x[0], x[1], x[2]))
    if len(all_trials)> args.max_trials:
        all_trials = all_trials[:args.max_trials]

    # 2) For each filter in args.filters => do debug on these trials
    filter_list = [f.strip().lower() for f in args.filters.split(',')]
    for ftype in filter_list:
        print(f"Testing IMU fusion with filter={ftype} ...")
        for (subj,act,tno,acc_fp, gyro_fp, sk_fp) in all_trials:
            print(f"=== Debugging trial S{subj}A{act}T{tno} ===")
            result = debug_trial(
                accel_fp= acc_fp,
                gyro_fp= gyro_fp,
                skel_fp= sk_fp,
                filter_type= ftype,
                window_size_sec= args.window_size,
                stride_sec= args.stride,
                wrist_idx= args.wrist_idx,
                do_plot= args.plot
            )
            if result is None:
                print(f"Skipping S{subj}A{act}T{tno} => no data or alignment fail.")
            else:
                # Possibly store or log result
                # e.g. write to npz or just log
                pass

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Debug completed.\n")
    print(f"Debug complete! Results saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()

