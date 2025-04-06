#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from fastdtw import fastdtw
import scipy.spatial.distance as dist
from sklearn.metrics import mean_squared_error
import pandas as pd

from utils.imu_fusion import (
    MadgwickFilter, KalmanFilter, ExtendedKalmanFilter,
    align_sensor_data, apply_lowpass_filter
)
from utils.processor.base import csvloader

###############################################################################
# 1) CSV loading & splitting
###############################################################################
def load_csv_and_split(file_path):
    data = csvloader(file_path)
    if data is None or data.shape[0] < 3:
        return None, None
    if data.shape[1] >= 4:
        # col0 timestamps, col1..3 data
        ts = data[:,0]
        arr = data[:,1:4]
    else:
        # no timestamps => synthetic
        ts = np.arange(len(data))
        arr = data[:,0:3]
    return arr, ts


###############################################################################
# 2) Skeleton loading for param optimization
#    - Optionally compute skeleton accelerations from positions
###############################################################################
def load_skeleton_data(file_path, wrist_joint_idx=9, fs=30.0, assume_positions=True):
    d = csvloader(file_path)
    if d is None or d.shape[0]<5:
        return None, None

    # check if we have timestamps
    t = None
    if d.shape[1]>=4:  # probably first col is time
        t = d[:,0]
        # the rest might be skeleton coords
        # we assume 1 + (some number*3) columns => we want the wrist
        start_col = 1 + wrist_joint_idx*3
        if d.shape[1]< start_col+3:
            return None, None
        raw = d[:, start_col:start_col+3]
    else:
        # no timestamp => synthetic
        t = np.arange(len(d))
        start_col = wrist_joint_idx*3
        if d.shape[1]< start_col+3:
            return None, None
        raw = d[:, start_col:start_col+3]

    # resample to fs
    t = t - t[0]
    if t[-1]>1e9:
        t = t / 1e9
    from scipy.interpolate import interp1d
    start, end = t[0], t[-1]
    if end<=start:
        return None, None
    newt = np.arange(start, end, 1/fs)
    if len(newt)<3:
        return None, None

    newpos = np.zeros((len(newt),3))
    for i in range(3):
        f = interp1d(t, raw[:,i], bounds_error=False,
                     fill_value=(raw[0,i], raw[-1,i]))
        newpos[:,i] = f(newt)

    if assume_positions:
        # compute acceleration
        vel = np.gradient(newpos, axis=0)*fs
        acc = np.gradient(vel, axis=0)*fs
        acc = acc - np.mean(acc, axis=0)  # remove offset
        return acc, newt
    else:
        return newpos, newt

###############################################################################
# 3) DTW routines for 3D or magnitude
###############################################################################
def dtw_3d(signal1, signal2):
    """DTW on Nx3 signals => returns distance, alignment path, plus metrics."""
    if signal1 is None or signal2 is None or len(signal1)<3 or len(signal2)<3:
        return None,None,None,None
    # Normalize each axis
    s1_mean, s2_mean = np.mean(signal1, axis=0), np.mean(signal2, axis=0)
    s1_std, s2_std = np.std(signal1, axis=0), np.std(signal2, axis=0)
    s1_std[s1_std<1e-8], s2_std[s2_std<1e-8] = 1,1
    s1_norm = (signal1 - s1_mean)/s1_std
    s2_norm = (signal2 - s2_mean)/s2_std
    distance, path = fastdtw(s1_norm, s2_norm, dist=dist.euclidean)
    path = np.array(path)
    idx1, idx2 = path[:,0], path[:,1]
    align1, align2 = signal1[idx1], signal2[idx2]
    # compute RMSE, corr, etc.
    min_len = min(len(signal1), len(signal2))
    rmse_full = np.sqrt(mean_squared_error(signal1[:min_len].flatten(),
                                          signal2[:min_len].flatten()))
    corr_vals = []
    for i in range(3):
        c = np.corrcoef(signal1[:min_len,i], signal2[:min_len,i])[0,1]
        corr_vals.append(c)
    corr = np.mean(corr_vals)
    aligned_rmse = np.sqrt(mean_squared_error(align1.flatten(),
                                              align2.flatten()))
    return distance, (rmse_full, corr, aligned_rmse), align1, align2

def dtw_magnitude(signal1, signal2):
    """DTW on magnitude only => orientation-agnostic."""
    if signal1 is None or signal2 is None or len(signal1)<3 or len(signal2)<3:
        return None,None,None,None
    mag1 = np.linalg.norm(signal1, axis=1)
    mag2 = np.linalg.norm(signal2, axis=1)
    distance, path = fastdtw(mag1, mag2, dist=dist.euclidean)
    path = np.array(path)
    idx1, idx2 = path[:,0], path[:,1]
    align1, align2 = mag1[idx1], mag2[idx2]
    min_len = min(len(mag1), len(mag2))
    rmse_full = np.sqrt(mean_squared_error(mag1[:min_len], mag2[:min_len]))
    c = np.corrcoef(mag1[:min_len], mag2[:min_len])[0,1]
    aligned_rmse = np.sqrt(mean_squared_error(align1, align2))
    return distance, (rmse_full, c, aligned_rmse), align1, align2

###############################################################################
# 4) Evaluate filter with param, but now we can optionally use skeleton to pick best param
###############################################################################
def evaluate_filter_with_skl(acc_data, gyro_data, skl_data=None,
                             filter_type='madgwick',
                             param_values=None, is_linear_acc=True, fs=30.0,
                             dtw_mode='3d'):
    """
    Return a dict with param->some dtw metrics. If 'skl_data' is provided,
    we measure how well the filtered watch data aligns with skeleton (3d or magnitude).
    """
    if acc_data is None or gyro_data is None or len(acc_data)<3 or len(gyro_data)<3:
        return None
    from scipy.spatial.transform import Rotation

    if param_values is None:
        if filter_type=='madgwick':
            param_values = [0.05, 0.1, 0.15, 0.2, 0.3]
        elif filter_type=='kalman':
            param_values = [1e-6,5e-6,1e-5,5e-5,1e-4]
        elif filter_type=='ekf':
            param_values = [0.01, 0.05, 0.1, 0.2, 0.3]
        else:
            return None

    # We'll store results => param -> { dtw_distance, rmse, correlation,... with skeleton }
    results = {}

    # Prepare filter objects
    for pv in param_values:
        if filter_type=='madgwick':
            fobj = MadgwickFilter(beta=pv)
        elif filter_type=='kalman':
            fobj = KalmanFilter(process_noise=pv)
        elif filter_type=='ekf':
            fobj = ExtendedKalmanFilter(measurement_noise=pv)
        else:
            return None

        # step: do the orientation
        before = acc_data.copy()
        quats = np.zeros((len(before),4))
        for i in range(len(before)):
            a = before[i]
            g = gyro_data[i]
            if is_linear_acc:
                a2 = a.copy()
                a2[2]+=9.81
                quats[i] = fobj.update(a2,g)
            else:
                quats[i] = fobj.update(a,g)
        after = np.zeros_like(before)
        for i in range(len(before)):
            q = quats[i]
            r = Rotation.from_quat([q[1],q[2],q[3],q[0]])
            after[i] = r.apply(before[i])
        if not is_linear_acc:
            after = after - np.array([0,0,9.81])

        # If skeleton is not provided => just store normal dtw metrics of (before vs after)
        # If skeleton is provided => measure alignment of 'after' vs skeleton
        dt = {}
        if skl_data is None:
            # do watch before vs after 3D dtw as a fallback
            dist, (rmse,corr,aligned_rmse), _,_ = dtw_3d(before, after)
            dt = {
                'distance': dist, 'rmse': rmse, 'corr': corr, 'aligned_rmse': aligned_rmse
            }
        else:
            # measure alignment of after vs skeleton
            if dtw_mode=='3d':
                dist,(rmse,corr,aligned_rmse),_,_ = dtw_3d(after, skl_data)
            else:
                dist,(rmse,corr,aligned_rmse),_,_ = dtw_magnitude(after, skl_data)
            dt = {
                'distance': dist, 'rmse': rmse, 'corr': corr, 'aligned_rmse': aligned_rmse
            }

        results[pv] = dt

    return results

###############################################################################
# 5) Main script that merges everything:
#    1) Load watch data + skeleton (if any)
#    2) Possibly do alignment BEFORE filtering
#    3) For each filter type, param search using skeleton data if available
#    4) Compare best param on after-filter alignment
#    5) Generate final plots
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Full robust test with 3D or magnitude alignment, using skeleton if available")
    parser.add_argument('--acc-file', required=True, help='Watch accelerometer CSV path')
    parser.add_argument('--gyro-file', required=True, help='Watch gyroscope CSV path')
    parser.add_argument('--skl-file', default=None, help='Optional skeleton CSV for param optimization or alignment')
    parser.add_argument('--fs', type=float, default=30.0, help='Sampling freq')
    parser.add_argument('--is-linear-acc', action='store_true', help='Is watch data already gravity-removed')
    parser.add_argument('--filter-type', default='all', help='Which filter(s) to run: madgwick|kalman|ekf|all')
    parser.add_argument('--dtw-mode', choices=['3d','magnitude'], default='3d',
                        help='Use 3D or magnitude dtw for param optimization with skeleton')
    parser.add_argument('--align-before-filter', action='store_true',
                        help='If set, do a watch-skeleton alignment before orientation filtering.')
    parser.add_argument('--out-dir', default='full_evaluation_results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ########################################################
    # (A) Load watch data, skeleton data, align them to fs
    ########################################################
    acc_raw, acc_ts = load_csv_and_split(args.acc_file)
    gyr_raw, gyr_ts = load_csv_and_split(args.gyro_file)
    if acc_raw is None or gyr_raw is None:
        print("Error: watch CSV invalid or empty.")
        return 1

    # align sensor data to target freq
    watch_acc, watch_gyr, watch_t = align_sensor_data(acc_raw, gyr_raw, acc_ts, gyr_ts, args.fs)
    if watch_acc is None or watch_gyr is None:
        print("Error: cannot align watch accelerometer & gyro.")
        return 1

    skl_data = None
    skl_t = None
    if args.skl_file and os.path.exists(args.skl_file):
        skl_data, skl_time = load_skeleton_data(args.skl_file, fs=args.fs, wrist_joint_idx=9, assume_positions=True)
        if skl_data is None:
            print("Warning: skeleton data invalid - ignoring skeleton usage.")
            skl_data = None

    # If user wants alignment BEFORE filter:
    if args.align_before_filter and skl_data is not None:
        print("Aligning watch <-> skeleton BEFORE filtering, using dtw_mode:", args.dtw_mode)
        if args.dtw_mode=='magnitude':
            dist,(rmse,corr,aligned_rmse),w_align,skl_align = dtw_magnitude(watch_acc, skl_data)
        else:
            dist,(rmse,corr,aligned_rmse),w_align,skl_align = dtw_3d(watch_acc, skl_data)
        print(f"Before-filter DTW => dist={dist:.2f}, rmse={rmse:.2f}, corr={corr:.2f}, aligned_rmse={aligned_rmse:.2f}")
        # Optionally overwrite watch_acc/skl_data with the aligned data
        # but that can produce mismatch in shape/time. For simplicity, let's skip.
        # watch_acc = w_align
        # skl_data = skl_align

    ########################################################
    # (B) Evaluate filters & param search (with skeleton if available)
    ########################################################
    # We will store best param for each filter type
    best_params = {}

    filter_types = []
    if args.filter_type=='all':
        filter_types = ['madgwick','kalman','ekf']
    else:
        filter_types = [args.filter_type]

    for ft in filter_types:
        # do param search
        print(f"[{ft.upper()}] Searching params using skeleton data if available. dtw_mode={args.dtw_mode}")
        param_results = evaluate_filter_with_skl(watch_acc, watch_gyr,
                                                 skl_data=skl_data,
                                                 filter_type=ft,
                                                 param_values=None,
                                                 is_linear_acc=args.is_linear_acc,
                                                 fs=args.fs,
                                                 dtw_mode=args.dtw_mode)
        if param_results is None:
            print(f"Warning: {ft} filter param search returned None. Skipping.")
            continue
        # param_results => { param: {distance, rmse, corr, aligned_rmse} }
        # we can pick the best param e.g. min 'aligned_rmse'
        best_p = None
        best_val = 1e9
        for p in param_results:
            met = param_results[p]
            if met['aligned_rmse']<best_val:
                best_val = met['aligned_rmse']
                best_p = p
        best_params[ft] = best_p
        # save out a CSV with param search
        param_rows = []
        for p in param_results:
            dist = param_results[p]['distance']
            rmse = param_results[p]['rmse']
            corr = param_results[p]['corr']
            armse = param_results[p]['aligned_rmse']
            param_rows.append([p, dist, rmse, corr, armse])
        pdf = pd.DataFrame(param_rows, columns=["Param","Distance","RMSE","Corr","AlignedRMSE"])
        pdf.to_csv(os.path.join(args.out_dir, f"{ft}_param_search.csv"), index=False)
        print(f"[{ft.upper()}] best param => {best_p}, aligned_rmse={best_val:.4f}")

    ########################################################
    # (C) Compare final filters with chosen best param
    #     do "after filtering" alignment with skeleton, produce final plots
    ########################################################
    summary_rows = []
    from scipy.spatial.transform import Rotation

    for ft in filter_types:
        if ft not in best_params:
            print(f"No best param for {ft}, skipping final compare.")
            continue
        param = best_params[ft]

        # create filter with that param
        if ft=='madgwick':
            fobj = MadgwickFilter(beta=param)
        elif ft=='kalman':
            fobj = KalmanFilter(process_noise=param)
        elif ft=='ekf':
            fobj = ExtendedKalmanFilter(measurement_noise=param)

        # apply filter
        acc_before = watch_acc.copy()
        quats = np.zeros((len(acc_before),4))
        for i in range(len(acc_before)):
            a = acc_before[i]
            g = watch_gyr[i]
            if args.is_linear_acc:
                a2 = a.copy()
                a2[2]+=9.81
                quats[i] = fobj.update(a2,g)
            else:
                quats[i] = fobj.update(a,g)
        acc_after = np.zeros_like(acc_before)
        for i in range(len(acc_before)):
            q = quats[i]
            r = Rotation.from_quat([q[1],q[2],q[3],q[0]])
            acc_after[i] = r.apply(acc_before[i])
        if not args.is_linear_acc:
            acc_after = acc_after - np.array([0,0,9.81])

        # if skeleton is available => do dtw again after filtering => final
        if skl_data is not None:
            if args.dtw_mode=='magnitude':
                dist,(rmse,corr,armse),_,_ = dtw_magnitude(acc_after, skl_data)
            else:
                dist,(rmse,corr,armse),_,_ = dtw_3d(acc_after, skl_data)
            summary_rows.append([ft, param, dist, rmse, corr, armse])
        else:
            # if no skeleton => do watch before vs after
            dist,(rmse,corr,armse),_,_ = dtw_3d(acc_before, acc_after)
            summary_rows.append([ft, param, dist, rmse, corr, armse])

        # optional: plot the final 3D lines, etc.
        # You can replicate your "plot 3D alignment" or "plot signals" code here if desired
        # to produce final comparisons. We'll keep it short here.

    # Save final summary
    if summary_rows:
        cdf = pd.DataFrame(summary_rows, columns=["Filter","Param","DTWdist","RMSE","Corr","AlignedRMSE"])
        cdf.to_csv(os.path.join(args.out_dir, "final_filter_comparison.csv"), index=False)
        print("Final filter comparison saved => final_filter_comparison.csv")

    print("All done. See results in:", args.out_dir)
    return 0

if __name__=='__main__':
    sys.exit(main())

