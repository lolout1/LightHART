#!/usr/bin/env python3
"""
debug_imu_fusion.py

Tool for visualizing and debugging the IMU fusion implementations.
This script:
1) Loads watch (accelerometer, gyroscope) + skeleton CSV data from data directory
2) Applies different IMU fusion filters (Standard, EKF, UKF)
3) Visualizes results with comparative plots of orientation over time
4) Allows analysis of filter performance on different activities

Usage:
    python debug_imu_fusion.py --data_dir data/smartfallmm --subjects 29,30 --filters standard,ekf,ukf 
                               --actions 10,11 --plot --output_dir debug_output
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
import json
import time
from collections import defaultdict

# Import utilities for data loading and processing
from utils.processor.base_quat import (
    parse_watch_csv,
    create_skeleton_timestamps,
    robust_align_modalities
)

# Import Kalman filter implementations
from utils.imu_fusion import (
    StandardKalmanIMU,
    ExtendedKalmanIMU,
    UnscentedKalmanIMU,
    extract_orientation_from_skeleton
)

def load_data(accel_path, gyro_path=None, skel_path=None):
    """
    Load sensor data from CSV files.
    
    Args:
        accel_path: Path to accelerometer CSV
        gyro_path: Path to gyroscope CSV (optional)
        skel_path: Path to skeleton CSV (optional)
    
    Returns:
        Dictionary with loaded data
    """
    result = {}
    
    # Load accelerometer
    try:
        accel_data = parse_watch_csv(accel_path)
        if accel_data.shape[0] > 0:
            result['accel_data'] = accel_data
            result['accel_timestamps'] = accel_data[:, 0]
        else:
            print(f"Warning: Empty accelerometer data in {accel_path}")
            return None
    except Exception as e:
        print(f"Error loading accelerometer data: {e}")
        return None
    
    # Load gyroscope if available
    if gyro_path and os.path.exists(gyro_path):
        try:
            gyro_data = parse_watch_csv(gyro_path)
            if gyro_data.shape[0] > 0:
                result['gyro_data'] = gyro_data
            else:
                print(f"Warning: Empty gyroscope data in {gyro_path}")
        except Exception as e:
            print(f"Error loading gyroscope data: {e}")
    
    # Load skeleton if available
    if skel_path and os.path.exists(skel_path):
        try:
            df = pd.read_csv(skel_path, header=None)
            skel_array = df.values.astype(np.float32)
            
            # Add time column if not present
            if skel_array.shape[1] == 96:  # No time column
                skel_timestamps = np.arange(skel_array.shape[0]) / 30.0  # 30 fps
                result['skel_data'] = skel_array
                result['skel_timestamps'] = skel_timestamps
            else:
                result['skel_data'] = skel_array[:, 1:]  # Skip time column
                result['skel_timestamps'] = skel_array[:, 0]
        except Exception as e:
            print(f"Error loading skeleton data: {e}")
    
    return result

def align_modalities(data_dict, wrist_idx=9, method='dtw'):
    """
    Align IMU and skeleton data.
    
    Args:
        data_dict: Dictionary with data from load_data()
        wrist_idx: Index of wrist joint
        method: Alignment method
    
    Returns:
        Dictionary with aligned data
    """
    if 'accel_data' not in data_dict or 'skel_data' not in data_dict:
        return data_dict
    
    try:
        # Extract data
        accel_data = data_dict['accel_data'][:, 1:]  # Skip time column
        accel_timestamps = data_dict['accel_timestamps']
        skel_data = data_dict['skel_data']
        
        # Align
        aligned_imu, aligned_skel, aligned_ts = robust_align_modalities(
            accel_data, 
            skel_data, 
            accel_timestamps,
            method=method,
            wrist_idx=wrist_idx
        )
        
        if aligned_imu.shape[0] > 10 and aligned_skel.shape[0] > 10:
            data_dict['aligned_imu'] = aligned_imu
            data_dict['aligned_skel'] = aligned_skel
            data_dict['aligned_timestamps'] = aligned_ts
            
            # Extract reference orientations
            orientations = extract_orientation_from_skeleton(
                aligned_skel, wrist_idx=wrist_idx
            )
            data_dict['reference_orientations'] = orientations
            
            # If gyro data exists, align it too
            if 'gyro_data' in data_dict:
                gyro_data = data_dict['gyro_data']
                gyro_timestamps = gyro_data[:, 0]
                gyro_values = gyro_data[:, 1:]
                
                # Interpolate gyro to aligned timestamps
                from scipy.interpolate import interp1d
                try:
                    gyro_interp = interp1d(
                        gyro_timestamps,
                        gyro_values,
                        axis=0,
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    
                    aligned_gyro = gyro_interp(aligned_ts)
                    data_dict['aligned_gyro'] = aligned_gyro
                except Exception as e:
                    print(f"Error interpolating gyro: {e}")
                    # Create zero gyro values
                    data_dict['aligned_gyro'] = np.zeros_like(aligned_imu)
        else:
            print("Alignment produced insufficient data points.")
    except Exception as e:
        print(f"Error aligning modalities: {e}")
        import traceback
        traceback.print_exc()
    
    return data_dict

def apply_filters(data_dict, filter_types=['standard', 'ekf', 'ukf']):
    """
    Apply different Kalman filters to the data.
    
    Args:
        data_dict: Dictionary with data
        filter_types: List of filter types to apply
    
    Returns:
        Dictionary with filter results
    """
    results = {}
    
    # Check if we have the necessary data
    if 'accel_data' not in data_dict:
        print("Missing accelerometer data")
        return results
    
    # Use aligned data if available
    if 'aligned_imu' in data_dict and 'aligned_timestamps' in data_dict:
        accel_values = data_dict['aligned_imu']
        timestamps = data_dict['aligned_timestamps']
        
        if 'aligned_gyro' in data_dict:
            gyro_values = data_dict['aligned_gyro']
        else:
            gyro_values = np.zeros_like(accel_values)
        
        reference_orientations = data_dict.get('reference_orientations', None)
        reference_timestamps = data_dict.get('aligned_timestamps', None)
    else:
        # Use original data
        accel_values = data_dict['accel_data'][:, 1:]  # Skip time column
        timestamps = data_dict['accel_timestamps']
        
        if 'gyro_data' in data_dict:
            gyro_values = data_dict['gyro_data'][:, 1:]  # Skip time column
        else:
            gyro_values = np.zeros_like(accel_values)
        
        reference_orientations = None
        reference_timestamps = None
    
    # Apply each filter
    for filter_type in filter_types:
        print(f"Applying {filter_type} filter...")
        
        # Create filter
        if filter_type == 'standard':
            kf = StandardKalmanIMU()
        elif filter_type == 'ekf':
            kf = ExtendedKalmanIMU()
        elif filter_type == 'ukf':
            kf = UnscentedKalmanIMU()
        else:
            print(f"Unknown filter type: {filter_type}")
            continue
        
        # Set reference data if available
        if reference_orientations is not None and reference_timestamps is not None:
            kf.set_reference_data(reference_timestamps, reference_orientations)
        
        # Apply filter
        start_time = time.time()
        output = kf.process_sequence(accel_values, gyro_values, timestamps)
        end_time = time.time()
        
        # Store results
        results[filter_type] = {
            'output': output,  # [accel, gyro, quat, euler]
            'timestamps': timestamps,
            'processing_time': end_time - start_time
        }
    
    return results

def plot_results(data_dict, filter_results, output_dir, trial_info=""):
    """
    Generate plots for filter results.
    
    Args:
        data_dict: Original data dictionary
        filter_results: Results from apply_filters()
        output_dir: Directory to save plots
        trial_info: Information about the trial for title
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If we have reference orientations, include them for comparison
    has_reference = 'reference_orientations' in data_dict
    
    # 1. Plot Euler angles (roll, pitch, yaw) for each filter
    plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1)
    
    angles = ["Roll", "Pitch", "Yaw"]
    euler_idx = [-3, -2, -1]  # Last 3 columns in output are euler angles
    
    for i in range(3):
        ax = plt.subplot(gs[i, 0])
        
        # Plot each filter
        for filter_type, result in filter_results.items():
            timestamps = result['timestamps']
            euler = result['output'][:, euler_idx[i]]
            ax.plot(timestamps, euler, label=f"{filter_type}")
        
        # Plot reference if available
        if has_reference:
            ref_timestamps = data_dict['aligned_timestamps']
            ref_orientations = data_dict['reference_orientations']
            ax.plot(ref_timestamps, ref_orientations[:, i], 'k--', label='Skeleton Reference')
        
        ax.set_ylabel(f"{angles[i]} (rad)")
        ax.grid(True)
        
        if i == 0:
            plt.title(f"Orientation Comparison - {trial_info}")
        if i == 2:
            ax.set_xlabel("Time (s)")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"euler_comparison_{trial_info.replace(' ', '_')}.png"))
    
    # 2. Plot quaternion components
    plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 1)
    
    quat_components = ["w", "x", "y", "z"]
    # The quaternion components are at columns -7, -6, -5, -4 (before euler angles)
    quat_idx = [-7, -6, -5, -4]
    
    for i in range(4):
        ax = plt.subplot(gs[i, 0])
        
        # Plot each filter
        for filter_type, result in filter_results.items():
            timestamps = result['timestamps']
            # Make sure quaternion data is available in the output
            try:
                quat = result['output'][:, quat_idx[i]]
                ax.plot(timestamps, quat, label=f"{filter_type}")
            except IndexError:
                print(f"Warning: Quaternion component {quat_components[i]} not found in {filter_type} output")
        
        ax.set_ylabel(f"Quat {quat_components[i]}")
        ax.grid(True)
        
        if i == 0:
            plt.title(f"Quaternion Components - {trial_info}")
        if i == 3:
            ax.set_xlabel("Time (s)")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"quaternion_comparison_{trial_info.replace(' ', '_')}.png"))
    
    # 3. Plot accelerometer data for reference
    plt.figure(figsize=(10, 6))
    
    if 'aligned_imu' in data_dict:
        accel_data = data_dict['aligned_imu']
        timestamps = data_dict['aligned_timestamps']
    else:
        accel_data = data_dict['accel_data'][:, 1:]
        timestamps = data_dict['accel_timestamps']
    
    # Plot each axis
    plt.plot(timestamps, accel_data[:, 0], label='X')
    plt.plot(timestamps, accel_data[:, 1], label='Y')
    plt.plot(timestamps, accel_data[:, 2], label='Z')
    
    # Plot acceleration magnitude
    accel_mag = np.linalg.norm(accel_data, axis=1)
    plt.plot(timestamps, accel_mag, 'k--', label='Magnitude')
    
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(f"Accelerometer Data - {trial_info}")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"accelerometer_{trial_info.replace(' ', '_')}.png"))
    
    # 4. Plot filter performance comparison
    plt.figure(figsize=(10, 6))
    
    # Compute errors if we have reference data
    errors = {}
    if has_reference:
        ref_orientations = data_dict['reference_orientations']
        
        for filter_type, result in filter_results.items():
            euler = result['output'][:, -3:]  # Last 3 columns are euler angles
            
            # Calculate MSE
            mse = np.mean(np.sum((euler - ref_orientations)**2, axis=1))
            errors[filter_type] = mse
        
        # Plot MSE
        plt.bar(errors.keys(), errors.values())
        plt.ylabel("Mean Squared Error (rad²)")
        plt.title(f"Filter Orientation Error - {trial_info}")
        
        # Add timing information
        for i, (filter_type, result) in enumerate(filter_results.items()):
            plt.text(i, errors[filter_type] + 0.01, 
                     f"{result['processing_time']:.3f}s", 
                     ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"filter_error_{trial_info.replace(' ', '_')}.png"))
    
    # 5. Save results in json format
    results_summary = {
        'trial_info': trial_info,
        'processing_times': {k: v['processing_time'] for k, v in filter_results.items()},
        'data_points': len(timestamps) if 'timestamps' in locals() else 0,
        'errors': errors if has_reference else None
    }
    
    with open(os.path.join(output_dir, f"results_{trial_info.replace(' ', '_')}.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    return results_summary

def process_trial(accel_path, gyro_path=None, skel_path=None, filter_types=['standard', 'ekf', 'ukf'], 
                 wrist_idx=9, output_dir="debug_output", trial_info="", plot=True):
    """
    Process a single trial with different filters and generate visualizations.
    
    Args:
        accel_path: Path to accelerometer data
        gyro_path: Path to gyroscope data (optional)
        skel_path: Path to skeleton data (optional)
        filter_types: List of filter types to apply
        wrist_idx: Index of wrist joint
        output_dir: Directory to save results
        trial_info: Information about the trial
        plot: Whether to generate plots
    
    Returns:
        Dictionary with results summary
    """
    print(f"\n=== Processing {trial_info} ===")
    
    # 1. Load data
    data_dict = load_data(accel_path, gyro_path, skel_path)
    if data_dict is None:
        print(f"Failed to load data for {trial_info}")
        return None
    
    # 2. Align modalities if we have skeleton data
    if 'skel_data' in data_dict:
        data_dict = align_modalities(data_dict, wrist_idx, method='dtw')
    
    # 3. Apply filters
    filter_results = apply_filters(data_dict, filter_types)
    if not filter_results:
        print(f"Failed to apply filters for {trial_info}")
        return None
    
    # 4. Plot results
    if plot:
        results_summary = plot_results(data_dict, filter_results, output_dir, trial_info)
    else:
        # Just compute errors and timing
        results_summary = {
            'trial_info': trial_info,
            'processing_times': {k: v['processing_time'] for k, v in filter_results.items()},
            'data_points': len(filter_results[list(filter_results.keys())[0]]['timestamps'])
        }
        
        # Compute errors if we have reference data
        if 'reference_orientations' in data_dict:
            errors = {}
            ref_orientations = data_dict['reference_orientations']
            
            for filter_type, result in filter_results.items():
                euler = result['output'][:, -3:]  # Last 3 columns are euler angles
                mse = np.mean(np.sum((euler - ref_orientations)**2, axis=1))
                errors[filter_type] = mse
            
            results_summary['errors'] = errors
    
    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Debug IMU Fusion Pipeline")
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
    parser.add_argument("--wrist_idx", type=int, default=9,
                        help="Index of wrist joint for alignment (if needed).")
    parser.add_argument("--output_dir", type=str, default="debug_output",
                        help="Directory to store debug results.")
    parser.add_argument("--plot", action='store_true', help="Generate plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Running IMU fusion debug ===")
    
    # Parse arguments
    subject_list = [int(x.strip()) for x in args.subjects.split(',')]
    filter_list = [f.strip().lower() for f in args.filters.split(',')]
    
    if args.actions is not None:
        action_list = [int(x.strip()) for x in args.actions.split(',')]
    else:
        action_list = None
    
    # Scan for trial data
    base_accel = os.path.join(args.data_dir, "young", "accelerometer", "watch")
    base_gyro = os.path.join(args.data_dir, "young", "gyroscope", "watch")
    base_skel = os.path.join(args.data_dir, "young", "skeleton")
    
    all_trials = []
    
    # Find files with pattern "SxxAxxTxx.csv"
    for root, dirs, files in os.walk(base_accel):
        for f in files:
            if not f.endswith(".csv"):
                continue
            
            try:
                # Parse subject, action, trial from filename
                name = os.path.splitext(f)[0]
                if len(name) < 9 or not name.startswith('S'):
                    continue
                    
                subj = int(name[1:3])
                act = int(name[4:6])
                trial = int(name[7:9])
                
                # Filter by subject and action
                if subj not in subject_list:
                    continue
                    
                if action_list is not None and act not in action_list:
                    continue
                
                # Get file paths
                acc_fp = os.path.join(root, f)
                gyro_fp = os.path.join(base_gyro, f)
                skel_fp = os.path.join(base_skel, f)
                
                # Check if files exist
                if not os.path.exists(gyro_fp):
                    gyro_fp = None
                    
                if not os.path.exists(skel_fp):
                    skel_fp = None
                
                all_trials.append((subj, act, trial, acc_fp, gyro_fp, skel_fp))
            except:
                continue
    
    # Sort and limit to max_trials
    all_trials = sorted(all_trials, key=lambda x: (x[0], x[1], x[2]))
    if len(all_trials) > args.max_trials:
        all_trials = all_trials[:args.max_trials]
    
    print(f"Found {len(all_trials)} trials to process")
    
    all_results = []
    
    # Process each trial
    for subj, act, trial, accel_fp, gyro_fp, skel_fp in all_trials:
        trial_info = f"S{subj:02d}A{act:02d}T{trial:02d}"
        print(f"Processing {trial_info}...")
        
        results = process_trial(
            accel_path=accel_fp,
            gyro_path=gyro_fp,
            skel_path=skel_fp,
            filter_types=filter_list,
            wrist_idx=args.wrist_idx,
            output_dir=args.output_dir,
            trial_info=trial_info,
            plot=args.plot
        )
        
        if results is not None:
            all_results.append(results)
    
    # Generate summary
    if all_results:
        # Overall performance comparison
        if args.plot:
            plt.figure(figsize=(12, 8))
            
            # Average error by filter type
            filter_errors = {}
            filter_times = {}
            
            for filter_type in filter_list:
                errors = []
                times = []
                
                for result in all_results:
                    if 'errors' in result and result['errors'] is not None and filter_type in result['errors']:
                        errors.append(result['errors'][filter_type])
                    if filter_type in result['processing_times']:
                        times.append(result['processing_times'][filter_type])
                
                if errors:
                    filter_errors[filter_type] = np.mean(errors)
                if times:
                    filter_times[filter_type] = np.mean(times)
            
            # Plot errors
            if filter_errors:
                plt.subplot(2, 1, 1)
                plt.bar(filter_errors.keys(), filter_errors.values())
                plt.ylabel("Average MSE (rad²)")
                plt.title("Filter Performance Comparison")
                
                # Add numerical values
                for i, (k, v) in enumerate(filter_errors.items()):
                    plt.text(i, v + 0.01, f"{v:.6f}", ha='center')
            
            # Plot times
            if filter_times:
                plt.subplot(2, 1, 2)
                plt.bar(filter_times.keys(), filter_times.values())
                plt.ylabel("Average Processing Time (s)")
                
                # Add numerical values
                for i, (k, v) in enumerate(filter_times.items()):
                    plt.text(i, v + 0.01, f"{v:.3f}s", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "filter_overall_comparison.png"))
        
        # Save summary
        with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
            json.dump({
                'trials': len(all_results),
                'subjects': args.subjects,
                'actions': args.actions,
                'filters': args.filters,
                'results': all_results
            }, f, indent=2)
        
        print(f"Debug complete! Results saved to {args.output_dir}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
