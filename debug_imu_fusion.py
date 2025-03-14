#!/usr/bin/env python3
"""
debug_imu_fusion.py

Tool for testing and visualizing different IMU fusion algorithms for fall detection.

This script:
1) Loads various sensor data (accelerometer, gyroscope, skeleton) from SmartFallMM dataset
2) Applies different IMU fusion filters (Standard, Extended, Unscented Kalman)
3) Visualizes and analyzes the results for orientation estimation 
4) Generates detailed diagnostic reports for comparing filter performance

Features:
- Robust handling of missing or corrupted sensor data
- Proper alignment of variably sampled data
- Comprehensive error handling with graceful degradation
- Visualization of orientation estimates across different filters
- Analytical comparison of filter accuracy using skeleton as ground truth

Usage:
    python debug_imu_fusion.py --data_dir data/smartfallmm --subjects 29,30 
                               --actions 10,11 --filters standard,ekf,ukf 
                               --max_trials 5 --output_dir debug_output
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
import traceback
from collections import defaultdict
import glob
import logging
import re
from scipy.interpolate import interp1d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IMUFusionDebug")

# Import custom modules (update these to use the robust versions)
from utils.processor.base import (
    parse_watch_csv,
    create_skeleton_timestamps
)
from utils.imu_fusion_robust import (
    RobustStandardKalmanIMU, 
    RobustExtendedKalmanIMU, 
    RobustUnscentedKalmanIMU,
    extract_orientation_from_skeleton
)
from utils.enhanced_alignment import robust_align_modalities

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Debug IMU Fusion Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/smartfallmm",
                        help="Base directory for dataset (containing 'young'/'old' subdirs)")
    parser.add_argument("--subjects", type=str, default="29,30,31",
                        help="Comma-separated list of subject IDs to test")
    parser.add_argument("--actions", type=str, default=None,
                        help="Comma-separated list of action IDs to test (None=all)")
    parser.add_argument("--filters", type=str, default="standard,ekf,ukf",
                        help="Comma-separated list of filters to test: standard,ekf,ukf")
    parser.add_argument("--max_trials", type=int, default=5,
                        help="Maximum number of trials to debug")
    parser.add_argument("--window_size", type=float, default=4.0,
                        help="Window size in seconds for sliding windows")
    parser.add_argument("--wrist_idx", type=int, default=9,
                        help="Index of wrist joint in skeleton data")
    parser.add_argument("--output_dir", type=str, default="debug_output",
                        help="Directory to store results and visualizations")
    parser.add_argument("--no_plot", dest="plot", action='store_false', 
                        help="Skip generating plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def find_sensor_files(base_dir, subject_list, action_list=None):
    """
    Find sensor data files for specified subjects and actions.
    
    Args:
        base_dir: Dataset base directory
        subject_list: List of subject IDs to include
        action_list: Optional list of action IDs to filter
        
    Returns:
        Dictionary of file paths organized by subject, action, and trial
    """
    logger.info(f"Searching for sensor files: subjects={subject_list}, actions={action_list}")
    
    # Create nested defaultdict for organizing files
    file_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Track counts for logging
    counts = {
        'accelerometer': 0,
        'gyroscope': 0,
        'skeleton': 0,
        'matched': 0
    }
    
    # Get age group directories (young/old)
    age_groups = ['young', 'old']
    valid_age_dirs = []
    
    for age in age_groups:
        age_dir = os.path.join(base_dir, age)
        if os.path.isdir(age_dir):
            valid_age_dirs.append((age, age_dir))
    
    if not valid_age_dirs:
        logger.error(f"No valid age group directories found in {base_dir}")
        return file_dict, counts
    
    # Process each age group
    for age_name, age_dir in valid_age_dirs:
        # Find accelerometer files
        accel_dir = os.path.join(age_dir, "accelerometer", "watch")
        if os.path.isdir(accel_dir):
            for subj in subject_list:
                # Pattern for subject ID in filename
                subj_pattern = f"S{subj:02d}A"
                
                for file in glob.glob(os.path.join(accel_dir, f"{subj_pattern}*.csv")):
                    # Extract action ID and trial number
                    match = re.search(r'S(\d+)A(\d+)T(\d+)', os.path.basename(file))
                    if match:
                        s_id, a_id, t_id = map(int, match.groups())
                        
                        # Filter by action if specified
                        if action_list is not None and a_id not in action_list:
                            continue
                        
                        # Store file path
                        file_dict[s_id][a_id][t_id]['accelerometer'] = file
                        counts['accelerometer'] += 1
        
        # Find gyroscope files
        gyro_dir = os.path.join(age_dir, "gyroscope", "watch")
        if os.path.isdir(gyro_dir):
            for subj in subject_list:
                subj_pattern = f"S{subj:02d}A"
                
                for file in glob.glob(os.path.join(gyro_dir, f"{subj_pattern}*.csv")):
                    match = re.search(r'S(\d+)A(\d+)T(\d+)', os.path.basename(file))
                    if match:
                        s_id, a_id, t_id = map(int, match.groups())
                        
                        if action_list is not None and a_id not in action_list:
                            continue
                        
                        file_dict[s_id][a_id][t_id]['gyroscope'] = file
                        counts['gyroscope'] += 1
        
        # Find skeleton files
        skel_dir = os.path.join(age_dir, "skeleton")
        if os.path.isdir(skel_dir):
            for subj in subject_list:
                subj_pattern = f"S{subj:02d}A"
                
                for file in glob.glob(os.path.join(skel_dir, f"{subj_pattern}*.csv")):
                    match = re.search(r'S(\d+)A(\d+)T(\d+)', os.path.basename(file))
                    if match:
                        s_id, a_id, t_id = map(int, match.groups())
                        
                        if action_list is not None and a_id not in action_list:
                            continue
                        
                        file_dict[s_id][a_id][t_id]['skeleton'] = file
                        counts['skeleton'] += 1
    
    # Count matched trials (must have accelerometer at minimum)
    for s_id in file_dict:
        for a_id in file_dict[s_id]:
            for t_id in file_dict[s_id][a_id]:
                if 'accelerometer' in file_dict[s_id][a_id][t_id]:
                    counts['matched'] += 1
    
    logger.info(f"Found {counts['accelerometer']} accelerometer, {counts['gyroscope']} gyroscope, "
                f"and {counts['skeleton']} skeleton files")
    logger.info(f"Total matched trials with accelerometer: {counts['matched']}")
    
    return file_dict, counts

def get_trial_list(file_dict, max_trials=5, seed=42):
    """
    Create a list of trials to process, with balanced selection of activities.
    
    Args:
        file_dict: Dictionary of files by subject/action/trial
        max_trials: Maximum number of trials to select
        seed: Random seed for reproducibility
        
    Returns:
        List of (subject_id, action_id, trial_id, files_dict) tuples
    """
    np.random.seed(seed)
    
    # Categorize trials by activity type
    adl_trials = []  # Activities of Daily Living
    fall_trials = []  # Fall activities
    
    for s_id in file_dict:
        for a_id in file_dict[s_id]:
            for t_id in file_dict[s_id][a_id]:
                # Must have accelerometer at minimum
                if 'accelerometer' not in file_dict[s_id][a_id][t_id]:
                    continue
                
                # Categorize (activity IDs > 9 are falls)
                trial = (s_id, a_id, t_id, file_dict[s_id][a_id][t_id])
                if a_id > 9:
                    fall_trials.append(trial)
                else:
                    adl_trials.append(trial)
    
    # Shuffle trials
    np.random.shuffle(adl_trials)
    np.random.shuffle(fall_trials)
    
    # Select balanced set of trials
    num_fall = min(max_trials // 2, len(fall_trials))
    num_adl = min(max_trials - num_fall, len(adl_trials))
    
    # If one category is empty, use all from the other
    if num_fall == 0 and len(adl_trials) > 0:
        num_adl = min(max_trials, len(adl_trials))
    if num_adl == 0 and len(fall_trials) > 0:
        num_fall = min(max_trials, len(fall_trials))
    
    # Combine and shuffle
    selected_trials = fall_trials[:num_fall] + adl_trials[:num_adl]
    np.random.shuffle(selected_trials)
    
    logger.info(f"Selected {len(selected_trials)} trials "
                f"({num_fall} falls, {num_adl} ADLs)")
    
    return selected_trials

def load_sensor_data(file_paths):
    """
    Load sensor data from files with robust error handling.
    
    Args:
        file_paths: Dictionary with file paths for different sensors
        
    Returns:
        Dictionary with loaded sensor data
    """
    result = {}
    
    # Load accelerometer data (required)
    accel_path = file_paths.get('accelerometer')
    if accel_path and os.path.exists(accel_path):
        logger.info(f"Loading accelerometer data from {accel_path}")
        accel_data = parse_watch_csv(accel_path)
        
        if accel_data.shape[0] > 0:
            result['accel_data'] = accel_data
            result['accel_timestamps'] = accel_data[:, 0]
            logger.info(f"Loaded accelerometer data: {accel_data.shape}")
        else:
            logger.warning(f"Empty accelerometer data: {accel_path}")
            return None  # Skip trial if no accelerometer data
    else:
        logger.warning("No accelerometer file provided")
        return None  # Skip trial if no accelerometer data
    
    # Load gyroscope data (optional)
    gyro_path = file_paths.get('gyroscope')
    if gyro_path and os.path.exists(gyro_path):
        logger.info(f"Loading gyroscope data from {gyro_path}")
        gyro_data = parse_watch_csv(gyro_path)
        
        if gyro_data.shape[0] > 0:
            result['gyro_data'] = gyro_data
            result['gyro_timestamps'] = gyro_data[:, 0]
            logger.info(f"Loaded gyroscope data: {gyro_data.shape}")
        else:
            logger.warning(f"Empty gyroscope data: {gyro_path}")
    else:
        logger.info("No gyroscope data available")
    
    # Load skeleton data (optional)
    skel_path = file_paths.get('skeleton')
    if skel_path and os.path.exists(skel_path):
        logger.info(f"Loading skeleton data from {skel_path}")
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
                
            logger.info(f"Loaded skeleton data: {skel_array.shape}")
        except Exception as e:
            logger.warning(f"Error loading skeleton data: {e}")
    else:
        logger.info("No skeleton data available")
    
    return result

def align_sensor_data(data_dict, wrist_idx=9, method='dtw'):
    """
    Align sensor data across modalities.
    
    Args:
        data_dict: Dictionary with loaded sensor data
        wrist_idx: Index of wrist joint
        method: Alignment method
        
    Returns:
        Dictionary with aligned data
    """
    # Must have accelerometer data at minimum
    if 'accel_data' not in data_dict:
        logger.warning("No accelerometer data for alignment")
        return data_dict
    
    # If no additional modalities to align, return as is
    has_gyro = 'gyro_data' in data_dict
    has_skel = 'skel_data' in data_dict
    
    if not has_gyro and not has_skel:
        logger.info("No additional modalities to align with accelerometer")
        return data_dict
    
    result = data_dict.copy()
    
    # Extract accelerometer data
    accel_values = data_dict['accel_data'][:, 1:4]  # Skip time column
    accel_timestamps = data_dict['accel_timestamps']
    
    # If we have skeleton data, align it with accelerometer
    if has_skel:
        logger.info(f"Aligning skeleton with accelerometer using {method} method")
        
        # Get skeleton data
        skel_data = data_dict['skel_data']
        
        try:
            # Align using robust method
            aligned_accel, aligned_skel, aligned_ts = robust_align_modalities(
                accel_values,
                skel_data,
                accel_timestamps,
                method=method,
                wrist_idx=wrist_idx
            )
            
            if aligned_accel.shape[0] > 10 and aligned_skel.shape[0] > 10:
                logger.info(f"Alignment succeeded: {aligned_accel.shape[0]} points")
                
                # Store aligned data
                result['aligned_accel'] = aligned_accel
                result['aligned_skel'] = aligned_skel
                result['aligned_timestamps'] = aligned_ts
                
                # Extract reference orientations from skeleton
                logger.info("Extracting reference orientations from skeleton")
                try:
                    reference_orientations = extract_orientation_from_skeleton(
                        aligned_skel, wrist_idx=wrist_idx
                    )
                    
                    # Check for length mismatch and fix if needed
                    if len(reference_orientations) != len(aligned_ts):
                        logger.warning(f"Orientation length mismatch: {len(reference_orientations)} vs {len(aligned_ts)}")
                        # Use the shorter length
                        min_len = min(len(reference_orientations), len(aligned_ts))
                        result['reference_orientations'] = reference_orientations[:min_len]
                        result['aligned_accel'] = aligned_accel[:min_len]
                        result['aligned_skel'] = aligned_skel[:min_len]
                        result['aligned_timestamps'] = aligned_ts[:min_len]
                    else:
                        result['reference_orientations'] = reference_orientations
                    
                except Exception as e:
                    logger.warning(f"Error extracting orientations: {e}")
                
                # If gyroscope is available, align it with common timebase
                if has_gyro:
                    gyro_values = data_dict['gyro_data'][:, 1:4]  # Skip time column
                    gyro_timestamps = data_dict['gyro_timestamps']
                    
                    try:
                        # Interpolate gyro to aligned timestamps
                        gyro_interp = interp1d(
                            gyro_timestamps,
                            gyro_values,
                            axis=0,
                            bounds_error=False,
                            fill_value="extrapolate"
                        )
                        
                        aligned_gyro = gyro_interp(aligned_ts)
                        result['aligned_gyro'] = aligned_gyro[:min_len] if 'min_len' in locals() else aligned_gyro
                        
                    except Exception as e:
                        logger.warning(f"Gyro interpolation failed: {e}, using zeros")
                        # Create zeros array of matching length
                        result['aligned_gyro'] = np.zeros_like(result['aligned_accel'])
            else:
                logger.warning("Alignment failed to produce sufficient data points")
                
        except Exception as e:
            logger.warning(f"Error during alignment: {e}")
    
    # If we have gyro but no skeleton (or alignment failed), interpolate gyro to accel timestamps
    elif has_gyro and 'aligned_timestamps' not in result:
        logger.info("Aligning gyroscope with accelerometer timing")
        
        gyro_values = data_dict['gyro_data'][:, 1:4]  # Skip time column
        gyro_timestamps = data_dict['gyro_timestamps']
        
        try:
            # Interpolate gyro to accel timestamps
            gyro_interp = interp1d(
                gyro_timestamps,
                gyro_values,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            # Find common time range
            t_min = max(accel_timestamps[0], gyro_timestamps[0])
            t_max = min(accel_timestamps[-1], gyro_timestamps[-1])
            
            # Filter to common range
            mask = (accel_timestamps >= t_min) & (accel_timestamps <= t_max)
            
            if np.sum(mask) > 10:
                filtered_ts = accel_timestamps[mask]
                filtered_accel = accel_values[mask]
                
                # Interpolate gyro to this timebase
                aligned_gyro = gyro_interp(filtered_ts)
                
                # Store aligned data
                result['aligned_accel'] = filtered_accel
                result['aligned_gyro'] = aligned_gyro
                result['aligned_timestamps'] = filtered_ts
            else:
                logger.warning("Insufficient overlap between accelerometer and gyroscope")
                
        except Exception as e:
            logger.warning(f"Gyroscope alignment failed: {e}")
    
    return result

def apply_imu_fusion(data_dict, filter_types=['standard', 'ekf', 'ukf']):
    """
    Apply different IMU fusion filters to the data.
    
    Args:
        data_dict: Dictionary with aligned sensor data
        filter_types: List of filter types to apply
    
    Returns:
        Dictionary with filter results
    """
    # Check if we have aligned data
    if 'aligned_accel' not in data_dict or 'aligned_timestamps' not in data_dict:
        logger.warning("No aligned data available for fusion")
        return {}
    
    # Extract aligned data
    accel_values = data_dict['aligned_accel']
    timestamps = data_dict['aligned_timestamps']
    
    # Check if gyro data is available
    if 'aligned_gyro' in data_dict:
        gyro_values = data_dict['aligned_gyro']
    else:
        logger.warning("No aligned gyro data, using zeros")
        gyro_values = np.zeros_like(accel_values)
    
    # Ensure all data has the same length
    min_len = min(len(accel_values), len(gyro_values), len(timestamps))
    accel_values = accel_values[:min_len]
    gyro_values = gyro_values[:min_len]
    timestamps = timestamps[:min_len]
    
    # Get reference orientations if available
    reference_orientations = data_dict.get('reference_orientations')
    if reference_orientations is not None:
        reference_orientations = reference_orientations[:min_len]
    
    # Process with each filter type
    results = {}
    
    for filter_type in filter_types:
        logger.info(f"Applying {filter_type} filter...")
        
        try:
            # Create appropriate filter
            if filter_type == 'standard':
                imu_filter = RobustStandardKalmanIMU()
            elif filter_type == 'ekf':
                imu_filter = RobustExtendedKalmanIMU()
            else:  # ukf
                imu_filter = RobustUnscentedKalmanIMU()
            
            # Set reference data if available
            if reference_orientations is not None:
                imu_filter.set_reference_data(timestamps, reference_orientations)
            
            # Process the data sequence
            start_time = time.time()
            output = imu_filter.process_sequence(accel_values, gyro_values, timestamps)
            process_time = time.time() - start_time
            
            # Store results
            results[filter_type] = {
                'output': output,
                'timestamps': timestamps,
                'processing_time': process_time
            }
            
            logger.info(f"{filter_type} filter processing time: {process_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error applying {filter_type} filter: {e}")
            traceback.print_exc()
    
    return results

def calculate_metrics(filter_results, reference_orientations=None):
    """
    Calculate performance metrics for filter results.
    
    Args:
        filter_results: Dictionary with filter outputs
        reference_orientations: Optional ground truth orientations
        
    Returns:
        Dictionary with metrics for each filter
    """
    metrics = {}
    
    # If no reference orientations, just calculate processing time
    if reference_orientations is None:
        for filter_type, result in filter_results.items():
            metrics[filter_type] = {
                'processing_time': result['processing_time'],
                'num_samples': len(result['timestamps'])
            }
        return metrics
    
    # With reference orientations, calculate accuracy metrics
    for filter_type, result in filter_results.items():
        output = result['output']
        proc_time = result['processing_time']
        
        # Extract Euler angles (last 3 columns of output)
        euler = output[:, 10:13]
        
        # Ensure same length for comparison
        min_len = min(len(euler), len(reference_orientations))
        euler = euler[:min_len]
        ref = reference_orientations[:min_len]
        
        # Calculate mean squared error (MSE)
        mse = np.mean(np.sum((euler - ref)**2, axis=1))
        
        # Calculate normalized error (RMSE / range)
        rmse = np.sqrt(mse)
        angle_range = np.max(ref, axis=0) - np.min(ref, axis=0)
        angle_range = np.where(angle_range > 0.01, angle_range, 0.01)  # Avoid division by zero
        normalized_error = rmse / np.mean(angle_range)
        
        # Store metrics
        metrics[filter_type] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'normalized_error': float(normalized_error),
            'processing_time': proc_time,
            'num_samples': min_len
        }
    
    return metrics

def plot_filter_results(filter_results, reference_orientations=None, output_path=None):
    """
    Generate plots for filter results.
    
    Args:
        filter_results: Dictionary with filter outputs
        reference_orientations: Optional ground truth orientations
        output_path: Path to save plot
        
    Returns:
        Path to saved plot file
    """
    if not filter_results:
        logger.warning("No filter results to plot")
        return None
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Extract timestamps from first filter result
    first_filter = list(filter_results.keys())[0]
    timestamps = filter_results[first_filter]['timestamps']
    
    # Plot Euler angles (roll, pitch, yaw)
    angle_names = ["Roll", "Pitch", "Yaw"]
    for i in range(3):
        ax = plt.subplot(3, 1, i+1)
        
        # Plot each filter result
        for filter_type, result in filter_results.items():
            output = result['output']
            # Extract Euler angle (columns 10-12)
            euler = output[:, 10+i]
            ax.plot(timestamps, euler, label=filter_type)
        
        # Plot reference if available
        if reference_orientations is not None:
            min_len = min(len(timestamps), len(reference_orientations))
            ax.plot(timestamps[:min_len], reference_orientations[:min_len, i], 
                   'k--', label='Skeleton Reference')
        
        ax.set_ylabel(f"{angle_names[i]} (rad)")
        ax.set_xlabel("Time (s)" if i == 2 else "")
        ax.grid(True)
        
        # Add legend on last subplot
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def plot_performance_comparison(all_metrics, output_path=None):
    """
    Plot performance comparison across all trials.
    
    Args:
        all_metrics: List of metrics dictionaries for each trial
        output_path: Path to save plot
        
    Returns:
        Path to saved plot file
    """
    if not all_metrics:
        logger.warning("No metrics to plot")
        return None
    
    # Collect metrics by filter type
    filter_metrics = defaultdict(lambda: defaultdict(list))
    
    for trial_metrics in all_metrics:
        for filter_type, metrics in trial_metrics.items():
            for metric_name, value in metrics.items():
                filter_metrics[filter_type][metric_name].append(value)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot processing time
    plt.subplot(2, 2, 1)
    for filter_type in filter_metrics:
        times = filter_metrics[filter_type]['processing_time']
        plt.bar(filter_type, np.mean(times), yerr=np.std(times), capsize=5)
    plt.ylabel("Processing Time (s)")
    plt.title("Average Processing Time")
    
    # Plot error metrics if available
    if 'mse' in filter_metrics[list(filter_metrics.keys())[0]]:
        # MSE
        plt.subplot(2, 2, 2)
        for filter_type in filter_metrics:
            mse = filter_metrics[filter_type]['mse']
            plt.bar(filter_type, np.mean(mse), yerr=np.std(mse), capsize=5)
        plt.ylabel("Mean Squared Error (rad²)")
        plt.title("Average MSE")
        
        # RMSE
        plt.subplot(2, 2, 3)
        for filter_type in filter_metrics:
            rmse = filter_metrics[filter_type]['rmse']
            plt.bar(filter_type, np.mean(rmse), yerr=np.std(rmse), capsize=5)
        plt.ylabel("RMSE (rad)")
        plt.title("Average RMSE")
        
        # Normalized Error
        plt.subplot(2, 2, 4)
        for filter_type in filter_metrics:
            nerr = filter_metrics[filter_type]['normalized_error']
            plt.bar(filter_type, np.mean(nerr), yerr=np.std(nerr), capsize=5)
        plt.ylabel("Normalized Error")
        plt.title("Average Normalized Error")
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def generate_report(trial_info, data_dict, filter_results, metrics, output_dir):
    """
    Generate detailed report for a trial.
    
    Args:
        trial_info: Trial information string
        data_dict: Data dictionary
        filter_results: Filter results
        metrics: Performance metrics
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    report_path = os.path.join(output_dir, f"{trial_info.replace(' ', '_')}_report.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"IMU Fusion Debug Report\n")
        f.write(f"======================\n\n")
        f.write(f"Trial: {trial_info}\n\n")
        
        # Data information
        f.write(f"1. Data Summary\n")
        f.write(f"--------------\n")
        
        # Accelerometer
        if 'accel_data' in data_dict:
            accel_shape = data_dict['accel_data'].shape
            f.write(f"Accelerometer: {accel_shape[0]} samples, {accel_shape[1]} columns\n")
            f.write(f"  Time range: {data_dict['accel_timestamps'][0]:.2f} - "
                   f"{data_dict['accel_timestamps'][-1]:.2f} seconds\n")
        else:
            f.write("Accelerometer: Not available\n")
        
        # Gyroscope
        if 'gyro_data' in data_dict:
            gyro_shape = data_dict['gyro_data'].shape
            f.write(f"Gyroscope: {gyro_shape[0]} samples, {gyro_shape[1]} columns\n")
            f.write(f"  Time range: {data_dict['gyro_timestamps'][0]:.2f} - "
                   f"{data_dict['gyro_timestamps'][-1]:.2f} seconds\n")
        else:
            f.write("Gyroscope: Not available\n")
        
        # Skeleton
        if 'skel_data' in data_dict:
            skel_shape = data_dict['skel_data'].shape
            f.write(f"Skeleton: {skel_shape[0]} frames, {skel_shape[1]} features\n")
            f.write(f"  Time range: {data_dict['skel_timestamps'][0]:.2f} - "
                   f"{data_dict['skel_timestamps'][-1]:.2f} seconds\n")
        else:
            f.write("Skeleton: Not available\n")
        
        # Alignment results
        f.write(f"\n2. Alignment Results\n")
        f.write(f"-------------------\n")
        
        if 'aligned_accel' in data_dict:
            aligned_shape = data_dict['aligned_accel'].shape
            f.write(f"Aligned data: {aligned_shape[0]} samples\n")
            f.write(f"  Time range: {data_dict['aligned_timestamps'][0]:.2f} - "
                   f"{data_dict['aligned_timestamps'][-1]:.2f} seconds\n")
            
            if 'aligned_gyro' in data_dict:
                f.write(f"  Gyroscope aligned: Yes\n")
            else:
                f.write(f"  Gyroscope aligned: No (using zeros)\n")
            
            if 'reference_orientations' in data_dict:
                ref_shape = data_dict['reference_orientations'].shape
                f.write(f"  Reference orientations: {ref_shape[0]} samples extracted from skeleton\n")
            else:
                f.write(f"  Reference orientations: None\n")
        else:
            f.write("No aligned data available\n")
        
        # Filter results
        f.write(f"\n3. Filter Results\n")
        f.write(f"----------------\n")
        
        if not filter_results:
            f.write("No filter results available\n")
        else:
            for filter_type, result in filter_results.items():
                f.write(f"{filter_type} filter:\n")
                f.write(f"  Processing time: {result['processing_time']:.3f} seconds\n")
                f.write(f"  Output shape: {result['output'].shape}\n")
        
        # Performance metrics
        f.write(f"\n4. Performance Metrics\n")
        f.write(f"---------------------\n")
        
        if not metrics:
            f.write("No performance metrics available\n")
        else:
            # Create a table header
            f.write(f"{'Filter Type':<15} {'MSE':<10} {'RMSE':<10} {'Norm Error':<15} {'Time (s)':<10}\n")
            f.write("-" * 60 + "\n")
            
            for filter_type, m in metrics.items():
                # Format metrics with appropriate precision
                mse = f"{m.get('mse', 'N/A'):.6f}" if 'mse' in m else "N/A"
                rmse = f"{m.get('rmse', 'N/A'):.6f}" if 'rmse' in m else "N/A"
                norm_err = f"{m.get('normalized_error', 'N/A'):.6f}" if 'normalized_error' in m else "N/A"
                proc_time = f"{m['processing_time']:.3f}"
                
                f.write(f"{filter_type:<15} {mse:<10} {rmse:<10} {norm_err:<15} {proc_time:<10}\n")
        
        # Recommendations
        f.write(f"\n5. Analysis and Recommendations\n")
        f.write(f"------------------------------\n")
        
        if metrics:
            # Find best filter based on error if available, otherwise processing time
            if 'mse' in list(metrics.values())[0]:
                best_filter = min(metrics.items(), key=lambda x: x[1]['mse'])[0]
                f.write(f"Based on orientation accuracy, the {best_filter} filter performs best for this trial.\n")
            else:
                best_filter = min(metrics.items(), key=lambda x: x[1]['processing_time'])[0]
                f.write(f"Based on processing time, the {best_filter} filter performs best for this trial.\n")
            
            # Observations about gyroscope data
            if 'gyro_data' in data_dict:
                f.write(f"Gyroscope data is available and used for fusion.\n")
            else:
                f.write(f"Gyroscope data is not available. Performance could be improved with gyroscope data.\n")
            
            # Observations about skeleton reference
            if 'reference_orientations' in data_dict:
                f.write(f"Skeleton reference orientations are available for drift correction.\n")
            else:
                f.write(f"No skeleton reference is available. Drift correction could improve long-term accuracy.\n")
        else:
            f.write(f"Insufficient data to provide recommendations.\n")
    
    return report_path

def process_trial(trial_info, file_paths, output_dir, filter_types=None, wrist_idx=9, plot=True):
    """
    Process a single trial with all filters.
    
    Args:
        trial_info: Trial information string
        file_paths: Dictionary with file paths
        output_dir: Output directory
        filter_types: List of filter types to apply
        wrist_idx: Index of wrist joint
        plot: Whether to generate plots
        
    Returns:
        Trial metrics dictionary
    """
    if filter_types is None:
        filter_types = ['standard', 'ekf', 'ukf']
    
    logger.info(f"\n=== Processing {trial_info} ===")
    
    # Create trial directory
    trial_dir = os.path.join(output_dir, trial_info.replace(' ', '_'))
    os.makedirs(trial_dir, exist_ok=True)
    
    # Load sensor data
    data_dict = load_sensor_data(file_paths)
    
    if data_dict is None:
        logger.warning(f"No valid data for trial {trial_info}")
        return None
    
    # Align sensor data
    data_dict = align_sensor_data(data_dict, wrist_idx=wrist_idx)
    
    # Apply IMU fusion
    filter_results = apply_imu_fusion(data_dict, filter_types)
    
    if not filter_results:
        logger.warning(f"No filter results for trial {trial_info}")
        return None
    
    # Calculate metrics
    reference_orientations = data_dict.get('reference_orientations')
    metrics = calculate_metrics(filter_results, reference_orientations)
    
    # Generate plots if requested
    if plot:
        try:
            # Plot filter results
            plot_path = os.path.join(trial_dir, "orientation_comparison.png")
            plot_filter_results(filter_results, reference_orientations, plot_path)
            
            # Plot accelerometer data
            if 'accel_data' in data_dict:
                accel_plot_path = os.path.join(trial_dir, "accelerometer_data.png")
                plt.figure(figsize=(10, 6))
                
                accel_data = data_dict['accel_data'][:, 1:4]  # Skip time column
                timestamps = data_dict['accel_timestamps']
                
                plt.plot(timestamps, accel_data[:, 0], label='X')
                plt.plot(timestamps, accel_data[:, 1], label='Y')
                plt.plot(timestamps, accel_data[:, 2], label='Z')
                plt.plot(timestamps, np.linalg.norm(accel_data, axis=1), 'k--', label='Magnitude')
                
                plt.xlabel('Time (s)')
                plt.ylabel('Acceleration (m/s²)')
                plt.title('Accelerometer Data')
                plt.legend()
                plt.grid(True)
                
                plt.savefig(accel_plot_path, dpi=150)
                plt.close()
            
            # Plot metrics comparison
            metrics_plot_path = os.path.join(trial_dir, "metrics_comparison.png")
            
            plt.figure(figsize=(12, 6))
            
            # Processing time
            plt.subplot(1, 2, 1)
            plt.bar(metrics.keys(), [m['processing_time'] for m in metrics.values()])
            plt.ylabel('Processing Time (s)')
            plt.title('Filter Processing Time')
            
            # Error metrics if available
            if 'mse' in list(metrics.values())[0]:
                plt.subplot(1, 2, 2)
                plt.bar(metrics.keys(), [m['mse'] for m in metrics.values()])
                plt.ylabel('MSE (rad²)')
                plt.title('Orientation Error')
            
            plt.tight_layout()
            plt.savefig(metrics_plot_path, dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error generating plots: {e}")
            traceback.print_exc()
    
    # Generate report
    report_path = generate_report(trial_info, data_dict, filter_results, metrics, trial_dir)
    logger.info(f"Report generated: {report_path}")
    
    return metrics

def main():
    """Main function for IMU fusion debugging."""
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process filter types
    filter_types = [f.strip().lower() for f in args.filters.split(',')]
    logger.info(f"Using filter types: {filter_types}")
    
    # Parse subject and action lists
    subject_list = [int(s.strip()) for s in args.subjects.split(',')]
    action_list = None
    if args.actions:
        action_list = [int(a.strip()) for a in args.actions.split(',')]
    
    # Find sensor files
    file_dict, counts = find_sensor_files(args.data_dir, subject_list, action_list)
    
    if counts['matched'] == 0:
        logger.error("No matched trials found")
        return
    
    # Get list of trials to process
    trial_list = get_trial_list(file_dict, args.max_trials, args.seed)
    
    if not trial_list:
        logger.error("No trials selected for processing")
        return
    
    # Process each trial
    all_metrics = []
    
    for s_id, a_id, t_id, files in trial_list:
        trial_info = f"S{s_id:02d}A{a_id:02d}T{t_id:02d}"
        
        # Skip if no accelerometer data
        if 'accelerometer' not in files:
            logger.warning(f"No accelerometer data for {trial_info}, skipping")
            continue
        
        # Process trial
        metrics = process_trial(
            trial_info=trial_info,
            file_paths=files,
            output_dir=args.output_dir,
            filter_types=filter_types,
            wrist_idx=args.wrist_idx,
            plot=args.plot
        )
        
        if metrics:
            all_metrics.append(metrics)
    
    # Generate summary plots
    if args.plot and all_metrics:
        summary_path = os.path.join(args.output_dir, "filter_performance_summary.png")
        plot_performance_comparison(all_metrics, summary_path)
    
    # Save summary report
    summary_path = os.path.join(args.output_dir, "summary_report.txt")
    with open(summary_path, 'w') as f:
        f.write(f"IMU Fusion Summary Report\n")
        f.write(f"=======================\n\n")
        f.write(f"Total trials processed: {len(all_metrics)}\n")
        f.write(f"Filter types: {', '.join(filter_types)}\n\n")
        
        # Compute average metrics across all trials
        avg_metrics = defaultdict(lambda: defaultdict(list))
        
        for trial_metrics in all_metrics:
            for filter_type, metrics in trial_metrics.items():
                for metric_name, value in metrics.items():
                    avg_metrics[filter_type][metric_name].append(value)
        
        # Write summary table
        f.write(f"Average Performance Metrics:\n")
        f.write(f"--------------------------\n")
        f.write(f"{'Filter Type':<15} {'MSE':<12} {'RMSE':<12} {'Norm Error':<15} {'Time (s)':<12}\n")
        f.write("-" * 65 + "\n")
        
        for filter_type, metrics in avg_metrics.items():
            # Calculate averages with appropriate handling of N/A values
            mse_vals = metrics.get('mse', [])
            rmse_vals = metrics.get('rmse', [])
            nerr_vals = metrics.get('normalized_error', [])
            time_vals = metrics.get('processing_time', [])
            
            mse_str = f"{np.mean(mse_vals):.6f}" if mse_vals else "N/A"
            rmse_str = f"{np.mean(rmse_vals):.6f}" if rmse_vals else "N/A"
            nerr_str = f"{np.mean(nerr_vals):.6f}" if nerr_vals else "N/A"
            time_str = f"{np.mean(time_vals):.4f}" if time_vals else "N/A"
            
            f.write(f"{filter_type:<15} {mse_str:<12} {rmse_str:<12} {nerr_str:<15} {time_str:<12}\n")
        
        # Overall recommendations
        f.write(f"\nOverall Recommendations:\n")
        f.write(f"-----------------------\n")
        
        # Find best filter based on error if available
        if 'mse' in avg_metrics[list(avg_metrics.keys())[0]]:
            best_acc_filter = min(avg_metrics.items(), 
                               key=lambda x: np.mean(x[1].get('mse', [np.inf])))[0]
            
            f.write(f"For best orientation accuracy: Use the {best_acc_filter} filter\n")
        
        # Find fastest filter
        best_speed_filter = min(avg_metrics.items(),
                             key=lambda x: np.mean(x[1].get('processing_time', [np.inf])))[0]
        
        f.write(f"For fastest processing: Use the {best_speed_filter} filter\n")
        
        # General recommendations
        f.write(f"\nGeneral observations:\n")
        if not all('mse' in metrics for metrics in all_metrics):
            f.write(f"• Skeleton reference data was not available for all trials.\n")
            f.write(f"  Consider using skeleton data during training for drift correction.\n")
        
        if not all('gyro_data' in trial_metrics for trial_metrics in all_metrics):
            f.write(f"• Gyroscope data was not available for all trials.\n")
            f.write(f"  Gyroscope data significantly improves orientation estimation.\n")
        
        f.write(f"\nIntegration recommendations:\n")
        f.write(f"• For real-time use on wearable devices, the {best_speed_filter} filter offers the best balance of accuracy and efficiency.\n")
        f.write(f"• For offline analysis where processing time is less critical, the {best_acc_filter if 'best_acc_filter' in locals() else best_speed_filter} filter provides the most accurate results.\n")
        f.write(f"• Quaternion representation should be used internally for stable orientation tracking.\n")
        f.write(f"• Euler angles can be derived from quaternions for visualization and feature extraction.\n")
    
    logger.info(f"Summary report saved to {summary_path}")
    logger.info("IMU fusion debugging complete!")

if __name__ == "__main__":
    main()
