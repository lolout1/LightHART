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
- Robust handling of variable sampling rates
- Timestamp-based alignment of accelerometer and gyroscope
- Resampling to fixed 30Hz to match skeleton data
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
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import our robust alignment and filtering modules
# Import from the robust implementations directly
from utils.robust_alignment import (
    process_all_modalities as align_modalities,
    align_imu_sensors,
    resample_to_fixed_rate,
    align_imu_with_skeleton,
    extract_orientation_from_skeleton
)
from utils.processor.base import (
        parse_watch_csv
        )
from utils.imu_fusion_robust import (
    RobustStandardKalmanIMU,
    RobustExtendedKalmanIMU,
    RobustUnscentedKalmanIMU,
    calibrate_filter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("imu_debug.log")
    ]
)
logger = logging.getLogger("IMUFusionDebug")

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
    
    # Data source arguments
    parser.add_argument("--data_dir", type=str, default="data/smartfallmm",
                        help="Base directory for dataset (containing 'young'/'old' subdirs)")
    parser.add_argument("--subjects", type=str, default="29,30,31",
                        help="Comma-separated list of subject IDs to test")
    parser.add_argument("--actions", type=str, default=None,
                        help="Comma-separated list of action IDs to test (None=all)")
    
    # Filter selection
    parser.add_argument("--filters", type=str, default="standard,ekf,ukf",
                        help="Comma-separated list of filters to test: standard,ekf,ukf")
    
    # Processing parameters
    parser.add_argument("--max_trials", type=int, default=5,
                        help="Maximum number of trials to debug")
    parser.add_argument("--target_fps", type=float, default=30.0,
                        help="Target sampling rate for resampling (Hz)")
    parser.add_argument("--window_size", type=float, default=4.0,
                        help="Window size in seconds for sliding windows")
    parser.add_argument("--wrist_idx", type=int, default=9,
                        help="Index of wrist joint in skeleton data")
    parser.add_argument("--align_method", type=str, default="dtw",
                        help="Method for aligning IMU and skeleton (dtw, interpolation, crop)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="debug_output",
                        help="Directory to store results and visualizations")
    parser.add_argument("--no_plot", dest="plot", action='store_false', 
                        help="Skip generating plots")
    parser.add_argument("--interactive", action='store_true',
                        help="Show interactive plots (not just saving them)")
    
    # Advanced options
    parser.add_argument("--calibrate_filters", action='store_true',
                        help="Calibrate filters before testing")
    parser.add_argument("--robust_cov", action='store_true',
                        help="Use robust covariance estimation")
    parser.add_argument("--apply_antialiasing", action='store_true', default=True,
                        help="Apply anti-aliasing when resampling")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers for processing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Debug verbosity
    parser.add_argument("--verbose", action='store_true', 
                        help="Enable verbose logging")
    
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
    
    # Count matched trials (with both accelerometer and gyroscope)
    for s_id in file_dict:
        for a_id in file_dict[s_id]:
            for t_id in file_dict[s_id][a_id]:
                files = file_dict[s_id][a_id][t_id]
                if 'accelerometer' in files and 'gyroscope' in files:
                    counts['matched'] += 1
    
    logger.info(f"Found {counts['accelerometer']} accelerometer, {counts['gyroscope']} gyroscope, "
                f"and {counts['skeleton']} skeleton files")
    logger.info(f"Total matched trials (accel+gyro): {counts['matched']}")
    
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
                # Must have both accelerometer and gyroscope
                files = file_dict[s_id][a_id][t_id]
                if 'accelerometer' not in files or 'gyroscope' not in files:
                    continue
                
                # Categorize (activity IDs > 9 are falls)
                trial = (s_id, a_id, t_id, files)
                if a_id > 9:
                    fall_trials.append(trial)
                else:
                    adl_trials.append(trial)
    
    # Shuffle trials
    np.random.shuffle(fall_trials)
    np.random.shuffle(adl_trials)
    
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
            
            # Check for time column
            if skel_array.shape[1] == 96:  # No time column
                result['skel_data'] = skel_array
            else:
                result['skel_data'] = skel_array[:, 1:]  # Skip time column
                
            logger.info(f"Loaded skeleton data: {skel_array.shape}")
        except Exception as e:
            logger.warning(f"Error loading skeleton data: {e}")
    else:
        logger.info("No skeleton data available")
    
    return result

def align_and_resample_data(accel_data, gyro_data, skel_data=None, target_fps=30.0, 
                           apply_antialiasing=True):
    """
    Align and resample sensor data to a consistent time grid.
    
    Args:
        accel_data: Accelerometer data with time in first column
        gyro_data: Gyroscope data with time in first column
        skel_data: Optional skeleton data
        target_fps: Target sampling rate in Hz
        apply_antialiasing: Whether to apply anti-aliasing filter
        
    Returns:
        Dictionary with aligned and resampled data
    """
    logger.info(f"Aligning and resampling sensor data to {target_fps} Hz")
    
    try:
        # First, align IMU sensors (accel and gyro)
        imu_aligned = align_imu_sensors(accel_data, gyro_data)
        
        if not imu_aligned.get('success', False):
            logger.warning("Failed to align accelerometer and gyroscope data")
            return None
            
        # Extract aligned IMU data
        timestamps = imu_aligned['timestamps']
        accel_values = imu_aligned['accel']
        gyro_values = imu_aligned['gyro']
        
        # Resample to target frequency
        # Create a common time grid for resampling both sensors
        t_start = timestamps[0]
        t_end = timestamps[-1]
        duration = t_end - t_start
        num_samples = int(duration * target_fps) + 1
        common_time_grid = np.linspace(t_start, t_end, num_samples)
        
        logger.info(f"Resampling accelerometer data to {target_fps} Hz")
        accel_resampled = resample_to_fixed_rate(
            timestamps, 
            accel_values, 
            target_fps=target_fps,
            apply_antialiasing=apply_antialiasing
        )
        
        logger.info(f"Resampling gyroscope data to {target_fps} Hz")
        gyro_resampled = resample_to_fixed_rate(
            timestamps, 
            gyro_values, 
            target_fps=target_fps,
            apply_antialiasing=apply_antialiasing
        )
        
        if not accel_resampled.get('success', False) or not gyro_resampled.get('success', False):
            logger.warning("Failed to resample IMU data")
            return None
            
        # Create result dictionary with common time grid
        result = {
            'timestamps': common_time_grid,
            'accel': accel_resampled['values'],
            'gyro': gyro_resampled['values'],
        }
        
        # If skeleton data is provided, align it with resampled IMU data
        if skel_data is not None and skel_data.shape[0] > 0:
            # Create skeleton timestamps at 30 Hz if needed
            if skel_data.shape[1] == 96:  # No time column
                skel_timestamps = np.arange(skel_data.shape[0]) / target_fps
                skeleton_values = skel_data
            else:
                skel_timestamps = skel_data[:, 0]
                skeleton_values = skel_data[:, 1:]
                
            # Find common time range
            common_start = max(t_start, skel_timestamps[0])
            common_end = min(t_end, skel_timestamps[-1])
            
            # Check if we have enough overlap
            if common_end > common_start:
                # Filter common time grid
                imu_common_mask = (common_time_grid >= common_start) & (common_time_grid <= common_end)
                common_times = common_time_grid[imu_common_mask]
                
                if len(common_times) > 10:  # Ensure we have enough points
                    # Apply same mask to resampled data
                    common_accel = result['accel'][imu_common_mask]
                    common_gyro = result['gyro'][imu_common_mask]
                    
                    # Create interpolator for skeleton data
                    from scipy.interpolate import interp1d
                    skel_interp = interp1d(
                        skel_timestamps,
                        skeleton_values,
                        axis=0,
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    
                    # Resample skeleton to common timestamps
                    resampled_skel = skel_interp(common_times)
                    
                    # Update result with aligned data
                    result['timestamps'] = common_times
                    result['accel'] = common_accel
                    result['gyro'] = common_gyro
                    result['skeleton'] = resampled_skel
                    
                    logger.info(f"Successfully aligned all modalities: {len(common_times)} samples")
                else:
                    logger.warning("Insufficient overlap between IMU and skeleton")
            else:
                logger.warning("No time overlap between IMU and skeleton")
                
        return result
        
    except Exception as e:
        logger.error(f"Error during alignment and resampling: {e}")
        logger.error(traceback.format_exc())
        return None

def apply_imu_fusion(aligned_data, filter_types=None, wrist_idx=9):
    """
    Apply different IMU fusion filters to the aligned data.
    
    Args:
        aligned_data: Dictionary with aligned sensor data
        filter_types: List of filter types to apply
        wrist_idx: Index of wrist joint
    
    Returns:
        Dictionary with filter results
    """
    if filter_types is None:
        filter_types = ['standard', 'ekf', 'ukf']
    
    # Check if we have aligned data
    if aligned_data is None or 'accel' not in aligned_data:
        logger.warning("No aligned data available for fusion")
        return {}
    
    # Extract aligned data
    accel_values = aligned_data['accel']
    gyro_values = aligned_data['gyro']
    timestamps = aligned_data['timestamps']
    
    # Get reference orientations if available
    reference_orientations = None
    if 'skeleton' in aligned_data:
        try:
            reference_orientations = extract_orientation_from_skeleton(
                aligned_data['skeleton'],
                wrist_idx=wrist_idx
            )
            logger.info(f"Extracted reference orientations from skeleton: {reference_orientations.shape}")
        except Exception as e:
            logger.warning(f"Error extracting reference orientations: {e}")
    
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
            logger.error(traceback.format_exc())
    
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
    if reference_orientations is None or len(reference_orientations) == 0:
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
        
        # Extract Euler angles (columns 10-12 of output)
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
        
        # Calculate component-wise metrics
        mse_roll = np.mean((euler[:, 0] - ref[:, 0])**2)
        mse_pitch = np.mean((euler[:, 1] - ref[:, 1])**2)
        mse_yaw = np.mean((euler[:, 2] - ref[:, 2])**2)
        
        # Store metrics
        metrics[filter_type] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mse_roll': float(mse_roll),
            'mse_pitch': float(mse_pitch),
            'mse_yaw': float(mse_yaw),
            'normalized_error': float(normalized_error),
            'processing_time': proc_time,
            'num_samples': min_len
        }
    
    return metrics

def plot_filter_results(filter_results, reference_orientations=None, output_path=None, 
                        plot_title=None, show_plot=False):
    """
    Generate plots for filter results.
    
    Args:
        filter_results: Dictionary with filter outputs
        reference_orientations: Optional ground truth orientations
        output_path: Path to save plot
        plot_title: Optional title for the plot
        show_plot: Whether to display the plot interactively
        
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
    
    # Create color cycle and line styles for better differentiation
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':']
    
    # Calculate time in seconds from start
    if len(timestamps) > 0:
        time_sec = timestamps - timestamps[0]
    else:
        time_sec = []
    
    # Plot Euler angles (roll, pitch, yaw)
    angle_names = ["Roll", "Pitch", "Yaw"]
    for i in range(3):
        ax = plt.subplot(3, 1, i+1)
        
        # Plot each filter result with distinct formatting
        for j, (filter_type, result) in enumerate(filter_results.items()):
            output = result['output']
            
            # Extract Euler angle (columns 10-12)
            if output.shape[1] >= 13:  # Ensure the output has enough columns
                euler = output[:, 10+i]
                ax.plot(time_sec, euler, 
                       color=colors[j % len(colors)],
                       linestyle=line_styles[j % len(line_styles)],
                       linewidth=2,
                       label=filter_type)
            else:
                logger.warning(f"Output for {filter_type} doesn't have Euler angles")
        
        # Plot reference if available
        if reference_orientations is not None and len(reference_orientations) > 0:
            min_len = min(len(time_sec), len(reference_orientations))
            if min_len > 0:
                ax.plot(time_sec[:min_len], reference_orientations[:min_len, i], 
                       'k--', linewidth=1.5, label='Skeleton Reference')
        
        ax.set_ylabel(f"{angle_names[i]} (rad)")
        ax.set_xlabel("Time (s)" if i == 2 else "")
        ax.grid(True, alpha=0.3)
        
        # Add legend on first subplot
        if i == 0:
            ax.legend(loc='upper right')
        
        # Add title
        if i == 0 and plot_title:
            ax.set_title(plot_title, fontsize=14)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return output_path

def plot_performance_comparison(all_metrics, output_path=None, show_plot=False):
    """
    Plot performance comparison across all trials.
    
    Args:
        all_metrics: List of metrics dictionaries for each trial
        output_path: Path to save plot
        show_plot: Whether to display the plot interactively
        
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
                if isinstance(value, (int, float)):  # Only collect numeric metrics
                    filter_metrics[filter_type][metric_name].append(value)
    
    # Create figure with enhanced layout
    plt.figure(figsize=(15, 12))
    
    # Use a nice color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(filter_metrics)))
    
    # Plot processing time
    plt.subplot(2, 2, 1)
    filter_names = []
    times = []
    errors = []
    
    for i, (filter_type, metrics) in enumerate(filter_metrics.items()):
        if 'processing_time' in metrics and len(metrics['processing_time']) > 0:
            filter_names.append(filter_type)
            times.append(np.mean(metrics['processing_time']))
            errors.append(np.std(metrics['processing_time']))
    
    bars = plt.bar(filter_names, times, yerr=errors, capsize=5, color=colors[:len(filter_names)])
    plt.ylabel("Processing Time (s)")
    plt.title("Average Processing Time per Filter", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time:.3f}s', ha='center', va='bottom', fontsize=9)
    
    # Plot error metrics if available
    if any('mse' in metrics for metrics in filter_metrics.values()):
        # MSE
        plt.subplot(2, 2, 2)
        mse_data = []
        mse_errors = []
        
        for filter_type in filter_names:
            if 'mse' in filter_metrics[filter_type]:
                mse_values = filter_metrics[filter_type]['mse']
                mse_data.append(np.mean(mse_values))
                mse_errors.append(np.std(mse_values))
            else:
                mse_data.append(0)
                mse_errors.append(0)
        
        bars = plt.bar(filter_names, mse_data, yerr=mse_errors, capsize=5, color=colors[:len(filter_names)])
        plt.ylabel("Mean Squared Error (rad²)")
        plt.title("Average MSE (lower is better)", fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mse in zip(bars, mse_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mse:.4f}', ha='center', va='bottom', fontsize=9)
        
        # RMSE
        plt.subplot(2, 2, 3)
        rmse_data = []
        rmse_errors = []
        
        for filter_type in filter_names:
            if 'rmse' in filter_metrics[filter_type]:
                rmse_values = filter_metrics[filter_type]['rmse']
                rmse_data.append(np.mean(rmse_values))
                rmse_errors.append(np.std(rmse_values))
            else:
                rmse_data.append(0)
                rmse_errors.append(0)
        
        bars = plt.bar(filter_names, rmse_data, yerr=rmse_errors, capsize=5, color=colors[:len(filter_names)])
        plt.ylabel("RMSE (rad)")
        plt.title("Average RMSE (lower is better)", fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, rmse in zip(bars, rmse_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Normalized Error or Component-wise comparison
        plt.subplot(2, 2, 4)
        
        # Check if we have component-wise metrics
        if any('mse_roll' in metrics for metrics in filter_metrics.values()):
            # Create a grouped bar chart for component-wise errors
            width = 0.25
            x = np.arange(len(filter_names))
            
            roll_data = []
            pitch_data = []
            yaw_data = []
            
            for filter_type in filter_names:
                roll_data.append(np.mean(filter_metrics[filter_type].get('mse_roll', [0])))
                pitch_data.append(np.mean(filter_metrics[filter_type].get('mse_pitch', [0])))
                yaw_data.append(np.mean(filter_metrics[filter_type].get('mse_yaw', [0])))
            
            plt.bar(x - width, roll_data, width, label='Roll', color='#1f77b4')
            plt.bar(x, pitch_data, width, label='Pitch', color='#ff7f0e')
            plt.bar(x + width, yaw_data, width, label='Yaw', color='#2ca02c')
            
            plt.xlabel('Filter Type')
            plt.ylabel('MSE (rad²)')
            plt.title('Component-wise Errors', fontweight='bold')
            plt.xticks(x, filter_names)
            plt.legend()
            
        else:
            # Just show normalized error
            nerr_data = []
            nerr_errors = []
            
            for filter_type in filter_names:
                if 'normalized_error' in filter_metrics[filter_type]:
                    nerr_values = filter_metrics[filter_type]['normalized_error']
                    nerr_data.append(np.mean(nerr_values))
                    nerr_errors.append(np.std(nerr_values))
                else:
                    nerr_data.append(0)
                    nerr_errors.append(0)
            
            bars = plt.bar(filter_names, nerr_data, yerr=nerr_errors, capsize=5, color=colors[:len(filter_names)])
            plt.ylabel("Normalized Error")
            plt.title("Average Normalized Error (lower is better)", fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, nerr in zip(bars, nerr_data):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{nerr:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved performance comparison to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_path

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
        f.write(f"Trial: {trial_info}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data information
        f.write(f"1. Data Summary\n")
        f.write(f"--------------\n")
        
        # Accelerometer
        if 'accel_data' in data_dict:
            accel_shape = data_dict['accel_data'].shape
            f.write(f"Accelerometer: {accel_shape[0]} samples, {accel_shape[1]} columns\n")
            
            # Calculate sampling statistics if we have timestamps
            if accel_shape[0] > 1:
                accel_timestamps = data_dict['accel_data'][:, 0]
                dt = np.diff(accel_timestamps)
                f.write(f"  Sampling: Mean={np.mean(dt)*1000:.2f}ms, Min={np.min(dt)*1000:.2f}ms, "
                       f"Max={np.max(dt)*1000:.2f}ms, Std={np.std(dt)*1000:.2f}ms\n")
        else:
            f.write("Accelerometer: Not available\n")
        
        # Gyroscope
        if 'gyro_data' in data_dict:
            gyro_shape = data_dict['gyro_data'].shape
            f.write(f"Gyroscope: {gyro_shape[0]} samples, {gyro_shape[1]} columns\n")
            
            # Calculate sampling statistics if we have timestamps
            if gyro_shape[0] > 1:
                gyro_timestamps = data_dict['gyro_data'][:, 0]
                dt = np.diff(gyro_timestamps)
                f.write(f"  Sampling: Mean={np.mean(dt)*1000:.2f}ms, Min={np.min(dt)*1000:.2f}ms, "
                       f"Max={np.max(dt)*1000:.2f}ms, Std={np.std(dt)*1000:.2f}ms\n")
        else:
            f.write("Gyroscope: Not available\n")
        
        # Skeleton
        if 'skel_data' in data_dict:
            skel_shape = data_dict['skel_data'].shape
            f.write(f"Skeleton: {skel_shape[0]} frames, {skel_shape[1]} features\n")
        else:
            f.write("Skeleton: Not available\n")
        
        # Alignment results
        f.write(f"\n2. Alignment Results\n")
        f.write(f"-------------------\n")
        
        if 'aligned_data' in data_dict and data_dict['aligned_data'] is not None:
            aligned_data = data_dict['aligned_data']
            f.write(f"Aligned data: {len(aligned_data['timestamps'])} samples\n")
            
            # Acceleration statistics
            accel_mag = np.linalg.norm(aligned_data['accel'], axis=1)
            f.write(f"  Acceleration magnitude: Mean={np.mean(accel_mag):.2f}, Min={np.min(accel_mag):.2f}, "
                   f"Max={np.max(accel_mag):.2f}, Std={np.std(accel_mag):.2f}\n")
            
            # Gyroscope statistics
            gyro_mag = np.linalg.norm(aligned_data['gyro'], axis=1)
            f.write(f"  Angular velocity magnitude: Mean={np.mean(gyro_mag):.2f}, Min={np.min(gyro_mag):.2f}, "
                   f"Max={np.max(gyro_mag):.2f}, Std={np.std(gyro_mag):.2f}\n")
                
            if 'skeleton' in aligned_data:
                f.write(f"  Skeleton data included in alignment: Yes\n")
                
                # Extract reference orientations for reporting
                try:
                    ref_orient = extract_orientation_from_skeleton(aligned_data['skeleton'])
                    
                    # Orientation statistics
                    ref_means = np.mean(ref_orient, axis=0)
                    ref_stds = np.std(ref_orient, axis=0)
                    f.write(f"  Reference orientation statistics:\n")
                    f.write(f"    Roll:  Mean={ref_means[0]:.2f}, Std={ref_stds[0]:.2f}\n")
                    f.write(f"    Pitch: Mean={ref_means[1]:.2f}, Std={ref_stds[1]:.2f}\n")
                    f.write(f"    Yaw:   Mean={ref_means[2]:.2f}, Std={ref_stds[2]:.2f}\n")
                except Exception as e:
                    f.write(f"  Error extracting reference orientations: {e}\n")
            else:
                f.write(f"  Skeleton data included in alignment: No\n")
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
                
                # Extract orientation statistics
                if result['output'].shape[1] >= 13:  # Check for Euler angles
                    euler = result['output'][:, 10:13]
                    euler_means = np.mean(euler, axis=0)
                    euler_stds = np.std(euler, axis=0)
                    
                    f.write(f"  Orientation statistics:\n")
                    f.write(f"    Roll:  Mean={euler_means[0]:.2f}, Std={euler_stds[0]:.2f}\n")
                    f.write(f"    Pitch: Mean={euler_means[1]:.2f}, Std={euler_stds[1]:.2f}\n")
                    f.write(f"    Yaw:   Mean={euler_means[2]:.2f}, Std={euler_stds[2]:.2f}\n")
        
        # Performance metrics
        f.write(f"\n4. Performance Metrics\n")
        f.write(f"---------------------\n")
        
        if not metrics:
            f.write("No performance metrics available\n")
        else:
            # Create a table header
            f.write(f"{'Filter Type':<15} {'MSE':<10} {'RMSE':<10} {'Roll MSE':<10} {'Pitch MSE':<10} "
                   f"{'Yaw MSE':<10} {'Norm Error':<10} {'Time (s)':<10}\n")
            f.write("-" * 85 + "\n")
            
            for filter_type, m in metrics.items():
                # Format metrics with appropriate precision
                mse = f"{m.get('mse', 'N/A'):.6f}" if 'mse' in m else "N/A"
                rmse = f"{m.get('rmse', 'N/A'):.6f}" if 'rmse' in m else "N/A"
                mse_roll = f"{m.get('mse_roll', 'N/A'):.6f}" if 'mse_roll' in m else "N/A"
                mse_pitch = f"{m.get('mse_pitch', 'N/A'):.6f}" if 'mse_pitch' in m else "N/A"
                mse_yaw = f"{m.get('mse_yaw', 'N/A'):.6f}" if 'mse_yaw' in m else "N/A"
                norm_err = f"{m.get('normalized_error', 'N/A'):.6f}" if 'normalized_error' in m else "N/A"
                proc_time = f"{m['processing_time']:.3f}"
                
                f.write(f"{filter_type:<15} {mse:<10} {rmse:<10} {mse_roll:<10} {mse_pitch:<10} "
                       f"{mse_yaw:<10} {norm_err:<10} {proc_time:<10}\n")
        
        # Recommendations
        f.write(f"\n5. Analysis and Recommendations\n")
        f.write(f"------------------------------\n")
        
        if metrics:
            # Find best filter based on error if available, otherwise processing time
            if 'mse' in list(metrics.values())[0]:
                best_accuracy = min(metrics.items(), key=lambda x: x[1].get('mse', float('inf')))[0]
                best_speed = min(metrics.items(), key=lambda x: x[1]['processing_time'])[0]
                
                f.write(f"Based on orientation accuracy, the {best_accuracy} filter performs best for this trial.\n")
                f.write(f"Based on processing time, the {best_speed} filter performs best for this trial.\n")
                
                # Component-wise analysis
                if all(key in metrics[best_accuracy] for key in ['mse_roll', 'mse_pitch', 'mse_yaw']):
                    component_errors = {
                        'roll': metrics[best_accuracy]['mse_roll'],
                        'pitch': metrics[best_accuracy]['mse_pitch'],
                        'yaw': metrics[best_accuracy]['mse_yaw']
                    }
                    worst_component = max(component_errors.items(), key=lambda x: x[1])[0]
                    f.write(f"The {worst_component} component has the highest error, which is typical due to "
                           f"gyroscope drift in this axis.\n")
            else:
                best_speed = min(metrics.items(), key=lambda x: x[1]['processing_time'])[0]
                f.write(f"Based on processing time, the {best_speed} filter performs best for this trial.\n")
            
            # Observations about gyroscope data
            if 'aligned_data' in data_dict and data_dict['aligned_data'] is not None:
                gyro_mag = np.linalg.norm(data_dict['aligned_data']['gyro'], axis=1)
                if np.max(gyro_mag) > 3.0:
                    f.write(f"Gyroscope data shows significant angular motion (max {np.max(gyro_mag):.2f} rad/s), "
                           f"which helps with orientation estimation.\n")
                else:
                    f.write(f"Gyroscope data shows limited angular motion (max {np.max(gyro_mag):.2f} rad/s), "
                           f"which may affect orientation accuracy.\n")
            
            # Observations about skeleton reference
            if 'aligned_data' in data_dict and data_dict['aligned_data'] is not None and 'skeleton' in data_dict['aligned_data']:
                f.write(f"Skeleton reference orientations are available for drift correction.\n")
            else:
                f.write(f"No skeleton reference is available. Drift correction could improve long-term accuracy.\n")
            
            # Recommendations for this specific trial
            f.write(f"\nRecommendations for this trial:\n")
            
            # If we have significant errors
            if 'mse' in list(metrics.values())[0]:
                avg_mse = np.mean([m['mse'] for m in metrics.values() if 'mse' in m])
                if avg_mse > 0.1:  # High error threshold
                    f.write(f"- All filters show relatively high orientation errors for this trial.\n")
                    f.write(f"  Consider using a hybrid approach with occasional resets from skeleton data.\n")
                else:
                    f.write(f"- Orientation estimation is working well with the selected filters.\n")
            
            # Speed vs. accuracy tradeoff
            if 'mse' in list(metrics.values())[0] and 'best_accuracy' in locals() and 'best_speed' in locals() and best_accuracy != best_speed:
                speed_ratio = metrics[best_accuracy]['processing_time'] / metrics[best_speed]['processing_time']
                error_ratio = metrics[best_speed].get('mse', float('inf')) / metrics[best_accuracy].get('mse', 1.0)
                
                if speed_ratio > 2 and error_ratio < 1.5:
                    f.write(f"- Consider using the faster {best_speed} filter as it provides similar accuracy "
                           f"with {speed_ratio:.1f}x better performance.\n")
                elif error_ratio > 2:
                    f.write(f"- The more accurate {best_accuracy} filter is recommended despite being slower, "
                           f"as it reduces errors by {error_ratio:.1f}x.\n")
        else:
            f.write(f"Insufficient data to provide detailed recommendations.\n")
        
        # General recommendations for implementation
        f.write(f"\nGeneral Implementation Recommendations:\n")
        f.write(f"- When implementing in real-time, consider adaptive filter selection based on detected motion:\n")
        f.write(f"  • Use standard Kalman for low dynamics (sitting, standing)\n")
        f.write(f"  • Use EKF for moderate dynamics (walking, basic movements)\n")
        f.write(f"  • Use UKF for high dynamics (falls, rapid movements)\n")
        f.write(f"- Consider using a simplified filter for steady-state and switching to more complex\n")
        f.write(f"  filters only when significant motion is detected to save battery life.\n")
        f.write(f"- Use the robust filter implementations to avoid quaternion and matrix stability issues.\n")
    
    return report_path

def process_trial(args, trial_info, file_paths, output_dir, filter_types=None, wrist_idx=9, plot=True):
    """
    Process a single trial with all filters.
    
    Args:
        args: Command line arguments
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
    
    # Skip if no accelerometer or gyroscope data
    if 'accel_data' not in data_dict:
        logger.warning(f"No accelerometer data for trial {trial_info}")
        return None
    
    if 'gyro_data' not in data_dict:
        logger.warning(f"No gyroscope data for trial {trial_info}")
        return None
    
    # Align and resample data
    logger.info(f"Resampling accelerometer data to {args.target_fps} Hz")
    logger.info(f"Resampling gyroscope data to {args.target_fps} Hz")
    
    aligned_data = align_and_resample_data(
        data_dict['accel_data'],
        data_dict['gyro_data'],
        data_dict.get('skel_data'),
        target_fps=args.target_fps,
        apply_antialiasing=args.apply_antialiasing
    )
    
    # Store aligned data
    data_dict['aligned_data'] = aligned_data
    
    if aligned_data is None:
        logger.warning(f"Failed to align data for trial {trial_info}")
        return None
    
    # Apply IMU fusion
    filter_results = apply_imu_fusion(
        aligned_data, 
        filter_types, 
        wrist_idx=wrist_idx
    )
    
    if not filter_results:
        logger.warning(f"No filter results for trial {trial_info}")
        return None
    
    # Calculate metrics
    reference_orientations = None
    if 'skeleton' in aligned_data:
        try:
            reference_orientations = extract_orientation_from_skeleton(
                aligned_data['skeleton'],
                wrist_idx=wrist_idx
            )
        except Exception as e:
            logger.warning(f"Error extracting reference orientations: {e}")
    
    metrics = calculate_metrics(filter_results, reference_orientations)
    
    # Generate plots if requested
    if plot:
        try:
            # Plot filter results
            plot_path = os.path.join(trial_dir, "orientation_comparison.png")
            plot_filter_results(
                filter_results, 
                reference_orientations, 
                plot_path, 
                plot_title=f"Orientation Comparison - {trial_info}",
                show_plot=args.interactive
            )
            
            # Plot acceleration and angular velocity
            if aligned_data is not None:
                motion_plot_path = os.path.join(trial_dir, "motion_data.png")
                
                plt.figure(figsize=(15, 10))
                
                # Plot acceleration
                plt.subplot(2, 1, 1)
                plt.plot(aligned_data['timestamps'] - aligned_data['timestamps'][0], 
                         aligned_data['accel'][:, 0], 'r-', label='X')
                plt.plot(aligned_data['timestamps'] - aligned_data['timestamps'][0], 
                         aligned_data['accel'][:, 1], 'g-', label='Y')
                plt.plot(aligned_data['timestamps'] - aligned_data['timestamps'][0], 
                         aligned_data['accel'][:, 2], 'b-', label='Z')
                plt.plot(aligned_data['timestamps'] - aligned_data['timestamps'][0], 
                         np.linalg.norm(aligned_data['accel'], axis=1), 'k--', label='Magnitude')
                
                plt.title("Acceleration Data")
                plt.xlabel("Time (s)")
                plt.ylabel("Acceleration (m/s²)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Plot angular velocity
                plt.subplot(2, 1, 2)
                plt.plot(aligned_data['timestamps'] - aligned_data['timestamps'][0], 
                         aligned_data['gyro'][:, 0], 'r-', label='X')
                plt.plot(aligned_data['timestamps'] - aligned_data['timestamps'][0], 
                         aligned_data['gyro'][:, 1], 'g-', label='Y')
                plt.plot(aligned_data['timestamps'] - aligned_data['timestamps'][0], 
                         aligned_data['gyro'][:, 2], 'b-', label='Z')
                plt.plot(aligned_data['timestamps'] - aligned_data['timestamps'][0], 
                         np.linalg.norm(aligned_data['gyro'], axis=1), 'k--', label='Magnitude')
                
                plt.title("Angular Velocity Data")
                plt.xlabel("Time (s)")
                plt.ylabel("Angular Velocity (rad/s)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(motion_plot_path, dpi=150, bbox_inches='tight')
                
                if args.interactive:
                    plt.show()
                else:
                    plt.close()
            
            # Plot metrics comparison
            metrics_plot_path = os.path.join(trial_dir, "metrics_comparison.png")
            
            plt.figure(figsize=(12, 6))
            
            # Processing time
            plt.subplot(1, 2, 1)
            filter_names = list(metrics.keys())
            times = [metrics[f]['processing_time'] for f in filter_names]
            
            plt.bar(filter_names, times)
            plt.ylabel('Processing Time (s)')
            plt.title('Filter Processing Time')
            plt.grid(axis='y', alpha=0.3)
            
            # Error metrics if available
            if 'mse' in list(metrics.values())[0]:
                plt.subplot(1, 2, 2)
                mse_values = [metrics[f]['mse'] for f in filter_names]
                
                plt.bar(filter_names, mse_values)
                plt.ylabel('MSE (rad²)')
                plt.title('Orientation Error')
                plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
            
            if args.interactive:
                plt.show()
            else:
                plt.close()
            
        except Exception as e:
            logger.warning(f"Error generating plots: {e}")
            logger.warning(traceback.format_exc())
    
    # Generate report
    report_path = generate_report(trial_info, data_dict, filter_results, metrics, trial_dir)
    logger.info(f"Report generated: {report_path}")
    
    return metrics

def main():
    """Main function for IMU fusion debugging."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log all arguments
    logger.info("Starting IMU fusion debugging with parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Process filter types
    filter_types = [f.strip().lower() for f in args.filters.split(',')]
    logger.info(f"Using filter types: {filter_types}")
    
    # Parse subject and action lists
    subject_list = [int(s.strip()) for s in args.subjects.split(',')]
    action_list = None
    if args.actions:
        action_list = [int(a.strip()) for a in args.actions.split(',')]
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
            args=args,
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
        plot_performance_comparison(all_metrics, summary_path, show_plot=args.interactive)
    
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
                    if isinstance(value, (int, float)):  # Only collect numeric metrics
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
        
        # Component-wise analysis if available
        if any('mse_roll' in metrics for metrics in avg_metrics.values()):
            f.write("\nComponent-wise Analysis:\n")
            f.write("----------------------\n")
            f.write(f"{'Filter Type':<15} {'Roll MSE':<12} {'Pitch MSE':<12} {'Yaw MSE':<12}\n")
            f.write("-" * 55 + "\n")
            
            for filter_type, metrics in avg_metrics.items():
                roll_vals = metrics.get('mse_roll', [])
                pitch_vals = metrics.get('mse_pitch', [])
                yaw_vals = metrics.get('mse_yaw', [])
                
                roll_str = f"{np.mean(roll_vals):.6f}" if roll_vals else "N/A"
                pitch_str = f"{np.mean(pitch_vals):.6f}" if pitch_vals else "N/A"
                yaw_str = f"{np.mean(yaw_vals):.6f}" if yaw_vals else "N/A"
                
                f.write(f"{filter_type:<15} {roll_str:<12} {pitch_str:<12} {yaw_str:<12}\n")
        
        # Overall recommendations
        f.write(f"\nOverall Recommendations:\n")
        f.write(f"-----------------------\n")
        
        # Find best filter based on error if available
        if any('mse' in metrics for metrics in avg_metrics.values()):
            best_acc_filter = min(
                [k for k, v in avg_metrics.items() if 'mse' in v],
                key=lambda x: np.mean(avg_metrics[x].get('mse', [np.inf]))
            )
            
            f.write(f"• For best orientation accuracy: Use the {best_acc_filter} filter\n")
        
        # Find fastest filter
        if all('processing_time' in metrics for metrics in avg_metrics.values()):
            best_speed_filter = min(
                avg_metrics.keys(),
                key=lambda x: np.mean(avg_metrics[x].get('processing_time', [np.inf]))
            )
            
            f.write(f"• For fastest processing: Use the {best_speed_filter} filter\n")
        
        # Optimal algorithm for fall detection
        f.write(f"\nRecommendations for Fall Detection Implementation:\n")
        f.write(f"1. For real-time fall detection on smartwatch or phone devices:\n")
        
        if 'best_speed_filter' in locals() and 'best_acc_filter' in locals():
            if best_speed_filter == best_acc_filter:
                f.write(f"   The {best_speed_filter} filter provides the best balance of accuracy and efficiency.\n")
            else:
                # Calculate tradeoff
                speed_ratio = np.mean(avg_metrics[best_acc_filter]['processing_time']) / np.mean(avg_metrics[best_speed_filter]['processing_time'])
                if 'mse' in avg_metrics[best_speed_filter] and 'mse' in avg_metrics[best_acc_filter]:
                    error_ratio = np.mean(avg_metrics[best_speed_filter]['mse']) / np.mean(avg_metrics[best_acc_filter]['mse'])
                    
                    if speed_ratio > 2 and error_ratio < 1.5:
                        f.write(f"   The {best_speed_filter} filter is recommended, as it is {speed_ratio:.1f}x faster while maintaining\n")
                        f.write(f"   acceptable accuracy (only {error_ratio:.1f}x higher error).\n")
                    elif error_ratio > 2:
                        f.write(f"   If accuracy is critical, use the {best_acc_filter} filter which reduces errors by {error_ratio:.1f}x\n")
                        f.write(f"   at the cost of {speed_ratio:.1f}x slower processing.\n")
                    else:
                        f.write(f"   Consider an adaptive approach that switches between {best_speed_filter} (for routine movements)\n")
                        f.write(f"   and {best_acc_filter} (during potential fall events).\n")
                else:
                    f.write(f"   Consider using {best_speed_filter} for real-time processing, switching to\n")
                    f.write(f"   {best_acc_filter} only during high-motion events to conserve battery.\n")
        
        # Implementation recommendations
        f.write(f"\n2. Implementation Best Practices:\n")
        f.write(f"   • Use proper timestamp-based alignment for acc/gyro data, resampled to 30 Hz\n")
        f.write(f"   • Add robust quaternion normalization to prevent zero-norm errors\n")
        f.write(f"   • Implement covariance regularization to ensure matrix positive definiteness\n")
        f.write(f"   • Use adaptive sampling rates based on detected activity levels\n")
        f.write(f"   • Consider motion-based filter switching for optimal accuracy/efficiency tradeoff\n")
        
        # Integration with existing codebase
        f.write(f"\n3. Integration with Existing Codebase:\n")
        f.write(f"   • Maintain the current teacher-student knowledge distillation approach\n")
        f.write(f"   • Use skeleton data and other sensors during training only\n")
        f.write(f"   • For inference, implement the recommended IMU fusion approach for watch/phone data\n")
        f.write(f"   • The quaternion representation provides the most stable rotation tracking\n")
    
    logger.info(f"Summary report saved to {summary_path}")
    logger.info("IMU fusion debugging complete!")

if __name__ == "__main__":
    main()
