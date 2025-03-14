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
import traceback
from collections import defaultdict
import glob
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_imu_fusion")

# Import utilities for data loading and processing
from utils.processor.base import (
    parse_watch_csv,
    create_skeleton_timestamps,
    robust_align_modalities,
    extract_wrist_trajectory,
    match_trials
)

# Import standard Kalman filter implementations
from utils.imu_fusion import (
    StandardKalmanIMU,
    ExtendedKalmanIMU,
    UnscentedKalmanIMU,
    extract_orientation_from_skeleton
)

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

def find_sensor_files(base_dir, subject_list, action_list=None):
    """
    Find all relevant sensor files for the given subjects and actions.
    
    Args:
        base_dir: Base directory for dataset
        subject_list: List of subject IDs to include
        action_list: Optional list of action IDs to filter
        
    Returns:
        Tuple of (accel_files, gyro_files, skel_files)
    """
    # Define directories
    young_dir = os.path.join(base_dir, "young")
    old_dir = os.path.join(base_dir, "old")
    
    accel_files = []
    gyro_files = []
    skel_files = []
    
    # Search patterns for each subject
    for subj in subject_list:
        # Pattern for subject ID in filename
        subj_pattern = f"S{subj:02d}A"
        
        # Search for accelerometer files
        accel_dir = os.path.join(young_dir, "accelerometer", "watch")
        if os.path.exists(accel_dir):
            for file in glob.glob(os.path.join(accel_dir, f"{subj_pattern}*.csv")):
                accel_files.append(file)
        
        # Search for gyroscope files
        gyro_dir = os.path.join(young_dir, "gyroscope", "watch")
        if os.path.exists(gyro_dir):
            for file in glob.glob(os.path.join(gyro_dir, f"{subj_pattern}*.csv")):
                gyro_files.append(file)
        
        # Search for skeleton files
        skel_dir = os.path.join(young_dir, "skeleton")
        if os.path.exists(skel_dir):
            for file in glob.glob(os.path.join(skel_dir, f"{subj_pattern}*.csv")):
                skel_files.append(file)
    
    # Filter by action ID if specified
    if action_list:
        def filter_by_action(file_list):
            filtered = []
            for file in file_list:
                filename = os.path.basename(file)
                # Extract action ID from filename (e.g., S29A01T01.csv -> 01)
                match = re.search(r'S\d+A(\d+)T', filename)
                if match:
                    act_id = int(match.group(1))
                    if act_id in action_list:
                        filtered.append(file)
            return filtered
        
        accel_files = filter_by_action(accel_files)
        gyro_files = filter_by_action(gyro_files)
        skel_files = filter_by_action(skel_files)
    
    logger.info(f"Found {len(accel_files)} accelerometer files, {len(gyro_files)} gyroscope files, "
               f"and {len(skel_files)} skeleton files")
    
    return accel_files, gyro_files, skel_files

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
    logger.info(f"Loading accelerometer data from {accel_path}")
    accel_data = parse_watch_csv(accel_path)
    if accel_data.shape[0] > 0:
        result['accel_data'] = accel_data
        result['accel_timestamps'] = accel_data[:, 0]
        logger.info(f"Loaded accelerometer data: {accel_data.shape}")
    else:
        logger.warning(f"Empty accelerometer data in {accel_path}")
        return None
    
    # Load gyroscope if available
    if gyro_path and os.path.exists(gyro_path):
        logger.info(f"Loading gyroscope data from {gyro_path}")
        gyro_data = parse_watch_csv(gyro_path)
        if gyro_data.shape[0] > 0:
            result['gyro_data'] = gyro_data
            logger.info(f"Loaded gyroscope data: {gyro_data.shape}")
        else:
            logger.warning(f"Empty gyroscope data in {gyro_path}")
    
    # Load skeleton if available
    if skel_path and os.path.exists(skel_path):
        logger.info(f"Loading skeleton data from {skel_path}")
        try:
            df = pd.read_csv(skel_path, header=None)
            skel_array = df.values.astype(np.float32)
            
            # Add time column if not present
            if skel_array.shape[1] == 96:  # No time column
                result['skel_data'] = skel_array
                result['skel_timestamps'] = np.arange(skel_array.shape[0]) / 30.0  # 30 fps
                logger.info(f"Loaded skeleton data: {skel_array.shape}")
            else:
                result['skel_data'] = skel_array[:, 1:]  # Skip time column
                result['skel_timestamps'] = skel_array[:, 0]
                logger.info(f"Loaded skeleton data with time: {skel_array.shape}")
        except Exception as e:
            logger.error(f"Error loading skeleton data: {e}")
    
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
        logger.warning("Missing accelerometer or skeleton data for alignment")
        return data_dict
    
    try:
        # Extract data
        accel_data = data_dict['accel_data'][:, 1:]  # Skip time column
        accel_timestamps = data_dict['accel_timestamps']
        skel_data = data_dict['skel_data']
        
        logger.info(f"Aligning modalities using {method} method")
        
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
            
            logger.info(f"Alignment succeeded: {aligned_imu.shape[0]} points")
            
            # Extract reference orientations
            logger.info("Extracting reference orientations from skeleton")
            orientations = extract_orientation_from_skeleton(
                aligned_skel, wrist_idx=wrist_idx
            )
            
            # Ensure orientations have the same length as aligned data
            if len(orientations) != len(aligned_ts):
                logger.warning(f"Orientation length mismatch: {len(orientations)} vs {len(aligned_ts)}")
                # Trim to minimum length to ensure dimension match
                min_len = min(len(orientations), len(aligned_ts))
                orientations = orientations[:min_len]
                aligned_ts = aligned_ts[:min_len]
                aligned_imu = aligned_imu[:min_len]
                aligned_skel = aligned_skel[:min_len]
                
            data_dict['reference_orientations'] = orientations
            
            # If gyro data exists, align it too
            if 'gyro_data' in data_dict:
                gyro_data = data_dict['gyro_data']
                gyro_timestamps = gyro_data[:, 0]
                gyro_values = gyro_data[:, 1:]
                
                # Interpolate gyro to aligned timestamps
                from scipy.interpolate import interp1d
                try:
                    if len(gyro_timestamps) > 1:
                        logger.info("Interpolating gyroscope data to aligned timestamps")
                        gyro_interp = interp1d(
                            gyro_timestamps,
                            gyro_values,
                            axis=0,
                            bounds_error=False,
                            fill_value="extrapolate"
                        )
                        
                        aligned_gyro = gyro_interp(aligned_ts)
                        data_dict['aligned_gyro'] = aligned_gyro
                    else:
                        # Handle empty gyro case
                        logger.warning("Empty gyroscope data, using zeros")
                        data_dict['aligned_gyro'] = np.zeros((len(aligned_ts), 3))
                except Exception as e:
                    logger.error(f"Error interpolating gyro: {e}")
                    # Create zero gyro values
                    data_dict['aligned_gyro'] = np.zeros((len(aligned_ts), 3))
        else:
            logger.warning("Alignment produced insufficient data points.")
    except Exception as e:
        logger.error(f"Error aligning modalities: {e}")
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
        logger.error("Missing accelerometer data, cannot apply filters")
        return results
    
    # Use aligned data if available
    if 'aligned_imu' in data_dict and 'aligned_timestamps' in data_dict:
        accel_values = data_dict['aligned_imu']
        timestamps = data_dict['aligned_timestamps']
        
        if 'aligned_gyro' in data_dict:
            gyro_values = data_dict['aligned_gyro']
        else:
            logger.warning("No aligned gyro data, using zeros")
            gyro_values = np.zeros_like(accel_values)
        
        reference_orientations = data_dict.get('reference_orientations', None)
        reference_timestamps = data_dict.get('aligned_timestamps', None)
    else:
        # Use original data
        logger.info("Using original (non-aligned) data for filtering")
        accel_values = data_dict['accel_data'][:, 1:]  # Skip time column
        timestamps = data_dict['accel_timestamps']
        
        if 'gyro_data' in data_dict:
            gyro_values = data_dict['gyro_data'][:, 1:]  # Skip time column
        else:
            logger.warning("No gyroscope data, using zeros")
            gyro_values = np.zeros_like(accel_values)
        
        reference_orientations = None
        reference_timestamps = None
    
    # Apply each filter
    for filter_type in filter_types:
        logger.info(f"Applying {filter_type} filter...")
        
        # Create filter
        if filter_type == 'standard':
            kf = StandardKalmanIMU()
        elif filter_type == 'ekf':
            kf = ExtendedKalmanIMU()
        elif filter_type == 'ukf':
            kf = UnscentedKalmanIMU()
        else:
            logger.warning(f"Unknown filter type: {filter_type}")
            continue
        
        # Set reference data if available
        if reference_orientations is not None and reference_timestamps is not None:
            if len(reference_orientations) == len(reference_timestamps):
                logger.info(f"Setting reference data: {len(reference_timestamps)} points")
                kf.set_reference_data(reference_timestamps, reference_orientations)
        
        # Apply filter
        try:
            start_time = time.time()
            output = kf.process_sequence(accel_values, gyro_values, timestamps)
            end_time = time.time()
            
            # Store results
            results[filter_type] = {
                'output': output,  # [accel, gyro, quat, euler]
                'timestamps': timestamps,
                'processing_time': end_time - start_time
            }
            
            logger.info(f"{filter_type} filter processing time: {end_time - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error applying {filter_type} filter: {e}")
            traceback.print_exc()
    
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
    
    # 1. Plot Euler angles (roll, pitch, yaw) for each filter
    plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1)
    
    angles = ["Roll", "Pitch", "Yaw"]
    euler_idx = [-3, -2, -1]  # Last 3 columns in output are euler angles
    
    # Check if we have reference orientations
    has_reference = False
    if 'reference_orientations' in data_dict and 'aligned_timestamps' in data_dict:
        ref_orientations = data_dict['reference_orientations']
        ref_timestamps = data_dict['aligned_timestamps']
        
        # Validate lengths to avoid plotting errors
        if len(ref_orientations) == len(ref_timestamps) and len(ref_orientations) > 0:
            has_reference = True
            logger.info(f"Using {len(ref_orientations)} reference orientations for plotting")
    
    for i in range(3):
        ax = plt.subplot(gs[i, 0])
        
        # Plot each filter
        for filter_type, result in filter_results.items():
            timestamps = result['timestamps']
            try:
                if result['output'].shape[1] > abs(euler_idx[i]):
                    euler = result['output'][:, euler_idx[i]]
                    ax.plot(timestamps, euler, label=f"{filter_type}")
                else:
                    logger.warning(f"Missing Euler angle {angles[i]} in {filter_type} output")
            except Exception as e:
                logger.error(f"Error plotting Euler angle {angles[i]} for {filter_type}: {e}")
        
        # Plot reference if available
        if has_reference:
            try:
                if i < ref_orientations.shape[1]:
                    ax.plot(ref_timestamps, ref_orientations[:, i], 'k--', label='Skeleton Reference')
            except Exception as e:
                logger.error(f"Error plotting reference orientation: {e}")
        
        ax.set_ylabel(f"{angles[i]} (rad)")
        ax.grid(True)
        
        if i == 0:
            plt.title(f"Orientation Comparison - {trial_info}")
        if i == 2:
            ax.set_xlabel("Time (s)")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"euler_comparison_{trial_info.replace(' ', '_')}.png"))
    plt.close()
    
    # 2. Plot quaternion components
    plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 1)
    
    quat_components = ["w", "x", "y", "z"]
    quat_idx = [-7, -6, -5, -4]  # Typical positions of quaternion components
    
    for i in range(4):
        ax = plt.subplot(gs[i, 0])
        
        # Plot each filter
        for filter_type, result in filter_results.items():
            timestamps = result['timestamps']
            try:
                # Check if quaternion component exists in output
                if result['output'].shape[1] > abs(quat_idx[i]):
                    quat = result['output'][:, quat_idx[i]]
                    ax.plot(timestamps, quat, label=f"{filter_type}")
                else:
                    logger.warning(f"Missing quaternion component {quat_components[i]} in {filter_type} output")
            except Exception as e:
                logger.error(f"Error plotting quaternion component {quat_components[i]} for {filter_type}: {e}")
        
        ax.set_ylabel(f"Quat {quat_components[i]}")
        ax.grid(True)
        
        if i == 0:
            plt.title(f"Quaternion Components - {trial_info}")
        if i == 3:
            ax.set_xlabel("Time (s)")
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"quaternion_comparison_{trial_info.replace(' ', '_')}.png"))
    plt.close()
    
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
    plt.close()
    
    # 4. Plot filter performance comparison
    plt.figure(figsize=(10, 6))
    
    # Compute errors if we have reference data
    errors = {}
    if has_reference:
        for filter_type, result in filter_results.items():
            try:
                # Extract Euler angles from filter output
                euler = result['output'][:, -3:]  # Last 3 columns are euler angles
                
                # Ensure same length for comparison
                min_len = min(euler.shape[0], ref_orientations.shape[0])
                
                # Calculate MSE
                mse = np.mean(np.sum((euler[:min_len] - ref_orientations[:min_len]) ** 2, axis=1))
                errors[filter_type] = mse
                logger.info(f"MSE for {filter_type}: {mse:.6f}")
            except Exception as e:
                logger.error(f"Error calculating MSE for {filter_type}: {e}")
        
        # Plot MSE
        if errors:
            plt.bar(errors.keys(), errors.values())
            plt.ylabel("Mean Squared Error (rad²)")
            plt.title(f"Filter Orientation Error - {trial_info}")
            
            # Add timing information
            for i, (filter_type, result) in enumerate(filter_results.items()):
                if filter_type in errors:
                    plt.text(i, errors[filter_type] + 0.01, 
                            f"{result['processing_time']:.3f}s", 
                            ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"filter_error_{trial_info.replace(' ', '_')}.png"))
            plt.close()
    
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

def generate_report(data_dict, results_summary, output_dir, trial_info):
    """
    Generate a detailed text report about the trial processing.
    
    Args:
        data_dict: Data dictionary
        results_summary: Results from plot_results
        output_dir: Directory to save report
        trial_info: Trial information
    """
    report_path = os.path.join(output_dir, f"{trial_info.replace(' ', '_')}_report.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"Debug Report for {trial_info}\n")
        f.write("="*80 + "\n\n")
        
        # Data Loading Section
        f.write("1) Data Loading\n")
        f.write("-"*80 + "\n")
        
        if 'accel_data' in data_dict:
            accel_shape = data_dict['accel_data'].shape
            f.write(f" accel: {accel_shape[0]} samples, {accel_shape[1]} columns\n")
        else:
            f.write(" accel: missing\n")
            
        if 'gyro_data' in data_dict:
            gyro_shape = data_dict['gyro_data'].shape
            f.write(f" gyro: {gyro_shape[0]} samples, {gyro_shape[1]} columns\n")
        else:
            f.write(" gyro: missing\n")
            
        if 'skel_data' in data_dict:
            skel_shape = data_dict['skel_data'].shape
            f.write(f" skeleton: {skel_shape[0]} frames, {skel_shape[1]} features\n")
        else:
            f.write(" skeleton: missing\n")
        
        f.write("\n")
        
        # Alignment Section
        f.write("2) Modality Alignment\n")
        f.write("-"*80 + "\n")
        
        if 'aligned_imu' in data_dict:
            aligned_shape = data_dict['aligned_imu'].shape
            f.write(f" Aligned data: {aligned_shape[0]} samples\n")
            
            if 'reference_orientations' in data_dict:
                ref_shape = data_dict['reference_orientations'].shape
                f.write(f" Reference orientations: {ref_shape[0]} samples, {ref_shape[1]} angles\n")
        else:
            f.write(" No aligned data\n")
        
        f.write("\n")
        
        # Filtering Results
        f.write("3) Filter Results\n")
        f.write("-"*80 + "\n")
        
        if 'processing_times' in results_summary:
            for filter_type, proc_time in results_summary['processing_times'].items():
                f.write(f" {filter_type}: {proc_time:.4f} seconds\n")
                
            if 'errors' in results_summary and results_summary['errors']:
                f.write("\n MSE Comparison:\n")
                for filter_type, error in results_summary['errors'].items():
                    f.write(f" {filter_type}: {error:.6f} rad²\n")
        else:
            f.write(" No filter results\n")
            
    logger.info(f"Generated report: {report_path}")
    return report_path

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
    logger.info(f"\n=== Processing {trial_info} ===")
    
    # Create trial-specific output directory
    trial_dir = os.path.join(output_dir, trial_info.replace(' ', '_'))
    os.makedirs(trial_dir, exist_ok=True)
    
    # 1. Load data
    data_dict = load_data(accel_path, gyro_path, skel_path)
    if data_dict is None:
        logger.error(f"Failed to load data for {trial_info}")
        # Generate error report
        with open(os.path.join(output_dir, f"{trial_info.replace(' ', '_')}_report.txt"), 'w') as f:
            f.write(f"Debug Report for {trial_info}\n")
            f.write("="*80 + "\n\n")
            f.write("1) Data Loading\n")
            f.write("-"*80 + "\n")
            f.write(f" accel: {accel_path}\n")
            if gyro_path:
                f.write(f" gyro: {gyro_path}\n")
            if skel_path:
                f.write(f" skeleton: {skel_path}\n")
            f.write("\n")
            accel_data = parse_watch_csv(accel_path)
            f.write(f"accel => shape={accel_data.shape}\n")
            if gyro_path:
                gyro_data = parse_watch_csv(gyro_path)
                f.write(f"gyro => shape={gyro_data.shape}\n")
            if skel_path:
                try:
                    df = pd.read_csv(skel_path, header=None)
                    skel_array = df.values.astype(np.float32)
                    f.write(f"skel => shape={skel_array.shape}\n")
                except:
                    f.write("skel => error loading\n")
            f.write("ERROR: empty accel => abort.\n")
        return None
    
    # 2. Align modalities if we have skeleton data
    if 'skel_data' in data_dict:
        data_dict = align_modalities(data_dict, wrist_idx, method='dtw')
    
    # 3. Apply filters
    filter_results = apply_filters(data_dict, filter_types)
    if not filter_results:
        logger.error(f"Failed to apply filters for {trial_info}")
        return None
    
    # 4. Generate results
    results_summary = None
    if plot:
        try:
            results_summary = plot_results(data_dict, filter_results, trial_dir, trial_info)
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            traceback.print_exc()
    
    # If plotting failed or was not requested, still compute errors
    if results_summary is None:
        results_summary = {
            'trial_info': trial_info,
            'processing_times': {k: v['processing_time'] for k, v in filter_results.items()},
            'data_points': len(filter_results[list(filter_results.keys())[0]]['timestamps'])
        }
        
        # Compute errors if we have reference data
        if 'reference_orientations' in data_dict and len(data_dict['reference_orientations']) > 0:
            errors = {}
            ref_orientations = data_dict['reference_orientations']
            
            for filter_type, result in filter_results.items():
                try:
                    euler = result['output'][:, -3:]  # Last 3 columns are euler angles
                    min_len = min(euler.shape[0], ref_orientations.shape[0])
                    mse = np.mean(np.sum((euler[:min_len] - ref_orientations[:min_len]) ** 2, axis=1))
                    errors[filter_type] = mse
                except Exception as e:
                    logger.error(f"Error calculating MSE: {e}")
            
            results_summary['errors'] = errors
    
    # 5. Generate detailed report
    generate_report(data_dict, results_summary, output_dir, trial_info)
    
    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Debug IMU Fusion Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/smartfallmm",
                        help="Base directory for dataset (containing 'young'/'old', etc.)")
    parser.add_argument("--subjects", type=str, default="29,30,31",
                        help="Comma-separated list of subject IDs to test.")
    parser.add_argument("--actions", type=str, default=None,
                        help="Comma-separated list of action IDs to test or None => all.")
    parser.add_argument("--filters", type=str, default="standard,ekf,ukf",
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
    parser.add_argument("--report_dir", type=str, default="results",
                        help="Directory to store result reports.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    logger.info(f"=== Running IMU fusion debug ===")
    
    # Parse arguments
    subject_list = [int(x.strip()) for x in args.subjects.split(',')]
    filter_list = [f.strip().lower() for f in args.filters.split(',')]
    
    action_list = None
    if args.actions is not None:
        action_list = [int(x.strip()) for x in args.actions.split(',')]
    
    # Find sensor files
    accel_files, gyro_files, skel_files = find_sensor_files(
        args.data_dir, subject_list, action_list
    )
    
    # Match trials
    matched_trials = match_trials(accel_files, gyro_files, skel_files)
    
    # Limit to max_trials
    if len(matched_trials) > args.max_trials:
        matched_trials = matched_trials[:args.max_trials]
    
    logger.info(f"Found {len(matched_trials)} matched trials to process")
    
    all_results = []
    
    # Process each trial
    for subj, act, trial, accel_fp, gyro_fp, skel_fp in matched_trials:
        trial_info = f"S{subj:02d}A{act:02d}T{trial:02d}"
        logger.info(f"Processing {trial_info}...")
        
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
        if args.plot and len(all_results) > 0:
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
                    if 'processing_times' in result and filter_type in result['processing_times']:
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
            plt.close()
        
        # Save summary
        with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
            json.dump({
                'trials': len(all_results),
                'subjects': args.subjects,
                'actions': args.actions,
                'filters': args.filters,
                'results': all_results
            }, f, indent=2)
        
        logger.info(f"Debug complete! Results saved to {args.output_dir}")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()
