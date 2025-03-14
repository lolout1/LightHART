"""
robust_alignment.py

Robust alignment of multimodal sensor data with different sampling rates.
Handles specific challenges of the SmartFallMM dataset:
- Variable sampling rates in watch/phone IMU sensors
- Fixed 30Hz sampling for skeleton data
- Multiple alignment methods with fallbacks
- Specialized preprocessing for different sensor types

Main functions:
- align_imu_sensors: Aligns accelerometer and gyroscope using timestamps
- resample_to_fixed_rate: Resamples sensor data to a regular grid
- align_imu_with_skeleton: Aligns processed IMU data with skeleton data
- process_all_modalities: End-to-end processing pipeline for all sensors
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logger = logging.getLogger("RobustAlignment")

def align_imu_sensors(accel_data: np.ndarray, gyro_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Aligns accelerometer and gyroscope data using timestamps.
    
    Args:
        accel_data: Accelerometer data with timestamps in first column
        gyro_data: Gyroscope data with timestamps in first column
        
    Returns:
        Dictionary with aligned timestamps, accelerometer, and gyroscope data
    """
    if accel_data.shape[0] == 0 or gyro_data.shape[0] == 0:
        logger.warning("Empty accelerometer or gyroscope data")
        return {'success': False}
    
    # Extract timestamps
    accel_timestamps = accel_data[:, 0]
    gyro_timestamps = gyro_data[:, 0]
    
    # Find common time range
    t_start = max(accel_timestamps[0], gyro_timestamps[0])
    t_end = min(accel_timestamps[-1], gyro_timestamps[-1])
    
    if t_end <= t_start:
        logger.warning("No overlapping time range between accelerometer and gyroscope")
        return {'success': False}
    
    # Filter accelerometer data to common time range
    accel_mask = (accel_timestamps >= t_start) & (accel_timestamps <= t_end)
    filtered_accel_timestamps = accel_timestamps[accel_mask]
    filtered_accel_values = accel_data[accel_mask, 1:4]  # x, y, z values
    
    # Check if we have enough data after filtering
    if len(filtered_accel_timestamps) < 5:
        logger.warning("Insufficient accelerometer data points after filtering")
        return {'success': False}
    
    try:
        # Interpolate gyroscope data to accelerometer timestamps
        gyro_interp = interp1d(
            gyro_timestamps, 
            gyro_data[:, 1:4],  # x, y, z values
            axis=0, 
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        # Get gyro values at accelerometer timestamps
        aligned_gyro = gyro_interp(filtered_accel_timestamps)
        
        # Check for NaN values
        if np.any(np.isnan(aligned_gyro)):
            logger.warning("NaN values found after gyroscope interpolation, using nearest values")
            # Fall back to nearest-neighbor interpolation
            for i in range(3):
                col_data = gyro_data[:, i+1]
                nn_interp = interp1d(
                    gyro_timestamps,
                    col_data,
                    kind='nearest',
                    bounds_error=False,
                    fill_value=(col_data[0], col_data[-1])
                )
                aligned_gyro[:, i] = nn_interp(filtered_accel_timestamps)
        
        return {
            'success': True,
            'timestamps': filtered_accel_timestamps,
            'accel': filtered_accel_values,
            'gyro': aligned_gyro
        }
        
    except Exception as e:
        logger.error(f"Error during IMU alignment: {e}")
        return {'success': False}

def butter_lowpass(cutoff, fs, order=2):
    """Design a lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=2):
    """Apply a lowpass filter to the data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data, axis=0)

def resample_to_fixed_rate(timestamps: np.ndarray, values: np.ndarray, 
                          target_fps: float = 30.0, 
                          apply_antialiasing: bool = True) -> Dict[str, np.ndarray]:
    """
    Resamples sensor data to a fixed rate with optional anti-aliasing.
    
    Args:
        timestamps: Original timestamps
        values: Sensor values corresponding to timestamps
        target_fps: Target sampling rate in Hz
        apply_antialiasing: Whether to apply an anti-aliasing filter before resampling
        
    Returns:
        Dictionary with resampled timestamps and values
    """
    if len(timestamps) < 2:
        logger.warning("Insufficient data points for resampling")
        return {'success': False}
    
    try:
        # Create regular time grid at target_fps
        t_start = timestamps[0]
        t_end = timestamps[-1]
        duration = t_end - t_start
        num_samples = int(duration * target_fps) + 1
        regular_timestamps = np.linspace(t_start, t_end, num_samples)
        
        # Calculate original average sampling rate
        orig_fps = (len(timestamps) - 1) / duration
        
        # Apply anti-aliasing filter if requested and if downsampling
        filtered_values = values.copy()
        if apply_antialiasing and orig_fps > target_fps:
            # Set cutoff to Nyquist frequency of target rate
            cutoff = 0.4 * target_fps  # Slightly below half the target rate
            filtered_values = lowpass_filter(values, cutoff, orig_fps)
        
        # Interpolate to regular grid
        interp_func = interp1d(
            timestamps,
            filtered_values,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        resampled_values = interp_func(regular_timestamps)
        
        # Check for NaN values
        if np.any(np.isnan(resampled_values)):
            logger.warning("NaN values found after resampling, fixing with nearest interpolation")
            for i in range(values.shape[1]):
                mask = np.isnan(resampled_values[:, i])
                if np.any(mask):
                    nn_interp = interp1d(
                        timestamps,
                        values[:, i],
                        kind='nearest',
                        bounds_error=False,
                        fill_value=(values[0, i], values[-1, i])
                    )
                    resampled_values[mask, i] = nn_interp(regular_timestamps[mask])
        
        return {
            'success': True,
            'timestamps': regular_timestamps,
            'values': resampled_values
        }
        
    except Exception as e:
        logger.error(f"Error during resampling: {e}")
        return {'success': False}

def extract_wrist_trajectory(skeleton_data: np.ndarray, wrist_idx: int = 9, 
                           num_joints: int = 32) -> np.ndarray:
    """
    Extract wrist joint trajectory from skeleton data.
    
    Args:
        skeleton_data: Skeleton data with joint positions
        wrist_idx: Index of wrist joint
        num_joints: Total number of joints in skeleton
        
    Returns:
        Wrist trajectory
    """
    if skeleton_data.shape[1] < 3 * num_joints:
        logger.warning(f"Skeleton data has wrong format: {skeleton_data.shape}")
        return np.zeros((skeleton_data.shape[0], 3))
    
    # Extract wrist joint positions (x, y, z)
    start_col = wrist_idx * 3
    end_col = start_col + 3
    return skeleton_data[:, start_col:end_col]

def calculate_alignment_features(accel: np.ndarray, skel_data: np.ndarray, 
                               wrist_idx: int = 9, fps: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate features for alignment between IMU and skeleton data.
    
    Args:
        accel: Accelerometer data (after resampling)
        skel_data: Skeleton data
        wrist_idx: Index of wrist joint
        fps: Frame rate for calculating velocity
        
    Returns:
        Tuple of (IMU feature, skeleton feature)
    """
    # For IMU: use acceleration magnitude
    accel_mag = np.linalg.norm(accel, axis=1)
    
    # For skeleton: extract wrist joint and calculate velocity
    wrist_traj = extract_wrist_trajectory(skel_data, wrist_idx)
    wrist_vel = np.zeros(len(wrist_traj))
    
    # Calculate finite differences for velocity
    dt = 1.0 / fps
    for i in range(1, len(wrist_traj)):
        wrist_vel[i] = np.linalg.norm(wrist_traj[i] - wrist_traj[i-1]) / dt
    
    # Normalize both signals for better alignment
    norm_accel = (accel_mag - np.mean(accel_mag)) / (np.std(accel_mag) + 1e-8)
    norm_vel = (wrist_vel - np.mean(wrist_vel)) / (np.std(wrist_vel) + 1e-8)
    
    return norm_accel, norm_vel

def align_dtw(signal1: np.ndarray, signal2: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Align two signals using Dynamic Time Warping.
    
    Args:
        signal1: First signal
        signal2: Second signal
        
    Returns:
        Tuple of (indices1, indices2) matching points in the two signals
    """
    try:
        # Try using dtaidistance (faster)
        from dtaidistance import dtw
        path = dtw.warping_path(signal1, signal2)
        idx1, idx2 = zip(*path)
        return list(idx1), list(idx2)
    except ImportError:
        # Fall back to a simple implementation
        logger.warning("dtaidistance not found, using slower DTW implementation")
        
        # Simple DTW implementation
        n, m = len(signal1), len(signal2)
        dtw_matrix = np.zeros((n+1, m+1)) + np.inf
        dtw_matrix[0, 0] = 0
        
        # Fill the DTW matrix
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(signal1[i-1] - signal2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        # Backtrack to find the path
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            # Find the next move
            options = [
                (i-1, j),    # insertion
                (i, j-1),    # deletion
                (i-1, j-1)   # match
            ]
            min_idx = np.argmin([
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            ])
            i, j = options[min_idx]
        
        # Reverse the path
        path.reverse()
        
        if not path:
            return [], []
            
        idx1, idx2 = zip(*path)
        return list(idx1), list(idx2)

def remove_duplicates(indices1: List[int], indices2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Remove duplicate indices while preserving path order.
    
    Args:
        indices1: First sequence indices
        indices2: Second sequence indices
        
    Returns:
        Tuple of (unique_indices1, unique_indices2)
    """
    unique1, unique2 = [], []
    seen1, seen2 = set(), set()
    
    for i1, i2 in zip(indices1, indices2):
        if i1 not in seen1:
            unique1.append(i1)
            seen1.add(i1)
        if i2 not in seen2:
            unique2.append(i2)
            seen2.add(i2)
    
    return unique1, unique2

def align_imu_with_skeleton(imu_data: Dict[str, np.ndarray], 
                           skeleton_data: np.ndarray, 
                           wrist_idx: int = 9, 
                           method: str = 'dtw') -> Dict[str, Any]:
    """
    Align IMU data with skeleton data.
    
    Args:
        imu_data: Dictionary with timestamps, accel, gyro from previous steps
        skeleton_data: Skeleton joint positions
        wrist_idx: Index of wrist joint
        method: Alignment method ('dtw', 'interpolation', 'crop')
        
    Returns:
        Dictionary with aligned data
    """
    if not imu_data.get('success', False):
        logger.warning("IMU data not properly processed")
        return {'success': False}
    
    # Get data from input
    timestamps = imu_data['timestamps']
    accel = imu_data['accel']
    gyro = imu_data['gyro']
    
    if len(skeleton_data) < 5:
        logger.warning("Insufficient skeleton data points")
        return {'success': False}
    
    # Create skeleton timestamps at 30 Hz if needed
    if skeleton_data.shape[1] == 96:  # No time column
        skeleton_timestamps = np.arange(len(skeleton_data)) / 30.0
        skeleton_values = skeleton_data
    else:  # First column is time
        skeleton_timestamps = skeleton_data[:, 0]
        skeleton_values = skeleton_data[:, 1:]
    
    if method == 'dtw':
        try:
            # Calculate features for alignment
            accel_feat, skel_feat = calculate_alignment_features(
                accel, skeleton_values, wrist_idx, 30.0
            )
            
            # Perform DTW alignment
            imu_indices, skel_indices = align_dtw(accel_feat, skel_feat)
            
            # Remove duplicates
            imu_unique, skel_unique = remove_duplicates(imu_indices, skel_indices)
            
            # Check if alignment was successful
            if len(imu_unique) < 5 or len(skel_unique) < 5:
                logger.warning("DTW alignment produced too few points")
                method = 'interpolation'  # Fall back to interpolation
            else:
                logger.info(f"DTW alignment successful: {len(imu_unique)} IMU points, {len(skel_unique)} skeleton points")
                
                # Extract aligned data
                aligned_timestamps = timestamps[imu_unique]
                aligned_accel = accel[imu_unique]
                aligned_gyro = gyro[imu_unique]
                aligned_skeleton = skeleton_values[skel_unique]
                
                return {
                    'success': True,
                    'method': 'dtw',
                    'timestamps': aligned_timestamps,
                    'accel': aligned_accel,
                    'gyro': aligned_gyro,
                    'skeleton': aligned_skeleton
                }
                
        except Exception as e:
            logger.error(f"DTW alignment failed: {e}")
            method = 'interpolation'  # Fall back to interpolation
    
    if method == 'interpolation':
        try:
            # Find common time range
            t_start = max(timestamps[0], skeleton_timestamps[0])
            t_end = min(timestamps[-1], skeleton_timestamps[-1])
            
            if t_end <= t_start:
                logger.warning("No overlapping time range")
                return {'success': False}
            
            # Filter IMU data to common range
            imu_mask = (timestamps >= t_start) & (timestamps <= t_end)
            filtered_timestamps = timestamps[imu_mask]
            filtered_accel = accel[imu_mask]
            filtered_gyro = gyro[imu_mask]
            
            # Interpolate skeleton to IMU timestamps
            skel_interp = interp1d(
                skeleton_timestamps,
                skeleton_values,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            interpolated_skeleton = skel_interp(filtered_timestamps)
            
            # Check for NaN values
            if np.any(np.isnan(interpolated_skeleton)):
                logger.warning("NaN values in interpolated skeleton, applying fixes")
                for i in range(interpolated_skeleton.shape[1]):
                    mask = np.isnan(interpolated_skeleton[:, i])
                    if np.any(mask):
                        nn_interp = interp1d(
                            skeleton_timestamps,
                            skeleton_values[:, i],
                            kind='nearest',
                            bounds_error=False,
                            fill_value=(skeleton_values[0, i], skeleton_values[-1, i])
                        )
                        interpolated_skeleton[mask, i] = nn_interp(filtered_timestamps[mask])
            
            logger.info(f"Interpolation alignment successful: {len(filtered_timestamps)} points")
            
            return {
                'success': True,
                'method': 'interpolation',
                'timestamps': filtered_timestamps,
                'accel': filtered_accel,
                'gyro': filtered_gyro,
                'skeleton': interpolated_skeleton
            }
            
        except Exception as e:
            logger.error(f"Interpolation alignment failed: {e}")
            method = 'crop'  # Fall back to crop
    
    # Final fallback: simple crop method
    logger.info("Using basic crop alignment as final fallback")
    min_len = min(len(timestamps), len(skeleton_timestamps))
    
    if min_len < 5:
        logger.warning("Insufficient data for crop alignment")
        return {'success': False}
    
    return {
        'success': True,
        'method': 'crop',
        'timestamps': timestamps[:min_len],
        'accel': accel[:min_len],
        'gyro': gyro[:min_len],
        'skeleton': skeleton_values[:min_len] if len(skeleton_values) >= min_len else skeleton_values
    }

def extract_orientation_from_skeleton(skeleton_data: np.ndarray, wrist_idx: int = 9) -> np.ndarray:
    """
    Extract orientation information from skeleton for reference.
    
    Args:
        skeleton_data: Aligned skeleton data
        wrist_idx: Index of wrist joint
        
    Returns:
        Reference orientations (Euler angles)
    """
    from scipy.spatial.transform import Rotation as R
    
    n_frames = skeleton_data.shape[0]
    orientations = np.zeros((n_frames, 3))  # roll, pitch, yaw
    
    try:
        # Define key joint indices
        NECK = 2
        SPINE = 1
        RIGHT_SHOULDER = 8
        LEFT_SHOULDER = 4
        
        for i in range(n_frames):
            # Get joint positions
            joints = skeleton_data[i].reshape(-1, 3)
            
            if len(joints) > max(NECK, SPINE, RIGHT_SHOULDER, LEFT_SHOULDER, wrist_idx):
                # Calculate orthogonal basis vectors
                
                # Forward vector (spine direction)
                spine_vec = joints[NECK] - joints[SPINE]
                spine_vec = spine_vec / (np.linalg.norm(spine_vec) + 1e-6)
                
                # Right vector (between shoulders)
                shoulder_vec = joints[RIGHT_SHOULDER] - joints[LEFT_SHOULDER]
                shoulder_vec = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-6)
                
                # Up vector (cross product)
                up_vec = np.cross(shoulder_vec, spine_vec)
                up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-6)
                
                # Recalculate right vector for orthogonality
                right_vec = np.cross(spine_vec, up_vec)
                right_vec = right_vec / (np.linalg.norm(right_vec) + 1e-6)
                
                # Create rotation matrix from these vectors
                rot_matrix = np.column_stack([right_vec, up_vec, spine_vec])
                
                # Convert to Euler angles
                try:
                    r = R.from_matrix(rot_matrix)
                    orientations[i] = r.as_euler('xyz')
                except Exception as e:
                    # Use previous orientation if available
                    if i > 0:
                        orientations[i] = orientations[i-1]
                    logger.debug(f"Error converting rotation matrix: {e}")
    
    except Exception as e:
        logger.warning(f"Error extracting orientation from skeleton: {e}")
    
    return orientations

def process_all_modalities(accel_data: np.ndarray, gyro_data: np.ndarray, 
                         skeleton_data: Optional[np.ndarray] = None, 
                         target_fps: float = 30.0,
                         wrist_idx: int = 9,
                         alignment_method: str = 'dtw') -> Dict[str, Any]:
    """
    Complete end-to-end processing pipeline for sensor alignment.
    
    Args:
        accel_data: Accelerometer data with timestamps in first column
        gyro_data: Gyroscope data with timestamps in first column
        skeleton_data: Optional skeleton data
        target_fps: Target sampling rate for resampling
        wrist_idx: Index of wrist joint
        alignment_method: Method for aligning IMU with skeleton
        
    Returns:
        Dictionary with processed data
    """
    result = {'success': False}
    
    # Step 1: Align accelerometer and gyroscope
    imu_aligned = align_imu_sensors(accel_data, gyro_data)
    if not imu_aligned['success']:
        logger.warning("IMU sensor alignment failed")
        return result
    
    # Step 2: Resample to fixed rate
    resampled_accel = resample_to_fixed_rate(
        imu_aligned['timestamps'], 
        imu_aligned['accel'], 
        target_fps
    )
    
    resampled_gyro = resample_to_fixed_rate(
        imu_aligned['timestamps'], 
        imu_aligned['gyro'], 
        target_fps
    )
    
    if not resampled_accel['success'] or not resampled_gyro['success']:
        logger.warning("Resampling failed")
        return result
    
    # Combine resampled data
    resampled_imu = {
        'success': True,
        'timestamps': resampled_accel['timestamps'],
        'accel': resampled_accel['values'],
        'gyro': resampled_gyro['values']
    }
    
    # Step 3: If skeleton data is available, align it with IMU
    if skeleton_data is not None and skeleton_data.shape[0] > 0:
        aligned_data = align_imu_with_skeleton(
            resampled_imu,
            skeleton_data,
            wrist_idx,
            alignment_method
        )
        
        if aligned_data['success']:
            # Extract reference orientations for drift correction
            orientations = extract_orientation_from_skeleton(
                aligned_data['skeleton'],
                wrist_idx
            )
            
            aligned_data['reference_orientations'] = orientations
            return aligned_data
    
    # If no skeleton or alignment failed, return resampled IMU data
    return {
        'success': True,
        'method': 'imu_only',
        'timestamps': resampled_imu['timestamps'],
        'accel': resampled_imu['accel'],
        'gyro': resampled_imu['gyro']
    }
