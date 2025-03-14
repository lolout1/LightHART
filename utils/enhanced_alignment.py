# utils/enhanced_alignment.py (modified)

import numpy as np
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger("Alignment")
from utils.robust_alignment import process_all_modalities

def align_modalities(accel_data, gyro_data, skeleton_data=None, **kwargs):
    """Alias for process_all_modalities for backwards compatibility."""
    return process_all_modalities(accel_data, gyro_data, skeleton_data, **kwargs)
def robust_align_modalities(imu_data, skel_data, imu_timestamps=None, method='dtw', 
                           wrist_idx=9, min_samples=5):
    """
    Robustly align IMU and skeleton data with multiple fallback methods.
    
    Args:
        imu_data: IMU data (accelerometer or gyroscope values)
        skel_data: Skeleton data
        imu_timestamps: IMU timestamps (optional)
        method: Alignment method ('dtw', 'interpolation', 'crop')
        wrist_idx: Index of wrist joint in skeleton data
        min_samples: Minimum number of samples required
        
    Returns:
        Tuple of (aligned_imu, aligned_skel, aligned_timestamps)
    """
    # Handle empty inputs
    if imu_data.size == 0 or skel_data.size == 0:
        logger.warning("Empty data provided for alignment")
        return (np.zeros((0, max(3, imu_data.shape[1]) if imu_data.size > 0 else 3)), 
                np.zeros((0, skel_data.shape[1]) if skel_data.size > 0 else 96), 
                np.zeros(0))
    
    # Generate timestamps if not provided (assume 50 Hz for IMU)
    if imu_timestamps is None:
        imu_timestamps = np.arange(len(imu_data)) / 50.0
    
    # Extract wrist trajectory for alignment
    wrist_trajectory = extract_wrist_trajectory(skel_data, wrist_idx)
    
    # Try DTW first if requested
    if method == 'dtw':
        try:
            from dtaidistance import dtw
            
            # Extract features for alignment
            # For IMU: use acceleration magnitude
            imu_mag = np.linalg.norm(imu_data[:, :min(3, imu_data.shape[1])], axis=1)
            
            # Calculate wrist velocity
            wrist_vel = np.zeros(len(wrist_trajectory))
            if len(wrist_trajectory) > 1:
                dt = 1.0/30.0  # Assuming 30 fps for skeleton
                for i in range(1, len(wrist_trajectory)):
                    wrist_vel[i] = np.linalg.norm(wrist_trajectory[i] - wrist_trajectory[i-1]) / dt
            
            # Normalize sequences for DTW
            imu_norm = (imu_mag - np.mean(imu_mag)) / (np.std(imu_mag) + 1e-9)
            wrist_norm = (wrist_vel - np.mean(wrist_vel)) / (np.std(wrist_vel) + 1e-9)
            
            # Calculate DTW path
            path = dtw.warping_path(imu_norm, wrist_norm)
            imu_idx, skel_idx = zip(*path)
            
            # Remove duplicates while preserving order
            imu_unique, skel_unique = [], []
            seen_imu, seen_skel = set(), set()
            
            for i, s in zip(imu_idx, skel_idx):
                if i not in seen_imu:
                    imu_unique.append(i)
                    seen_imu.add(i)
                if s not in seen_skel:
                    skel_unique.append(s)
                    seen_skel.add(s)
            
            # Get aligned data
            aligned_imu = imu_data[imu_unique]
            aligned_skel = skel_data[skel_unique]
            aligned_ts = imu_timestamps[imu_unique]
            
            if len(aligned_imu) >= min_samples:
                logger.info(f"DTW alignment successful: {len(aligned_imu)} points")
                return aligned_imu, aligned_skel, aligned_ts
            else:
                logger.warning(f"DTW alignment produced insufficient points: {len(aligned_imu)} < {min_samples}")
        except Exception as e:
            logger.warning(f"DTW alignment failed: {e}, falling back to interpolation")
    
    # Try interpolation next
    try:
        # Create skeleton timestamps (30 fps)
        skel_timestamps = np.arange(len(skel_data)) / 30.0
        
        # Find common time range
        t_start = max(imu_timestamps[0], skel_timestamps[0])
        t_end = min(imu_timestamps[-1], skel_timestamps[-1])
        
        if t_end <= t_start:
            logger.warning("No overlapping time range between modalities")
            # Return correctly sized empty arrays
            return (np.zeros((0, imu_data.shape[1])), 
                    np.zeros((0, skel_data.shape[1])), 
                    np.zeros(0))
        
        # Filter IMU data to common range
        mask = (imu_timestamps >= t_start) & (imu_timestamps <= t_end)
        filtered_ts = imu_timestamps[mask]
        filtered_imu = imu_data[mask]
        
        if len(filtered_ts) < min_samples:
            logger.warning(f"Insufficient IMU points in common time range: {len(filtered_ts)} < {min_samples}")
            # Try crop method as last resort
            mask_skel = (skel_timestamps >= t_start) & (skel_timestamps <= t_end)
            if np.sum(mask_skel) < min_samples:
                # Not enough overlap, just use the shorter sequence
                min_len = min(len(imu_data), len(skel_data))
                if min_len < min_samples:
                    logger.warning("Insufficient samples for alignment, returning empty arrays")
                    return (np.zeros((0, imu_data.shape[1])), 
                            np.zeros((0, skel_data.shape[1])), 
                            np.zeros(0))
                return imu_data[:min_len], skel_data[:min_len], imu_timestamps[:min_len]
            
            # Use direct sample-to-sample mapping
            skel_in_range = skel_data[mask_skel]
            skel_ts_in_range = skel_timestamps[mask_skel]
            
            # Simple cropping - use the shorter sequence
            min_samples = min(len(filtered_imu), len(skel_in_range))
            return filtered_imu[:min_samples], skel_in_range[:min_samples], filtered_ts[:min_samples]
        
        # Interpolate skeleton data to IMU timestamps
        interp_funcs = []
        for i in range(skel_data.shape[1]):
            interp_funcs.append(
                interp1d(skel_timestamps, skel_data[:, i], 
                        bounds_error=False, fill_value="extrapolate")
            )
        
        # Apply interpolation
        interp_skel = np.zeros((len(filtered_ts), skel_data.shape[1]))
        for i, func in enumerate(interp_funcs):
            interp_skel[:, i] = func(filtered_ts)
        
        logger.info(f"Interpolation alignment successful: {len(filtered_ts)} points")
        return filtered_imu, interp_skel, filtered_ts
        
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}, falling back to crop alignment")
    
    # Last resort: simple crop alignment
    logger.info("Using basic crop alignment as fallback")
    min_len = min(len(imu_data), len(skel_data))
    
    if min_len < min_samples:
        logger.warning(f"Insufficient samples after cropping: {min_len} < {min_samples}")
        return (np.zeros((0, imu_data.shape[1])), 
                np.zeros((0, skel_data.shape[1])), 
                np.zeros(0))
    
    return imu_data[:min_len], skel_data[:min_len], imu_timestamps[:min_len]

def extract_wrist_trajectory(skeleton_data, wrist_idx=9, joint_dim=3):
    """
    Extract wrist joint trajectory from skeleton data.
    
    Args:
        skeleton_data: Skeleton data of shape (n_frames, n_features)
        wrist_idx: Index of wrist joint
        joint_dim: Number of dimensions per joint (default: 3 for x,y,z)
        
    Returns:
        Wrist trajectory of shape (n_frames, joint_dim)
    """
    if skeleton_data.shape[1] < (wrist_idx + 1) * joint_dim:
        logger.warning(f"Skeleton data has insufficient dimensions for wrist_idx={wrist_idx}")
        return np.zeros((skeleton_data.shape[0], joint_dim))
    
    start_idx = wrist_idx * joint_dim
    end_idx = start_idx + joint_dim
    return skeleton_data[:, start_idx:end_idx]
