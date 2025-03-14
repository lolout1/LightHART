"""
Enhanced alignment module for SmartFallMM.
Provides robust alignment between different modalities with different sampling rates.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import logging
import traceback

logger = logging.getLogger(__name__)

def detect_activity_segments(data, threshold=2.0, min_duration_sec=0.5, timestamps=None):
    """
    Detect segments of activity in sensor data.
    
    Args:
        data: Sensor data of shape (n_samples, n_features)
        threshold: Threshold for activity detection
        min_duration_sec: Minimum activity duration in seconds
        timestamps: Timestamps corresponding to data
        
    Returns:
        List of (start_time, end_time) tuples for active segments
    """
    if timestamps is None:
        # Create synthetic timestamps at assumed 30Hz
        timestamps = np.arange(len(data)) / 30.0
    
    # Calculate magnitude of signal (for acceleration or joint movement)
    if data.shape[1] >= 3:
        # For acceleration or 3D positions
        magnitude = np.linalg.norm(data[:, :3], axis=1)
    else:
        # Use the data as is
        magnitude = data[:, 0]
    
    # Calculate rolling variance
    window_size = max(int(min_duration_sec * 30), 5)  # Assuming ~30Hz
    rolling_var = np.zeros_like(magnitude)
    
    for i in range(len(magnitude)):
        start_idx = max(0, i - window_size)
        rolling_var[i] = np.var(magnitude[start_idx:i+1])
    
    # Detect active segments
    active = rolling_var > threshold
    
    # Find transitions
    transitions = np.where(np.diff(active.astype(int)))[0]
    
    # Group into start/end pairs
    segments = []
    if len(transitions) >= 2:
        # Check if first transition is start or end
        if active[0]:
            # First segment starts before data begins
            segments.append((timestamps[0], timestamps[transitions[0] + 1]))
            transitions = transitions[1:]
            
        # Process remaining transitions in pairs
        for i in range(0, len(transitions) - 1, 2):
            if i + 1 < len(transitions):
                start_idx = transitions[i] + 1
                end_idx = transitions[i + 1] + 1
                segments.append((timestamps[start_idx], timestamps[end_idx]))
        
        # Check if last segment is active
        if len(transitions) % 2 == 1:
            start_idx = transitions[-1] + 1
            segments.append((timestamps[start_idx], timestamps[-1]))
    elif active.any():
        # One continuous active segment
        segments.append((timestamps[0], timestamps[-1]))
    
    # Filter segments by minimum duration
    segments = [(start, end) for start, end in segments if end - start >= min_duration_sec]
    
    return segments

def extract_wrist_trajectory(skeleton_data, wrist_idx=9):
    """
    Extract wrist joint trajectory from skeleton data.
    
    Args:
        skeleton_data: Skeleton data of shape (n_frames, n_features)
        wrist_idx: Index of wrist joint
        
    Returns:
        Wrist trajectory of shape (n_frames, 3)
    """
    n_frames = skeleton_data.shape[0]
    
    # Each joint usually has 3 coordinates (x, y, z)
    joint_dim = 3
    
    # Extract wrist joint coordinates
    if skeleton_data.shape[1] >= (wrist_idx + 1) * joint_dim:
        start_idx = wrist_idx * joint_dim
        end_idx = start_idx + joint_dim
        wrist_trajectory = skeleton_data[:, start_idx:end_idx]
    else:
        # Fallback if wrist joint not found
        logger.warning(f"Wrist joint (idx={wrist_idx}) not found in skeleton data")
        wrist_trajectory = np.zeros((n_frames, 3))
    
    return wrist_trajectory

def calculate_joint_velocity(joint_trajectory, timestamps=None):
    """
    Calculate joint velocity from position trajectory.
    
    Args:
        joint_trajectory: Joint position trajectory of shape (n_frames, 3)
        timestamps: Optional timestamps for accurate derivative
        
    Returns:
        Joint velocity of shape (n_frames, 3)
    """
    if timestamps is None:
        # Assume constant sample rate (30Hz)
        dt = 1.0 / 30.0
        velocity = np.zeros_like(joint_trajectory)
        velocity[1:] = (joint_trajectory[1:] - joint_trajectory[:-1]) / dt
    else:
        # Use actual time differences
        velocity = np.zeros_like(joint_trajectory)
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                velocity[i] = (joint_trajectory[i] - joint_trajectory[i-1]) / dt
            else:
                velocity[i] = velocity[i-1]  # Repeat previous if dt = 0
    
    return velocity

def enhanced_align_modalities(imu_data, skel_data, imu_timestamps=None, skel_fps=30.0, 
                             wrist_idx=9, return_all=False):
    """
    Enhanced alignment between IMU and skeleton modalities.
    
    Args:
        imu_data: IMU data of shape (n_imu_samples, n_features)
        skel_data: Skeleton data of shape (n_skel_frames, n_features)
        imu_timestamps: IMU timestamps (or None to create synthetic ones)
        skel_fps: Frame rate for skeleton data
        wrist_idx: Index of wrist joint
        return_all: If True, return all alignment info
        
    Returns:
        Tuple of (aligned_imu, aligned_skel, aligned_timestamps, reference_orientations)
        or dict with all alignment info if return_all=True
    """
    # 1. Create timestamps if they don't exist
    if imu_timestamps is None:
        imu_timestamps = np.arange(len(imu_data)) / 50.0  # Assuming 50Hz IMU
    
    skel_timestamps = np.arange(len(skel_data)) / skel_fps
    
    # 2. Detect activity segments
    try:
        imu_segments = detect_activity_segments(imu_data, threshold=1.5, 
                                            timestamps=imu_timestamps)
        
        skel_segments = detect_activity_segments(skel_data, threshold=0.05,
                                            timestamps=skel_timestamps)
        
        logger.info(f"Detected {len(imu_segments)} IMU segments and {len(skel_segments)} skeleton segments")
    except Exception as e:
        logger.warning(f"Segment detection failed: {e}")
        traceback.print_exc()
        # Basic fallback
        imu_segments = [(imu_timestamps[0], imu_timestamps[-1])]
        skel_segments = [(skel_timestamps[0], skel_timestamps[-1])]
    
    # 3. Try DTW alignment if segments align
    try:
        from dtaidistance import dtw
        
        # Extract wrist trajectory for skeleton
        wrist_trajectory = extract_wrist_trajectory(skel_data, wrist_idx)
        wrist_velocity = calculate_joint_velocity(wrist_trajectory, skel_timestamps)
        wrist_velocity_mag = np.linalg.norm(wrist_velocity, axis=1)
        
        # Calculate IMU magnitude
        imu_feat_dim = min(3, imu_data.shape[1])
        imu_mag = np.linalg.norm(imu_data[:, :imu_feat_dim], axis=1)
        
        # Normalize for DTW
        norm_imu = (imu_mag - np.mean(imu_mag)) / (np.std(imu_mag) + 1e-6)
        norm_vel = (wrist_velocity_mag - np.mean(wrist_velocity_mag)) / (np.std(wrist_velocity_mag) + 1e-6)
        
        # Apply DTW
        path = dtw.warping_path(norm_imu, norm_vel)
        
        # Extract aligned indices
        imu_idx, skel_idx = zip(*path)
        
        # Remove duplicates while preserving order
        imu_unique = []
        skel_unique = []
        seen_imu = set()
        seen_skel = set()
        
        for i, s in zip(imu_idx, skel_idx):
            if i not in seen_imu:
                imu_unique.append(i)
                seen_imu.add(i)
            if s not in seen_skel:
                skel_unique.append(s)
                seen_skel.add(s)
        
        # Create aligned arrays
        aligned_imu = imu_data[imu_unique]
        aligned_skel = skel_data[skel_unique]
        aligned_ts = imu_timestamps[imu_unique]
        
        # Extract reference orientation from skeleton
        from utils.imu_fusion import extract_orientation_from_skeleton
        reference_orientations = extract_orientation_from_skeleton(
            aligned_skel, wrist_idx=wrist_idx
        )
        
        if return_all:
            return {
                'aligned_imu': aligned_imu,
                'aligned_skel': aligned_skel,
                'aligned_timestamps': aligned_ts,
                'reference_orientations': reference_orientations,
                'success': True
            }
        else:
            return aligned_imu, aligned_skel, aligned_ts, reference_orientations
            
    except Exception as e:
        logger.warning(f"DTW alignment failed: {e}, falling back to interpolation")
        traceback.print_exc()
    
    # 4. Try interpolation if DTW fails
    try:
        # Find common time range
        t_min = max(imu_timestamps[0], skel_timestamps[0])
        t_max = min(imu_timestamps[-1], skel_timestamps[-1])
        
        if t_max <= t_min:
            logger.warning("No overlapping time range for modalities")
            if return_all:
                return {
                    'aligned_imu': np.zeros((0, imu_data.shape[1])),
                    'aligned_skel': np.zeros((0, skel_data.shape[1])),
                    'aligned_timestamps': np.zeros(0),
                    'reference_orientations': np.zeros((0, 3)),
                    'success': False
                }
            else:
                return np.zeros((0, imu_data.shape[1])), np.zeros((0, skel_data.shape[1])), np.zeros(0), np.zeros((0, 3))
        
        # Filter to common range
        imu_mask = (imu_timestamps >= t_min) & (imu_timestamps <= t_max)
        filtered_imu = imu_data[imu_mask]
        filtered_ts = imu_timestamps[imu_mask]
        
        # Interpolate skeleton to IMU timestamps
        skel_interp = interp1d(
            skel_timestamps,
            skel_data,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        interp_skel = skel_interp(filtered_ts)
        
        # Extract reference orientation from skeleton
        from utils.imu_fusion import extract_orientation_from_skeleton
        reference_orientations = extract_orientation_from_skeleton(
            interp_skel, wrist_idx=wrist_idx
        )
        
        if return_all:
            return {
                'aligned_imu': filtered_imu,
                'aligned_skel': interp_skel,
                'aligned_timestamps': filtered_ts,
                'reference_orientations': reference_orientations,
                'success': True
            }
        else:
            return filtered_imu, interp_skel, filtered_ts, reference_orientations
    
    except Exception as e:
        logger.warning(f"Interpolation alignment failed: {e}, falling back to basic alignment")
        traceback.print_exc()
    
    # 5. Basic alignment as last resort
    min_len = min(len(imu_data), len(skel_data))
    aligned_imu = imu_data[:min_len]
    aligned_skel = skel_data[:min_len]
    
    if imu_timestamps is not None:
        aligned_ts = imu_timestamps[:min_len]
    else:
        aligned_ts = np.arange(min_len) / 50.0  # Assuming 50Hz
    
    # Extract reference orientation
    try:
        from utils.imu_fusion import extract_orientation_from_skeleton
        reference_orientations = extract_orientation_from_skeleton(
            aligned_skel, wrist_idx=wrist_idx
        )
    except Exception as e:
        logger.warning(f"Failed to extract orientations: {e}")
        reference_orientations = np.zeros((min_len, 3))
    
    if return_all:
        return {
            'aligned_imu': aligned_imu,
            'aligned_skel': aligned_skel,
            'aligned_timestamps': aligned_ts,
            'reference_orientations': reference_orientations,
            'success': True if min_len > 10 else False
        }
    else:
        return aligned_imu, aligned_skel, aligned_ts, reference_orientations
