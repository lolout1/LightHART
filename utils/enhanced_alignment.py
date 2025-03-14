# utils/enhanced_alignment.py

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import logging
from dtaidistance import dtw  # You may need to install this package

logger = logging.getLogger("ModalityAlignment")

def create_skeleton_timestamps(skel_array, fps=30.0):
    """Create timestamps for skeleton data that lacks them."""
    n_frames = skel_array.shape[0]
    timestamps = np.arange(n_frames) / fps
    return timestamps

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

def find_corresponding_segments(imu_segments, skeleton_segments, max_offset=1.0):
    """
    Find corresponding activity segments between IMU and skeleton data.
    
    Args:
        imu_segments: List of (start, end) tuples for IMU data
        skeleton_segments: List of (start, end) tuples for skeleton data
        max_offset: Maximum allowed time difference between segment boundaries
        
    Returns:
        List of matching segment pairs ((imu_start, imu_end), (skel_start, skel_end))
    """
    matching_segments = []
    
    for i_start, i_end in imu_segments:
        for s_start, s_end in skeleton_segments:
            # Check if segments overlap significantly
            overlap_start = max(i_start, s_start)
            overlap_end = min(i_end, s_end)
            
            if overlap_end > overlap_start:
                # Calculate overlap ratio
                imu_duration = i_end - i_start
                skel_duration = s_end - s_start
                overlap_duration = overlap_end - overlap_start
                
                min_duration = min(imu_duration, skel_duration)
                overlap_ratio = overlap_duration / min_duration
                
                # Check if boundaries are reasonably close
                start_diff = abs(i_start - s_start)
                end_diff = abs(i_end - s_end)
                
                if (overlap_ratio > 0.7 and 
                    start_diff < max_offset and 
                    end_diff < max_offset):
                    matching_segments.append(((i_start, i_end), (s_start, s_end)))
    
    return matching_segments

def align_segments_with_dtw(imu_data, skel_data, imu_timestamps, skel_timestamps, 
                           matching_segments, wrist_idx=9):
    """
    Align IMU and skeleton data segments using DTW.
    
    Args:
        imu_data: IMU data of shape (n_imu_samples, n_features)
        skel_data: Skeleton data of shape (n_skel_frames, n_features)
        imu_timestamps: IMU timestamps
        skel_timestamps: Skeleton timestamps
        matching_segments: List of matching segment pairs
        wrist_idx: Index of wrist joint
        
    Returns:
        List of aligned data pairs for each matching segment
    """
    aligned_segments = []
    
    for (i_start, i_end), (s_start, s_end) in matching_segments:
        # Extract segment data
        i_mask = (imu_timestamps >= i_start) & (imu_timestamps <= i_end)
        s_mask = (skel_timestamps >= s_start) & (skel_timestamps <= s_end)
        
        imu_segment = imu_data[i_mask]
        imu_segment_ts = imu_timestamps[i_mask]
        
        skel_segment = skel_data[s_mask]
        skel_segment_ts = skel_timestamps[s_mask]
        
        if len(imu_segment) < 5 or len(skel_segment) < 5:
            logger.warning(f"Segment too short to align: IMU={len(imu_segment)}, Skel={len(skel_segment)}")
            continue
            
        # Extract wrist trajectory and calculate velocity
        wrist_trajectory = extract_wrist_trajectory(skel_segment, wrist_idx)
        wrist_velocity = calculate_joint_velocity(wrist_trajectory, skel_segment_ts)
        wrist_velocity_mag = np.linalg.norm(wrist_velocity, axis=1)
        
        # Calculate acceleration magnitude for IMU
        if imu_segment.shape[1] >= 3:
            accel_mag = np.linalg.norm(imu_segment[:, :3], axis=1)
        else:
            accel_mag = imu_segment[:, 0]
        
        # Normalize signals for DTW
        norm_accel = (accel_mag - np.mean(accel_mag)) / (np.std(accel_mag) + 1e-6)
        norm_vel = (wrist_velocity_mag - np.mean(wrist_velocity_mag)) / (np.std(wrist_velocity_mag) + 1e-6)
        
        # Apply DTW
        try:
            path = dtw.warping_path(norm_accel, norm_vel)
            
            # Extract aligned indices
            imu_idx, skel_idx = zip(*path)
            
            # Remove duplicates while preserving order
            imu_unique_idx = []
            skel_unique_idx = []
            seen_imu = set()
            seen_skel = set()
            
            for i, s in zip(imu_idx, skel_idx):
                if i not in seen_imu:
                    imu_unique_idx.append(i)
                    seen_imu.add(i)
                if s not in seen_skel:
                    skel_unique_idx.append(s)
                    seen_skel.add(s)
            
            # Create aligned arrays
            aligned_imu = imu_segment[imu_unique_idx]
            aligned_imu_ts = imu_segment_ts[imu_unique_idx]
            
            aligned_skel = skel_segment[skel_unique_idx]
            aligned_skel_ts = skel_segment_ts[skel_unique_idx]
            
            aligned_segments.append({
                'imu_data': aligned_imu,
                'imu_timestamps': aligned_imu_ts,
                'skel_data': aligned_skel,
                'skel_timestamps': aligned_skel_ts,
                'start_time': min(aligned_imu_ts[0], aligned_skel_ts[0]),
                'end_time': max(aligned_imu_ts[-1], aligned_skel_ts[-1])
            })
            
        except Exception as e:
            logger.error(f"DTW alignment failed: {e}")
            continue
    
    return aligned_segments

def extract_orientation_from_skeleton(skel_data, timestamps=None, fps=30.0, wrist_idx=9):
    """
    Extract orientation from skeleton wrist trajectory.
    
    Args:
        skel_data: Skeleton data with joint positions
        timestamps: Optional timestamps for skeleton data
        fps: Frame rate for skeleton data (used if timestamps not provided)
        wrist_idx: Index of wrist joint
        
    Returns:
        Tuple of (timestamps, orientations)
    """
    if timestamps is None:
        timestamps = create_skeleton_timestamps(skel_data, fps)
    
    # Extract wrist trajectory
    wrist_trajectory = extract_wrist_trajectory(skel_data, wrist_idx)
    
    # Calculate velocity and acceleration
    wrist_velocity = calculate_joint_velocity(wrist_trajectory, timestamps)
    
    # Calculate acceleration
    wrist_accel = np.zeros_like(wrist_velocity)
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt > 0:
            wrist_accel[i] = (wrist_velocity[i] - wrist_velocity[i-1]) / dt
    
    # Estimate orientation using velocity direction
    orientations = np.zeros((len(skel_data), 3))  # roll, pitch, yaw
    
    for i in range(len(skel_data)):
        # Only calculate orientation if there's sufficient movement
        vel_mag = np.linalg.norm(wrist_velocity[i])
        if vel_mag > 0.1:  # Threshold for meaningful movement
            # Normalize velocity
            vel_dir = wrist_velocity[i] / vel_mag
            
            # Calculate orientation (simplified approach)
            # Pitch (up/down angle)
            pitch = np.arcsin(-vel_dir[1])  # y-axis is typically up
            
            # Yaw (left/right angle)
            yaw = np.arctan2(vel_dir[0], vel_dir[2])
            
            # Roll is harder to estimate from position only
            # Here we're using acceleration direction perpendicular to velocity
            if i > 0 and np.linalg.norm(wrist_accel[i]) > 0.1:
                # Project acceleration onto plane perpendicular to velocity
                acc_perp = wrist_accel[i] - vel_dir * np.dot(wrist_accel[i], vel_dir)
                acc_perp_mag = np.linalg.norm(acc_perp)
                
                if acc_perp_mag > 0.1:
                    # Roll around velocity axis
                    acc_perp_norm = acc_perp / acc_perp_mag
                    roll = np.arctan2(np.dot(acc_perp_norm, np.cross([0, 1, 0], vel_dir)), 
                                      np.dot(acc_perp_norm, [0, 1, 0]))
                else:
                    roll = orientations[i-1, 0]  # Maintain previous roll
            else:
                roll = orientations[i-1, 0] if i > 0 else 0.0
                
            orientations[i] = [roll, pitch, yaw]
        elif i > 0:
            # If no movement, maintain previous orientation
            orientations[i] = orientations[i-1]
    
    return timestamps, orientations

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
    
    skel_timestamps = create_skeleton_timestamps(skel_data, skel_fps)
    
    # 2. Detect activity segments
    imu_segments = detect_activity_segments(imu_data, threshold=1.5, 
                                           timestamps=imu_timestamps)
    
    skel_segments = detect_activity_segments(skel_data, threshold=0.05,
                                           timestamps=skel_timestamps)
    
    logger.info(f"Detected {len(imu_segments)} IMU segments and {len(skel_segments)} skeleton segments")
    
    # 3. Find corresponding segments
    matching_segments = find_corresponding_segments(imu_segments, skel_segments)
    
    logger.info(f"Found {len(matching_segments)} matching segments between modalities")
    
    if not matching_segments:
        logger.warning("No matching segments found - using basic alignment")
        # Fall back to basic alignment (taking the minimum length)
        common_length = min(len(imu_data), len(skel_data))
        aligned_imu = imu_data[:common_length]
        aligned_skel = skel_data[:common_length]
        aligned_ts = imu_timestamps[:common_length]
        
        # Extract reference orientation from skeleton
        ref_timestamps, ref_orientations = extract_orientation_from_skeleton(
            aligned_skel, aligned_ts, wrist_idx=wrist_idx
        )
        
        if return_all:
            return {
                'aligned_imu': aligned_imu,
                'aligned_skel': aligned_skel,
                'aligned_timestamps': aligned_ts,
                'reference_timestamps': ref_timestamps,
                'reference_orientations': ref_orientations,
                'success': False
            }
        else:
            return aligned_imu, aligned_skel, aligned_ts, ref_orientations
    
    # 4. Align segments with DTW
    aligned_segments = align_segments_with_dtw(
        imu_data, skel_data, imu_timestamps, skel_timestamps, 
        matching_segments, wrist_idx
    )
    
    if not aligned_segments:
        logger.warning("Segment alignment failed - using basic alignment")
        # Fall back to basic alignment
        common_length = min(len(imu_data), len(skel_data))
        aligned_imu = imu_data[:common_length]
        aligned_skel = skel_data[:common_length]
        aligned_ts = imu_timestamps[:common_length]
        
        # Extract reference orientation from skeleton
        ref_timestamps, ref_orientations = extract_orientation_from_skeleton(
            aligned_skel, aligned_ts, wrist_idx=wrist_idx
        )
        
        if return_all:
            return {
                'aligned_imu': aligned_imu,
                'aligned_skel': aligned_skel,
                'aligned_timestamps': aligned_ts,
                'reference_timestamps': ref_timestamps,
                'reference_orientations': ref_orientations,
                'success': False
            }
        else:
            return aligned_imu, aligned_skel, aligned_ts, ref_orientations
    
    # 5. Combine aligned segments
    # For simplicity, we'll use the largest segment
    largest_segment = max(aligned_segments, key=lambda x: len(x['imu_data']))
    
    aligned_imu = largest_segment['imu_data']
    aligned_skel = largest_segment['skel_data']
    aligned_ts = largest_segment['imu_timestamps']
    
    # 6. Extract reference orientation from skeleton
    ref_timestamps, ref_orientations = extract_orientation_from_skeleton(
        aligned_skel, largest_segment['skel_timestamps'], wrist_idx=wrist_idx
    )
    
    if return_all:
        return {
            'aligned_imu': aligned_imu,
            'aligned_skel': aligned_skel,
            'aligned_timestamps': aligned_ts,
            'reference_timestamps': ref_timestamps,
            'reference_orientations': ref_orientations,
            'aligned_segments': aligned_segments,
            'success': True
        }
    else:
        return aligned_imu, aligned_skel, aligned_ts, ref_orientations
