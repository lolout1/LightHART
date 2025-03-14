# utils/enhanced_imu_fusion.py (start)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
import logging

logger = logging.getLogger("SensorFusion")

class EnhancedIMUFusionBase:
    """Base class for IMU fusion with drift correction capabilities."""
    
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=0.1, 
                 gyro_bias_noise=0.01, drift_correction_weight=0.1):
        """
        Initialize the IMU fusion base.
        
        Args:
            dt: Default time step in seconds
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            gyro_bias_noise: Gyroscope bias noise variance
            drift_correction_weight: Weight for skeleton-based drift correction
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gyro_bias_noise = gyro_bias_noise
        self.drift_correction_weight = drift_correction_weight
        self.name = "Enhanced Base"
        self.calibrated = False
        
        # Reference data for drift correction
        self.reference_timestamps = None
        self.reference_orientations = None
        self.use_reference = False
        
        # Fall detection specific parameters
        self.fall_detection_features = True  # Enable fall-specific features
        
    def initialize(self):
        """Initialize the filter - implemented by subclasses."""
        raise NotImplementedError
        
    def set_reference_data(self, timestamps, orientations):
        """
        Set reference orientation data from skeleton.
        
        Args:
            timestamps: Array of reference timestamps
            orientations: Array of reference orientations (euler angles)
        """
        if timestamps is None or orientations is None or len(timestamps) == 0:
            self.use_reference = False
            return
            
        self.reference_timestamps = timestamps
        self.reference_orientations = orientations
        self.use_reference = True
        logger.info(f"Reference data set: {len(timestamps)} points")
        
    def get_reference_orientation(self, timestamp):
        """
        Get reference orientation at a specific timestamp.
        
        Args:
            timestamp: Time point to get reference orientation
            
        Returns:
            Reference orientation or None if not available
        """
        if not self.use_reference:
            return None
            
        # Check if timestamp is within range
        if (timestamp < self.reference_timestamps[0] or 
            timestamp > self.reference_timestamps[-1]):
            return None
            
        # Interpolate reference orientation
        interp_func = interp1d(
            self.reference_timestamps,
            self.reference_orientations,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        return interp_func(timestamp)
        
    def apply_drift_correction(self, estimated_orientation, timestamp):
        """
        Apply drift correction using reference orientation.
        
        Args:
            estimated_orientation: Orientation from Kalman filter
            timestamp: Current timestamp
            
        Returns:
            Corrected orientation
        """
        if not self.use_reference:
            return estimated_orientation
            
        reference = self.get_reference_orientation(timestamp)
        if reference is None:
            return estimated_orientation
            
        # Apply weighted correction
        w = self.drift_correction_weight
        corrected = (1 - w) * estimated_orientation + w * reference
        return corrected
    
    def extract_fall_features(self, accel, gyro, orientation, prev_accel=None):
        """
        Extract fall-specific features.
        
        Args:
            accel: Current acceleration (3,)
            gyro: Current angular velocity (3,)
            orientation: Current orientation (3,) (euler angles)
            prev_accel: Previous acceleration or None
            
        Returns:
            Dictionary of fall-specific features
        """
        if not self.fall_detection_features:
            return {}
            
        features = {}
        
        # 1. Vertical acceleration component (using orientation)
        r = R.from_euler('xyz', orientation)
        world_accel = r.apply(accel)
        features['vert_accel'] = world_accel[2]  # Vertical component
        
        # 2. Orientation change rate - useful for detecting sudden posture changes
        features['pitch'] = orientation[1]  # Pitch angle
        
        # 3. Impact detection - sudden changes in acceleration magnitude
        accel_mag = np.linalg.norm(accel)
        features['accel_magnitude'] = accel_mag
        
        # 4. Jerk - rate of acceleration change (first derivative of acceleration)
        if prev_accel is not None:
            jerk = (accel - prev_accel) / self.dt
            features['jerk_magnitude'] = np.linalg.norm(jerk)
        
        return features
