# test_imu_fusion.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.imu_fusion import align_sensor_data, process_imu_data, extract_features_from_window

def create_test_data():
    """Create synthetic sensor data for testing"""
    # Create timestamps
    timestamps = np.arange(0, 10, 0.01)  # 10 seconds at 100Hz
    
    # Create synthetic accelerometer data
    acc_data = pd.DataFrame()
    acc_data['timestamp'] = timestamps
    acc_data['x'] = np.sin(timestamps * 5)
    acc_data['y'] = np.cos(timestamps * 5)
    acc_data['z'] = np.sin(timestamps * 2)
    
    # Create synthetic gyroscope data with slightly different sampling
    gyro_timestamps = np.arange(0.005, 10, 0.015)  # Different sampling rate
    gyro_data = pd.DataFrame()
    gyro_data['timestamp'] = gyro_timestamps
    gyro_data['x'] = np.sin(gyro_timestamps * 3)
    gyro_data['y'] = np.cos(gyro_timestamps * 3)
    gyro_data['z'] = np.sin(gyro_timestamps * 1.5)
    
    return acc_data, gyro_data

def test_alignment():
    """Test sensor alignment functionality"""
    print("Testing sensor alignment...")
    acc_data, gyro_data = create_test_data()
    
    # Align the sensor data
    aligned_acc, aligned_gyro, aligned_times = align_sensor_data(acc_data, gyro_data)
    
    print(f"Original acc samples: {len(acc_data)}")
    print(f"Original gyro samples: {len(gyro_data)}")
    print(f"Aligned samples: {len(aligned_acc)}")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc_data['timestamp'], acc_data['x'], 'b.', label='Original Acc X')
    plt.plot(gyro_data['timestamp'], gyro_data['x'], 'r.', label='Original Gyro X')
    plt.legend()
    plt.title('Original Data')
    
    plt.subplot(2, 1, 2)
    plt.plot(aligned_times, aligned_acc[:, 0], 'b-', label='Aligned Acc X')
    plt.plot(aligned_times, aligned_gyro[:, 0], 'r-', label='Aligned Gyro X')
    plt.legend()
    plt.title('Aligned Data')
    
    plt.tight_layout()
    plt.savefig('alignment_test.png')
    plt.close()
    
    print("Alignment test completed. Check 'alignment_test.png' for results.")
    
    return aligned_acc, aligned_gyro, aligned_times

def test_processing():
    """Test IMU data processing functionality"""
    print("Testing IMU data processing...")
    aligned_acc, aligned_gyro, aligned_times = test_alignment()
    
    # Process the aligned data
    results = process_imu_data(
        acc_data=aligned_acc, 
        gyro_data=aligned_gyro, 
        timestamps=aligned_times,
        filter_type='madgwick', 
        return_features=True
    )
    
    # Check the results
    print(f"Quaternion shape: {results['quaternion'].shape}")
    print(f"Linear acceleration shape: {results['linear_acceleration'].shape}")
    print(f"Feature vector size: {len(results['fusion_features'])}")
    
    # Plot quaternion components
    plt.figure(figsize=(12, 6))
    q = results['quaternion']
    plt.plot(aligned_times, q[:, 0], label='w')
    plt.plot(aligned_times, q[:, 1], label='x')
    plt.plot(aligned_times, q[:, 2], label='y')
    plt.plot(aligned_times, q[:, 3], label='z')
    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Value')
    plt.legend()
    plt.title('Quaternion Components')
    plt.savefig('quaternion_test.png')
    plt.close()
    
    print("Processing test completed. Check 'quaternion_test.png' for results.")

if __name__ == "__main__":
    test_processing()
