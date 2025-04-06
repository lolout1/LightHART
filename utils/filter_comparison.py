"""
Filter Comparison Utilities for Fall Detection

This module provides utilities for comparing different sensor fusion filters
for fall detection applications using wearable sensors.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json
import time
from scipy.signal import butter, filtfilt

# Import orientation filters
from utils.imu_fusion import (
    MadgwickFilter, 
    KalmanFilter, 
    ExtendedKalmanFilter, 
    process_imu_data
)

def compare_filter_accuracy(acc_data, gyro_data, timestamps=None, filter_types=None):
    """
    Compare the accuracy of different filter types on accelerometer and gyroscope data.
    
    Args:
        acc_data: Accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        timestamps: Optional timestamps for variable rate sampling
        filter_types: List of filter types to compare (default: madgwick, comp, kalman, ekf, ukf)
        
    Returns:
        Dictionary with orientation estimates for each filter type
    """
    if filter_types is None:
        filter_types = ['madgwick', 'comp', 'kalman', 'ekf', 'ukf']
    
    results = {}
    
    for filter_type in filter_types:
        # Create filter instance
        if filter_type == 'madgwick':
            orientation_filter = MadgwickFilter()
        elif filter_type == 'kalman':
            orientation_filter = KalmanFilter()
        elif filter_type == 'ekf':
            orientation_filter = ExtendedKalmanFilter()
        elif filter_type == 'ukf':
            orientation_filter = UnscentedKalmanFilter()
        else:
            print(f"Unknown filter type: {filter_type}")
            continue
        
        # Process data and record time
        start_time = time.time()
        quaternions = []
        
        for i in range(len(acc_data)):
            acc = acc_data[i]
            gyro = gyro_data[i]
            ts = timestamps[i] if timestamps is not None else None
            
            q = orientation_filter.update(acc, gyro, ts)
            quaternions.append(q)
        
        processing_time = time.time() - start_time
        
        # Store results
        results[filter_type] = {
            'quaternions': np.array(quaternions),
            'processing_time': processing_time
        }
    
    return results

def visualize_filter_comparison(filter_results, output_dir=None):
    """
    Visualize the comparison between different orientation filters.
    
    Args:
        filter_results: Dictionary with results from compare_filter_accuracy
        output_dir: Optional directory to save visualizations
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for quaternion components
    plt.figure(figsize=(15, 10))
    
    # Plot quaternion components
    component_names = ['w', 'x', 'y', 'z']
    for i, component in enumerate(component_names):
        plt.subplot(2, 2, i+1)
        
        for filter_type, data in filter_results.items():
            quaternions = data['quaternions']
            plt.plot(quaternions[:, i], label=f'{filter_type}')
        
        plt.title(f'Quaternion {component} Component')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'quaternion_comparison.png'))
    
    # Create processing time comparison
    plt.figure(figsize=(10, 6))
    
    filter_types = list(filter_results.keys())
    processing_times = [data['processing_time'] for data in filter_results.values()]
    
    plt.bar(filter_types, processing_times, color='skyblue')
    plt.title('Processing Time by Filter Type')
    plt.xlabel('Filter Type')
    plt.ylabel('Processing Time (seconds)')
    plt.grid(axis='y', alpha=0.3)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'processing_time_comparison.png'))
    
    # Create Euler angles visualization
    plt.figure(figsize=(15, 10))
    
    for filter_type, data in filter_results.items():
        quaternions = data['quaternions']
        
        # Convert quaternions to Euler angles
        euler_angles = np.zeros((len(quaternions), 3))
        for i, q in enumerate(quaternions):
            # Convert quaternion [w,x,y,z] to scipy rotation [x,y,z,w]
            scipy_q = [q[1], q[2], q[3], q[0]]
            
            # Use scipy to convert to Euler angles
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_quat(scipy_q)
            euler = rotation.as_euler('xyz', degrees=True)
            euler_angles[i] = euler
        
        # Add to results
        filter_results[filter_type]['euler_angles'] = euler_angles
    
    # Plot Euler angles
    angle_names = ['Roll', 'Pitch', 'Yaw']
    for i, angle in enumerate(angle_names):
        plt.subplot(3, 1, i+1)
        
        for filter_type, data in filter_results.items():
            euler_angles = data['euler_angles']
            plt.plot(euler_angles[:, i], label=f'{filter_type}')
        
        plt.title(f'{angle} Angle')
        plt.xlabel('Sample')
        plt.ylabel('Angle (degrees)')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'euler_angles_comparison.png'))
    
    return filter_results

def compare_filter_features(acc_data, gyro_data, timestamps=None, filter_types=None):
    """
    Compare features extracted from different filter types.
    
    Args:
        acc_data: Accelerometer data [n_samples, 3]
        gyro_data: Gyroscope data [n_samples, 3]
        timestamps: Optional timestamps for variable rate sampling
        filter_types: List of filter types to compare
        
    Returns:
        Dictionary with extracted features for each filter type
    """
    if filter_types is None:
        filter_types = ['madgwick', 'comp', 'kalman', 'ekf', 'ukf']
    
    features = {}
    
    for filter_type in filter_types:
        # Process data with specified filter type
        results = process_imu_data(
            acc_data=acc_data,
            gyro_data=gyro_data,
            timestamps=timestamps,
            filter_type=filter_type,
            return_features=True
        )
        
        # Store extracted features
        features[filter_type] = {
            'quaternions': results['quaternion'],
            'linear_acceleration': results['linear_acceleration'],
            'fusion_features': results['fusion_features']
        }
    
    return features

def visualize_feature_comparison(feature_results, output_dir=None):
    """
    Visualize the comparison of features extracted from different filters.
    
    Args:
        feature_results: Dictionary with results from compare_filter_features
        output_dir: Optional directory to save visualizations
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Compare linear acceleration magnitude
    plt.figure(figsize=(12, 6))
    
    for filter_type, data in feature_results.items():
        linear_acc = data['linear_acceleration']
        # Calculate magnitude
        magnitude = np.sqrt(np.sum(linear_acc**2, axis=1))
        plt.plot(magnitude, label=f'{filter_type}')
    
    plt.title('Linear Acceleration Magnitude')
    plt.xlabel('Sample')
    plt.ylabel('Magnitude (m/s²)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'linear_acc_magnitude.png'))
    
    # Compare extracted features
    plt.figure(figsize=(15, 10))
    
    # Get feature descriptions
    feature_names = [
        "Acc Mean X", "Acc Mean Y", "Acc Mean Z",
        "Acc Std X", "Acc Std Y", "Acc Std Z",
        "Acc Max X", "Acc Max Y", "Acc Max Z",
        "Acc Min X", "Acc Min Y", "Acc Min Z",
        "Acc Mag Mean", "Acc Mag Std", "Acc Mag Max",
        "Gyro Mean X", "Gyro Mean Y", "Gyro Mean Z",
        "Gyro Std X", "Gyro Std Y", "Gyro Std Z",
        "Gyro Max X", "Gyro Max Y", "Gyro Max Z",
        "Jerk Mag Mean", "Jerk Mag Max",
        "Euler Mean Roll", "Euler Mean Pitch", "Euler Mean Yaw",
        "Euler Std Roll", "Euler Std Pitch", "Euler Std Yaw",
        "Angle Rate Mean", "Angle Rate Max"
    ]
    
    # Add FFT features
    for axis in ["X", "Y", "Z"]:
        for stat in ["Max", "Mean", "Var"]:
            feature_names.append(f"FFT {axis} {stat}")
    
    # Make sure we have the right number of features
    feature_names = feature_names[:43]  # Limit to 43 features
    
    # Select a subset of key features to visualize
    key_features = [0, 3, 6, 12, 13, 14, 15, 18, 21, 24, 25, 26, 27, 28, 32, 33]
    key_names = [feature_names[i] for i in key_features]
    
    # Create bar chart for each key feature
    x = np.arange(len(key_names))
    width = 0.8 / len(feature_results)
    
    for i, (filter_type, data) in enumerate(feature_results.items()):
        fusion_features = data['fusion_features']
        selected_features = [fusion_features[j] for j in key_features]
        
        offset = width * i - width * (len(feature_results) - 1) / 2
        plt.bar(x + offset, selected_features, width, label=filter_type)
    
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.title('Key Features by Filter Type')
    plt.xticks(x, key_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_comparison.png'))
    
    return feature_results

def compare_filter_results_from_directory(results_dir, filter_types=None):
    """
    Compare the accuracy of different filter types based on training results.
    
    Args:
        results_dir: Directory containing the results for different filters
        filter_types: List of filter types to compare (default: madgwick, comp, kalman, ekf, ukf)
        
    Returns:
        Dictionary with accuracy metrics for each filter
    """
    if filter_types is None:
        filter_types = ['madgwick', 'comp', 'kalman', 'ekf', 'ukf']
    
    results = {}
    
    for filter_type in filter_types:
        # Path to test results file
        test_result_path = os.path.join(results_dir, filter_type, "test_result.txt")
        
        if os.path.exists(test_result_path):
            with open(test_result_path, 'r') as f:
                metrics = {}
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
            
            results[filter_type] = metrics
    
    return results

def create_comparison_visualizations(comparison_file: str, output_dir: str):
    """
    Create visualizations from filter comparison results.
    
    Args:
        comparison_file: Path to CSV file with comparison results
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load comparison data
    try:
        df = pd.read_csv(comparison_file)
        
        # Bar chart for accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(df['filter_type'], df['accuracy'], color='skyblue')
        plt.title('Accuracy by Filter Type')
        plt.xlabel('Filter Type')
        plt.ylabel('Accuracy')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.0)  # Assuming accuracy is between 0 and 1
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
        plt.close()
        
        # Bar chart for F1 score
        plt.figure(figsize=(10, 6))
        plt.bar(df['filter_type'], df['f1_score'], color='lightgreen')
        plt.title('F1 Score by Filter Type')
        plt.xlabel('Filter Type')
        plt.ylabel('F1 Score')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.0)  # Assuming F1 is between 0 and 1
        plt.savefig(os.path.join(output_dir, 'f1_comparison.png'))
        plt.close()
        
        # Combined metrics chart
        plt.figure(figsize=(12, 8))
        width = 0.2
        x = np.arange(len(df['filter_type']))
        
        plt.bar(x - width*1.5, df['accuracy'], width, label='Accuracy', color='skyblue')
        plt.bar(x - width/2, df['f1_score'], width, label='F1 Score', color='lightgreen')
        plt.bar(x + width/2, df['precision'], width, label='Precision', color='salmon')
        plt.bar(x + width*1.5, df['recall'], width, label='Recall', color='gold')
        
        plt.xlabel('Filter Type')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Filter Type')
        plt.xticks(x, df['filter_type'])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        return False

def generate_report(comparison_file: str, output_dir: str):
    """
    Generate a markdown report from filter comparison results.
    
    Args:
        comparison_file: Path to CSV file with comparison results
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report file
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "filter_comparison_report.md")
    
    try:
        # Load comparison data
        df = pd.read_csv(comparison_file)
        
        # Find best filter by F1 score
        best_row = df.loc[df['f1_score'].idxmax()]
        best_filter = best_row['filter_type']
        best_f1 = best_row['f1_score']
        
        with open(report_path, 'w') as f:
            f.write("# Sensor Fusion Filter Comparison Report\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"Best performing filter: **{best_filter.upper()}** with F1 Score of {best_f1:.4f}\n\n")
            
            f.write("## Comparison Table\n\n")
            f.write("| Filter | Accuracy | F1 Score | Precision | Recall |\n")
            f.write("|--------|----------|----------|-----------|--------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['filter_type']} | {row['accuracy']:.4f} | {row['f1_score']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |\n")
            
            f.write("\n\n## Visualizations\n\n")
            f.write("### Combined Metrics\n\n")
            f.write("![Combined Metrics](combined_metrics.png)\n\n")
            
            f.write("### F1 Score Comparison\n\n")
            f.write("![F1 Score Comparison](f1_comparison.png)\n\n")
            
            f.write("## Filter Characteristics\n\n")
            
            f.write("### Madgwick Filter\n")
            f.write("- Fast and efficient orientation tracking\n")
            f.write("- Uses gradient descent optimization\n")
            f.write("- Good for real-time applications\n\n")
            
            
            f.write("### Kalman Filter\n")
            f.write("- Optimal for linear systems with Gaussian noise\n")
            f.write("- Provides good tracking with proper tuning\n")
            f.write("- Moderate computational requirements\n\n")
            
            f.write("### Extended Kalman Filter (EKF)\n")
            f.write("- Handles non-linear systems through linearization\n")
            f.write("- More accurate than basic Kalman for complex motions\n")
            f.write("- Higher computational cost than basic Kalman\n\n")
            
            
            f.write("## Recommendations\n\n")
            f.write(f"Based on the performance analysis, we recommend using the **{best_filter.upper()}** filter for fall detection applications on wearable devices.\n\n")
            
            if best_filter == "madgwick":
                f.write("The Madgwick filter offers an excellent balance between accuracy and computational efficiency, making it ideal for real-time applications on resource-constrained devices like smartwatches.\n\n")
            elif best_filter in ["ekf", "ukf"]:
                f.write("This advanced Kalman filter variant provides superior accuracy for complex motion patterns, which is crucial for reliable fall detection. The higher computational cost may require optimization for deployment on wearable devices.\n\n")
            
        return report_path
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

def process_comparison_results(results_dir: str):
    """
    Process the results of a filter comparison experiment.
    
    Args:
        results_dir: Directory containing the experiment results
        
    Returns:
        Boolean indicating success
    """
    # Path to the comparison CSV file
    comparison_file = os.path.join(results_dir, "comparison.csv")
    
    if not os.path.exists(comparison_file):
        print(f"Error: Comparison file not found at {comparison_file}")
        return False
    
    # Create visualization directory
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create visualizations
    create_comparison_visualizations(comparison_file, viz_dir)
    
    # Generate report
    report_path = generate_report(comparison_file, viz_dir)
    
    if report_path:
        print(f"Filter comparison report generated: {report_path}")
        return True
    else:
        print("Failed to generate filter comparison report")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process filter comparison results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing comparison results")
    
    args = parser.parse_args()
    process_comparison_results(args.results_dir)
