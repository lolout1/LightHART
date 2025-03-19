"""
IMU Fusion Filter Comparison Module

This module provides utilities for comparing different sensor fusion filters
for fall detection applications using wearable sensors. It includes visualization,
analysis, and reporting tools.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import json
import time
import seaborn as sns
import traceback
from scipy.signal import butter, filtfilt
import logging

# Configure logging
logger = logging.getLogger("filter_comparison")

def load_filter_results(results_dir: str) -> pd.DataFrame:
    """
    Load comparison results from CSV file.
    
    Args:
        results_dir: Directory containing the results
        
    Returns:
        DataFrame with filter comparison metrics
    """
    comparison_file = os.path.join(results_dir, "comparison.csv")
    if not os.path.exists(comparison_file):
        logger.error(f"Comparison file not found: {comparison_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(comparison_file)
        logger.info(f"Loaded comparison data with {len(df)} filter types")
        return df
    except Exception as e:
        logger.error(f"Error loading comparison data: {str(e)}")
        return pd.DataFrame()

def create_metric_comparison_plot(df: pd.DataFrame, metric: str, output_dir: str):
    """
    Create a bar plot comparing a specific metric across filter types.
    
    Args:
        df: DataFrame with filter comparison metrics
        metric: Metric to compare ('accuracy', 'f1', etc.)
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by metric value
    sorted_df = df.sort_values(by=metric, ascending=False)
    
    # Create bar plot
    ax = sns.barplot(x='filter_type', y=metric, data=sorted_df, palette='viridis')
    
    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            value + 0.01, 
            f'{value:.4f}', 
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    # Formatting
    plt.title(f'Comparison of {metric.capitalize()} by Filter Type', fontsize=14)
    plt.xlabel('Filter Type', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.ylim(0, min(1.0, df[metric].max() * 1.2))  # Set appropriate y-limit
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{metric}_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Created {metric} comparison plot: {output_path}")
    
    return output_path

def create_combined_metrics_plot(df: pd.DataFrame, output_dir: str):
    """
    Create a combined plot with multiple metrics for each filter type.
    
    Args:
        df: DataFrame with filter comparison metrics
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Sort by overall performance (average of metrics)
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
    df['overall'] = df[metrics].mean(axis=1)
    sorted_df = df.sort_values(by='overall', ascending=False)
    
    # Set up positions for grouped bars
    filter_types = sorted_df['filter_type'].values
    x = np.arange(len(filter_types))
    width = 0.15
    
    # Plot each metric as a group of bars
    for i, metric in enumerate(metrics):
        offset = (i - 2) * width
        plt.bar(x + offset, sorted_df[metric], width, label=metric.capitalize())
    
    # Formatting
    plt.title('Comparison of All Metrics by Filter Type', fontsize=14)
    plt.xlabel('Filter Type', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, filter_types)
    plt.ylim(0, 1.0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(metrics))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "combined_metrics.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Created combined metrics plot: {output_path}")
    
    return output_path

def create_radar_chart(df: pd.DataFrame, output_dir: str):
    """
    Create a radar chart comparing all filters across metrics.
    
    Args:
        df: DataFrame with filter comparison metrics
        output_dir: Directory to save the plot
    """
    # Metrics to include in radar chart
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
    
    # Get filter types
    filter_types = df['filter_type'].values
    
    # Set up radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Set number of angles (metrics)
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Set labels for each angle
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics])
    
    # Set ylim
    ax.set_ylim(0, 1)
    
    # Plot each filter type
    for i, filter_type in enumerate(filter_types):
        # Get values for this filter
        values = df.loc[df['filter_type'] == filter_type, metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, label=filter_type)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Radar Chart of Filter Performance', fontsize=15)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "radar_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Created radar chart: {output_path}")
    
    return output_path

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
    # Import here to avoid circular imports
    from utils.imu_fusion import (
        MadgwickFilter, 
        ComplementaryFilter, 
        KalmanFilter, 
        ExtendedKalmanFilter, 
        UnscentedKalmanFilter
    )
    
    if filter_types is None:
        filter_types = ['madgwick', 'comp', 'kalman', 'ekf', 'ukf']
    
    results = {}
    
    for filter_type in filter_types:
        # Create filter instance
        if filter_type == 'madgwick':
            orientation_filter = MadgwickFilter()
        elif filter_type == 'comp':
            orientation_filter = ComplementaryFilter()
        elif filter_type == 'kalman':
            orientation_filter = KalmanFilter()
        elif filter_type == 'ekf':
            orientation_filter = ExtendedKalmanFilter()
        elif filter_type == 'ukf':
            orientation_filter = UnscentedKalmanFilter()
        else:
            logger.warning(f"Unknown filter type: {filter_type}")
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

def generate_report(df: pd.DataFrame, results_dir: str):
    """
    Generate a detailed HTML report of filter comparison results.
    
    Args:
        df: DataFrame with filter comparison metrics
        results_dir: Directory for saving the report
    """
    # Create output directory for report
    report_dir = os.path.join(results_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Load fold information if available
    fold_data = {}
    for filter_type in df['filter_type'].values:
        cv_summary_path = os.path.join(results_dir, f"{filter_type}_filter", "cv_summary.json")
        if os.path.exists(cv_summary_path):
            try:
                with open(cv_summary_path, 'r') as f:
                    cv_data = json.load(f)
                    fold_data[filter_type] = cv_data.get('fold_metrics', [])
            except Exception as e:
                logger.error(f"Error loading CV summary for {filter_type}: {str(e)}")
    
    # Start HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IMU Filter Comparison for Fall Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .highlight { background-color: #e8f4f8; font-weight: bold; }
            img { max-width: 100%; height: auto; margin: 20px 0; }
            .container { display: flex; flex-wrap: wrap; justify-content: space-between; }
            .chart { width: 48%; margin-bottom: 20px; }
            .filter-section { margin-top: 40px; border-top: 1px solid #eee; padding-top: 20px; }
        </style>
    </head>
    <body>
        <h1>IMU Filter Comparison for Fall Detection</h1>
        <p>This report compares the performance of different IMU fusion filters for fall detection.</p>
    """
    
    # Add summary table
    html_content += """
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Filter Type</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Balanced Accuracy</th>
            </tr>
    """
    
    # Sort by F1 score
    sorted_df = df.sort_values(by='f1', ascending=False)
    best_filter = sorted_df.iloc[0]['filter_type']
    
    for _, row in sorted_df.iterrows():
        filter_type = row['filter_type']
        highlight = 'highlight' if filter_type == best_filter else ''
        
        html_content += f"""
            <tr class="{highlight}">
                <td>{filter_type}</td>
                <td>{row['accuracy']:.4f}</td>
                <td>{row['f1']:.4f}</td>
                <td>{row['precision']:.4f}</td>
                <td>{row['recall']:.4f}</td>
                <td>{row['balanced_accuracy']:.4f}</td>
            </tr>
        """
    
    html_content += """
        </table>
    """
    
    # Create and add visualizations
    html_content += """
        <h2>Visualization</h2>
        <div class="container">
    """
    
    # Create individual metric plots
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']:
        plot_path = create_metric_comparison_plot(df, metric, report_dir)
        rel_path = os.path.basename(plot_path)
        html_content += f"""
            <div class="chart">
                <img src="{rel_path}" alt="{metric} comparison">
            </div>
        """
    
    # Create combined plot and radar chart
    combined_plot = create_combined_metrics_plot(df, report_dir)
    radar_plot = create_radar_chart(df, report_dir)
    
    html_content += f"""
        </div>
        <h3>Combined Metrics</h3>
        <img src="{os.path.basename(combined_plot)}" alt="Combined metrics">
        
        <h3>Radar Chart</h3>
        <img src="{os.path.basename(radar_plot)}" alt="Radar chart">
    """
    
    # Add detailed filter descriptions
    html_content += """
        <h2>Filter Descriptions</h2>
        
        <div class="filter-section">
            <h3>Madgwick Filter</h3>
            <p>The Madgwick filter is a popular orientation filter specifically designed for IMUs. It:</p>
            <ul>
                <li>Uses gradient descent optimization to estimate orientation</li>
                <li>Is computationally efficient, making it suitable for real-time applications</li>
                <li>Provides good performance across various motion types</li>
                <li>Handles the quaternion normalization constraint directly</li>
                <li>Is widely used in wearable applications</li>
            </ul>
        </div>
        
        <div class="filter-section">
            <h3>Complementary Filter</h3>
            <p>The Complementary filter combines sensor data in the frequency domain. It:</p>
            <ul>
                <li>Uses high-pass filtering for gyroscope data and low-pass for accelerometer</li>
                <li>Has very low computational requirements</li>
                <li>Is simple to implement and tune</li>
                <li>May struggle with complex motions like falls</li>
                <li>Works well for steady-state or slowly changing orientations</li>
            </ul>
        </div>
        
        <div class="filter-section">
            <h3>Kalman Filter</h3>
            <p>The standard Kalman filter is a recursive estimator for linear systems. For IMU fusion:</p>
            <ul>
                <li>It provides optimal estimation for linear systems with Gaussian noise</li>
                <li>It has moderate computational complexity</li>
                <li>It handles sensor noise well through statistical modeling</li>
                <li>It has limited ability to handle the nonlinearities in orientation tracking</li>
                <li>It works well for small angle changes where linearization is valid</li>
            </ul>
        </div>
        
        <div class="filter-section">
            <h3>Extended Kalman Filter (EKF)</h3>
            <p>The Extended Kalman Filter extends the standard Kalman filter to nonlinear systems through linearization. For IMU fusion:</p>
            <ul>
                <li>It linearizes the nonlinear orientation dynamics using Jacobian matrices</li>
                <li>It better handles quaternion dynamics than the standard Kalman filter</li>
                <li>It provides a good balance of accuracy and computational cost</li>
                <li>It may diverge in highly nonlinear motions if poorly tuned</li>
                <li>It can track gyroscope bias effectively</li>
            </ul>
        </div>
        
        <div class="filter-section">
            <h3>Unscented Kalman Filter (UKF)</h3>
            <p>The Unscented Kalman Filter uses a deterministic sampling approach to handle nonlinearities. For IMU fusion:</p>
            <ul>
                <li>It uses sigma points to represent the probability distributions</li>
                <li>It doesn't require explicit Jacobian calculations, unlike the EKF</li>
                <li>It provides better theoretical handling of nonlinearities in fall detection</li>
                <li>It has the highest computational requirements among the Kalman variants</li>
                <li>It is more robust to initialization errors and large state changes</li>
            </ul>
        </div>
    """
    
    # Add recommendations based on best filter
    html_content += f"""
        <h2>Recommendations</h2>
        <p>Based on the performance comparison, the <strong>{best_filter}</strong> filter provides the best performance for fall detection with wearable sensors, with the highest F1 score of {sorted_df.iloc[0]['f1']:.4f}.</p>
    """
    
    if best_filter == 'ukf':
        html_content += """
        <p>The Unscented Kalman Filter performs best because:</p>
        <ul>
            <li>It effectively handles the highly nonlinear nature of fall motions without linearization</li>
            <li>Its sigma point approach better captures the rapid orientation changes characteristic of falls</li>
            <li>It maintains robustness to sensor noise during high-dynamic movements</li>
            <li>It better preserves the quaternion unit norm constraint throughout orientation tracking</li>
        </ul>
        <p>Despite its higher computational cost, the UKF provides sufficient performance for real-time processing on modern smartwatches, and the accuracy benefits outweigh the additional processing requirements for critical fall detection applications.</p>
        """
    elif best_filter == 'ekf':
        html_content += """
        <p>The Extended Kalman Filter provides the best balance between accuracy and computational efficiency for fall detection because:</p>
        <ul>
            <li>Its linearization approach adequately captures fall dynamics while being computationally efficient</li>
            <li>It effectively handles gyroscope drift and bias during orientation tracking</li>
            <li>It's well-suited for the variable sampling rates typical of smartwatch sensors</li>
            <li>It provides better accuracy than simpler filters while being less computationally intensive than the UKF</li>
        </ul>
        <p>The EKF is a good choice for real-time applications on wearable devices with limited processing power and battery constraints.</p>
        """
    elif best_filter == 'kalman':
        html_content += """
        <p>The standard Kalman Filter performs surprisingly well for fall detection because:</p>
        <ul>
            <li>Its simplicity provides excellent computational efficiency</li>
            <li>For short-duration events like falls, linearization errors are limited</li>
            <li>It's robust to sensor noise, which is significant in consumer-grade IMUs</li>
            <li>It has the lowest computational overhead, making it suitable for battery-constrained devices</li>
        </ul>
        <p>The standard Kalman filter offers a good balance of performance and efficiency, especially when implemented with quaternion corrections to handle orientation constraints.</p>
        """
    else:  # madgwick or comp
        html_content += """
        <p>The Madgwick/Complementary Filter performs best because:</p>
        <ul>
            <li>It's specifically designed for IMU orientation tracking with efficiency in mind</li>
            <li>Its approach to handling orientation constraints is effective for fall motion patterns</li>
            <li>It's computationally efficient for real-time processing on constrained devices</li>
            <li>It handles the variably sampled data from smartwatches effectively</li>
        </ul>
        <p>This filter is a solid choice for wearable applications where battery life and real-time performance are critical considerations.</p>
        """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write the report
    report_path = os.path.join(report_dir, "filter_comparison_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated detailed HTML report: {report_path}")
    
    return report_path

def summarize_filter_performance(results_dir):
    """
    Generate a comprehensive summary of filter performance
    
    Args:
        results_dir: Directory containing comparison results
    """
    comparison_file = os.path.join(results_dir, "comparison.csv")
    if not os.path.exists(comparison_file):
        print(f"Error: Comparison file not found at {comparison_file}")
        return
    
    # Load comparison data
    df = pd.read_csv(comparison_file)
    
    # Find best filter for each metric
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
    best_filters = {}
    
    for metric in metrics:
        best_idx = df[metric].idxmax()
        best_filters[metric] = {
            'filter': df.loc[best_idx, 'filter_type'],
            'value': df.loc[best_idx, metric]
        }
    
    # Calculate overall ranking
    df['rank_sum'] = 0
    for metric in metrics:
        df[f'rank_{metric}'] = df[metric].rank(ascending=False)
        df['rank_sum'] += df[f'rank_{metric}']
    
    best_overall = df.loc[df['rank_sum'].idxmin(), 'filter_type']
    
    # Print summary
    print("\n===== FILTER PERFORMANCE SUMMARY =====")
    print(f"Best overall filter: {best_overall}")
    print("\nBest filter by metric:")
    for metric, data in best_filters.items():
        print(f"  {metric.capitalize()}: {data['filter']} ({data['value']:.4f})")
    
    print("\nFull comparison:")
    for _, row in df.sort_values('rank_sum').iterrows():
        print(f"  {row['filter_type']}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}, Balanced Acc={row['balanced_accuracy']:.4f}")
    
    # Create full report
    report_path = os.path.join(results_dir, "filter_report.md")
    with open(report_path, 'w') as f:
        f.write("# IMU Filter Comparison for Fall Detection\n\n")
        
        f.write("## Best Performing Filters\n\n")
        f.write(f"**Best overall filter:** {best_overall}\n\n")
        
        f.write("| Metric | Best Filter | Value |\n")
        f.write("|--------|-------------|-------|\n")
        for metric, data in best_filters.items():
            f.write(f"| {metric.capitalize()} | {data['filter']} | {data['value']:.4f} |\n")
        
        f.write("\n## Complete Results\n\n")
        f.write("| Filter | Accuracy | F1 Score | Precision | Recall | Balanced Accuracy |\n")
        f.write("|--------|----------|----------|-----------|--------|------------------|\n")
        
        for _, row in df.sort_values('rank_sum').iterrows():
            f.write(f"| {row['filter_type']} | {row['accuracy']:.4f} | {row['f1']:.4f} | " +
                   f"{row['precision']:.4f} | {row['recall']:.4f} | {row['balanced_accuracy']:.4f} |\n")
        
        f.write("\n## Filter Characteristics\n\n")
        
        # Add detailed descriptions of each filter
        f.write("### Madgwick Filter\n")
        f.write("- Gradient descent-based orientation filter\n")
        f.write("- Computationally efficient\n")
        f.write("- Good performance across various motion types\n")
        f.write("- Widely used in wearable applications\n\n")
        
        f.write("### Complementary Filter\n")
        f.write("- Frequency domain fusion (high-pass for gyro, low-pass for accelerometer)\n")
        f.write("- Very low computational requirements\n")
        f.write("- Simple implementation\n")
        f.write("- May struggle with complex motions\n\n")
        
        f.write("### Kalman Filter\n")
        f.write("- Optimal for linear systems with Gaussian noise\n")
        f.write("- Moderate computational complexity\n")
        f.write("- Handles sensor noise well\n")
        f.write("- Limited ability to handle nonlinearities\n\n")
        
        f.write("### Extended Kalman Filter (EKF)\n")
        f.write("- Nonlinear extension of Kalman filter using local linearization\n")
        f.write("- Better handles quaternion dynamics\n")
        f.write("- Good balance of accuracy and computational cost\n")
        f.write("- May diverge in highly nonlinear motions\n\n")
        
        f.write("### Unscented Kalman Filter (UKF)\n")
        f.write("- Uses sigma points to represent probability distributions\n")
        f.write("- No need for explicit Jacobian calculations\n")
        f.write("- Best theoretical handling of nonlinearities\n")
        f.write("- Highest computational requirements\n\n")
        
        # Add recommendations based on best filter
        f.write("## Recommendations\n\n")
        if best_overall == 'ukf':
            f.write("The Unscented Kalman Filter performs best for fall detection because it effectively handles the highly nonlinear nature of fall motions. Its sigma point approach better captures the rapid orientation changes characteristic of falls, while maintaining robustness to sensor noise.\n\n")
        elif best_overall == 'ekf':
            f.write("The Extended Kalman Filter provides the best balance between accuracy and computational efficiency for fall detection. Its linearization approach adequately captures fall dynamics while being more computationally efficient than the UKF, making it suitable for wearable devices with limited processing power.\n\n")
        elif best_overall == 'madgwick':
            f.write("The Madgwick filter performs best, demonstrating that its gradient descent approach is well-suited for fall detection. Its efficiency and ability to quickly converge to correct orientation estimates make it ideal for real-time applications on wearable devices.\n\n")
        
    print(f"\nDetailed report saved to {report_path}")
    
    return report_path

def analyze_filter_performance(results_dir: str):
    """
    Perform comprehensive analysis of filter performance.
    
    Args:
        results_dir: Directory containing the results
    """
    # Load comparison data
    df = load_filter_results(results_dir)
    
    if df.empty:
        logger.error("No comparison data available for analysis")
        return
    
    # Create visualization directory
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate individual metric plots
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']:
        create_metric_comparison_plot(df, metric, viz_dir)
    
    # Generate combined metrics plot
    create_combined_metrics_plot(df, viz_dir)
    
    # Generate radar chart
    create_radar_chart(df, viz_dir)
    
    # Generate detailed report
    report_path = generate_report(df, results_dir)
    
    # Generate plain markdown report
    markdown_report = summarize_filter_performance(results_dir)
    
    logger.info(f"Filter performance analysis completed. Reports generated at:")
    logger.info(f"  - HTML Report: {report_path}")
    logger.info(f"  - Markdown Report: {markdown_report}")

def process_comparison_results(results_dir: str):
    """
    Process the results of a filter comparison experiment.
    
    Args:
        results_dir: Directory containing the experiment results
        
    Returns:
        Boolean indicating success
    """
    try:
        # Path to the comparison CSV file
        comparison_file = os.path.join(results_dir, "comparison.csv")
        
        if not os.path.exists(comparison_file):
            logger.error(f"Comparison file not found at {comparison_file}")
            return False
        
        # Create visualization directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load data
        df = pd.read_csv(comparison_file)
        
        # Create visualizations
        for metric in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']:
            create_metric_comparison_plot(df, metric, viz_dir)
        
        # Create combined plots
        create_combined_metrics_plot(df, viz_dir)
        create_radar_chart(df, viz_dir)
        
        # Generate report
        report_path = generate_report(df, results_dir)
        summarize_filter_performance(results_dir)
        
        logger.info(f"Filter comparison processing complete. Report at {report_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing comparison results: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process filter comparison results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing comparison results")
    parser.add_argument("--action", type=str, default="process", choices=["process", "analyze", "summarize"],
                        help="Action to perform (process, analyze, summarize)")
    
    args = parser.parse_args()
    
    if args.action == "process":
        process_comparison_results(args.results_dir)
    elif args.action == "analyze":
        analyze_filter_performance(args.results_dir)
    elif args.action == "summarize":
        summarize_filter_performance(args.results_dir)
