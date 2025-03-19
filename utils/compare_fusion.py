# utils/compare_fusion_methods.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Optional

def load_results(base_dir: str, filter_types: List[str]) -> pd.DataFrame:
    """
    Load and combine results from different filter experiments.
    
    Args:
        base_dir: Base directory containing experiment results
        filter_types: List of filter types to include
        
    Returns:
        DataFrame with combined results
    """
    all_data = []
    
    for filter_type in filter_types:
        # Path to results file
        result_path = os.path.join(base_dir, f"{filter_type}_model", "test_result.txt")
        
        if not os.path.exists(result_path):
            print(f"Warning: No results found for {filter_type} at {result_path}")
            continue
        
        # Load metrics from file
        metrics = {}
        with open(result_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    metric_name = parts[0]
                    metric_value = float(parts[1])
                    metrics[metric_name] = metric_value
        
        # Add filter type to metrics
        metrics['filter_type'] = filter_type
        all_data.append(metrics)
    
    # Convert to DataFrame
    if all_data:
        return pd.DataFrame(all_data)
    else:
        return pd.DataFrame()

def create_comparison_plots(df: pd.DataFrame, output_dir: str):
    """
    Create comparison plots for different filter types.
    
    Args:
        df: DataFrame with filter results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select key metrics to visualize
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'balanced_accuracy']
    
    # Create individual metric plots
    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in results")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Sort by metric value
        sorted_df = df.sort_values(by=metric, ascending=False)
        
        # Create bar plot
        sns.barplot(x='filter_type', y=metric, data=sorted_df, palette='viridis')
        
        # Add value labels
        for i, v in enumerate(sorted_df[metric]):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.title(f'{metric.capitalize()} by Filter Type')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Filter Type')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()
    
    # Create combined plot
    plt.figure(figsize=(12, 8))
    
    # Calculate positions for grouped bars
    filter_types = df['filter_type'].unique()
    x = np.arange(len(filter_types))
    width = 0.15
    
    # Plot each metric as a group
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
            
        values = []
        for filter_type in filter_types:
            value = df.loc[df['filter_type'] == filter_type, metric].values
            values.append(value[0] if len(value) > 0 else 0)
        
        plt.bar(x + (i - len(metrics)/2) * width, values, width, label=metric.capitalize())
    
    plt.xlabel('Filter Type')
    plt.ylabel('Score')
    plt.title('Comparison of Filter Types Across Metrics')
    plt.xticks(x, filter_types)
    plt.legend()
    plt.tight_layout()
    
    # Save combined plot
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))
    plt.close()
    
    # Create radar chart for comprehensive comparison
    plt.figure(figsize=(10, 8))
    
    # Set up radar chart parameters
    categories = metrics
    n = len(categories)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Add category labels
    plt.xticks(angles[:-1], categories)
    
    # Plot each filter type
    for filter_type in filter_types:
        values = []
        for metric in metrics:
            if metric in df.columns:
                value = df.loc[df['filter_type'] == filter_type, metric].values
                values.append(value[0] if len(value) > 0 else 0)
            else:
                values.append(0)
        
        # Close the loop
        values += values[:1]
        
        # Plot this filter type
        ax.plot(angles, values, linewidth=1, label=filter_type)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Filter Type Comparison')
    
    # Save radar chart
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'))
    plt.close()

def create_comparison_report(df: pd.DataFrame, output_dir: str):
    """
    Create a comprehensive report comparing filter types.
    
    Args:
        df: DataFrame with filter results
        output_dir: Directory to save report
    """
    # Identify the best filter for each metric
    best_filters = {}
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'balanced_accuracy']
    
    for metric in metrics:
        if metric in df.columns:
            idx = df[metric].idxmax()
            if idx is not None:
                best_filters[metric] = {
                    'filter': df.loc[idx, 'filter_type'],
                    'value': df.loc[idx, metric]
                }
    
    # Calculate overall score by averaging all metrics
    df['overall_score'] = df[[m for m in metrics if m in df.columns]].mean(axis=1)
    best_idx = df['overall_score'].idxmax()
    best_overall = df.loc[best_idx, 'filter_type'] if best_idx is not None else None
    
    # Create report
    report_path = os.path.join(output_dir, 'filter_comparison_report.md')
    with open(report_path, 'w') as f:
        f.write('# Filter Type Comparison Report\n\n')
        
        if best_overall:
            f.write(f'## Best Overall Filter: {best_overall}\n\n')
        
        f.write('## Best Filter by Metric\n\n')
        f.write('| Metric | Best Filter | Value |\n')
        f.write('|--------|-------------|-------|\n')
        
        for metric, data in best_filters.items():
            f.write(f"| {metric.capitalize()} | {data['filter']} | {data['value']:.4f} |\n")
        
        f.write('\n## All Results\n\n')
        f.write('| Filter Type | ' + ' | '.join(m.capitalize() for m in metrics if m in df.columns) + ' | Overall Score |\n')
        f.write('|------------|-' + '-|-'.join('-' * len(m) for m in metrics if m in df.columns) + '-|---------------|\n')
        
        for _, row in df.sort_values('overall_score', ascending=False).iterrows():
            f.write(f"| {row['filter_type']} | " + 
                   ' | '.join(f"{row[m]:.4f}" for m in metrics if m in df.columns) + 
                   f" | {row['overall_score']:.4f} |\n")
        
        f.write('\n## Filter Characteristics\n\n')
        
        # Add descriptions for each filter type
        f.write('### Madgwick Filter\n')
        f.write('- Gradient descent-based orientation filter\n')
        f.write('- Computationally efficient\n')
        f.write('- Good performance across various motion types\n')
        f.write('- Widely used in wearable applications\n\n')
        
        f.write('### Complementary Filter\n')
        f.write('- Frequency domain fusion (high-pass for gyro, low-pass for accelerometer)\n')
        f.write('- Very low computational requirements\n')
        f.write('- Simple implementation\n')
        f.write('- May struggle with complex motions\n\n')
        
        f.write('### Kalman Filter\n')
        f.write('- Optimal for linear systems with Gaussian noise\n')
        f.write('- Moderate computational complexity\n')
        f.write('- Handles sensor noise well\n')
        f.write('- Limited ability to handle nonlinearities\n\n')
        
        f.write('### Extended Kalman Filter (EKF)\n')
        f.write('- Nonlinear extension of Kalman filter using local linearization\n')
        f.write('- Better handles quaternion dynamics\n')
        f.write('- Good balance of accuracy and computational cost\n')
        f.write('- May diverge in highly nonlinear motions\n\n')
        
        f.write('### Unscented Kalman Filter (UKF)\n')
        f.write('- Uses sigma points to represent probability distributions\n')
        f.write('- No need for explicit Jacobian calculations\n')
        f.write('- Best theoretical handling of nonlinearities\n')
        f.write('- Highest computational requirements\n\n')
        
        # Add recommendations
        f.write('## Recommendations\n\n')
        
        if best_overall == 'ukf':
            f.write('The Unscented Kalman Filter (UKF) provides the best overall performance for fall detection. Its ability to handle the highly nonlinear nature of fall motions without linearization makes it particularly effective. The sigma point approach accurately captures the probability distribution during rapid orientation changes, which is crucial for detecting the sudden movements in falls.\n\n')
            f.write('While the UKF has the highest computational requirements of the filters tested, modern wearable devices likely have sufficient processing power for real-time implementation. For battery-constrained applications, the Extended Kalman Filter (EKF) can be considered as an alternative that balances performance and efficiency.\n')
        elif best_overall == 'ekf':
            f.write('The Extended Kalman Filter (EKF) provides the best balance between accuracy and computational efficiency for fall detection. Its linearization approach adequately captures fall dynamics while being more computationally efficient than the UKF. This makes it suitable for wearable devices with limited processing power.\n\n')
            f.write('The EKF\'s ability to estimate and correct for gyroscope bias is particularly beneficial for long-term monitoring applications where sensor drift can be a concern. For applications requiring the absolute highest detection accuracy regardless of computational cost, the UKF may be considered instead.\n')
        elif best_overall == 'madgwick':
            f.write('The Madgwick filter provides the best overall performance for fall detection, which is notable given its computational simplicity compared to Kalman filter variants. Its gradient descent optimization approach efficiently handles quaternion constraints and quickly converges to accurate orientation estimates during the rapid movements characteristic of falls.\n\n')
            f.write('The filter\'s excellent performance combined with its low computational requirements makes it particularly well-suited for real-time processing on battery-constrained wearable devices. This suggests that for fall detection applications, the simpler approach of the Madgwick filter may be preferable to more complex Kalman-based methods.\n')
        elif best_overall == 'kalman':
            f.write('The standard Kalman filter surprisingly outperforms more complex filters for this fall detection application. This suggests that the linearized motion model is sufficient for capturing the dynamics of falls, at least in the controlled environment of the dataset.\n\n')
            f.write('The Kalman filter\'s balance of good performance and moderate computational complexity makes it an excellent choice for wearable devices. Its robustness to sensor noise is particularly beneficial when working with consumer-grade IMUs found in smartwatches and similar devices.\n')
        elif best_overall == 'comp':
            f.write('The Complementary filter\'s strong performance demonstrates that simple frequency-domain fusion approaches can be effective for fall detection. Its extremely low computational requirements make it ideal for highly resource-constrained devices.\n\n')
            f.write('The filter works by applying a high-pass filter to gyroscope data and a low-pass filter to accelerometer data, effectively separating short-term dynamics from long-term drift. This simple approach appears well-suited to detecting the sudden orientation changes that occur during falls.\n')
    
    print(f"Report saved to {report_path}")
    return report_path

def main(results_dir: str, output_dir: str, filter_types: Optional[List[str]] = None):
    """
    Run the complete analysis pipeline.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save analysis outputs
        filter_types: Optional list of filter types to include (default: all available)
    """
    if filter_types is None:
        filter_types = ['madgwick', 'comp', 'kalman', 'ekf', 'ukf']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from {results_dir}")
    results_df = load_results(results_dir, filter_types)
    
    if results_df.empty:
        print("No results found. Ensure experiments have completed successfully.")
        return
    
    print(f"Found results for {len(results_df)} filter types")
    print(results_df)
    
    print("Creating comparison plots")
    create_comparison_plots(results_df, output_dir)
    
    print("Creating comparison report")
    create_comparison_report(results_df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare different fusion methods")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="fusion_comparison", help="Directory to save analysis outputs")
    parser.add_argument("--filter-types", type=str, nargs="+", help="Filter types to include in comparison")
    
    args = parser.parse_args()
    main(args.results_dir, args.output_dir, args.filter_types)
