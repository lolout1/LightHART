# utils/filter_comparison.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_ind, f_oneway
from sklearn.metrics import confusion_matrix

def load_fold_results(results_dir, filter_type):
    """Load cross-validation results for a specific filter type."""
    try:
        model_dir = os.path.join(results_dir, filter_type)
        
        # Try to load from cv_summary.json first (preferred format)
        summary_path = os.path.join(model_dir, 'cv_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                data = json.load(f)
                return data
        
        # Otherwise try to load from fold_scores.csv
        scores_path = os.path.join(model_dir, 'fold_scores.csv')
        if os.path.exists(scores_path):
            return pd.read_csv(scores_path)
        
        # If no results found, return None
        print(f"No results found for {filter_type}")
        return None
    except Exception as e:
        print(f"Error loading results for {filter_type}: {e}")
        return None

def extract_metrics(data):
    """Extract metrics from loaded result data."""
    metrics = {}
    
    # Handle cv_summary.json format
    if isinstance(data, dict) and 'average_metrics' in data:
        avg = data['average_metrics']
        metrics = {
            'accuracy': avg.get('accuracy', 0),
            'accuracy_std': avg.get('accuracy_std', 0),
            'f1': avg.get('f1', 0),
            'f1_std': avg.get('f1_std', 0),
            'precision': avg.get('precision', 0), 
            'precision_std': avg.get('precision_std', 0),
            'recall': avg.get('recall', 0),
            'recall_std': avg.get('recall_std', 0),
            'balanced_accuracy': avg.get('balanced_accuracy', 0),
            'balanced_accuracy_std': avg.get('balanced_accuracy_std', 0),
        }
        
        # Extract per-fold metrics if available
        if 'fold_metrics' in data:
            metrics['folds'] = data['fold_metrics']
    
    # Handle fold_scores.csv format
    elif isinstance(data, pd.DataFrame):
        metrics = {
            'accuracy': data['accuracy'].mean(),
            'accuracy_std': data['accuracy'].std(),
            'f1': data['f1_score'].mean(),
            'f1_std': data['f1_score'].std(),
            'precision': data['precision'].mean() if 'precision' in data.columns else 0,
            'precision_std': data['precision'].std() if 'precision' in data.columns else 0,
            'recall': data['recall'].mean() if 'recall' in data.columns else 0,
            'recall_std': data['recall'].std() if 'recall' in data.columns else 0,
        }
        metrics['folds'] = data.to_dict('records')
    
    return metrics

def create_metrics_table(filter_metrics):
    """Create a pandas DataFrame of metrics for all filters."""
    table_data = []
    
    for filter_type, metrics in filter_metrics.items():
        row = {
            'Filter Type': filter_type,
            'Accuracy (%)': f"{metrics['accuracy']:.2f} ± {metrics['accuracy_std']:.2f}",
            'F1 Score': f"{metrics['f1']:.2f} ± {metrics['f1_std']:.2f}",
            'Precision (%)': f"{metrics['precision']:.2f} ± {metrics['precision_std']:.2f}",
            'Recall (%)': f"{metrics['recall']:.2f} ± {metrics['recall_std']:.2f}",
            'Balanced Acc (%)': f"{metrics.get('balanced_accuracy', 0):.2f} ± {metrics.get('balanced_accuracy_std', 0):.2f}",
        }
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def plot_metrics_comparison(filter_metrics, save_path):
    """Create a bar chart comparing key metrics across filters."""
    # Prepare data in suitable format for plotting
    filters = list(filter_metrics.keys())
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
    metric_labels = ['Accuracy (%)', 'F1 Score', 'Precision (%)', 'Recall (%)', 'Balanced Acc (%)']
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    width = 0.15  # width of the bars
    x = np.arange(len(metrics))  # the label locations
    
    # Plot each filter as a group of bars
    for i, filter_type in enumerate(filters):
        values = []
        errors = []
        for metric in metrics:
            if metric in filter_metrics[filter_type]:
                values.append(filter_metrics[filter_type][metric])
                errors.append(filter_metrics[filter_type].get(f"{metric}_std", 0))
            else:
                values.append(0)
                errors.append(0)
        
        plt.bar(x + i*width, values, width, label=filter_type.capitalize(), 
                yerr=errors, capsize=5)
    
    # Customize the plot
    plt.ylabel('Score')
    plt.title('Performance Metrics by Filter Type')
    plt.xticks(x + width * (len(filters) - 1) / 2, metric_labels)
    plt.ylim(0, 100)  # Set y-axis to percentage scale
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Metrics comparison chart saved to {save_path}")

def plot_fold_performance(filter_metrics, save_path):
    """Plot per-fold performance across filters."""
    plt.figure(figsize=(15, 8))
    
    # Extract fold data
    data = []
    for filter_type, metrics in filter_metrics.items():
        if 'folds' in metrics:
            for fold in metrics['folds']:
                # Handle both formats (dict and fold_metrics format)
                if isinstance(fold, dict):
                    # Standard dict format
                    if 'fold' in fold:
                        fold_num = fold['fold']
                    else:
                        # Extract from name if possible
                        fold_num = next((i+1 for i, key in enumerate(fold.keys()) 
                                        if 'fold' in str(key).lower()), 0)
                    
                    row = {
                        'Filter Type': filter_type,
                        'Fold': fold_num,
                        'Accuracy': fold.get('accuracy', 0),
                        'F1 Score': fold.get('f1', fold.get('f1_score', 0)),
                        'Precision': fold.get('precision', 0),
                        'Recall': fold.get('recall', 0)
                    }
                    data.append(row)
    
    if not data:
        print("No fold data available for plotting")
        return
    
    df = pd.DataFrame(data)
    
    # Create multiplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        sns.barplot(x='Fold', y=metric, hue='Filter Type', data=df, ax=ax)
        ax.set_title(f'{metric} by Fold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Only include legend for the bottom-right plot
        if i != 3:
            ax.get_legend().remove()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Fold performance chart saved to {save_path}")

def perform_statistical_analysis(filter_metrics):
    """Perform statistical analysis on the results."""
    # Extract fold-level data for analysis
    analysis_data = []
    
    for filter_type, metrics in filter_metrics.items():
        if 'folds' in metrics:
            for fold in metrics['folds']:
                if isinstance(fold, dict):
                    # Extract key metrics
                    row = {
                        'filter_type': filter_type,
                        'fold': fold.get('fold', 0),
                        'accuracy': fold.get('accuracy', 0),
                        'f1': fold.get('f1', fold.get('f1_score', 0)),
                        'precision': fold.get('precision', 0),
                        'recall': fold.get('recall', 0),
                        'balanced_accuracy': fold.get('balanced_accuracy', 0)
                    }
                    analysis_data.append(row)
    
    if not analysis_data:
        return "No fold-level data available for statistical analysis"
    
    df = pd.DataFrame(analysis_data)
    
    # Perform statistical analyses
    results = []
    
    # 1. ANOVA for each metric
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
    for metric in metrics:
        if metric in df.columns:
            # Prepare groups for ANOVA
            groups = [group[metric].values for name, group in df.groupby('filter_type')]
            
            # Only perform ANOVA if we have enough groups
            if len(groups) < 2:
                results.append(f"ANOVA for {metric}: Not enough filter types for comparison")
                continue
            
            try:
                f_val, p_val = stats.f_oneway(*groups)
                significance = "significant" if p_val < 0.05 else "not significant"
                results.append(f"ANOVA for {metric}: F={f_val:.4f}, p={p_val:.4f} ({significance})")
                
                # 2. Pairwise t-tests if ANOVA is significant
                if p_val < 0.05:
                    filter_types = df['filter_type'].unique()
                    for i in range(len(filter_types)):
                        for j in range(i+1, len(filter_types)):
                            type1, type2 = filter_types[i], filter_types[j]
                            try:
                                group1 = df[df['filter_type'] == type1][metric].values
                                group2 = df[df['filter_type'] == type2][metric].values
                                
                                t_stat, t_p = stats.ttest_ind(group1, group2)
                                t_sig = "significant" if t_p < 0.05 else "not significant"
                                results.append(f"  {type1} vs {type2} ({metric}): t={t_stat:.4f}, p={t_p:.4f} ({t_sig})")
                            except Exception as e:
                                results.append(f"  {type1} vs {type2} ({metric}): Error in t-test: {e}")
            except Exception as e:
                results.append(f"ANOVA for {metric}: Error: {e}")
    
    return "\n".join(results)

def plot_confusion_matrices(results_dir, filter_types, save_path):
    """Plot confusion matrices for each filter side by side."""
    cms = {}
    
    for filter_type in filter_types:
        model_dir = os.path.join(results_dir, filter_type)
        cm_path = os.path.join(model_dir, 'confusion_matrix.png')
        
        if os.path.exists(cm_path):
            try:
                # Load the confusion matrix image
                img = plt.imread(cm_path)
                cms[filter_type] = img
            except Exception as e:
                print(f"Error loading confusion matrix for {filter_type}: {e}")
    
    if not cms:
        print("No confusion matrices found")
        return
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(cms), figsize=(5*len(cms), 5))
    
    # If only one filter, axes will not be an array
    if len(cms) == 1:
        axes = [axes]
    
    for i, (filter_type, img) in enumerate(cms.items()):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{filter_type.capitalize()}")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Confusion matrix comparison saved to {save_path}")

def generate_report(filter_metrics, stats_analysis, save_path):
    """Generate a comprehensive report of the results."""
    metrics_table = create_metrics_table(filter_metrics)
    
    with open(save_path, 'w') as f:
        f.write("# IMU Fusion Filter Comparison Report\n\n")
        f.write("## Performance Metrics Summary\n\n")
        f.write(metrics_table.to_markdown(index=False))
        
        f.write("\n\n## Statistical Analysis\n\n")
        f.write("```\n")
        f.write(stats_analysis)
        f.write("\n```\n\n")
        
        f.write("\n\n## Filter Characteristics\n\n")
        f.write("### Madgwick Filter\n")
        f.write("- Computationally efficient quaternion-based orientation filter\n")
        f.write("- Uses gradient descent algorithm to compute orientation from accelerometer and gyroscope\n")
        f.write("- Good at handling gyroscope drift with accelerometer corrections\n")
        
        f.write("\n### Kalman Filter\n")
        f.write("- Classical state estimation algorithm\n")
        f.write("- Optimal for linear systems with Gaussian noise\n")
        f.write("- Balances between gyroscope integration and accelerometer measurements\n")
        
        f.write("\n### Extended Kalman Filter (EKF)\n")
        f.write("- Nonlinear extension of the Kalman filter\n")
        f.write("- Linearizes the state transition and observation models\n")
        f.write("- Better suited for orientation estimation than basic Kalman filter\n")
        
        f.write("\n### Unscented Kalman Filter (UKF)\n")
        f.write("- Advanced nonlinear Kalman filter variant\n")
        f.write("- Uses sigma points to capture mean and covariance through nonlinear transformations\n")
        f.write("- Generally provides better accuracy than EKF at the cost of higher computation\n")
        
        f.write("\n\n## Conclusion\n\n")
        # Find the best filter based on F1 score
        best_filter = max(filter_metrics.items(), key=lambda x: x[1]['f1'])[0]
        f.write(f"Based on the cross-validation results, the **{best_filter.upper()} filter** provides the best overall performance ")
        f.write("for this fall detection task. This is indicated by the highest F1 score, which balances precision and recall.\n\n")
        
        # Generate specific observations based on the metrics
        # Higher accuracy filter
        accuracy_filter = max(filter_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        f.write(f"The {accuracy_filter.upper()} filter achieves the highest accuracy at {filter_metrics[accuracy_filter]['accuracy']:.2f}%.\n\n")
        
        # Higher recall (important for fall detection)
        recall_filter = max(filter_metrics.items(), key=lambda x: x[1]['recall'])[0]
        f.write(f"For maximum sensitivity to falls, the {recall_filter.upper()} filter demonstrates the highest recall ")
        f.write(f"at {filter_metrics[recall_filter]['recall']:.2f}%, making it potentially better at detecting all fall events.\n\n")
        
        # Higher precision
        precision_filter = max(filter_metrics.items(), key=lambda x: x[1]['precision'])[0]
        f.write(f"If minimizing false alarms is a priority, the {precision_filter.upper()} filter shows the highest precision ")
        f.write(f"at {filter_metrics[precision_filter]['precision']:.2f}%, resulting in fewer false positives.\n\n")
        
        f.write("These results highlight the importance of choosing the appropriate filter based on the specific requirements ")
        f.write("of the fall detection application. For a balanced system, the F1 score provides a good metric, while applications ")
        f.write("requiring high sensitivity may prioritize recall, and those needing to minimize false alarms would focus on precision.")
    
    print(f"Comprehensive report saved to {save_path}")

def plot_feature_importance_comparison(results_dir, save_path):
    """Plot comparison of different feature combinations (acc-only, acc-quat, acc-gyro-quat)."""
    try:
        # Look for results from different model configurations
        config_dirs = {}
        
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path):
                # Check for features in the name
                if "acc_only" in item.lower():
                    config_dirs["accelerometer_only"] = item_path
                elif "acc_quat" in item.lower() and "gyro" not in item.lower():
                    config_dirs["acc_quaternion"] = item_path
                elif ("acc_gyro" in item.lower() or "full" in item.lower() or 
                      any(f in item.lower() for f in ["ekf", "madgwick", "kalman", "ukf"])):
                    config_dirs["full_fusion"] = item_path
        
        if not config_dirs:
            print("No feature configuration directories found")
            return
        
        # Load metrics
        feature_metrics = {}
        
        for config_name, config_dir in config_dirs.items():
            if os.path.exists(os.path.join(config_dir, "cv_summary.json")):
                with open(os.path.join(config_dir, "cv_summary.json"), 'r') as f:
                    data = json.load(f)
                    metrics = data.get("average_metrics", {})
                    feature_metrics[config_name] = {
                        "accuracy": metrics.get("accuracy", 0),
                        "f1": metrics.get("f1", 0),
                        "precision": metrics.get("precision", 0),
                        "recall": metrics.get("recall", 0),
                    }
        
        if not feature_metrics:
            print("No valid metrics found for feature configurations")
            return
        
        # Create bar plot for feature importance
        plt.figure(figsize=(12, 8))
        
        configs = list(feature_metrics.keys())
        metrics = ["accuracy", "f1", "precision", "recall"]
        label_map = {
            "accuracy": "Accuracy (%)",
            "f1": "F1 Score",
            "precision": "Precision (%)",
            "recall": "Recall (%)"
        }
        
        x = np.arange(len(metrics))
        width = 0.25
        
        # Create bars
        for i, config in enumerate(configs):
            values = [feature_metrics[config].get(metric, 0) for metric in metrics]
            plt.bar(x + (i - len(configs)/2 + 0.5) * width, values, width, label=config.replace("_", " ").title())
        
        # Customize plot
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Comparison of Feature Combinations')
        plt.xticks(x, [label_map[m] for m in metrics])
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Feature importance comparison saved to {save_path}")
        
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
        import traceback
        traceback.print_exc()

def process_comparison_results(results_dir, filter_types=None):
    """
    Process and visualize results from multiple filter runs.
    
    Args:
        results_dir: Directory containing results subdirectories
        filter_types: Optional list of filter types to analyze (otherwise detect automatically)
    """
    # Create visualization directory
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Detect available filter types if not specified
    if filter_types is None:
        filter_types = []
        for item in os.listdir(results_dir):
            if os.path.isdir(os.path.join(results_dir, item)) and item != "visualizations":
                # Check if this is likely a filter results directory
                if os.path.exists(os.path.join(results_dir, item, "cv_summary.json")) or \
                   os.path.exists(os.path.join(results_dir, item, "fold_scores.csv")):
                    filter_types.append(item)
    
    if not filter_types:
        print("No filter results found for analysis")
        return
    
    print(f"Analyzing results for filters: {', '.join(filter_types)}")
    
    # Load results for each filter type
    filter_metrics = {}
    for filter_type in filter_types:
        data = load_fold_results(results_dir, filter_type)
        if data:
            metrics = extract_metrics(data)
            filter_metrics[filter_type] = metrics
    
    if not filter_metrics:
        print("No valid metrics found for any filter")
        return
    
    # Generate visualizations
    plot_metrics_comparison(filter_metrics, os.path.join(vis_dir, "metrics_comparison.png"))
    plot_fold_performance(filter_metrics, os.path.join(vis_dir, "fold_performance.png"))
    plot_confusion_matrices(results_dir, filter_types, os.path.join(vis_dir, "confusion_matrices.png"))
    
    # Compare feature importance if available
    plot_feature_importance_comparison(results_dir, os.path.join(vis_dir, "feature_importance.png"))
    
    # Statistical analysis
    stats_analysis = perform_statistical_analysis(filter_metrics)
    
    # Generate report
    generate_report(filter_metrics, stats_analysis, os.path.join(results_dir, "filter_comparison_report.md"))
    
    # Save summary CSV
    summary_data = []
    for filter_type, metrics in filter_metrics.items():
        row = {
            'filter_type': filter_type,
            'accuracy': metrics['accuracy'],
            'accuracy_std': metrics['accuracy_std'],
            'f1': metrics['f1'],
            'f1_std': metrics['f1_std'],
            'precision': metrics['precision'],
            'precision_std': metrics['precision_std'],
            'recall': metrics['recall'],
            'recall_std': metrics['recall_std'],
            'balanced_accuracy': metrics.get('balanced_accuracy', 0),
            'balanced_accuracy_std': metrics.get('balanced_accuracy_std', 0)
        }
        summary_data.append(row)
    
    pd.DataFrame(summary_data).to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)
    
    print(f"Analysis complete. Results saved to {results_dir}")
    print(f"Report available at: {os.path.join(results_dir, 'filter_comparison_report.md')}")
