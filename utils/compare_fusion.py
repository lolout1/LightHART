import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import balanced_accuracy_score, classification_report
from scipy.stats import wilcoxon, friedmanchisquare, ttest_rel
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
import torch
import os
import time
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("filter_evaluation")

def comprehensive_filter_evaluation(filter_results, test_data, output_dir=None, metrics=None, 
                                   significance_threshold=0.05, n_jobs=-1, visualize=True):
    """
    Comprehensive evaluation of IMU fusion filter performance for fall detection.
    
    This function performs a detailed analysis of filter performance, including standard
    classification metrics, fall-specific metrics, statistical significance testing,
    visualization, and feature importance analysis.
    
    Args:
        filter_results: Dictionary mapping filter types to their predictions or models
                        Format: {filter_type: predictions} or {filter_type: model}
        test_data: Dictionary containing test dataset information
                  Required keys: 'features', 'labels', 'timestamps'
                  Optional keys: 'window_indices', 'subject_ids'
        output_dir: Directory to save evaluation results and visualizations
        metrics: List of metrics to calculate (default: accuracy, precision, recall, f1, balanced_accuracy)
        significance_threshold: p-value threshold for statistical significance
        n_jobs: Number of parallel jobs for computation (-1 for all CPUs)
        visualize: Whether to generate and save visualizations
        
    Returns:
        Dictionary containing comprehensive evaluation results:
        - 'metrics': Performance metrics for each filter
        - 'significance': Statistical significance test results
        - 'feature_importance': Feature importance analysis
        - 'plots': Paths to generated visualization files
    """
    start_time = time.time()
    logger.info(f"Starting comprehensive evaluation of {len(filter_results)} filter types")
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
        
    # Set default metrics if not provided
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']
    
    # Check if we have predictions or models
    using_predictions = isinstance(next(iter(filter_results.values())), (np.ndarray, list))
    
    # Initialize results containers
    performance_metrics = {filter_type: {} for filter_type in filter_results}
    predictions = {}
    probabilities = {}
    confusion_matrices = {}
    fall_metrics = {}
    latency_metrics = {}
    plot_paths = {}
    
    # Extract true labels and timestamps
    y_true = test_data['labels']
    timestamps = test_data.get('timestamps', None)
    
    # 1. Get predictions for each filter type
    logger.info("Computing predictions and performance metrics")
    for filter_type, value in filter_results.items():
        if using_predictions:
            # We already have predictions
            y_pred = value
            y_prob = None  # We might not have probability scores
        else:
            # We have a model, need to get predictions
            model = value
            features = test_data['features']
            
            # Check if model uses PyTorch or scikit-learn API
            if hasattr(model, 'predict'):
                # scikit-learn style API
                y_pred = model.predict(features)
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(features)[:, 1]
                    except:
                        y_prob = None
                else:
                    y_prob = None
            else:
                # Assume PyTorch model
                with torch.no_grad():
                    if isinstance(features, np.ndarray):
                        features = torch.FloatTensor(features)
                    outputs = model(features)
                    if isinstance(outputs, torch.Tensor):
                        # Get class predictions
                        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                        # Get probability scores if possible
                        if outputs.shape[1] >= 2:  # Binary or multi-class
                            y_prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        else:
                            y_prob = torch.sigmoid(outputs).cpu().numpy()
                    else:
                        # Handle dictionary output
                        y_pred = outputs.get('predictions', np.zeros_like(y_true))
                        y_prob = outputs.get('probabilities', None)
        
        # Store predictions and probabilities
        predictions[filter_type] = y_pred
        probabilities[filter_type] = y_prob
        
        # 2. Calculate standard metrics
        for metric in metrics:
            if metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, average='binary', zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, average='binary', zero_division=0)
            elif metric == 'f1':
                score = f1_score(y_true, y_pred, average='binary', zero_division=0)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true, y_pred)
            else:
                warnings.warn(f"Unknown metric: {metric}, skipping.")
                continue
                
            performance_metrics[filter_type][metric] = score
        
        # 3. Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        confusion_matrices[filter_type] = cm
        
        # 4. Calculate fall-specific metrics
        fall_metrics[filter_type] = calculate_fall_specific_metrics(y_true, y_pred)
        
        # 5. Calculate latency metrics if timestamps are available
        if timestamps is not None:
            latency_metrics[filter_type] = calculate_detection_latency(
                y_true, y_pred, timestamps, window_indices=test_data.get('window_indices', None)
            )
        
        # Add detailed metrics to performance_metrics
        performance_metrics[filter_type].update(fall_metrics[filter_type])
        if timestamps is not None:
            performance_metrics[filter_type].update(latency_metrics[filter_type])
    
    # 6. Statistical significance testing
    logger.info("Performing statistical significance testing")
    significance_results = perform_significance_testing(
        predictions, y_true, filter_results.keys(), threshold=significance_threshold
    )
    
    # 7. Feature importance analysis (if models are available)
    logger.info("Analyzing feature importance")
    feature_importance = {}
    if not using_predictions and 'feature_names' in test_data:
        feature_importance = analyze_feature_importance(
            filter_results, test_data, output_dir, feature_names=test_data['feature_names']
        )
    
    # 8. Visualizations
    if visualize:
        logger.info("Creating visualizations")
        # Performance metrics comparison
        plot_paths['metrics_comparison'] = create_metrics_comparison_plot(
            performance_metrics, metrics, output_dir
        )
        
        # Confusion matrices visualization
        plot_paths['confusion_matrices'] = create_confusion_matrices_plot(
            confusion_matrices, output_dir
        )
        
        # ROC curves (if probabilities are available)
        if any(probabilities.values()):
            plot_paths['roc_curves'] = create_roc_curves_plot(
                probabilities, y_true, output_dir
            )
        
        # PR curves (if probabilities are available)
        if any(probabilities.values()):
            plot_paths['pr_curves'] = create_precision_recall_curves_plot(
                probabilities, y_true, output_dir
            )
        
        # Fall metrics comparison
        plot_paths['fall_metrics'] = create_fall_metrics_plot(
            fall_metrics, output_dir
        )
        
        # Latency comparison (if available)
        if timestamps is not None:
            plot_paths['latency'] = create_latency_comparison_plot(
                latency_metrics, output_dir
            )
    
    # 9. Save detailed results to CSV
    if output_dir is not None:
        # Convert performance metrics to DataFrame and save
        metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
        metrics_df.to_csv(os.path.join(output_dir, "tables", "performance_metrics.csv"))
        
        # Save significance results
        significance_df = pd.DataFrame(significance_results)
        significance_df.to_csv(os.path.join(output_dir, "tables", "significance_tests.csv"))
        
        # Save feature importance if available
        if feature_importance:
            feature_imp_df = pd.DataFrame(feature_importance)
            feature_imp_df.to_csv(os.path.join(output_dir, "tables", "feature_importance.csv"))
    
    # 10. Generate summary report
    if output_dir is not None:
        generate_summary_report(
            performance_metrics, 
            significance_results,
            confusion_matrices,
            feature_importance,
            plot_paths,
            output_dir
        )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    
    # Return comprehensive results
    return {
        'metrics': performance_metrics,
        'confusion_matrices': confusion_matrices,
        'significance': significance_results,
        'feature_importance': feature_importance,
        'plots': plot_paths
    }

def calculate_fall_specific_metrics(y_true, y_pred):
    """
    Calculate fall-specific performance metrics.
    
    Args:
        y_true: True labels (1 for falls, 0 for non-falls)
        y_pred: Predicted labels
        
    Returns:
        Dictionary of fall-specific metrics
    """
    # Calculate true positives, false positives, etc.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Fall detection rate (also known as sensitivity or recall for falls)
    fall_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # False alarm rate (false positive rate)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Fall detection accuracy (balanced accuracy for falls)
    fall_detection_accuracy = (fall_detection_rate + (1 - false_alarm_rate)) / 2
    
    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Fall F1 score (harmonic mean of precision and recall for falls)
    fall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fall_f1 = 2 * (fall_precision * fall_detection_rate) / (fall_precision + fall_detection_rate) if (fall_precision + fall_detection_rate) > 0 else 0
    
    # Compute the Matthews Correlation Coefficient (MCC)
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 1
    mcc = numerator / denominator
    
    return {
        'fall_detection_rate': fall_detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'fall_detection_accuracy': fall_detection_accuracy,
        'specificity': specificity,
        'fall_precision': fall_precision,
        'fall_f1': fall_f1,
        'mcc': mcc
    }

def calculate_detection_latency(y_true, y_pred, timestamps, window_indices=None):
    """
    Calculate fall detection latency metrics.
    
    Args:
        y_true: True labels (1 for falls, 0 for non-falls)
        y_pred: Predicted labels
        timestamps: Array of timestamps for each window
        window_indices: Optional array indicating the original indices of windows
        
    Returns:
        Dictionary of latency metrics
    """
    if window_indices is None:
        window_indices = np.arange(len(y_true))
    
    # Find fall events (transitions from 0 to 1 in ground truth)
    fall_events = []
    fall_start_times = []
    current_fall = None
    fall_start_time = None
    
    # Group consecutive fall windows
    for i in range(len(y_true)):
        if y_true[i] == 1:  # Fall window
            if current_fall is None:  # New fall starts
                current_fall = [window_indices[i]]
                fall_start_time = timestamps[i]
            else:  # Continue current fall
                current_fall.append(window_indices[i])
        else:  # Non-fall window
            if current_fall is not None:  # Fall event ended
                fall_events.append(current_fall)
                fall_start_times.append(fall_start_time)
                current_fall = None
                fall_start_time = None
    
    # Add last fall event if exists
    if current_fall is not None:
        fall_events.append(current_fall)
        fall_start_times.append(fall_start_time)
    
    # Calculate detection latency for each fall
    latencies = []
    detected_falls = 0
    
    for fall_idx, (fall_windows, start_time) in enumerate(zip(fall_events, fall_start_times)):
        # Check if any window in this fall event was detected
        detected = False
        detection_time = None
        
        for window in fall_windows:
            # Find this window's position in the prediction array
            window_pos = np.where(window_indices == window)[0]
            if len(window_pos) > 0 and y_pred[window_pos[0]] == 1:
                detected = True
                detection_time = timestamps[window_pos[0]]
                break
        
        if detected:
            detected_falls += 1
            latency = max(0, (detection_time - start_time))
            latencies.append(latency)
    
    # Calculate metrics
    detection_rate = detected_falls / len(fall_events) if len(fall_events) > 0 else 0
    mean_latency = np.mean(latencies) if latencies else float('inf')
    median_latency = np.median(latencies) if latencies else float('inf')
    min_latency = np.min(latencies) if latencies else float('inf')
    max_latency = np.max(latencies) if latencies else float('inf')
    
    return {
        'fall_detection_rate': detection_rate,
        'mean_latency_ms': mean_latency * 1000 if mean_latency != float('inf') else float('inf'),
        'median_latency_ms': median_latency * 1000 if median_latency != float('inf') else float('inf'),
        'min_latency_ms': min_latency * 1000 if min_latency != float('inf') else float('inf'),
        'max_latency_ms': max_latency * 1000 if max_latency != float('inf') else float('inf')
    }

def perform_significance_testing(predictions, y_true, filter_types, threshold=0.05):
    """
    Perform statistical significance testing to compare filter performance.
    
    Args:
        predictions: Dictionary mapping filter types to their predictions
        y_true: True labels
        filter_types: List of filter types to compare
        threshold: p-value threshold for significance
        
    Returns:
        List of dictionaries with pairwise comparison results
    """
    results = []
    
    # Compute all pairwise comparisons
    for i, filter1 in enumerate(filter_types):
        for j, filter2 in enumerate(filter_types):
            if i >= j:  # Skip same filter and redundant comparisons
                continue
                
            # Get predictions
            y_pred1 = predictions[filter1]
            y_pred2 = predictions[filter2]
            
            # McNemar's test for paired binary classifications
            try:
                table = np.zeros((2, 2), dtype=int)
                # Both correct
                table[0, 0] = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
                # Filter 1 correct, Filter 2 wrong
                table[0, 1] = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
                # Filter 1 wrong, Filter 2 correct
                table[1, 0] = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
                # Both wrong
                table[1, 1] = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
                
                mcnemar_result = mcnemar(table, exact=True)
                p_value = mcnemar_result.pvalue
                
                # Test null hypothesis that the two filters perform equally well
                # If p < threshold, reject null hypothesis (filters perform differently)
                significant = p_value < threshold
                
                # Determine which filter performed better in case of significance
                better_filter = None
                if significant:
                    # Compare overall accuracy
                    acc1 = accuracy_score(y_true, y_pred1)
                    acc2 = accuracy_score(y_true, y_pred2)
                    better_filter = filter1 if acc1 > acc2 else filter2
                
                results.append({
                    'filter1': filter1,
                    'filter2': filter2,
                    'statistic': mcnemar_result.statistic,
                    'p_value': p_value,
                    'significant': significant,
                    'better_filter': better_filter
                })
            except Exception as e:
                logger.warning(f"Error in McNemar's test for {filter1} vs {filter2}: {e}")
                # Add a placeholder result
                results.append({
                    'filter1': filter1,
                    'filter2': filter2,
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'significant': False,
                    'better_filter': None,
                    'error': str(e)
                })
    
    return results

def analyze_feature_importance(filter_results, test_data, output_dir=None, feature_names=None):
    """
    Analyze feature importance for each filter model.
    
    Args:
        filter_results: Dictionary mapping filter types to their models
        test_data: Dictionary containing test dataset information
        output_dir: Directory to save feature importance visualizations
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping filter types to their feature importance scores
    """
    feature_importance = {}
    
    # If feature names not provided, generate generic names
    if feature_names is None:
        n_features = test_data['features'].shape[1] if 'features' in test_data else 0
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    for filter_type, model in filter_results.items():
        # Extract feature importance if available
        importance_scores = None
        
        # Check different methods for accessing feature importance
        if hasattr(model, 'feature_importances_'):  # Tree-based models
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):  # Linear models
            importance_scores = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        elif hasattr(model, 'feature_importance_'):  # Custom attribute
            importance_scores = model.feature_importance_
        
        # Store importance scores if available
        if importance_scores is not None:
            # Ensure consistent length with feature names
            if len(importance_scores) == len(feature_names):
                # Create dictionary of feature name to importance score
                feature_imp = {name: score for name, score in zip(feature_names, importance_scores)}
                
                # Sort by importance (descending)
                feature_imp = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))
                
                feature_importance[filter_type] = feature_imp
                
                # Create visualization if output directory is provided
                if output_dir is not None:
                    create_feature_importance_plot(feature_imp, filter_type, output_dir)
    
    return feature_importance

def create_metrics_comparison_plot(performance_metrics, metrics, output_dir=None):
    """
    Create a bar plot comparing performance metrics across filters.
    
    Args:
        performance_metrics: Dictionary mapping filter types to their metrics
        metrics: List of metrics to include in the plot
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file or None if not saved
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    filter_types = list(performance_metrics.keys())
    x = np.arange(len(filter_types))
    width = 0.8 / len(metrics)  # Bar width
    
    # Plot each metric as a separate bar
    for i, metric in enumerate(metrics):
        values = [performance_metrics[filter_type].get(metric, 0) for filter_type in filter_types]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    # Customize plot
    plt.xlabel('Filter Type')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x + width * (len(metrics) - 1) / 2, filter_types)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot if output directory provided
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "metrics_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_confusion_matrices_plot(confusion_matrices, output_dir=None):
    """
    Create a visualization of confusion matrices for all filters.
    
    Args:
        confusion_matrices: Dictionary mapping filter types to their confusion matrices
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file or None if not saved
    """
    n_filters = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_filters, figsize=(5*n_filters, 4))
    
    # Handle case with only one filter
    if n_filters == 1:
        axes = [axes]
    
    # Plot each confusion matrix
    for ax, (filter_type, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{filter_type}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(['Non-Fall', 'Fall'])
        ax.set_yticklabels(['Non-Fall', 'Fall'])
    
    plt.tight_layout()
    
    # Save plot if output directory provided
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "confusion_matrices.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_roc_curves_plot(probabilities, y_true, output_dir=None):
    """
    Create ROC curves for all filters.
    
    Args:
        probabilities: Dictionary mapping filter types to their probability predictions
        y_true: True labels
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file or None if not saved
    """
    plt.figure(figsize=(10, 8))
    
    # For each filter with probabilities
    for filter_type, y_prob in probabilities.items():
        if y_prob is None:
            continue
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{filter_type} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # Save plot if output directory provided
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "roc_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_precision_recall_curves_plot(probabilities, y_true, output_dir=None):
    """
    Create Precision-Recall curves for all filters.
    
    Args:
        probabilities: Dictionary mapping filter types to their probability predictions
        y_true: True labels
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file or None if not saved
    """
    plt.figure(figsize=(10, 8))
    
    # For each filter with probabilities
    for filter_type, y_prob in probabilities.items():
        if y_prob is None:
            continue
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, lw=2, label=f'{filter_type} (AUC = {pr_auc:.3f})')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    
    # Save plot if output directory provided
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "precision_recall_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_fall_metrics_plot(fall_metrics, output_dir=None):
    """
    Create a visualization of fall-specific metrics for all filters.
    
    Args:
        fall_metrics: Dictionary mapping filter types to their fall-specific metrics
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file or None if not saved
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    filter_types = list(fall_metrics.keys())
    metrics_to_plot = ['fall_detection_rate', 'false_alarm_rate', 'fall_detection_accuracy', 'fall_f1']
    x = np.arange(len(filter_types))
    width = 0.8 / len(metrics_to_plot)  # Bar width
    
    # Plot each metric as a separate bar
    for i, metric in enumerate(metrics_to_plot):
        values = [fall_metrics[filter_type].get(metric, 0) for filter_type in filter_types]
        plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    # Customize plot
    plt.xlabel('Filter Type')
    plt.ylabel('Score')
    plt.title('Fall-Specific Metrics Comparison')
    plt.xticks(x + width * (len(metrics_to_plot) - 1) / 2, filter_types)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot if output directory provided
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "fall_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_latency_comparison_plot(latency_metrics, output_dir=None):
    """
    Create a visualization of detection latency for all filters.
    
    Args:
        latency_metrics: Dictionary mapping filter types to their latency metrics
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file or None if not saved
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    filter_types = list(latency_metrics.keys())
    latency_data = []
    
    # Extract mean latency values, handling infinite values
    for filter_type in filter_types:
        mean_latency = latency_metrics[filter_type].get('mean_latency_ms', float('inf'))
        if mean_latency == float('inf'):
            mean_latency = None
        latency_data.append(mean_latency)
    
    # Create bar chart
    plt.bar(filter_types, latency_data)
    
    # Customize plot
    plt.xlabel('Filter Type')
    plt.ylabel('Mean Detection Latency (ms)')
    plt.title('Fall Detection Latency Comparison')
    plt.grid(axis='y', alpha=0.3)
    
    # Add detection rate as text on each bar
    for i, filter_type in enumerate(filter_types):
        detection_rate = latency_metrics[filter_type].get('fall_detection_rate', 0)
        plt.text(i, 10, f'DR: {detection_rate:.2f}', ha='center')
    
    # Save plot if output directory provided
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "latency_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_feature_importance_plot(feature_importance, filter_type, output_dir):
    """
    Create a bar plot of feature importance scores.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        filter_type: Name of the filter
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved plot file
    """
    plt.figure(figsize=(12, 8))
    
    # Sort features by importance
    features = list(feature_importance.keys())
    scores = list(feature_importance.values())
    
    # Limit to top 20 features if there are many
    if len(features) > 20:
        indices = np.argsort(scores)[-20:]
        features = [features[i] for i in indices]
        scores = [scores[i] for i in indices]
    
    # Create horizontal bar chart
    plt.barh(features, scores)
    
    # Customize plot
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for {filter_type} Filter')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "plots", f"feature_importance_{filter_type}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_summary_report(performance_metrics, significance_results, confusion_matrices, 
                           feature_importance, plot_paths, output_dir):
    """
    Generate a comprehensive summary report in HTML format.
    
    Args:
        performance_metrics: Dictionary mapping filter types to their metrics
        significance_results: List of pairwise significance test results
        confusion_matrices: Dictionary mapping filter types to their confusion matrices
        feature_importance: Dictionary mapping filter types to their feature importance
        plot_paths: Dictionary mapping plot types to file paths
        output_dir: Directory to save the report
    """
    # Create HTML report
    report_path = os.path.join(output_dir, "filter_comparison_report.html")
    
    with open(report_path, 'w') as f:
        # Write HTML header
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IMU Filter Comparison Report</title>
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
        """)
        
        # Write performance metrics table
        f.write("""
            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Filter Type</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Fall Detection Rate</th>
                    <th>False Alarm Rate</th>
                </tr>
        """)
        
        # Find best filter by F1 score
        best_filter = max(performance_metrics.items(), key=lambda x: x[1].get('f1', 0))[0]
        
        for filter_type, metrics in performance_metrics.items():
            highlight = 'highlight' if filter_type == best_filter else ''
            f.write(f"""
                <tr class="{highlight}">
                    <td>{filter_type}</td>
                    <td>{metrics.get('accuracy', 0):.4f}</td>
                    <td>{metrics.get('f1', 0):.4f}</td>
                    <td>{metrics.get('precision', 0):.4f}</td>
                    <td>{metrics.get('recall', 0):.4f}</td>
                    <td>{metrics.get('fall_detection_rate', 0):.4f}</td>
                    <td>{metrics.get('false_alarm_rate', 0):.4f}</td>
                </tr>
            """)
        
        f.write("</table>")
        
        # Write significance testing results
        if significance_results:
            f.write("""
                <h2>Statistical Significance</h2>
                <p>This table shows pairwise comparisons between filters. Significant differences (p < 0.05) are highlighted.</p>
                <table>
                    <tr>
                        <th>Filter 1</th>
                        <th>Filter 2</th>
                        <th>p-value</th>
                        <th>Significant Difference</th>
                        <th>Better Filter</th>
                    </tr>
            """)
            
            for result in significance_results:
                significant = 'Yes' if result.get('significant', False) else 'No'
                better = result.get('better_filter', 'N/A')
                p_value = result.get('p_value', 1.0)
                
                highlight = 'highlight' if result.get('significant', False) else ''
                
                f.write(f"""
                    <tr class="{highlight}">
                        <td>{result.get('filter1', 'N/A')}</td>
                        <td>{result.get('filter2', 'N/A')}</td>
                        <td>{p_value:.4f}</td>
                        <td>{significant}</td>
                        <td>{better}</td>
                    </tr>
                """)
            
            f.write("</table>")
        
        # Add visualizations
        f.write("<h2>Visualizations</h2>")
        f.write('<div class="container">')
        
        for plot_name, plot_path in plot_paths.items():
            if plot_path:
                # Get relative path for the HTML
                rel_path = os.path.relpath(plot_path, output_dir)
                f.write(f"""
                    <div class="chart">
                        <h3>{plot_name.replace('_', ' ').title()}</h3>
                        <img src="{rel_path}" alt="{plot_name}">
                    </div>
                """)
        
        f.write('</div>')
        
        # Add filter descriptions
        f.write("""
            <h2>Filter Descriptions</h2>
            
            <div class="filter-section">
                <h3>Madgwick Filter</h3>
                <p>The Madgwick filter uses a computationally efficient algorithm for orientation estimation, combining accelerometer and gyroscope data:</p>
                <ul>
                    <li>Uses gradient descent optimization to minimize errors</li>
                    <li>Provides good performance across various motion types</li>
                    <li>Computationally efficient, making it suitable for real-time applications</li>
                    <li>Specifically designed to handle gyroscope drift</li>
                    <li>Widely used in wearable applications</li>
                </ul>
            </div>
            
            <div class="filter-section">
                <h3>Complementary Filter</h3>
                <p>The Complementary filter fuses sensor data in the frequency domain:</p>
                <ul>
                    <li>Uses high-pass filtering for gyroscope data and low-pass for accelerometer</li>
                    <li>Extremely low computational requirements</li>
                    <li>Simple to implement and tune</li>
                    <li>Works well for steady-state motion but may struggle with rapid changes</li>
                    <li>Less effective for complex motions like falls</li>
                </ul>
            </div>
            
            <div class="filter-section">
                <h3>Kalman Filter</h3>
                <p>The standard Kalman filter is a recursive estimator for linear systems:</p>
                <ul>
                    <li>Provides optimal estimation for linear systems with Gaussian noise</li>
                    <li>Moderate computational complexity</li>
                    <li>Handles sensor noise well through statistical modeling</li>
                    <li>Limited ability to handle nonlinearities in orientation tracking</li>
                    <li>Works well for small angle changes</li>
                </ul>
            </div>
            
            <div class="filter-section">
                <h3>Extended Kalman Filter (EKF)</h3>
                <p>The Extended Kalman Filter extends the Kalman filter to nonlinear systems:</p>
                <ul>
                    <li>Linearizes the nonlinear orientation dynamics using Jacobian matrices</li>
                    <li>Better handles quaternion dynamics than the standard Kalman filter</li>
                    <li>Good balance of accuracy and computational cost</li>
                    <li>May diverge during highly nonlinear motions if poorly tuned</li>
                    <li>Can track gyroscope bias effectively</li>
                </ul>
            </div>
            
            <div class="filter-section">
                <h3>Unscented Kalman Filter (UKF)</h3>
                <p>The Unscented Kalman Filter uses deterministic sampling to handle nonlinearities:</p>
                <ul>
                    <li>Uses sigma points to represent the probability distributions</li>
                    <li>Doesn't require explicit Jacobian calculations</li>
                    <li>Better theoretical handling of nonlinearities in fall detection</li>
                    <li>Higher computational requirements</li>
                    <li>More robust to initialization errors and large state changes</li>
                </ul>
            </div>
        """)
        
        # Add recommendations based on best filter
        f.write(f"""
            <h2>Recommendations</h2>
            <p>Based on the performance comparison, the <strong>{best_filter}</strong> filter provides the best overall performance for fall detection.</p>
        """)
        
        # Add conclusion based on which filter performed best
        if best_filter == 'ukf':
            f.write("""
            <p>The Unscented Kalman Filter performs best because:</p>
            <ul>
                <li>It effectively handles the highly nonlinear nature of fall motions</li>
                <li>Its sigma point approach better captures the rapid orientation changes characteristic of falls</li>
                <li>It maintains robustness to sensor noise during high-dynamic movements</li>
                <li>It better preserves the quaternion unit norm constraint throughout orientation tracking</li>
            </ul>
            <p>Despite its higher computational cost, the UKF provides sufficient performance for real-time processing on modern smartwatches, and the accuracy benefits outweigh the additional processing requirements for critical fall detection applications.</p>
            """)
        elif best_filter == 'ekf':
            f.write("""
            <p>The Extended Kalman Filter provides the best balance between accuracy and computational efficiency because:</p>
            <ul>
                <li>Its linearization approach adequately captures fall dynamics while being computationally efficient</li>
                <li>It effectively handles gyroscope drift and bias during orientation tracking</li>
                <li>It's well-suited for the variable sampling rates typical of smartwatch sensors</li>
                <li>It provides better accuracy than simpler filters while being less computationally intensive than the UKF</li>
            </ul>
            <p>The EKF is a good choice for real-time applications on wearable devices with limited processing power and battery constraints.</p>
            """)
        elif best_filter == 'kalman':
            f.write("""
            <p>The standard Kalman Filter performs surprisingly well for fall detection because:</p>
            <ul>
                <li>Its simplicity provides excellent computational efficiency</li>
                <li>For short-duration events like falls, linearization errors are limited</li>
                <li>It's robust to sensor noise, which is significant in consumer-grade IMUs</li>
                <li>It has the lowest computational overhead, making it suitable for battery-constrained devices</li>
            </ul>
            <p>The standard Kalman filter offers a good balance of performance and efficiency, especially when implemented with quaternion corrections to handle orientation constraints.</p>
            """)
        elif best_filter == 'madgwick':
            f.write("""
            <p>The Madgwick Filter performs best because:</p>
            <ul>
                <li>It's specifically designed for IMU orientation tracking with efficiency in mind</li>
                <li>Its approach to handling orientation constraints is effective for fall motion patterns</li>
                <li>It's computationally efficient for real-time processing on constrained devices</li>
                <li>It handles the variably sampled data from smartwatches effectively</li>
                <li>Its gradient descent algorithm provides good convergence during rapid orientation changes</li>
            </ul>
            <p>This filter is a solid choice for wearable applications where battery life and real-time performance are critical considerations.</p>
            """)
        elif best_filter == 'comp':
            f.write("""
            <p>The Complementary Filter performs best because:</p>
            <ul>
                <li>Its frequency-domain approach effectively separates noise from actual motion</li>
                <li>It's extremely lightweight, making it ideal for resource-constrained devices</li>
                <li>It handles the specific motion patterns in this dataset particularly well</li>
                <li>It's simple to implement and maintain in embedded systems</li>
            </ul>
            <p>This filter provides a good balance between computational efficiency and accuracy for this specific application context.</p>
            """)
        
        # Close HTML
        f.write("""
        </body>
        </html>
        """)
    
    logger.info(f"Generated comprehensive report at: {report_path}")
    return report_path
