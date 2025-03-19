import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import balanced_accuracy_score, classification_report
from scipy.stats import wilcoxon, friedmanchisquare, ttest_rel, chi2
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
import torch
import os
import time
import warnings
import traceback
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("filter_evaluation")

def comprehensive_filter_evaluation(filter_results, test_data, output_dir=None, metrics=None, 
                                   significance_threshold=0.05, n_jobs=-1, visualize=True):
    start_time = time.time()
    logger.info(f"Starting comprehensive evaluation of {len(filter_results)} filter types")
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
        
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']
    
    using_predictions = isinstance(next(iter(filter_results.values())), (np.ndarray, list))
    
    performance_metrics = {filter_type: {} for filter_type in filter_results}
    predictions = {}
    probabilities = {}
    confusion_matrices = {}
    fall_metrics = {}
    latency_metrics = {}
    plot_paths = {}
    
    y_true = test_data['labels']
    timestamps = test_data.get('timestamps', None)
    
    logger.info("Computing predictions and performance metrics")
    for filter_type, value in filter_results.items():
        if using_predictions:
            y_pred = value
            y_prob = None
        else:
            model = value
            features = test_data['features']
            
            if hasattr(model, 'predict'):
                y_pred = model.predict(features)
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(features)[:, 1]
                    except:
                        y_prob = None
                else:
                    y_prob = None
            else:
                with torch.no_grad():
                    if isinstance(features, np.ndarray):
                        features = torch.FloatTensor(features)
                    outputs = model(features)
                    if isinstance(outputs, torch.Tensor):
                        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                        if outputs.shape[1] >= 2:
                            y_prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        else:
                            y_prob = torch.sigmoid(outputs).cpu().numpy()
                    else:
                        y_pred = outputs.get('predictions', np.zeros_like(y_true))
                        y_prob = outputs.get('probabilities', None)
        
        predictions[filter_type] = y_pred
        probabilities[filter_type] = y_prob
        
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
        
        cm = confusion_matrix(y_true, y_pred)
        confusion_matrices[filter_type] = cm
        
        fall_metrics[filter_type] = calculate_fall_specific_metrics(y_true, y_pred)
        
        if timestamps is not None:
            latency_metrics[filter_type] = calculate_detection_latency(
                y_true, y_pred, timestamps, window_indices=test_data.get('window_indices', None)
            )
        
        performance_metrics[filter_type].update(fall_metrics[filter_type])
        if timestamps is not None:
            performance_metrics[filter_type].update(latency_metrics[filter_type])
    
    logger.info("Performing statistical significance testing")
    significance_results = perform_significance_testing(
        predictions, y_true, filter_results.keys(), threshold=significance_threshold
    )
    
    logger.info("Analyzing feature importance")
    feature_importance = {}
    if not using_predictions and 'feature_names' in test_data:
        feature_importance = analyze_feature_importance(
            filter_results, test_data, output_dir, feature_names=test_data['feature_names']
        )
    
    if visualize:
        logger.info("Creating visualizations")
        plot_paths['metrics_comparison'] = create_metrics_comparison_plot(
            performance_metrics, metrics, output_dir
        )
        
        plot_paths['confusion_matrices'] = create_confusion_matrices_plot(
            confusion_matrices, output_dir
        )
        
        if any(probabilities.values()):
            plot_paths['roc_curves'] = create_roc_curves_plot(
                probabilities, y_true, output_dir
            )
        
        if any(probabilities.values()):
            plot_paths['pr_curves'] = create_precision_recall_curves_plot(
                probabilities, y_true, output_dir
            )
        
        plot_paths['fall_metrics'] = create_fall_metrics_plot(
            fall_metrics, output_dir
        )
        
        if timestamps is not None:
            plot_paths['latency'] = create_latency_comparison_plot(
                latency_metrics, output_dir
            )
    
    if output_dir is not None:
        metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
        metrics_df.to_csv(os.path.join(output_dir, "tables", "performance_metrics.csv"))
        
        significance_df = pd.DataFrame(significance_results)
        significance_df.to_csv(os.path.join(output_dir, "tables", "significance_tests.csv"))
        
        if feature_importance:
            feature_imp_df = pd.DataFrame(feature_importance)
            feature_imp_df.to_csv(os.path.join(output_dir, "tables", "feature_importance.csv"))
    
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
    
    return {
        'metrics': performance_metrics,
        'confusion_matrices': confusion_matrices,
        'significance': significance_results,
        'feature_importance': feature_importance,
        'plots': plot_paths
    }

def calculate_fall_specific_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    fall_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fall_detection_accuracy = (fall_detection_rate + (1 - false_alarm_rate)) / 2
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fall_f1 = 2 * (fall_precision * fall_detection_rate) / (fall_precision + fall_detection_rate) if (fall_precision + fall_detection_rate) > 0 else 0
    
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
    if window_indices is None:
        window_indices = np.arange(len(y_true))
    
    fall_events = []
    fall_start_times = []
    current_fall = None
    fall_start_time = None
    
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if current_fall is None:
                current_fall = [window_indices[i]]
                fall_start_time = timestamps[i]
            else:
                current_fall.append(window_indices[i])
        else:
            if current_fall is not None:
                fall_events.append(current_fall)
                fall_start_times.append(fall_start_time)
                current_fall = None
                fall_start_time = None
    
    if current_fall is not None:
        fall_events.append(current_fall)
        fall_start_times.append(fall_start_time)
    
    latencies = []
    detected_falls = 0
    
    for fall_idx, (fall_windows, start_time) in enumerate(zip(fall_events, fall_start_times)):
        detected = False
        detection_time = None
        
        for window in fall_windows:
            window_pos = np.where(window_indices == window)[0]
            if len(window_pos) > 0 and y_pred[window_pos[0]] == 1:
                detected = True
                detection_time = timestamps[window_pos[0]]
                break
        
        if detected:
            detected_falls += 1
            latency = max(0, (detection_time - start_time))
            latencies.append(latency)
    
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
    results = []
    
    for i, filter1 in enumerate(filter_types):
        for j, filter2 in enumerate(filter_types):
            if i >= j:
                continue
                
            y_pred1 = predictions[filter1]
            y_pred2 = predictions[filter2]
            
            try:
                table = np.zeros((2, 2), dtype=int)
                table[0, 0] = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
                table[0, 1] = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
                table[1, 0] = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
                table[1, 1] = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
                
                mcnemar_result = mcnemar(table, exact=True)
                p_value = mcnemar_result.pvalue
                
                significant = p_value < threshold
                
                better_filter = None
                if significant:
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
    feature_importance = {}
    
    if feature_names is None:
        n_features = test_data['features'].shape[1] if 'features' in test_data else 0
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    for filter_type, model in filter_results.items():
        importance_scores = None
        
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        elif hasattr(model, 'feature_importance_'):
            importance_scores = model.feature_importance_
        
        if importance_scores is not None:
            if len(importance_scores) == len(feature_names):
                feature_imp = {name: score for name, score in zip(feature_names, importance_scores)}
                feature_imp = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True))
                feature_importance[filter_type] = feature_imp
                
                if output_dir is not None:
                    create_feature_importance_plot(feature_imp, filter_type, output_dir)
    
    return feature_importance

def create_metrics_comparison_plot(performance_metrics, metrics, output_dir=None):
    plt.figure(figsize=(12, 8))
    
    filter_types = list(performance_metrics.keys())
    x = np.arange(len(filter_types))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        values = [performance_metrics[filter_type].get(metric, 0) for filter_type in filter_types]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    plt.xlabel('Filter Type')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x + width * (len(metrics) - 1) / 2, filter_types)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "metrics_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_confusion_matrices_plot(confusion_matrices, output_dir=None):
    n_filters = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_filters, figsize=(5*n_filters, 4))
    
    if n_filters == 1:
        axes = [axes]
    
    for ax, (filter_type, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{filter_type}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(['Non-Fall', 'Fall'])
        ax.set_yticklabels(['Non-Fall', 'Fall'])
    
    plt.tight_layout()
    
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "confusion_matrices.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_roc_curves_plot(probabilities, y_true, output_dir=None):
    plt.figure(figsize=(10, 8))
    
    for filter_type, y_prob in probabilities.items():
        if y_prob is None:
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{filter_type} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "roc_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_precision_recall_curves_plot(probabilities, y_true, output_dir=None):
    plt.figure(figsize=(10, 8))
    
    for filter_type, y_prob in probabilities.items():
        if y_prob is None:
            continue
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, lw=2, label=f'{filter_type} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "precision_recall_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_fall_metrics_plot(fall_metrics, output_dir=None):
    plt.figure(figsize=(12, 8))
    
    filter_types = list(fall_metrics.keys())
    metrics_to_plot = ['fall_detection_rate', 'false_alarm_rate', 'fall_detection_accuracy', 'fall_f1']
    x = np.arange(len(filter_types))
    width = 0.8 / len(metrics_to_plot)
    
    for i, metric in enumerate(metrics_to_plot):
        values = [fall_metrics[filter_type].get(metric, 0) for filter_type in filter_types]
        plt.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    plt.xlabel('Filter Type')
    plt.ylabel('Score')
    plt.title('Fall-Specific Metrics Comparison')
    plt.xticks(x + width * (len(metrics_to_plot) - 1) / 2, filter_types)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "fall_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_latency_comparison_plot(latency_metrics, output_dir=None):
    plt.figure(figsize=(12, 8))
    
    filter_types = list(latency_metrics.keys())
    latency_data = []
    
    for filter_type in filter_types:
        mean_latency = latency_metrics[filter_type].get('mean_latency_ms', float('inf'))
        if mean_latency == float('inf'):
            mean_latency = None
        latency_data.append(mean_latency)
    
    plt.bar(filter_types, latency_data)
    
    plt.xlabel('Filter Type')
    plt.ylabel('Mean Detection Latency (ms)')
    plt.title('Fall Detection Latency Comparison')
    plt.grid(axis='y', alpha=0.3)
    
    for i, filter_type in enumerate(filter_types):
        detection_rate = latency_metrics[filter_type].get('fall_detection_rate', 0)
        plt.text(i, 10, f'DR: {detection_rate:.2f}', ha='center')
    
    if output_dir is not None:
        plot_path = os.path.join(output_dir, "plots", "latency_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    plt.close()
    return None

def create_feature_importance_plot(feature_importance, filter_type, output_dir):
    plt.figure(figsize=(12, 8))
    
    features = list(feature_importance.keys())
    scores = list(feature_importance.values())
    
    if len(features) > 20:
        indices = np.argsort(scores)[-20:]
        features = [features[i] for i in indices]
        scores = [scores[i] for i in indices]
    
    plt.barh(features, scores)
    
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for {filter_type} Filter')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "plots", f"feature_importance_{filter_type}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_summary_report(performance_metrics, significance_results, confusion_matrices, 
                           feature_importance, plot_paths, output_dir):
    report_path = os.path.join(output_dir, "filter_comparison_report.html")
    
    with open(report_path, 'w') as f:
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
        
        f.write("<h2>Visualizations</h2>")
        f.write('<div class="container">')
        
        for plot_name, plot_path in plot_paths.items():
            if plot_path:
                rel_path = os.path.relpath(plot_path, output_dir)
                f.write(f"""
                    <div class="chart">
                        <h3>{plot_name.replace('_', ' ').title()}</h3>
                        <img src="{rel_path}" alt="{plot_name}">
                    </div>
                """)
        
        f.write('</div>')
        
        f.write("""
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
        """)
        
        f.write(f"""
            <h2>Recommendations</h2>
            <p>Based on the performance comparison, the <strong>{best_filter}</strong> filter provides the best overall performance for fall detection.</p>
        """)
        
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
        
        f.write("""
        </body>
        </html>
        """)
    
    logger.info(f"Generated comprehensive report at: {report_path}")
    return report_path

def process_comparison_results(results_dir):
    try:
        comparison_file = os.path.join(results_dir, "comparison.csv")
        
        if not os.path.exists(comparison_file):
            logger.error(f"Comparison file not found at {comparison_file}")
            return False
        
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        df = pd.read_csv(comparison_file)
        
        logger.info(f"Processing comparison results for {len(df)} filter types")
        
        for metric in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']:
            if metric in df.columns:
                create_metric_comparison_from_df(df, metric, viz_dir)
        
        create_combined_metrics_from_df(df, viz_dir)
        
        available_metrics = [m for m in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy'] if m in df.columns]
        if len(available_metrics) >= 3:
            create_radar_chart_from_df(df, available_metrics, viz_dir)
        
        generate_summary_report_from_df(df, results_dir)
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing comparison results: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def create_metric_comparison_from_df(df, metric, output_dir):
    plt.figure(figsize=(10, 6))
    
    sorted_df = df.sort_values(by=metric, ascending=False)
    
    ax = sns.barplot(x='filter_type', y=metric, data=sorted_df, palette='viridis')
    
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
    
    plt.title(f'Comparison of {metric.capitalize()} by Filter Type', fontsize=14)
    plt.xlabel('Filter Type', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.ylim(0, min(1.0, df[metric].max() * 1.2))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{metric}_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def create_combined_metrics_from_df(df, output_dir):
    plt.figure(figsize=(12, 8))
    
    non_metric_cols = ['model', 'filter_type']
    metrics = [col for col in df.columns if col not in non_metric_cols]
    
    df['overall'] = df[metrics].mean(axis=1)
    sorted_df = df.sort_values(by='overall', ascending=False)
    
    filter_types = sorted_df['filter_type'].values
    x = np.arange(len(filter_types))
    width = 0.15 if len(metrics) <= 5 else 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        if metric == 'overall':
            continue
        offset = (i - (len(metrics)-1)/2) * width
        plt.bar(x + offset, sorted_df[metric], width, label=metric.capitalize())
    
    plt.title('Comparison of All Metrics by Filter Type', fontsize=14)
    plt.xlabel('Filter Type', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, filter_types)
    plt.ylim(0, 1.0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(metrics))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "combined_metrics.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def create_radar_chart_from_df(df, metrics, output_dir):
    filter_types = df['filter_type'].values
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics])
    
    ax.set_ylim(0, 1)
    
    for i, filter_type in enumerate(filter_types):
        values = df.loc[df['filter_type'] == filter_type, metrics].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, label=filter_type)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Radar Chart of Filter Performance', fontsize=15)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "radar_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def generate_summary_report_from_df(df, results_dir):
    report_dir = os.path.join(results_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
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
    
    sorted_df = df.copy()
    if 'f1' in df.columns:
        sorted_df = df.sort_values(by='f1', ascending=False)
    elif 'accuracy' in df.columns:
        sorted_df = df.sort_values(by='accuracy', ascending=False)
    
    best_filter = sorted_df.iloc[0]['filter_type']
    
    for _, row in sorted_df.iterrows():
        filter_type = row['filter_type']
        highlight = 'highlight' if filter_type == best_filter else ''
        
        html_content += f"""
            <tr class="{highlight}">
                <td>{filter_type}</td>
                <td>{row.get('accuracy', 'N/A')}</td>
                <td>{row.get('f1', 'N/A')}</td>
                <td>{row.get('precision', 'N/A')}</td>
                <td>{row.get('recall', 'N/A')}</td>
                <td>{row.get('balanced_accuracy', 'N/A')}</td>
            </tr>
        """
    
    html_content += """
        </table>
    """
    
    html_content += """
        <h2>Visualization</h2>
        <p>The following visualizations compare the performance of different filters:</p>
        
        <div class="container">
            <div class="chart">
                <h3>Combined Metrics</h3>
                <img src="../visualizations/combined_metrics.png" alt="Combined metrics">
            </div>
            
            <div class="chart">
                <h3>Radar Chart</h3>
                <img src="../visualizations/radar_chart.png" alt="Radar chart">
            </div>
        </div>
        
        <h3>Individual Metrics</h3>
        <div class="container">
    """
    
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
    for metric in metrics:
        if metric in df.columns:
            html_content += f"""
                <div class="chart">
                    <img src="../visualizations/{metric}_comparison.png" alt="{metric} comparison">
                </div>
            """
    
    html_content += """
        </div>
    """
    
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
    
    html_content += f"""
        <h2>Recommendations</h2>
        <p>Based on the performance comparison, the <strong>{best_filter}</strong> filter provides the best performance for fall detection with wearable sensors.</p>
    """
    
    if best_filter.lower() == 'ukf':
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
    elif best_filter.lower() == 'ekf':
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
    elif best_filter.lower() == 'kalman':
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
    elif best_filter.lower() == 'madgwick':
        html_content += """
        <p>The Madgwick Filter performs best because:</p>
        <ul>
            <li>It's specifically designed for IMU orientation tracking with efficiency in mind</li>
            <li>Its approach to handling orientation constraints is effective for fall motion patterns</li>
            <li>It's computationally efficient for real-time processing on constrained devices</li>
            <li>It handles the variably sampled data from smartwatches effectively</li>
            <li>Its gradient descent algorithm provides good convergence during rapid orientation changes</li>
        </ul>
        <p>This filter is a solid choice for wearable applications where battery life and real-time performance are critical considerations.</p>
        """
    elif best_filter.lower() == 'comp':
        html_content += """
        <p>The Complementary Filter performs best because:</p>
        <ul>
            <li>Its frequency-domain approach effectively separates noise from actual motion</li>
            <li>It's extremely lightweight, making it ideal for resource-constrained devices</li>
            <li>It handles the specific motion patterns in this dataset particularly well</li>
            <li>It's simple to implement and maintain in embedded systems</li>
        </ul>
        <p>This filter provides a good balance between computational efficiency and accuracy for this specific application context.</p>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    report_path = os.path.join(report_dir, "filter_comparison_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path
