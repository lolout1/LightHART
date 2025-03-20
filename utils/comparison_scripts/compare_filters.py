#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import argparse

def load_filter_results(results_dir, filter_types):
    all_results = []
    for filter_type in filter_types:
        filter_dir = os.path.join(results_dir, f"{filter_type}_model")
        cv_summary_path = os.path.join(filter_dir, "cv_summary.json")
        if not os.path.exists(cv_summary_path):
            print(f"Warning: No summary file found for {filter_type}")
            continue
        try:
            with open(cv_summary_path, 'r') as f:
                summary = json.load(f)
            avg_metrics = summary.get('average_metrics', {})
            row = {
                'filter_type': filter_type,
                'accuracy': avg_metrics.get('accuracy', 0),
                'accuracy_std': avg_metrics.get('accuracy_std', 0),
                'f1': avg_metrics.get('f1', 0),
                'f1_std': avg_metrics.get('f1_std', 0),
                'precision': avg_metrics.get('precision', 0),
                'precision_std': avg_metrics.get('precision_std', 0),
                'recall': avg_metrics.get('recall', 0),
                'recall_std': avg_metrics.get('recall_std', 0),
                'balanced_accuracy': avg_metrics.get('balanced_accuracy', 0),
                'balanced_accuracy_std': avg_metrics.get('balanced_accuracy_std', 0)
            }
            fold_metrics = summary.get('fold_metrics', [])
            for i, fold in enumerate(fold_metrics):
                fold_num = i + 1
                row[f'fold{fold_num}_accuracy'] = fold.get('accuracy', 0)
                row[f'fold{fold_num}_f1'] = fold.get('f1', 0)
                row[f'fold{fold_num}_precision'] = fold.get('precision', 0)
                row[f'fold{fold_num}_recall'] = fold.get('recall', 0)
            all_results.append(row)
        except Exception as e:
            print(f"Error loading results for {filter_type}: {e}")
    return pd.DataFrame(all_results) if all_results else pd.DataFrame(columns=['filter_type', 'accuracy', 'f1', 'precision', 'recall'])

def create_comparison_chart(df, output_dir):
    if df.empty:
        print("No data to visualize")
        return
    plt.figure(figsize=(14, 10))
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
    filters = df['filter_type'].tolist()
    x = np.arange(len(filters))
    width = 0.15
    for i, metric in enumerate(metrics):
        values = df[metric].values
        std_values = df[f'{metric}_std'].values if f'{metric}_std' in df.columns else np.zeros_like(values)
        plt.bar(x + width * (i - len(metrics)/2 + 0.5), values, width, label=metric.capitalize(), yerr=std_values, capsize=3)
    plt.xlabel('Filter Type', fontweight='bold', fontsize=12)
    plt.ylabel('Score (%)', fontweight='bold', fontsize=12)
    plt.title('IMU Fusion Filter Comparison', fontweight='bold', fontsize=16)
    plt.xticks(x, filters, fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, metric in enumerate(metrics):
        for j, value in enumerate(df[metric].values):
            plt.text(j + width * (i - len(metrics)/2 + 0.5), value + 1, f"{value:.1f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'filter_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Comparison chart saved to {output_path}")
    plt.figure(figsize=(15, 10))
    num_folds = sum(1 for col in df.columns if col.startswith('fold') and col.endswith('_f1'))
    for filter_idx, filter_type in enumerate(filters):
        filter_data = df[df['filter_type'] == filter_type]
        fold_f1 = [filter_data[f'fold{i+1}_f1'].values[0] for i in range(num_folds)]
        fold_acc = [filter_data[f'fold{i+1}_accuracy'].values[0] for i in range(num_folds)]
        plt.subplot(len(filters), 2, filter_idx*2 + 1)
        plt.bar(range(1, num_folds+1), fold_f1, color='blue', alpha=0.7)
        plt.title(f'{filter_type} - F1 Score by Fold')
        plt.xlabel('Fold')
        plt.ylabel('F1 Score')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.subplot(len(filters), 2, filter_idx*2 + 2)
        plt.bar(range(1, num_folds+1), fold_acc, color='green', alpha=0.7)
        plt.title(f'{filter_type} - Accuracy by Fold')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fold_output_path = os.path.join(output_dir, 'fold_comparison.png')
    plt.savefig(fold_output_path, dpi=300)
    plt.close()
    print(f"Fold comparison chart saved to {fold_output_path}")

def create_comparison_report(df, output_path):
    if df.empty:
        with open(output_path, 'w') as f:
            f.write("# IMU Fusion Filter Comparison\n\nNo results available.\n")
        return
    with open(output_path, 'w') as f:
        f.write("# IMU Fusion Filter Comparison Results\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Filter Type | Accuracy | F1 Score | Precision | Recall | Balanced Accuracy |\n")
        f.write("|-------------|----------|----------|-----------|--------|------------------|\n")
        for _, row in df.iterrows():
            filter_type = row['filter_type']
            accuracy = f"{row['accuracy']:.2f}% ± {row['accuracy_std']:.2f}%"
            f1 = f"{row['f1']:.2f} ± {row['f1_std']:.2f}"
            precision = f"{row['precision']:.2f}% ± {row['precision_std']:.2f}%"
            recall = f"{row['recall']:.2f}% ± {row['recall_std']:.2f}%"
            bal_acc = f"{row['balanced_accuracy']:.2f}% ± {row['balanced_accuracy_std']:.2f}%"
            f.write(f"| {filter_type} | {accuracy} | {f1} | {precision} | {recall} | {bal_acc} |\n")
        f.write("\n## Best Performing Filter by Metric\n\n")
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        for metric in metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_filter = df.loc[best_idx, 'filter_type']
                best_value = df.loc[best_idx, metric]
                best_std = df.loc[best_idx, f'{metric}_std'] if f'{metric}_std' in df.columns else 0
                f.write(f"- **{metric.capitalize()}**: {best_filter} ({best_value:.2f}% ± {best_std:.2f}%)\n")
        f.write("\n## Fold-by-Fold Results\n\n")
        num_folds = sum(1 for col in df.columns if col.startswith('fold') and col.endswith('_f1'))
        for filter_type in df['filter_type']:
            filter_data = df[df['filter_type'] == filter_type]
            f.write(f"### {filter_type}\n\n")
            f.write("| Fold | Accuracy | F1 Score | Precision | Recall |\n")
            f.write("|------|----------|----------|-----------|--------|\n")
            for fold in range(1, num_folds+1):
                acc = filter_data[f'fold{fold}_accuracy'].values[0]
                f1 = filter_data[f'fold{fold}_f1'].values[0]
                prec = filter_data[f'fold{fold}_precision'].values[0]
                rec = filter_data[f'fold{fold}_recall'].values[0]
                f.write(f"| {fold} | {acc:.2f}% | {f1:.2f} | {prec:.2f}% | {rec:.2f}% |\n")
            f.write("\n")
        f.write("\n## Filter Descriptions\n\n")
        f.write("- **Madgwick**: A computationally efficient orientation filter using gradient descent.\n")
        f.write("- **Kalman**: Standard Kalman filter for optimal sensor fusion.\n")
        f.write("- **EKF**: Extended Kalman Filter for non-linear orientation estimation.\n")
    print(f"Comparison report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare IMU fusion filter performance')
    parser.add_argument('--results-dir', required=True, help='Results directory')
    parser.add_argument('--output-csv', required=True, help='Output CSV file')
    parser.add_argument('--filter-types', nargs='+', default=['madgwick', 'kalman', 'ekf'], help='Filter types to compare')
    args = parser.parse_args()
    results_df = load_filter_results(args.results_dir, args.filter_types)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    vis_dir = os.path.join(args.results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    create_comparison_chart(results_df, vis_dir)
    report_path = os.path.join(vis_dir, 'comparison_report.md')
    create_comparison_report(results_df, report_path)
    print("\n===== IMU FUSION FILTER COMPARISON =====")
    if not results_df.empty:
        for _, row in results_df.iterrows():
            filter_type = row['filter_type']
            print(f"\n{filter_type.upper()} FILTER:")
            print(f"  Accuracy:          {row['accuracy']:.2f}% ± {row['accuracy_std']:.2f}%")
            print(f"  F1 Score:          {row['f1']:.2f} ± {row['f1_std']:.2f}")
            print(f"  Precision:         {row['precision']:.2f}% ± {row['precision_std']:.2f}%")
            print(f"  Recall:            {row['recall']:.2f}% ± {row['recall_std']:.2f}%")
            print(f"  Balanced Accuracy: {row['balanced_accuracy']:.2f}% ± {row['balanced_accuracy_std']:.2f}%")
    else:
        print("No results available to display")
    print("\n=========================================")

if __name__ == '__main__':
    main()
