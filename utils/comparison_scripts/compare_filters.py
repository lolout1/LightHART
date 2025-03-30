#!/usr/bin/env python3
import os, sys, json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from typing import Dict, List, Any
import argparse

def load_filter_results(results_dir, filter_types):
    all_results = []
    for filter_type in filter_types:
        filter_dir = os.path.join(results_dir, f"{filter_type}_model")
        cv_summary_path = os.path.join(filter_dir, "test_summary.json")
        if not os.path.exists(cv_summary_path):
            print(f"Warning: No summary file found for {filter_type}")
            continue
        try:
            with open(cv_summary_path, 'r') as f: summary = json.load(f)
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
            all_results.append(row)
        except Exception as e:
            print(f"Error loading results for {filter_type}: {e}")
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

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

def main():
    parser = argparse.ArgumentParser(description='Compare IMU fusion filter performance')
    parser.add_argument('--results-dir', required=True, help='Results directory')
    parser.add_argument('--output-csv', required=True, help='Output CSV file')
    parser.add_argument('--filter-types', nargs='+', default=['madgwick', 'kalman', 'ekf', 'none'], help='Filter types to compare')
    args = parser.parse_args()
    results_df = load_filter_results(args.results_dir, args.filter_types)
    if not results_df.empty:
        results_df.to_csv(args.output_csv, index=False)
        vis_dir = os.path.join(args.results_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        create_comparison_chart(results_df, vis_dir)
        
        print("\n===== IMU FUSION FILTER COMPARISON =====")
        for _, row in results_df.iterrows():
            filter_type = row['filter_type']
            print(f"\n{filter_type.upper()} FILTER:")
            print(f"  Accuracy:          {row['accuracy']:.2f}% ± {row['accuracy_std']:.2f}%")
            print(f"  F1 Score:          {row['f1']:.2f} ± {row['f1_std']:.2f}")
            print(f"  Precision:         {row['precision']:.2f}% ± {row['precision_std']:.2f}%")
            print(f"  Recall:            {row['recall']:.2f}% ± {row['recall_std']:.2f}%")
            print(f"  Balanced Accuracy: {row['balanced_accuracy']:.2f}% ± {row['balanced_accuracy_std']:.2f}%")
        print("\n=========================================")
        
        best_f1_idx = results_df['f1'].idxmax()
        best_f1_filter = results_df.loc[best_f1_idx, 'filter_type']
        print(f"\nBest performing filter (F1 Score): {best_f1_filter.upper()}")
    else:
        print("No results available to display")

if __name__ == '__main__':
    main()
