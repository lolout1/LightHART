#!/bin/bash
#
# Enhanced IMU Filter Comparison Script with Cross-Validation
# - Trains models with three different IMU fusion filters:
#   1. Madgwick (baseline)
#   2. Kalman
#   3. Extended Kalman Filter (EKF)
# - Performs 5-fold cross-validation for each filter
# - Collects and displays comprehensive metrics across all folds

# Enable strict error handling
set -e                  # Exit immediately if a command fails
set -o pipefail         # Pipeline fails if any command fails
set -u                  # Treat unset variables as errors

# Configuration variables
DEVICE="0,1"                # GPU devices
BASE_LR=0.001              # Learning rate
WEIGHT_DECAY=0.001          # Weight decay
NUM_EPOCHS=100               # Number of epochs
PATIENCE=15                 # Early stopping patience
SEED=42                     # Random seed for reproducibility
BATCH_SIZE=16               # Batch size
RESULTS_DIR="filter_comparison_results"  # Results directory
CONFIG_DIR="config/filter_comparison"    # Configuration directory
UTILS_DIR="utils/comparison_scripts"     # Utility scripts directory
REPORT_FILE="${RESULTS_DIR}/comparison_results.csv"  # CSV file for comparison

# Create necessary directories
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/visualizations"
mkdir -p "${CONFIG_DIR}"
mkdir -p "${UTILS_DIR}"

# Function to get current timestamp
timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# Logging function with timestamps
log() {
    local level="$1"
    local msg="$2"
    echo "[$(timestamp)] [${level}] ${msg}"
    echo "[$(timestamp)] [${level}] ${msg}" >> "${RESULTS_DIR}/logs/training.log"
}

# Check if previous command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        log "ERROR" "$1"
        return 1
    fi
    return 0
}

# Create a Python script for aggregating and comparing results
cat > "${UTILS_DIR}/compare_filters.py" << 'EOF'
#!/usr/bin/env python3
"""
Compare performance metrics across different IMU fusion filters.
Aggregates results from cross-validation and generates visualizations.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import argparse

def load_filter_results(results_dir: str, filter_types: List[str]) -> pd.DataFrame:
    """Load results from all filters into a single DataFrame."""
    all_results = []
    
    # Load data from each filter
    for filter_type in filter_types:
        filter_dir = os.path.join(results_dir, f"{filter_type}_model")
        cv_summary_path = os.path.join(filter_dir, "cv_summary.json")
        
        if not os.path.exists(cv_summary_path):
            print(f"Warning: No summary file found for {filter_type} at {cv_summary_path}")
            continue
            
        try:
            # Load summary JSON
            with open(cv_summary_path, 'r') as f:
                summary = json.load(f)
            
            # Extract average metrics
            avg_metrics = summary.get('average_metrics', {})
            
            # Create row for this filter
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
            
            # Add fold-specific data
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
    
    # Convert to DataFrame
    if all_results:
        return pd.DataFrame(all_results)
    else:
        return pd.DataFrame(columns=['filter_type', 'accuracy', 'f1', 'precision', 'recall'])

def create_comparison_chart(df: pd.DataFrame, output_dir: str) -> None:
    """Create comparison charts for all metrics."""
    if df.empty:
        print("No data to visualize")
        return
    
    # Set up figure
    plt.figure(figsize=(14, 10))
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
    
    # Create positions for bars
    filters = df['filter_type'].tolist()
    x = np.arange(len(filters))
    width = 0.15
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = df[metric].values
        std_values = df[f'{metric}_std'].values if f'{metric}_std' in df.columns else np.zeros_like(values)
        
        plt.bar(x + width * (i - len(metrics)/2 + 0.5), values, 
                width, label=metric.capitalize(), 
                yerr=std_values, capsize=3)
    
    # Add labels and legend
    plt.xlabel('Filter Type', fontweight='bold', fontsize=12)
    plt.ylabel('Score (%)', fontweight='bold', fontsize=12)
    plt.title('IMU Fusion Filter Comparison', fontweight='bold', fontsize=16)
    plt.xticks(x, filters, fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, metric in enumerate(metrics):
        for j, value in enumerate(df[metric].values):
            plt.text(j + width * (i - len(metrics)/2 + 0.5), value + 1, 
                     f"{value:.1f}", ha='center', va='bottom', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'filter_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Comparison chart saved to {output_path}")
    
    # Create fold-specific charts
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

def create_comparison_report(df: pd.DataFrame, output_path: str) -> None:
    """Create a detailed comparison report in markdown format."""
    if df.empty:
        with open(output_path, 'w') as f:
            f.write("# IMU Fusion Filter Comparison\n\nNo results available.\n")
        return
        
    with open(output_path, 'w') as f:
        # Write header
        f.write("# IMU Fusion Filter Comparison Results\n\n")
        
        # Write summary table
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
        
        # Write best performer for each metric
        f.write("\n## Best Performing Filter by Metric\n\n")
        
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        for metric in metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_filter = df.loc[best_idx, 'filter_type']
                best_value = df.loc[best_idx, metric]
                best_std = df.loc[best_idx, f'{metric}_std'] if f'{metric}_std' in df.columns else 0
                
                f.write(f"- **{metric.capitalize()}**: {best_filter} ({best_value:.2f}% ± {best_std:.2f}%)\n")
        
        # Write fold-specific results
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
        
        # Add filter descriptions
        f.write("\n## Filter Descriptions\n\n")
        f.write("- **Madgwick**: A computationally efficient orientation filter that uses gradient descent for gyroscope drift correction via accelerometer data.\n")
        f.write("- **Kalman**: Standard Kalman filter implementation that optimally combines accelerometer and gyroscope data using a linearized model.\n")
        f.write("- **EKF (Extended Kalman Filter)**: Advanced Kalman filter that better handles non-linear relationships in orientation estimation using Jacobian matrices.\n")
    
    print(f"Comparison report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare IMU fusion filter performance')
    parser.add_argument('--results-dir', required=True, help='Results directory')
    parser.add_argument('--output-csv', required=True, help='Output CSV file')
    parser.add_argument('--filter-types', nargs='+', default=['madgwick', 'kalman', 'ekf'], 
                        help='Filter types to compare')
    
    args = parser.parse_args()
    
    # Load and combine results
    results_df = load_filter_results(args.results_dir, args.filter_types)
    
    # Save to CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
    # Create visualizations
    vis_dir = os.path.join(args.results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    create_comparison_chart(results_df, vis_dir)
    
    # Create report
    report_path = os.path.join(vis_dir, 'comparison_report.md')
    create_comparison_report(results_df, report_path)
    
    # Print summary to console
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
EOF

# Create config creator function
create_filter_config() {
    local filter_type="$1"
    local output_file="$2"
    
    log "INFO" "Creating configuration for ${filter_type} filter: ${output_file}"
    
    cat > "${output_file}" << EOF
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm

# Include all subjects for comprehensive evaluation
subjects: [29, 30, 31, 33, 45, 46, 34, 37, 39, 38, 43, 35, 36, 44, 32]

model_args:
  num_layers: 3
  embed_dim: 48
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: 64
  mocap_frames: 64
  num_heads: 8
  fusion_type: 'concat'
  dropout: 0.3
  use_batch_norm: true
  feature_dim: 144  # 48 * 3 for concatenation fusion

dataset_args:
  mode: 'sliding_window'
  max_length: 64
  task: 'fd'
  modalities: ['accelerometer', 'gyroscope']
  age_group: ['young']
  sensors: ['watch']
  fusion_options:
    enabled: true
    filter_type: '${filter_type}'
    process_per_window: true
    acc_threshold: 3.0
    gyro_threshold: 1.0
    visualize: false
    save_aligned: true

batch_size: ${BATCH_SIZE}
test_batch_size: ${BATCH_SIZE}
val_batch_size: ${BATCH_SIZE}
num_epoch: ${NUM_EPOCHS}

feeder: Feeder.Make_Dataset.UTD_mm
train_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: true

val_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: true

test_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: false

seed: ${SEED}
optimizer: adamw
base_lr: ${BASE_LR}
weight_decay: ${WEIGHT_DECAY}

kfold:
  enabled: true
  num_folds: 5
  fold_assignments:
    - [43, 35, 36]  # Fold 1: ~38.3% falls
    - [44, 34, 32]  # Fold 2: ~39.7% falls
    - [45, 37, 38]  # Fold 3: ~44.8% falls
    - [46, 29, 31]  # Fold 4: ~41.4% falls
    - [30, 39]      # Fold 5: ~43.3% falls
EOF

    # Check if config file was created successfully
    if [ -f "${output_file}" ]; then
        log "INFO" "Successfully created config file: ${output_file}"
        return 0
    else
        log "ERROR" "Failed to create config file: ${output_file}"
        return 1
    fi
}

# Function to train a model with specific filter
train_filter_model() {
    local filter_type="$1"
    local config_file="$2"
    local output_dir="${RESULTS_DIR}/${filter_type}_model"
    
    log "INFO" "========================================================="
    log "INFO" "STARTING TRAINING FOR ${filter_type^^} FILTER"
    log "INFO" "========================================================="
    
    # Create output directory
    mkdir -p "${output_dir}/logs"
    
    # Check if config file exists
    if [ ! -f "${config_file}" ]; then
        log "ERROR" "Config file does not exist: ${config_file}"
        return 1
    fi
    
    # Run training with cross-validation
    log "INFO" "Training model with ${filter_type} filter"
    CUDA_VISIBLE_DEVICES=${DEVICE} python main.py \
        --config "${config_file}" \
        --work-dir "${output_dir}" \
        --model-saved-name "${filter_type}_model" \
        --device 0 1 \
        --multi-gpu True \
        --kfold True \
        --num-folds 5 \
        --patience ${PATIENCE} \
        --parallel-threads 30 \
        --num-epoch ${NUM_EPOCHS} \
        --run-comparison True 2>&1 | tee "${output_dir}/logs/training.log"
    
    train_status=$?
    if [ ${train_status} -ne 0 ]; then
        log "WARNING" "Training process exited with status ${train_status}"
        
        # Try to recover cross-validation summary if missing
        if [ ! -f "${output_dir}/cv_summary.json" ]; then
            log "INFO" "Attempting to recover cross-validation summary from fold results"
            python "${UTILS_DIR}/recover_cv_summary.py" \
                   --output-dir "${output_dir}" \
                   --filter-type "${filter_type}"
        fi
    fi
    
    # Ensure we have a cv_summary.json for reporting
    if [ ! -f "${output_dir}/cv_summary.json" ]; then
        log "WARNING" "No cross-validation summary found for ${filter_type}"
        # Create minimal summary to avoid breaking the comparison
        echo "{\"filter_type\":\"${filter_type}\",\"average_metrics\":{\"accuracy\":0,\"f1\":0,\"precision\":0,\"recall\":0,\"balanced_accuracy\":0},\"fold_metrics\":[]}" > "${output_dir}/cv_summary.json"
    fi
    
    log "INFO" "Training complete for ${filter_type} filter"
    return 0
}

# Function to create recovery script for CV summary
create_cv_recovery_script() {
    cat > "${UTILS_DIR}/recover_cv_summary.py" << 'EOF'
#!/usr/bin/env python3
"""
Recover cross-validation summary from individual fold results.
"""
import os
import json
import argparse
import numpy as np
import glob
from typing import List, Dict, Any

def load_fold_results(output_dir: str) -> List[Dict[str, Any]]:
    fold_metrics = []
    
    fold_dirs = sorted(glob.glob(os.path.join(output_dir, "fold_*")))
    for i, fold_dir in enumerate(fold_dirs, 1):
        results_file = os.path.join(fold_dir, "test_results.json")
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Add fold number
                results["fold"] = i
                
                fold_metrics.append(results)
                print(f"Loaded results from {results_file}")
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
    
    return fold_metrics

def create_cv_summary(fold_metrics: List[Dict[str, Any]], filter_type: str) -> Dict[str, Any]:
    if not fold_metrics:
        return {
            "filter_type": filter_type,
            "average_metrics": {
                "accuracy": 0,
                "f1": 0,
                "precision": 0,
                "recall": 0,
                "balanced_accuracy": 0
            },
            "fold_metrics": []
        }
    
    # Calculate average metrics
    metrics = ["accuracy", "f1", "precision", "recall", "balanced_accuracy"]
    avg_metrics = {}
    
    for metric in metrics:
        values = [fold.get(metric, 0) for fold in fold_metrics]
        if values:
            avg_metrics[metric] = float(np.mean(values))
            avg_metrics[f"{metric}_std"] = float(np.std(values))
        else:
            avg_metrics[metric] = 0
            avg_metrics[f"{metric}_std"] = 0
    
    # Create summary
    cv_summary = {
        "filter_type": filter_type,
        "average_metrics": avg_metrics,
        "fold_metrics": fold_metrics
    }
    
    return cv_summary

def main():
    parser = argparse.ArgumentParser(description="Recover CV summary from fold results")
    parser.add_argument("--output-dir", required=True, help="Model output directory")
    parser.add_argument("--filter-type", required=True, help="Filter type (madgwick, kalman, ekf)")
    
    args = parser.parse_args()
    
    # Load fold results
    fold_metrics = load_fold_results(args.output_dir)
    
    # Create CV summary
    cv_summary = create_cv_summary(fold_metrics, args.filter_type)
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "cv_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"Recovered CV summary saved to {summary_path}")

if __name__ == "__main__":
    main()
EOF

    chmod +x "${UTILS_DIR}/recover_cv_summary.py"
}

# Main function to run the entire training and comparison
run_filter_comparison() {
    log "INFO" "Starting comprehensive IMU filter comparison"
    
    # Create results directory
    mkdir -p "${RESULTS_DIR}"
    
    # Initialize comparison CSV
    echo "filter_type,accuracy,f1,precision,recall,balanced_accuracy" > "${REPORT_FILE}"
    
    # Make utilities executable
    chmod +x "${UTILS_DIR}/compare_filters.py"
    create_cv_recovery_script
    
    # Create filter configs
    create_filter_config "madgwick" "${CONFIG_DIR}/madgwick.yaml"
    create_filter_config "kalman" "${CONFIG_DIR}/kalman.yaml"
    create_filter_config "ekf" "${CONFIG_DIR}/ekf.yaml"
    
    # Step 1: Train with Madgwick filter (baseline)
    log "INFO" "============= TRAINING WITH MADGWICK FILTER (BASELINE) ============="
    train_filter_model "madgwick" "${CONFIG_DIR}/madgwick.yaml"
    
    # Step 2: Train with Kalman filter
    log "INFO" "============= TRAINING WITH KALMAN FILTER ============="
    train_filter_model "kalman" "${CONFIG_DIR}/kalman.yaml"
    
    # Step 3: Train with EKF (Extended Kalman Filter)
    log "INFO" "============= TRAINING WITH EXTENDED KALMAN FILTER ============="
    train_filter_model "ekf" "${CONFIG_DIR}/ekf.yaml"
    
    # Step 4: Generate comprehensive comparison
    log "INFO" "============= GENERATING COMPARISON REPORT ============="
    python "${UTILS_DIR}/compare_filters.py" \
           --results-dir "${RESULTS_DIR}" \
           --output-csv "${REPORT_FILE}" \
           --filter-types madgwick kalman ekf
    
    # Final message
    log "INFO" "Filter comparison complete. Results available in:"
    log "INFO" "- ${REPORT_FILE}"
    log "INFO" "- ${RESULTS_DIR}/visualizations/comparison_report.md"
    log "INFO" "- ${RESULTS_DIR}/visualizations/filter_comparison.png"
}

# Run the main function
run_filter_comparison
