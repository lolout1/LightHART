#!/bin/bash
set -e
set -o pipefail
set -u

DEVICE="0,1"
BASE_LR=0.0005
WEIGHT_DECAY=0.001
NUM_EPOCHS=100
PATIENCE=15
SEED=42
BATCH_SIZE=16
RESULTS_DIR="filter_comparison_results"
CONFIG_DIR="config/filter_comparison"
UTILS_DIR="utils/comparison_scripts"
REPORT_FILE="${RESULTS_DIR}/comparison_results.csv"
CACHE_ENABLED="true"
CACHE_DIR="processed_data"

mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/visualizations"
mkdir -p "${CONFIG_DIR}"
mkdir -p "${UTILS_DIR}"
[ "${CACHE_ENABLED}" = "true" ] && mkdir -p "${CACHE_DIR}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

log() {
    local level="$1"
    local msg="$2"
    echo "[$(timestamp)] [${level}] ${msg}"
    echo "[$(timestamp)] [${level}] ${msg}" >> "${RESULTS_DIR}/logs/training.log"
}

check_status() {
    if [ $? -ne 0 ]; then
        log "ERROR" "$1"
        return 1
    fi
    return 0
}

cat > "${UTILS_DIR}/compare_filters.py" << 'EOF'
#!/usr/bin/env python3
import os, sys, json, pandas as pd, numpy as np, matplotlib.pyplot as plt
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
        f.write("- **UKF**: Unscented Kalman Filter for highly accurate non-linear state estimation with better uncertainty handling.\n")
    print(f"Comparison report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare IMU fusion filter performance')
    parser.add_argument('--results-dir', required=True, help='Results directory')
    parser.add_argument('--output-csv', required=True, help='Output CSV file')
    parser.add_argument('--filter-types', nargs='+', default=['madgwick', 'kalman', 'ekf', 'ukf'], help='Filter types to compare')
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
EOF

create_cv_recovery_script() {
    cat > "${UTILS_DIR}/recover_cv_summary.py" << 'EOF'
#!/usr/bin/env python3
import os, json, argparse, numpy as np, glob
from typing import List, Dict, Any

def load_fold_results(output_dir: str) -> List[Dict[str, Any]]:
    fold_metrics = []
    fold_dirs = sorted(glob.glob(os.path.join(output_dir, "fold_*")))
    for i, fold_dir in enumerate(fold_dirs, 1):
        results_file = os.path.join(fold_dir, "validation_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f: results = json.load(f)
                results["fold"] = i
                fold_metrics.append(results)
                print(f"Loaded results from {results_file}")
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
    return fold_metrics

def create_cv_summary(fold_metrics: List[Dict[str, Any]], filter_type: str) -> Dict[str, Any]:
    if not fold_metrics:
        return {"filter_type": filter_type, "average_metrics": {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0, "balanced_accuracy": 0}, "fold_metrics": []}
    
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
            
    return {"filter_type": filter_type, "average_metrics": avg_metrics, "fold_metrics": fold_metrics}

def main():
    parser = argparse.ArgumentParser(description="Recover CV summary from fold results")
    parser.add_argument("--output-dir", required=True, help="Model output directory")
    parser.add_argument("--filter-type", required=True, help="Filter type (madgwick, kalman, ekf, ukf)")
    args = parser.parse_args()
    
    fold_metrics = load_fold_results(args.output_dir)
    cv_summary = create_cv_summary(fold_metrics, args.filter_type)
    
    summary_path = os.path.join(args.output_dir, "cv_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
        
    print(f"Recovered CV summary saved to {summary_path}")

if __name__ == "__main__":
    main()
EOF
    chmod +x "${UTILS_DIR}/recover_cv_summary.py"
}

create_filter_config() {
    local filter_type="$1"
    local output_file="$2"
    local cache_dir="${CACHE_DIR}/${filter_type}"
    [ "${CACHE_ENABLED}" = "true" ] && mkdir -p "${cache_dir}"
    
    cat > "${output_file}" << EOF
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm
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
  feature_dim: 144

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
    preserve_filter_state: true
EOF

    case "${filter_type}" in
        madgwick)
            cat >> "${output_file}" << EOF
    beta: 0.15
    acc_threshold: 3.0
    gyro_threshold: 1.0
EOF
            ;;
        kalman)
            cat >> "${output_file}" << EOF
    process_noise: 5e-5
    measurement_noise: 0.1
    acc_threshold: 3.0
    gyro_threshold: 1.0
EOF
            ;;
        ekf)
            cat >> "${output_file}" << EOF
    process_noise: 1e-5
    measurement_noise: 0.05
    acc_threshold: 3.0
    gyro_threshold: 1.0
EOF
            ;;
        ukf)
            cat >> "${output_file}" << EOF
    alpha: 0.15
    beta: 2.0
    kappa: 1.0
    acc_threshold: 3.0
    gyro_threshold: 1.0
EOF
            ;;
    esac
    
    if [ "${CACHE_ENABLED}" = "true" ]; then
        cat >> "${output_file}" << EOF
    use_cache: true
    cache_dir: "${cache_dir}"
EOF
    fi
    
    cat >> "${output_file}" << EOF
    visualize: false
    save_aligned: true

batch_size: ${BATCH_SIZE}
val_batch_size: ${BATCH_SIZE}
num_epoch: ${NUM_EPOCHS}

feeder: Feeder.Make_Dataset.UTD_mm
train_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: true

val_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: true

seed: ${SEED}
optimizer: adamw
base_lr: ${BASE_LR}
weight_decay: ${WEIGHT_DECAY}

kfold:
  enabled: true
  num_folds: 5
  fold_assignments:
    - [43, 35, 36]
    - [44, 34, 32]
    - [45, 37, 38]
    - [46, 29, 31]
    - [30, 39]
EOF

    [ -f "${output_file}" ] && return 0 || return 1
}

train_filter_model() {
    local filter_type="$1"
    local config_file="$2"
    local output_dir="${RESULTS_DIR}/${filter_type}_model"
    
    log "INFO" "========================================================="
    log "INFO" "STARTING TRAINING FOR ${filter_type^^} FILTER"
    log "INFO" "========================================================="
    
    mkdir -p "${output_dir}/logs"
    
    if [ ! -f "${config_file}" ]; then
        log "ERROR" "Config file does not exist: ${config_file}"
        return 1
    fi
    
    log "INFO" "Training model with ${filter_type} filter"
    CUDA_VISIBLE_DEVICES=${DEVICE} python main.py \
        --config "${config_file}" \
        --work-dir "${output_dir}" \
        --model-saved-name "${filter_type}_model" \
        --device 0 1 \
        --multi-gpu True \
        --patience ${PATIENCE} \
        --parallel-threads 30 \
        --num-epoch ${NUM_EPOCHS} \
        --run-comparison True 2>&1 | tee "${output_dir}/logs/training.log"
    
    train_status=$?
    if [ ${train_status} -ne 0 ]; then
        log "WARNING" "Training process exited with status ${train_status}"
        if [ ! -f "${output_dir}/cv_summary.json" ]; then
            log "INFO" "Attempting to recover cross-validation summary from fold results"
            python "${UTILS_DIR}/recover_cv_summary.py" \
                   --output-dir "${output_dir}" \
                   --filter-type "${filter_type}"
        fi
    fi
    
    if [ ! -f "${output_dir}/cv_summary.json" ]; then
        log "WARNING" "No cross-validation summary found for ${filter_type}"
        echo "{\"filter_type\":\"${filter_type}\",\"average_metrics\":{\"accuracy\":0,\"f1\":0,\"precision\":0,\"recall\":0,\"balanced_accuracy\":0},\"fold_metrics\":[]}" > "${output_dir}/cv_summary.json"
    fi
    
    log "INFO" "Training complete for ${filter_type} filter"
    return 0
}

run_filter_comparison() {
    log "INFO" "Starting IMU filter comparison"
    
    mkdir -p "${RESULTS_DIR}"
    
    echo "filter_type,accuracy,f1,precision,recall,balanced_accuracy" > "${REPORT_FILE}"
    
    chmod +x "${UTILS_DIR}/compare_filters.py"
    
    create_cv_recovery_script
    
    create_filter_config "madgwick" "${CONFIG_DIR}/madgwick.yaml"
    create_filter_config "kalman" "${CONFIG_DIR}/kalman.yaml"
    create_filter_config "ekf" "${CONFIG_DIR}/ekf.yaml"
    create_filter_config "ukf" "${CONFIG_DIR}/ukf.yaml"
    
    log "INFO" "============= TRAINING WITH MADGWICK FILTER (BASELINE) ============="
    train_filter_model "madgwick" "${CONFIG_DIR}/madgwick.yaml"
    
    log "INFO" "============= TRAINING WITH KALMAN FILTER ============="
    train_filter_model "kalman" "${CONFIG_DIR}/kalman.yaml"
    
    log "INFO" "============= TRAINING WITH EXTENDED KALMAN FILTER ============="
    train_filter_model "ekf" "${CONFIG_DIR}/ekf.yaml"
    
    log "INFO" "============= TRAINING WITH UNSCENTED KALMAN FILTER ============="
    train_filter_model "ukf" "${CONFIG_DIR}/ukf.yaml"
    
    log "INFO" "============= GENERATING COMPARISON REPORT ============="
    python "${UTILS_DIR}/compare_filters.py" \
           --results-dir "${RESULTS_DIR}" \
           --output-csv "${REPORT_FILE}" \
           --filter-types madgwick kalman ekf ukf
    
    log "INFO" "Filter comparison complete. Results available in:"
    log "INFO" "- ${REPORT_FILE}"
    log "INFO" "- ${RESULTS_DIR}/visualizations/comparison_report.md"
    log "INFO" "- ${RESULTS_DIR}/visualizations/filter_comparison.png"
}

run_filter_comparison
