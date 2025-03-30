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

create_filter_config() {
    local filter_type="$1"
    local output_file="$2"
    local cache_dir="${CACHE_DIR}/${filter_type}"
    [ "${CACHE_ENABLED}" = "true" ] && mkdir -p "${cache_dir}"
    
    cat > "${output_file}" << EOF
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm
subjects: [32, 39, 30, 31, 33, 34, 35, 37, 43, 44, 45, 36, 29]
val_subjects: [38, 46]
permanent_train: [45, 36, 29]

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
    process_per_window: false
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
        none)
            cat >> "${output_file}" << EOF
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
rotate_tests: true
test_combinations: true
EOF

    [ -f "${output_file}" ] && return 0 || return 1
}

create_comparison_script() {
    cat > "${UTILS_DIR}/compare_filters.py" << 'EOF'
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
EOF
    chmod +x "${UTILS_DIR}/compare_filters.py"
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
        --parallel-threads 42 \
        --num-epoch ${NUM_EPOCHS} 2>&1 | tee "${output_dir}/logs/training.log"
    
    log "INFO" "Training complete for ${filter_type} filter"
    return 0
}

run_filter_comparison() {
    log "INFO" "Starting IMU filter comparison"
    
    mkdir -p "${RESULTS_DIR}"
    echo "filter_type,accuracy,f1,precision,recall,balanced_accuracy" > "${REPORT_FILE}"
    
    create_comparison_script
    
    create_filter_config "madgwick" "${CONFIG_DIR}/madgwick.yaml"
    create_filter_config "kalman" "${CONFIG_DIR}/kalman.yaml"
    create_filter_config "ekf" "${CONFIG_DIR}/ekf.yaml"
    create_filter_config "none" "${CONFIG_DIR}/none.yaml"
    
    log "INFO" "============= TRAINING WITH MADGWICK FILTER (BASELINE) ============="
    train_filter_model "madgwick" "${CONFIG_DIR}/madgwick.yaml"
    
    log "INFO" "============= TRAINING WITH KALMAN FILTER ============="
    train_filter_model "kalman" "${CONFIG_DIR}/kalman.yaml"
    
    log "INFO" "============= TRAINING WITH EXTENDED KALMAN FILTER ============="
    train_filter_model "ekf" "${CONFIG_DIR}/ekf.yaml"
    
    log "INFO" "============= TRAINING WITH NO QUATERNION (ACC+GYRO ONLY) ============="
    train_filter_model "none" "${CONFIG_DIR}/none.yaml"
    
    log "INFO" "============= GENERATING COMPARISON REPORT ============="
    python "${UTILS_DIR}/compare_filters.py" \
           --results-dir "${RESULTS_DIR}" \
           --output-csv "${REPORT_FILE}" \
           --filter-types madgwick kalman ekf none
    
    log "INFO" "Filter comparison complete. Results available in:"
    log "INFO" "- ${REPORT_FILE}"
    log "INFO" "- ${RESULTS_DIR}/visualizations/filter_comparison.png"
}

run_filter_comparison
