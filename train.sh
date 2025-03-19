#!/bin/bash

# Script to compare different filter types for IMU fusion in fall detection
# Ensures correct propagation of filter type through the pipeline

# Set strict error handling
set -e

# Configuration
DEVICE="0,1"  # GPU devices
BASE_LR=0.0005
WEIGHT_DECAY=0.001
NUM_EPOCHS=60
RESULTS_DIR="filter_comparison_results"
CONFIG_DIR="config/filter_comparison"

# Create directories
mkdir -p $RESULTS_DIR
mkdir -p $CONFIG_DIR
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR/visualizations"

# Log function with timestamp
log() {
    local level=$1
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $2"
    echo "$msg"
    echo "$msg" >> "$RESULTS_DIR/logs/main.log"
}

# Create configuration file for a specific filter
create_config() {
    local config_file=$1
    local filter_type=$2
    
    log "INFO" "Creating config for $filter_type filter: $config_file"
    
    cat > $config_file << EOL
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
  feature_dim: 144  # 48 * 3 for concat fusion

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
    acc_threshold: 3.0
    gyro_threshold: 1.0
    visualize: false
    save_aligned: false

batch_size: 16
test_batch_size: 16
val_batch_size: 16
num_epoch: ${NUM_EPOCHS}

feeder: Feeder.Make_Dataset.UTD_mm
train_feeder_args:
  batch_size: 16
  drop_last: true

val_feeder_args:
  batch_size: 16
  drop_last: true

test_feeder_args:
  batch_size: 16
  drop_last: false

seed: 42
optimizer: adamw
base_lr: ${BASE_LR}
weight_decay: ${WEIGHT_DECAY}

kfold:
  enabled: true
  num_folds: 5
  fold_assignments:
    - [43, 35, 36]  # Fold 1
    - [44, 34, 32]  # Fold 2
    - [45, 37, 38]  # Fold 3
    - [46, 29, 31]  # Fold 4
    - [30, 39]      # Fold 5
EOL
}

# Train a model with a specific filter
train_model() {
    local config_file=$1
    local model_name=$2
    local filter_type=$3
    local output_dir="$RESULTS_DIR/$model_name"
    
    log "INFO" "Training model with $filter_type filter: $model_name"
    mkdir -p "$output_dir"
    
    # Run training with cross-validation
    CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
        --config $config_file \
        --work-dir $output_dir \
        --model-saved-name $model_name \
        --device 0 1 \
        --multi-gpu True \
        --kfold True \
        --parallel-threads 40 \
        --num-epoch $NUM_EPOCHS \
        --patience 15 2>&1 | tee "$RESULTS_DIR/logs/${model_name}_train.log"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "ERROR" "Training failed with exit code $status"
        
        # Create a patch for the JSON issue if that was the problem
        if grep -q "NameError: name 'json' is not defined" "$RESULTS_DIR/logs/${model_name}_train.log"; then
            log "INFO" "Detected missing json import, creating summary file manually"
            python -c "
import json
import os
import glob
import numpy as np

# Collect data from individual folds
metrics = {
    'accuracy': [],
    'f1': [],
    'precision': [],
    'recall': [],
    'balanced_accuracy': []
}

fold_metrics = []
fold_dirs = sorted(glob.glob('$output_dir/fold_*'))

for i, fold_dir in enumerate(fold_dirs, 1):
    # Check if fold has a test_results.json file
    test_file = os.path.join(fold_dir, 'test_results.json')
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            try:
                data = json.load(f)
                fold_result = {
                    'fold': i,
                    'test_subjects': [], # We don't have this info, leave empty
                    'accuracy': data.get('accuracy', 0),
                    'f1': data.get('f1', 0),
                    'precision': data.get('precision', 0),
                    'recall': data.get('recall', 0),
                    'balanced_accuracy': data.get('balanced_accuracy', 0)
                }
                fold_metrics.append(fold_result)
                
                # Add to averages
                for key in metrics.keys():
                    metrics[key].append(data.get(key, 0))
            except:
                print(f'Error reading {test_file}')

# Calculate averages and std deviations
avg_metrics = {}
for key, values in metrics.items():
    if values:
        avg_metrics[key] = np.mean(values)
        avg_metrics[key + '_std'] = np.std(values)
    else:
        avg_metrics[key] = 0
        avg_metrics[key + '_std'] = 0

# Create summary
cv_summary = {
    'fold_metrics': fold_metrics,
    'average_metrics': avg_metrics,
    'filter_type': '$filter_type'
}

# Save to file
with open('$output_dir/cv_summary.json', 'w') as f:
    json.dump(cv_summary, f, indent=2)
print(f'Created summary file for {len(fold_metrics)} folds')
"
        fi
        
        # Continue despite error so we can process other filters
        log "WARNING" "Continuing to next filter despite training error"
    fi
    
    # Extract and log key metrics if summary exists
    if [ -f "$output_dir/cv_summary.json" ]; then
        log "INFO" "Cross-validation results for $model_name ($filter_type):"
        
        # Extract metrics using Python
        python -c "
import json
import sys
try:
    with open('$output_dir/cv_summary.json') as f:
        data = json.load(f)
        metrics = data['average_metrics']
        print(f\"Accuracy: {metrics.get('accuracy', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}\")
        print(f\"F1 score: {metrics.get('f1', 0):.4f} ± {metrics.get('f1_std', 0):.4f}\")
        print(f\"Precision: {metrics.get('precision', 0):.4f} ± {metrics.get('precision_std', 0):.4f}\")
        print(f\"Recall: {metrics.get('recall', 0):.4f} ± {metrics.get('recall_std', 0):.4f}\")
        
        # Save to comparison CSV
        with open('$RESULTS_DIR/comparison.csv', 'a') as csv:
            csv.write(f\"{model_name},{filter_type},{metrics.get('accuracy', 0):.6f},{metrics.get('f1', 0):.6f},{metrics.get('precision', 0):.6f},{metrics.get('recall', 0):.6f},{metrics.get('balanced_accuracy', 0):.6f}\\n\")
except Exception as e:
    print(f\"Error reading CV summary: {str(e)}\")
    # Create an empty entry in the comparison table
    with open('$RESULTS_DIR/comparison.csv', 'a') as csv:
        csv.write(f\"{model_name},{filter_type},0.0,0.0,0.0,0.0,0.0\\n\")
"
    else
        log "ERROR" "No cross-validation summary found for $model_name"
        # Create an empty entry in the comparison table
        echo "$model_name,$filter_type,0.0,0.0,0.0,0.0,0.0" >> "$RESULTS_DIR/comparison.csv"
    fi
    
    return 0
}

# Main function to run the filter comparison
main() {
    log "INFO" "Starting comprehensive filter comparison for IMU fusion"
    
    # Create comparison CSV header
    echo "model,filter_type,accuracy,f1,precision,recall,balanced_accuracy" > "$RESULTS_DIR/comparison.csv"
    
    # Create configurations for each filter type
    create_config "$CONFIG_DIR/madgwick.yaml" "madgwick"
    create_config "$CONFIG_DIR/kalman.yaml" "kalman"
    create_config "$CONFIG_DIR/ekf.yaml" "ekf"
    
    # Train models with different filter types
    log "INFO" "============= TRAINING WITH MADGWICK FILTER (BASELINE) ============="
    train_model "$CONFIG_DIR/madgwick.yaml" "madgwick_model" "madgwick"
    
    log "INFO" "============= TRAINING WITH STANDARD KALMAN FILTER ============="
    train_model "$CONFIG_DIR/kalman.yaml" "kalman_model" "kalman"
    
    log "INFO" "============= TRAINING WITH EXTENDED KALMAN FILTER ============="
    train_model "$CONFIG_DIR/ekf.yaml" "ekf_model" "ekf"
    
    # Generate comparison visualizations and report
    log "INFO" "Generating comparative analysis"
    python -c "
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory
os.makedirs('$RESULTS_DIR/visualizations', exist_ok=True)

# Load comparison data
try:
    comp_data = pd.read_csv('$RESULTS_DIR/comparison.csv')
    
    # Create bar chart for all metrics
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
    plt.figure(figsize=(14, 10))
    
    # Set bar positions
    x = np.arange(len(comp_data['filter_type']))
    width = 0.15
    multiplier = 0
    
    # Plot each metric as a group of bars
    for metric in metrics:
        offset = width * multiplier
        plt.bar(x + offset, comp_data[metric], width, label=metric.capitalize())
        multiplier += 1
    
    # Add labels and legend
    plt.xlabel('Filter Type', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('IMU Fusion Filter Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x + width * 2, comp_data['filter_type'], fontsize=10, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('$RESULTS_DIR/visualizations/filter_comparison.png', dpi=300)
    
    # Create a detailed metrics table
    with open('$RESULTS_DIR/visualizations/comparison_summary.md', 'w') as f:
        f.write('# IMU Fusion Filter Comparison Results\n\n')
        f.write('## Performance Metrics\n\n')
        f.write(comp_data.to_markdown(index=False))
        
        # Find best filter for each metric
        f.write('\n\n## Best Performing Filter by Metric\n\n')
        for metric in metrics:
            best_idx = comp_data[metric].idxmax()
            best_filter = comp_data.loc[best_idx, 'filter_type']
            best_value = comp_data.loc[best_idx, metric]
            f.write(f'- **{metric.capitalize()}**: {best_filter} ({best_value:.4f})\n')
    
    print('Successfully generated comparison visualizations and summary')
except Exception as e:
    print(f'Error generating comparison: {str(e)}')
"
    
    log "INFO" "Filter comparison completed successfully"
    log "INFO" "Results available in $RESULTS_DIR"
    log "INFO" "See $RESULTS_DIR/visualizations for comparative analysis"
}

# Run the main function
main
