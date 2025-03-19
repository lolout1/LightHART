#!/bin/bash

# Script to compare different filter types for IMU fusion in fall detection
# Fixed to ensure compatibility with main.py arguments

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

# Include cross-validation settings directly in config file
# Instead of passing them as command line arguments
kfold:
  enabled: true
  num_folds: 5
  fold_assignments:
    - [43, 35, 36]  # Fold 1
    - [44, 34, 32]  # Fold 2
    - [45, 37, 38]  # Fold 3
    - [46, 29, 31]  # Fold 4
    - [30, 39]      # Fold 5

# Early stopping settings
early_stopping:
  enabled: true
  patience: 15
  monitor: val_loss

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
    visualize: true

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
EOL
}

# Train a model with a specific filter
train_model() {
    local config_file=$1
    local model_name=$2
    local filter_type=$3
    local output_dir="$RESULTS_DIR/${model_name}"
    
    log "INFO" "Training model with $filter_type filter: $model_name"
    mkdir -p "$output_dir"
    
    # Run training with cross-validation - FIXED: removed problematic arguments
    CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
        --config $config_file \
        --work-dir $output_dir \
        --model-saved-name $model_name \
        --device 0 1 \
        --multi-gpu True \
        --parallel-threads 48 \
        --num-epoch $NUM_EPOCHS 2>&1 | tee "$RESULTS_DIR/logs/${model_name}_train.log"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "ERROR" "Training failed with exit code $status"
        return 1
    fi
    
    # Extract and log key metrics
    if [ -f "$output_dir/cv_summary.json" ]; then
        log "INFO" "Cross-validation results for $model_name ($filter_type):"
        
        # Extract metrics using Python
        python -c "
import json
import os
import sys

cv_file = '$output_dir/cv_summary.json'
comparison_file = '$RESULTS_DIR/comparison.csv'

# Create comparison file with header if it doesn't exist
if not os.path.exists(comparison_file):
    with open(comparison_file, 'w') as f:
        f.write('model,filter_type,accuracy,f1,precision,recall,balanced_accuracy\\n')

try:
    with open(cv_file) as f:
        data = json.load(f)
        metrics = data.get('average_metrics', {})
        print(f\"Accuracy: {metrics.get('accuracy', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}\")
        print(f\"F1 score: {metrics.get('f1', 0):.4f} ± {metrics.get('f1_std', 0):.4f}\")
        print(f\"Precision: {metrics.get('precision', 0):.4f} ± {metrics.get('precision_std', 0):.4f}\")
        print(f\"Recall: {metrics.get('recall', 0):.4f} ± {metrics.get('recall_std', 0):.4f}\")
        
        # Append to comparison CSV
        with open(comparison_file, 'a') as csv:
            csv.write(f\"'$model_name','$filter_type',{metrics.get('accuracy', 0):.6f},{metrics.get('f1', 0):.6f},{metrics.get('precision', 0):.6f},{metrics.get('recall', 0):.6f},{metrics.get('balanced_accuracy', 0):.6f}\\n\")
except Exception as e:
    print(f\"Error processing results: {e}\")
    sys.exit(1)
"
    else
        log "WARNING" "No cross-validation summary found for $model_name. Creating empty entry in comparison file."
        echo "'$model_name','$filter_type',0.0,0.0,0.0,0.0,0.0" >> "$RESULTS_DIR/comparison.csv"
    fi
    
    return 0
}

# Generate comparison visualizations
generate_visualizations() {
    if [ ! -f "$RESULTS_DIR/comparison.csv" ]; then
        log "ERROR" "No comparison data found. Skipping visualizations."
        return 1
    fi
    
    log "INFO" "Generating comparative visualizations..."
    mkdir -p "$RESULTS_DIR/visualizations"
    
    # Use the utility script to generate visualizations and report
    python -c "
import sys
sys.path.append('.')
try:
    from utils.filter_comparison import process_comparison_results
    success = process_comparison_results('$RESULTS_DIR')
    if success:
        print('Visualizations and report generated successfully')
    else:
        print('Failed to generate visualizations')
except Exception as e:
    print(f'Error generating visualizations: {e}')
    sys.exit(1)
"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "ERROR" "Visualization generation failed with exit code $status"
        return 1
    fi
    
    log "INFO" "Visualizations saved to $RESULTS_DIR/visualizations"
    return 0
}

# Main function to run the filter comparison
main() {
    log "INFO" "Starting comprehensive filter comparison for IMU fusion"
    
    # Test filter implementations first
    log "INFO" "================ TESTING FILTER IMPLEMENTATIONS ================"
    log "INFO" "Testing filter implementations"
    log "INFO" "Running filter tests..."
    
    # Simple test of filter implementations 
    python -c "
import sys
sys.path.append('.')
try:
    from utils.imu_fusion import MadgwickFilter, ComplementaryFilter, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
    import numpy as np
    
    # Create test data
    acc = np.array([0, 0, 9.81])
    gyro = np.array([0.1, 0.2, 0.3])
    
    # Test each filter
    filters = {
        'madgwick': MadgwickFilter(),
        'comp': ComplementaryFilter(),
        'kalman': KalmanFilter(),
        'ekf': ExtendedKalmanFilter(),
        'ukf': UnscentedKalmanFilter()
    }
    
    for name, filter_obj in filters.items():
        q = filter_obj.update(acc, gyro)
        print(f'{name} quaternion: {q}')
    
    print('All filters functioning correctly')
except Exception as e:
    print(f'Filter test failed: {e}')
    sys.exit(1)
"
    
    local status=$?
    if [ $status -ne 0 ]; then
        log "ERROR" "Filter implementation tests failed with exit code $status"
        return 1
    fi
    
    log "INFO" "Filter implementation tests completed"
    
    # Create configuration files for each filter type
    create_config "$CONFIG_DIR/madgwick.yaml" "madgwick"
    create_config "$CONFIG_DIR/comp.yaml" "comp"
    create_config "$CONFIG_DIR/kalman.yaml" "kalman"
    create_config "$CONFIG_DIR/ekf.yaml" "ekf"
    create_config "$CONFIG_DIR/ukf.yaml" "ukf"
    
    # Create empty comparison file
    echo "model,filter_type,accuracy,f1,precision,recall,balanced_accuracy" > "$RESULTS_DIR/comparison.csv"
    
    # Train models with different filter types
    log "INFO" "============= TRAINING WITH MADGWICK FILTER (BASELINE) ============="
    train_model "$CONFIG_DIR/madgwick.yaml" "madgwick_model" "madgwick"
    
    log "INFO" "============= TRAINING WITH COMPLEMENTARY FILTER ============="
    train_model "$CONFIG_DIR/comp.yaml" "comp_model" "comp"
    
    log "INFO" "============= TRAINING WITH STANDARD KALMAN FILTER ============="
    train_model "$CONFIG_DIR/kalman.yaml" "kalman_model" "kalman"
    
    log "INFO" "============= TRAINING WITH EXTENDED KALMAN FILTER ============="
    train_model "$CONFIG_DIR/ekf.yaml" "ekf_model" "ekf"
    
    log "INFO" "============= TRAINING WITH UNSCENTED KALMAN FILTER ============="
    train_model "$CONFIG_DIR/ukf.yaml" "ukf_model" "ukf"
    
    # Generate comparison visualizations and report
    generate_visualizations
    
    log "INFO" "============= FILTER COMPARISON SUMMARY ============="
    # Print summary table
    column -t -s, "$RESULTS_DIR/comparison.csv"
    
    log "INFO" "Filter comparison completed. Results saved to $RESULTS_DIR"
    log "INFO" "See $RESULTS_DIR/report/filter_comparison_report.html for detailed analysis"
}

# Run the main function
main
