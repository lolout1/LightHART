#!/bin/bash

# Comprehensive training script for fall detection with multiple filter types
# Includes thorough debugging, visualization, and performance comparison

# Set strict error handling
set -e

# Set common parameters
DEVICE=0
BASE_LR=0.0005
WEIGHT_DECAY=0.001
NUM_EPOCHS=60
RESULT_DIR="results_comparison"
LOG_DIR="debug_logs"
VISUALIZATION_DIR="filter_visualizations"

# Create directories
mkdir -p $RESULT_DIR
mkdir -p $LOG_DIR
mkdir -p $VISUALIZATION_DIR

# Log function with timestamp
log() {
    local level=$1
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $2"
    echo "$msg"
    echo "$msg" >> "$LOG_DIR/training.log"
}

# Error handler
handle_error() {
    local exit_code=$?
    local line_number=$1
    log "ERROR" "Command failed at line $line_number with exit code $exit_code"
    run_diagnostics
}
trap 'handle_error $LINENO' ERR

# Diagnostics function
run_diagnostics() {
    log "INFO" "Running diagnostics..."

    # Check Python dependencies
    log "INFO" "Checking dependencies..."
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
    python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
    python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    
    # Check CUDA
    python -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA devices: {torch.cuda.device_count()}"); print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}")'
    
    # Check dataset file structure
    log "INFO" "Checking dataset structure..."
    ls -la data/smartfallmm/young/
    
    # Test IMU fusion module
    log "INFO" "Testing IMU fusion module..."
    python -c "
from utils.imu_fusion import MadgwickFilter, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, process_imu_data
import numpy as np

# Create dummy data
acc = np.array([[0, 0, 9.8], [0.1, 0, 9.7], [0.2, 0.1, 9.7]])
gyro = np.array([[0, 0, 0], [0.01, 0.01, 0], [0.02, 0.02, 0]])

# Test filters
madgwick = MadgwickFilter()
q1 = madgwick.update(acc[0], gyro[0])
q2 = madgwick.update(acc[1], gyro[1])
print(f'Madgwick quaternion: {q2}')

kalman = KalmanFilter()
q1 = kalman.update(acc[0], gyro[0])
q2 = kalman.update(acc[1], gyro[1])
print(f'Kalman quaternion: {q2}')

print('IMU fusion module tests passed!')
"
}

# Function to generate configuration file
create_config() {
    local config_file=$1
    local filter_type=$2
    local embed_dim=$3
    local num_layers=$4
    local num_heads=$5
    local fusion_type=$6
    
    log "INFO" "Creating config file: $config_file"
    
    cat > $config_file << EOL
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm

# Fall detection subjects
subjects: [29, 30, 31, 33, 45, 46, 34, 37, 39, 38, 43, 35, 36, 44, 32]

model_args:
  num_layers: ${num_layers}
  embed_dim: ${embed_dim}
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: 128
  mocap_frames: 64
  num_heads: ${num_heads}
  fusion_type: '${fusion_type}'
  dropout: 0.3
  use_batch_norm: true

dataset_args:
  mode: 'sliding_window'
  max_length: 128
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

# dataloader
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

# Function to train model
train_model() {
    local config_file=$1
    local model_name=$2
    local work_dir="${RESULT_DIR}/${model_name}"
    local filter_type=$3
    
    log "INFO" "Training model: $model_name with $filter_type filter"
    
    # Create work directory
    mkdir -p $work_dir
    
    # Run training
    log "INFO" "Starting training..."
    python main.py \
        --config $config_file \
        --work-dir $work_dir \
        --model-saved-name $model_name \
        --device $DEVICE \
        --base-lr $BASE_LR \
        --weight-decay $WEIGHT_DECAY \
        --num-epoch $NUM_EPOCHS \
        --include-val True 2>&1 | tee -a "${LOG_DIR}/${model_name}_train.log"
    
    local train_status=$?
    if [ $train_status -ne 0 ]; then
        log "ERROR" "Training failed with exit code $train_status"
        return 1
    fi
    
    # Run testing
    log "INFO" "Starting testing..."
    python main.py \
        --config $config_file \
        --work-dir $work_dir \
        --weights "${work_dir}/${model_name}.pt" \
        --device $DEVICE \
        --phase 'test' 2>&1 | tee -a "${LOG_DIR}/${model_name}_test.log"
    
    local test_status=$?
    if [ $test_status -ne 0 ]; then
        log "ERROR" "Testing failed with exit code $test_status"
        return 1
    fi
    
    # Extract metrics
    local accuracy=$(grep -Po "Test accuracy: \K[0-9.]+%" "${LOG_DIR}/${model_name}_test.log" | tail -1 | tr -d '%')
    local f1=$(grep -Po "Test F-Score: \K[0-9.]+" "${LOG_DIR}/${model_name}_test.log" | tail -1)
    local precision=$(grep -Po "Test precision: \K[0-9.]+" "${LOG_DIR}/${model_name}_test.log" | tail -1)
    local recall=$(grep -Po "Test recall: \K[0-9.]+" "${LOG_DIR}/${model_name}_test.log" | tail -1)
    
    log "INFO" "Results for $model_name:"
    log "INFO" "  Accuracy:  ${accuracy}%"
    log "INFO" "  F1-Score:  ${f1}"
    log "INFO" "  Precision: ${precision}"
    log "INFO" "  Recall:    ${recall}"
    
    # Add to comparison table
    echo "$model_name,$filter_type,$accuracy,$f1,$precision,$recall" >> "${RESULT_DIR}/comparison.csv"
}

# Function to generate plots
create_comparison_plots() {
    log "INFO" "Generating comparison plots..."
    
    python -c "
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read comparison data
df = pd.read_csv('${RESULT_DIR}/comparison.csv')

# Set figure size
plt.figure(figsize=(12, 8))

# Bar width
bar_width = 0.2
r1 = np.arange(len(df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Create grouped bar chart
plt.bar(r1, df['accuracy'], width=bar_width, label='Accuracy', color='blue', alpha=0.7)
plt.bar(r2, df['f1'], width=bar_width, label='F1 Score', color='green', alpha=0.7)
plt.bar(r3, df['precision'], width=bar_width, label='Precision', color='red', alpha=0.7)
plt.bar(r4, df['recall'], width=bar_width, label='Recall', color='purple', alpha=0.7)

# Add labels and title
plt.xlabel('Model Configuration')
plt.ylabel('Score')
plt.title('Performance Metrics by Model and Filter Type')
plt.xticks([r + bar_width*1.5 for r in range(len(df))], df['model'])
plt.legend()

# Add value labels on top of bars
for i, v in enumerate(df['accuracy']):
    plt.text(r1[i], v + 0.01, f'{v:.3f}', ha='center', va='bottom', rotation=90)
for i, v in enumerate(df['f1']):
    plt.text(r2[i], v + 0.01, f'{v:.3f}', ha='center', va='bottom', rotation=90)
for i, v in enumerate(df['precision']):
    plt.text(r3[i], v + 0.01, f'{v:.3f}', ha='center', va='bottom', rotation=90)
for i, v in enumerate(df['recall']):
    plt.text(r4[i], v + 0.01, f'{v:.3f}', ha='center', va='bottom', rotation=90)

# Save figure
plt.tight_layout()
plt.savefig('${RESULT_DIR}/performance_comparison.png', dpi=300)
plt.close()

print('Comparison plot generated at ${RESULT_DIR}/performance_comparison.png')
"
}

# Main function
main() {
    # Log start
    log "INFO" "Starting comprehensive fall detection training and filter comparison"
    
    # Run diagnostics
    run_diagnostics
    
    # Create comparison table header
    echo "model,filter_type,accuracy,f1,precision,recall" > "${RESULT_DIR}/comparison.csv"
    
    # Train with Madgwick filter
    create_config "config/smartfallmm/madgwick_fusion.yaml" "madgwick" 32 3 8 "concat"
    train_model "config/smartfallmm/madgwick_fusion.yaml" "madgwick_model" "madgwick"
    
    # Train with standard Kalman filter
    create_config "config/smartfallmm/kalman_fusion.yaml" "kalman" 32 3 8 "concat"
    train_model "config/smartfallmm/kalman_fusion.yaml" "kalman_model" "kalman"
    
    # Train with Extended Kalman filter
    create_config "config/smartfallmm/ekf_fusion.yaml" "ekf" 32 3 8 "concat"
    train_model "config/smartfallmm/ekf_fusion.yaml" "ekf_model" "ekf"
    
    # Train with Unscented Kalman filter
    create_config "config/smartfallmm/ukf_fusion.yaml" "ukf" 32 3 8 "concat"
    train_model "config/smartfallmm/ukf_fusion.yaml" "ukf_model" "ukf"
    
    # Train baseline accelerometer-only model for comparison
    create_config "config/smartfallmm/acc_only.yaml" "madgwick" 32 3 8 "acc_only"
    train_model "config/smartfallmm/acc_only.yaml" "acc_only_model" "none"
    
    # Generate comparison visualizations
    create_comparison_plots
    
    # Print final results
    log "INFO" "===== FINAL RESULTS ====="
    cat "${RESULT_DIR}/comparison.csv" | column -t -s ','
    log "INFO" "Complete results saved to ${RESULT_DIR}/comparison.csv"
    log "INFO" "Performance visualization saved to ${RESULT_DIR}/performance_comparison.png"
    log "INFO" "Training completed successfully"
}

# Execute main function
main
