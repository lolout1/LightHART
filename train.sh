#!/bin/bash

# Comprehensive training script for fall detection with multiple filter types
# Includes thorough debugging, visualization, and performance comparison

# Set strict error handling
set -e

# Set common parameters
DEVICE="0,1"  # Use both GPUs
BASE_LR=0.005
WEIGHT_DECAY=0.001
NUM_EPOCHS=120
RESULT_DIR="results_comparison"
LOG_DIR="debug_logs"
VISUALIZATION_DIR="filter_visualizations"
CONFIG_DIR="config/smartfallmm"

# Create directories
mkdir -p $RESULT_DIR
mkdir -p $LOG_DIR
mkdir -p $VISUALIZATION_DIR
mkdir -p $CONFIG_DIR

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
    python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA device: None')
"
    
    # Check dataset file structure
    log "INFO" "Checking dataset structure..."
    ls -la data/smartfallmm/young/
    
    # Test IMU fusion module
    log "INFO" "Testing IMU fusion module..."
    python -c "
from utils.imu_fusion import MadgwickFilter, ComplementaryFilter, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test')
logger.info('Testing IMU fusion filters')
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
for name, f in filters.items():
    q = f.update(acc, gyro)
    print(f'{name} quaternion: {q}')
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
    local dropout=$6
    local fusion_type=$7
    
    log "INFO" "Creating config file: $config_file"
    
    cat > $config_file << EOL
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm

# Include all subjects for better generalization
subjects: [29, 30, 31, 33, 45, 46]

model_args:
  num_layers: ${num_layers}
  embed_dim: ${embed_dim}
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: 64
  mocap_frames: 64
  num_heads: ${num_heads}
  fusion_type: '${fusion_type}'
  dropout: ${dropout}
  use_batch_norm: true

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
    local filter_type=$3
    local work_dir="${RESULT_DIR}/${model_name}"
    
    log "INFO" "========================================================"
    log "INFO" "Training model: $model_name with $filter_type filter"
    log "INFO" "========================================================"
    
    # Create work directory
    mkdir -p $work_dir
    
    # Log detailed information
    log "INFO" "Using devices: $DEVICE"
    log "INFO" "Config file: $config_file"
    log "INFO" "Output directory: $work_dir"
    
    # Run training with parallel threads
    log "INFO" "Starting training phase..."
    CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
        --config $config_file \
        --work-dir $work_dir \
        --model-saved-name $model_name \
        --device 0 1 \
        --base-lr $BASE_LR \
        --weight-decay $WEIGHT_DECAY \
        --num-epoch $NUM_EPOCHS \
        --multi-gpu True \
        --parallel-threads 48 \
        --include-val True 2>&1 | tee "${LOG_DIR}/${model_name}_train.log"
    
    local train_status=$?
    if [ $train_status -ne 0 ]; then
        log "ERROR" "Training failed with exit code $train_status"
        log "ERROR" "Check ${LOG_DIR}/${model_name}_train.log for details"
        return 1
    fi
    
    # Verify model file exists before testing
    if [ ! -f "${work_dir}/${model_name}.pt" ]; then
        log "ERROR" "Model file not found: ${work_dir}/${model_name}.pt"
        log "ERROR" "Training may have failed to save the model"
        return 1
    fi
    
    # Run testing only if training succeeded
    log "INFO" "Starting testing phase..."
    CUDA_VISIBLE_DEVICES=${DEVICE} python main.py \
        --config $config_file \
        --work-dir $work_dir \
        --weights "${work_dir}/${model_name}.pt" \
        --device 0 \
        --phase 'test' \
        --parallel-threads 48 2>&1 | tee "${LOG_DIR}/${model_name}_test.log"
    
    local test_status=$?
    if [ $test_status -ne 0 ]; then
        log "ERROR" "Testing failed with exit code $test_status"
        log "ERROR" "Check ${LOG_DIR}/${model_name}_test.log for details"
        return 1
    fi
    
    # Extract metrics from test_result.txt
    log "INFO" "Extracting performance metrics..."
    local test_result="${work_dir}/test_result.txt"
    
    if [ -f "$test_result" ]; then
        local accuracy=$(grep "accuracy" "$test_result" | cut -d' ' -f2)
        local f1=$(grep "f1_score" "$test_result" | cut -d' ' -f2)
        local precision=$(grep "precision" "$test_result" | cut -d' ' -f2)
        local recall=$(grep "recall" "$test_result" | cut -d' ' -f2)
        
        log "INFO" "Results for $model_name:"
        log "INFO" "  Accuracy:  ${accuracy}%"
        log "INFO" "  F1-Score:  ${f1}"
        log "INFO" "  Precision: ${precision}%"
        log "INFO" "  Recall:    ${recall}%"
        
        # Add to comparison table
        echo "$model_name,$filter_type,$accuracy,$f1,$precision,$recall" >> "${RESULT_DIR}/comparison.csv"
    else
        log "ERROR" "Test results file not found: $test_result"
        return 1
    fi
    
    log "INFO" "Model training and evaluation completed successfully"
    return 0
}

# Main function
main() {
    # Log start
    log "INFO" "Starting comprehensive fall detection training and filter comparison"
    log "INFO" "Using CUDA_VISIBLE_DEVICES=$DEVICE"
    
    # Run diagnostics
    run_diagnostics
    
    # Create comparison table header
    echo "model,filter_type,accuracy,f1,precision,recall" > "${RESULT_DIR}/comparison.csv"
    
    # Train with Madgwick filter
    log "INFO" "======================================================"
    log "INFO" "Starting Madgwick filter training"
    log "INFO" "======================================================"
    create_config "${CONFIG_DIR}/madgwick_filter.yaml" "madgwick" 32 2 8 0.3 "concat"
    train_model "${CONFIG_DIR}/madgwick_filter.yaml" "madgwick_model" "madgwick"
    
    # Train with Complementary filter
    log "INFO" "======================================================"
    log "INFO" "Starting Complementary filter training"
    log "INFO" "======================================================"
    create_config "${CONFIG_DIR}/comp_filter.yaml" "comp" 32 2 8 0.3 "concat"
    train_model "${CONFIG_DIR}/comp_filter.yaml" "comp_model" "comp"
    
    # Train with standard Kalman filter
    log "INFO" "======================================================"
    log "INFO" "Starting Kalman filter training"
    log "INFO" "======================================================"
    create_config "${CONFIG_DIR}/kalman_filter.yaml" "kalman" 48 3 8 0.25 "concat"
    train_model "${CONFIG_DIR}/kalman_filter.yaml" "kalman_model" "kalman"
    
    # Train with Extended Kalman filter
    log "INFO" "======================================================"
    log "INFO" "Starting Extended Kalman filter training"
    log "INFO" "======================================================"
    create_config "${CONFIG_DIR}/ekf_filter.yaml" "ekf" 48 3 8 0.25 "concat"
    train_model "${CONFIG_DIR}/ekf_filter.yaml" "ekf_model" "ekf"
    
    # Train with Unscented Kalman filter
    log "INFO" "======================================================"
    log "INFO" "Starting Unscented Kalman filter training"
    log "INFO" "======================================================"
    create_config "${CONFIG_DIR}/ukf_filter.yaml" "ukf" 64 3 12 0.2 "concat"
    train_model "${CONFIG_DIR}/ukf_filter.yaml" "ukf_model" "ukf"
    
    # Print final results
    log "INFO" "======================================================"
    log "INFO" "===== FINAL RESULTS ====="
    cat "${RESULT_DIR}/comparison.csv" | column -t -s ','
    log "INFO" "Complete results saved to ${RESULT_DIR}/comparison.csv"
    log "INFO" "Training completed successfully"
    log "INFO" "======================================================"
}

# Execute main function
main
