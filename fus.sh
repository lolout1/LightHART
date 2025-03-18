#!/bin/bash

# Comprehensive training script for fall detection with multiple filter types
# Includes thorough debugging, visualization, and performance comparison

# Set strict error handling
set -e

# Set common parameters
DEVICE="0,1"  # Use both GPUs
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
    
    # Check CUDA - Using heredoc syntax
    python << 'PYEOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA device: None")
PYEOF
    
    # Check dataset file structure
    log "INFO" "Checking dataset structure..."
    ls -la data/smartfallmm/young/
    
    # Test IMU fusion module - Using heredoc syntax with explicit marker
    log "INFO" "Testing IMU fusion module..."
    python << 'PYEOF'
import sys
import traceback
try:
    # Import the specific filter classes we want to test
    from utils.imu_fusion import (
        MadgwickFilter, 
        KalmanFilter, 
        ExtendedKalmanFilter, 
        UnscentedKalmanFilter, 
        process_imu_data
    )
    import numpy as np
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("imu_test")
    logger.info("Starting IMU fusion test")

    # Create dummy data
    acc = np.array([[0, 0, 9.8], [0.1, 0, 9.7], [0.2, 0.1, 9.7]])
    gyro = np.array([[0, 0, 0], [0.01, 0.01, 0], [0.02, 0.02, 0]])
    logger.info(f"Created test data: acc={acc.shape}, gyro={gyro.shape}")

    # Test only the filters that we've explicitly imported
    filter_classes = {
        'madgwick': MadgwickFilter,
        'kalman': KalmanFilter,
        'ekf': ExtendedKalmanFilter,
        'ukf': UnscentedKalmanFilter
    }
    
    for name, filter_class in filter_classes.items():
        logger.info(f"Testing {name} filter")
        filter_obj = filter_class()
        q1 = filter_obj.update(acc[0], gyro[0])
        q2 = filter_obj.update(acc[1], gyro[1])
        print(f"{name} quaternion: {q2}")
    
    # Test process_imu_data function
    logger.info("Testing process_imu_data function")
    result = process_imu_data(
        acc_data=acc, 
        gyro_data=gyro, 
        filter_type='madgwick',
        return_features=True
    )
    logger.info(f"process_imu_data result keys: {list(result.keys())}")
    if 'quaternion' in result:
        logger.info(f"quaternion shape: {result['quaternion'].shape}")
    
    print("IMU fusion module tests passed!")
except Exception as e:
    print(f"IMU fusion test failed: {str(e)}")
    traceback.print_exc()
    # Don't exit with error here - we want the script to continue
PYEOF
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

# Include all subjects for better generalization
subjects: [29, 30, 31, 33, 45, 46]

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


# Function to generate comparison plots
# Function to train model with enhanced error detection
train_model() {
    local config_file=$1
    local model_name=$2
    local work_dir="${RESULT_DIR}/${model_name}"
    local filter_type=$3
    
    log "INFO" "========================================================"
    log "INFO" "Training model: $model_name with $filter_type filter"
    log "INFO" "========================================================"
    
    # Create work directory
    mkdir -p $work_dir
    
    # Parse device string into array for correct argument passing
    # Convert comma-separated string to space-separated for main.py
    IFS=',' read -ra DEVICE_ARRAY <<< "$DEVICE"
    DEVICE_ARGS=""
    for dev in "${DEVICE_ARRAY[@]}"; do
        DEVICE_ARGS="$DEVICE_ARGS $dev"
    done
    
    # Log detailed information
    log "INFO" "Using devices:$DEVICE_ARGS"
    log "INFO" "Config file: $config_file"
    log "INFO" "Output directory: $work_dir"
    
    # Run training with proper device arguments
    log "INFO" "Starting training phase..."
    CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
        --config $config_file \
        --work-dir $work_dir \
        --model-saved-name $model_name \
        --device$DEVICE_ARGS \
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
        # Don't proceed to testing if training failed
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
    CUDA_VISIBLE_DEVICES=${DEVICE_ARRAY[0]} python main.py \
        --config $config_file \
        --work-dir $work_dir \
        --weights "${work_dir}/${model_name}.pt" \
        --device ${DEVICE_ARRAY[0]} \
        --phase 'test' 2>&1 | tee "${LOG_DIR}/${model_name}_test.log"
    
    local test_status=$?
    if [ $test_status -ne 0 ]; then
        log "ERROR" "Testing failed with exit code $test_status"
        log "ERROR" "Check ${LOG_DIR}/${model_name}_test.log for details"
        return 1
    fi
    
    # Extract metrics with robust error handling
    log "INFO" "Extracting performance metrics..."
    local test_log="${LOG_DIR}/${model_name}_test.log"
    
    # Use safer extraction with null checks
    if [ -f "$test_log" ]; then
        local accuracy=$(grep -oP "Test accuracy: \K[0-9.]+%" "$test_log" 2>/dev/null | tail -1 | tr -d '%' || echo "N/A")
        local f1=$(grep -oP "Test F.*Score: \K[0-9.]+" "$test_log" 2>/dev/null | tail -1 || echo "N/A")
        local precision=$(grep -oP "Test precision: \K[0-9.]+" "$test_log" 2>/dev/null | tail -1 || echo "N/A")
        local recall=$(grep -oP "Test recall: \K[0-9.]+" "$test_log" 2>/dev/null | tail -1 || echo "N/A")
        
        log "INFO" "Results for $model_name:"
        log "INFO" "  Accuracy:  ${accuracy}%"
        log "INFO" "  F1-Score:  ${f1}"
        log "INFO" "  Precision: ${precision}"
        log "INFO" "  Recall:    ${recall}"
        
        # Add to comparison table
        echo "$model_name,$filter_type,$accuracy,$f1,$precision,$recall" >> "${RESULT_DIR}/comparison.csv"
    else
        log "ERROR" "Test log file not found: $test_log"
        return 1
    fi
    
    log "INFO" "Model training and evaluation completed successfully"
    return 0
}
create_comparison_plots() {
    log "INFO" "Generating comparison plots..."
    
    # Use heredoc syntax for Python script
    python << 'PYEOF'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    # Read comparison data
    comparison_file = 'results_comparison/comparison.csv'
    if not os.path.exists(comparison_file):
        print(f"Error: Comparison file not found at {comparison_file}")
        exit(1)
    
    df = pd.read_csv(comparison_file)
    print(f"Loaded comparison data with {len(df)} rows")
    
    # Set figure size
    plt.figure(figsize=(12, 8))
    
    # Bar width
    bar_width = 0.2
    r1 = np.arange(len(df))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    # Make sure all values are numeric
    for col in ['accuracy', 'f1', 'precision', 'recall']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
    viz_dir = 'results_comparison'
    os.makedirs(viz_dir, exist_ok=True)
    plt.savefig(f'{viz_dir}/performance_comparison.png', dpi=300)
    plt.close()
    
    print(f"Comparison plot generated at {viz_dir}/performance_comparison.png")
except Exception as e:
    import traceback
    print(f"Error generating plot: {e}")
    traceback.print_exc()
PYEOF
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
    create_config "config/smartfallmm/madgwick_fusion.yaml" "madgwick" 32 3 8 "concat"
    train_model "config/smartfallmm/madgwick_fusion.yaml" "madgwick_model" "madgwick"
    
    # Train with standard Kalman filter
    log "INFO" "======================================================"
    log "INFO" "Starting Kalman filter training"
    log "INFO" "======================================================"
    create_config "config/smartfallmm/kalman_fusion.yaml" "kalman" 32 3 8 "concat"
    train_model "config/smartfallmm/kalman_fusion.yaml" "kalman_model" "kalman"
    
    # Train with Extended Kalman filter
    log "INFO" "======================================================"
    log "INFO" "Starting Extended Kalman filter training"
    log "INFO" "======================================================"
    create_config "config/smartfallmm/ekf_fusion.yaml" "ekf" 32 3 8 "concat"
    train_model "config/smartfallmm/ekf_fusion.yaml" "ekf_model" "ekf"
    
    # Train with Unscented Kalman filter
    log "INFO" "======================================================"
    log "INFO" "Starting Unscented Kalman filter training"
    log "INFO" "======================================================"
    create_config "config/smartfallmm/ukf_fusion.yaml" "ukf" 48 3 12 "concat"
    train_model "config/smartfallmm/ukf_fusion.yaml" "ukf_model" "ukf"
    
    # Train baseline accelerometer-only model for comparison
    log "INFO" "======================================================"
    log "INFO" "Starting baseline (no fusion) model training"
    log "INFO" "======================================================"
    create_config "config/smartfallmm/acc_only.yaml" "none" 32 3 8 "acc_only"
    train_model "config/smartfallmm/acc_only.yaml" "acc_only_model" "none"
    
    # Generate comparison visualizations
    create_comparison_plots
    
    # Print final results
    log "INFO" "======================================================"
    log "INFO" "===== FINAL RESULTS ====="
    cat "${RESULT_DIR}/comparison.csv" | column -t -s ','
    log "INFO" "Complete results saved to ${RESULT_DIR}/comparison.csv"
    log "INFO" "Performance visualization saved to ${RESULT_DIR}/performance_comparison.png"
    log "INFO" "Training completed successfully"
    log "INFO" "======================================================"
}

# Execute main function
main
