#!/bin/bash

# Enhanced debugging script for IMU filter training
# Adds improved diagnostics to identify why training isn't proceeding

# Set error handling
set -e

# Configuration
DEVICE="0,1"
BASE_LR=0.0005
WEIGHT_DECAY=0.001
NUM_EPOCHS=60
OUTPUT_DIR="filter_training_debug"
CONFIG_DIR="config/filter_debug"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $CONFIG_DIR
mkdir -p "$OUTPUT_DIR/logs"

# Log function with timestamp
log() {
    local level=$1
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $2"
    echo "$msg"
    echo "$msg" >> "$OUTPUT_DIR/logs/debug.log"
}

# Create simplified test configuration for debugging
create_debug_config() {
    local config_file=$1
    local filter_type=$2
    
    log "INFO" "Creating debug config for $filter_type filter: $config_file"
    
    cat > $config_file << EOL
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm

# Using a small subset of subjects for debugging
subjects: [29, 30, 31]

model_args:
  num_layers: 2
  embed_dim: 32
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: 64
  mocap_frames: 64
  num_heads: 4
  fusion_type: 'concat'
  dropout: 0.3
  use_batch_norm: true
  feature_dim: 96

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

batch_size: 16
test_batch_size: 16
val_batch_size: 16
num_epoch: 10

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

# Disable k-fold for simpler debugging
kfold:
  enabled: false
EOL
}

# Run training with extra debug flags
debug_train_model() {
    local config_file=$1
    local model_name=$2
    local filter_type=$3
    local output_dir="$OUTPUT_DIR/${model_name}"
    
    log "INFO" "Starting debug training for $filter_type filter"
    log "INFO" "Model: $model_name, Config: $config_file"
    
    mkdir -p "$output_dir"
    
    # First, try to identify the dataset loading issue
    log "DEBUG" "Testing dataset loading only..."
    
    CUDA_VISIBLE_DEVICES=$DEVICE python -c "
import yaml
import sys

try:
    from utils.dataset import prepare_smartfallmm, split_by_subjects
    
    # Load config
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a dummy arg object
    class Args:
        pass
    args = Args()
    
    # Copy config to args
    for key, value in config.items():
        setattr(args, key, value)
    
    # Try dataset preparation
    print('Preparing dataset...')
    builder = prepare_smartfallmm(args)
    
    # Try data splitting
    print('Splitting data for subjects:', args.subjects)
    fuse = args.dataset_args['fusion_options']['enabled']
    data = split_by_subjects(builder, args.subjects, fuse)
    
    # Check if any data was loaded
    if data:
        print('Data keys:', data.keys())
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f'{key} shape: {value.shape}')
            elif isinstance(value, list):
                print(f'{key} length: {len(value)}')
    else:
        print('No data was loaded!')
    
    print('Dataset test completed successfully')
    
except Exception as e:
    print(f'Error in dataset loading: {str(e)}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee "$OUTPUT_DIR/logs/${model_name}_dataset_test.log"
    
    dataset_status=$?
    if [ $dataset_status -ne 0 ]; then
        log "ERROR" "Dataset loading failed for $model_name - see log for details"
        return 1
    fi
    
    # Now run the actual training with verbose output
    log "INFO" "Starting training with verbose output..."
    
    # Flag to help trace execution path
    export PYTHONPATH=$(pwd):$PYTHONPATH
    
    CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
        --config $config_file \
        --work-dir $output_dir \
        --model-saved-name $model_name \
        --device 0 1 \
        --multi-gpu True \
        --kfold False \
        --parallel-threads 48 \
        --num-epoch 10 \
        --print-log True \
        --phase train \
        --verbose True 2>&1 | tee "$OUTPUT_DIR/logs/${model_name}_train.log"
    
    exit_code=$?
    log "INFO" "Training process completed with exit code: $exit_code"
    
    # Check for model file
    if [ -f "$output_dir/${model_name}.pt" ]; then
        log "SUCCESS" "Model file successfully created: $output_dir/${model_name}.pt"
        return 0
    else
        log "ERROR" "Model file was not created after training"
        
        # Check if any Python errors were logged
        if grep -q "Error\|Exception\|Traceback" "$OUTPUT_DIR/logs/${model_name}_train.log"; then
            log "ERROR" "Found error in training log:"
            grep -A 10 "Error\|Exception\|Traceback" "$OUTPUT_DIR/logs/${model_name}_train.log" | head -15
        fi
        
        # Check for log file created by the trainer
        if [ -f "$output_dir/log.txt" ]; then
            log "DEBUG" "Contents of trainer log file (last 20 lines):"
            tail -20 "$output_dir/log.txt" >> "$OUTPUT_DIR/logs/debug.log"
        else
            log "WARNING" "No trainer log file was created"
        fi
        
        return 1
    fi
}

# Main debug function
main() {
    log "INFO" "Starting IMU filter debugging process"
    
    # Diagnostics: Check environment
    log "DEBUG" "Python version:"
    python --version 2>&1 | tee -a "$OUTPUT_DIR/logs/debug.log"
    
    log "DEBUG" "PyTorch version and CUDA availability:"
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1 | tee -a "$OUTPUT_DIR/logs/debug.log"
    
    # Test with just one filter to simplify debugging
    create_debug_config "$CONFIG_DIR/madgwick_debug.yaml" "madgwick"
    
    # Check that the config file was created correctly
    log "DEBUG" "Config file content:"
    cat "$CONFIG_DIR/madgwick_debug.yaml" | tee -a "$OUTPUT_DIR/logs/debug.log"
    
    # Try running the debug training
    log "INFO" "=== STARTING DEBUG TRAINING WITH MADGWICK FILTER ==="
    if debug_train_model "$CONFIG_DIR/madgwick_debug.yaml" "madgwick_debug" "madgwick"; then
        log "SUCCESS" "Debug training completed successfully!"
    else
        log "ERROR" "Debug training failed - check logs for details"
    fi
    
    log "INFO" "Debug session complete. Check $OUTPUT_DIR/logs for detailed information."
}

# Run the debug session
main
