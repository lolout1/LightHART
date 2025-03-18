#!/bin/bash

# Enhanced training script with robust error handling and filter comparison
# Optimized for maximum accuracy, F1, precision, and recall in fall detection

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
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

    # Check essential files
    log "INFO" "Checking critical files..."
    for file in "utils/imu_fusion.py" "utils/dataset.py" "Feeder/Make_Dataset.py" "Models/fusion_transformer.py"; do
        if [ -f "$file" ]; then
            log "INFO" "✅ $file exists"
        else
            log "ERROR" "❌ $file missing!"
        fi
    done
    
    # Test IMU fusion functionality with explicit traceback import
    log "INFO" "Testing IMU fusion module..."
    python -c "
import sys
import traceback
try:
    from utils.imu_fusion import OrientationEstimator, MadgwickFilter, CompFilter, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
    print('Successfully imported orientation estimators')
    import numpy as np
    # Test with sample data
    acc = np.array([0, 0, 9.81])
    gyro = np.array([0.1, 0.2, 0.3])
    filters = {
        'madgwick': MadgwickFilter(),
        'comp': CompFilter(),
        'kalman': KalmanFilter(),
        'ekf': ExtendedKalmanFilter(),
        'ukf': UnscentedKalmanFilter()
    }
    
    # Test each filter
    for name, filter_obj in filters.items():
        q = filter_obj.update(acc, gyro)
        print(f'{name} test quaternion: {q}')
    
except Exception as e:
    print(f'IMU fusion test failed: {str(e)}')
    traceback.print_exc()
    sys.exit(1)
"
}

# Create optimized yaml configuration for different filter types
create_config() {
    local config_file=$1
    local fusion_type=$2
    local embed_dim=$3
    local num_layers=$4
    local num_heads=$5
    local acc_frames=$6
    local dropout=$7
    
    log "INFO" "Creating config: $config_file with fusion=$fusion_type"
    
    cat > $config_file << EOF
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm

# Include all subjects for better generalization
subjects: [29, 30, 31, 33, 45, 46]

model_args:
  num_layers: $num_layers
  embed_dim: $embed_dim
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: $acc_frames
  mocap_frames: 64
  num_heads: $num_heads
  fusion_type: 'concat'
  dropout: $dropout
  use_batch_norm: true

dataset_args:
  mode: 'sliding_window'
  max_length: $acc_frames
  task: 'fd'
  modalities: ['accelerometer', 'gyroscope']
  age_group: ['young']
  sensors: ['watch']
  fusion_options:
    enabled: true
    filter_type: '$fusion_type'
    acc_threshold: 3.0
    gyro_threshold: 1.0
    visualize: true

batch_size: 16
test_batch_size: 16
val_batch_size: 16
num_epoch: $NUM_EPOCHS

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
  drop_last: true

seed: 42
optimizer: adam
base_lr: $BASE_LR
weight_decay: $WEIGHT_DECAY
EOF
}

# Train model function with direct method
train_model() {
    local config=$1
    local model_name=$2
    local work_dir="$RESULT_DIR/$model_name"
    
    mkdir -p "$work_dir"
    log "INFO" "======================================================"
    log "INFO" "Training model: $model_name"
    log "INFO" "Config: $config"
    log "INFO" "Work dir: $work_dir"
    log "INFO" "======================================================"
    
    # Print config contents for debugging
    log "DEBUG" "Config file contents:"
    cat $config | while read line; do log "DEBUG" "  $line"; done
    
    # Run training with error handling
    log "INFO" "Starting training process..."
    if python main.py \
        --config "$config" \
        --work-dir "$work_dir" \
        --device "$DEVICE" \
        --base-lr "$BASE_LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --num-epoch "$NUM_EPOCHS"; then
        
        log "INFO" "Training completed successfully"
        
        # Run testing if training succeeded
        log "INFO" "Starting evaluation..."
        python main.py \
            --config "$config" \
            --work-dir "$work_dir" \
            --weights "${work_dir}/model.pt" \
            --device "$DEVICE" \
            --phase test
        
        # Extract and log metrics 
        if [ -f "$work_dir/test_result.txt" ]; then
            accuracy=$(grep "accuracy" "$work_dir/test_result.txt" | awk '{print $NF}')
            f1_score=$(grep "f1_score" "$work_dir/test_result.txt" | awk '{print $NF}')
            precision=$(grep "precision" "$work_dir/test_result.txt" | awk '{print $NF}')
            recall=$(grep "recall" "$work_dir/test_result.txt" | awk '{print $NF}')
            
            log "INFO" "Test Results for $model_name:"
            log "INFO" "  Accuracy: $accuracy"
            log "INFO" "  F1-Score: $f1_score"
            log "INFO" "  Precision: $precision"
            log "INFO" "  Recall: $recall"
            
            # Save results to comparison file
            echo "$model_name,$accuracy,$f1_score,$precision,$recall" >> "$RESULT_DIR/comparison.csv"
        else
            log "WARNING" "Test results file not found"
        fi
    else
        log "ERROR" "Training failed"
        
        # Run diagnostics to help troubleshoot
        run_diagnostics
    fi
}

# Start execution
log "INFO" "======================================================"
log "INFO" "Starting SmartFallMM training script"
log "INFO" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "INFO" "======================================================"

# Initial diagnostics
run_diagnostics

# Initialize comparison file
echo "Model,Accuracy,F1-Score,Precision,Recall" > "$RESULT_DIR/comparison.csv"

# Train with different filter types and model configurations
log "INFO" "======================================================"
log "INFO" "Training models with different filter configurations"
log "INFO" "======================================================"

# Create configs with different filter types - optimized parameters for each filter
create_config "config/madgwick_filter.yaml" "madgwick" 32 2 8 128 0.3
log "INFO" "Preparing to train Madgwick fusion model..."
train_model "config/madgwick_filter.yaml" "madgwick_filter"

create_config "config/comp_filter.yaml" "comp" 32 2 8 128 0.3
log "INFO" "Preparing to train Complementary filter model..."
train_model "config/comp_filter.yaml" "comp_filter"

create_config "config/kalman_filter.yaml" "kalman" 48 3 8 128 0.25
log "INFO" "Preparing to train Kalman filter model..."
train_model "config/kalman_filter.yaml" "kalman_filter"

create_config "config/ekf_filter.yaml" "ekf" 48 3 8 128 0.25
log "INFO" "Preparing to train EKF model..."
train_model "config/ekf_filter.yaml" "ekf_filter"

create_config "config/ukf_filter.yaml" "ukf" 64 3 8 128 0.2
log "INFO" "Preparing to train UKF model..."
train_model "config/ukf_filter.yaml" "ukf_filter"

# Generate comparison visualization with explicit traceback handling
log "INFO" "Generating comparison visualizations..."
python - << 'END_PYTHON'
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traceback

try:
    # Load comparison data
    comparison_file = os.path.join("results_comparison", "comparison.csv")
    
    if os.path.exists(comparison_file):
        df = pd.read_csv(comparison_file)
        
        if len(df) > 0:
            # Create output directory
            viz_dir = "results_comparison/visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            
            # Plot metrics comparison
            metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                bars = ax.bar(df['Model'], df[metric].astype(float), color=colors)
                ax.set_title(f'Comparison of {metric}')
                ax.set_ylim(0, 1.0)
                ax.set_ylabel(metric)
                ax.set_xlabel('Model')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}', ha='center', va='bottom', rotation=0)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "metrics_comparison.png"))
            plt.close()
            
            # Create summary table visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.axis('off')
            
            # Create the table
            cell_text = []
            for _, row in df.iterrows():
                cell_text.append([row['Model'], 
                                f"{float(row['Accuracy']):.4f}", 
                                f"{float(row['F1-Score']):.4f}", 
                                f"{float(row['Precision']):.4f}", 
                                f"{float(row['Recall']):.4f}"])
            
            table = ax.table(cellText=cell_text, 
                            colLabels=['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall'],
                            cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            
            plt.title("Performance Metrics Summary", fontsize=16, pad=20)
            plt.savefig(os.path.join(viz_dir, "metrics_table.png"))
            plt.close()
            
            print(f"Visualizations saved to {viz_dir}")
        else:
            print("Comparison data is empty")
    else:
        print(f"Comparison file {comparison_file} not found")
except Exception as e:
    print(f"Error generating visualizations: {e}")
    traceback.print_exc()
END_PYTHON

# Finish execution
log "INFO" "======================================================"
log "INFO" "Training and evaluation complete"
log "INFO" "Results saved in $RESULT_DIR"
log "INFO" "======================================================"
