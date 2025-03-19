#!/bin/bash

# Script to compare different models and filter types for IMU fusion in fall detection
# This script runs a comprehensive evaluation of:
# 1. Accelerometer-only model
# 2. Accelerometer + Quaternion model (with different filters)
# 3. Full sensor fusion model (acc + gyro + quaternion)

# Set strict error handling
set -e

# Configuration
DEVICE="0,1"  # GPU devices
BASE_LR=0.0005
WEIGHT_DECAY=0.001
NUM_EPOCHS=60
RESULTS_DIR="model_comparison_results"
CONFIG_DIR="config/model_comparison"

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

# Create configuration file for a model type
create_config() {
    local config_file=$1
    local model_type=$2
    local filter_type=${3:-"madgwick"}  # Default filter if not specified
    
    log "INFO" "Creating config for $model_type model with $filter_type filter: $config_file"
    
    # Common configuration
    cat > $config_file << EOL
dataset: smartfallmm

subjects: [29, 30, 31, 33, 45, 46, 34, 37, 39, 38, 43, 35, 36, 44, 32]

model_args:
  num_layers: 3
  embed_dim: 32
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: 64
  mocap_frames: 64
  num_heads: 4
  dropout: 0.3
  use_batch_norm: true
EOL

    # Model-specific configuration
    if [[ "$model_type" == "acc_only" ]]; then
        cat >> $config_file << EOL
model: Models.acc_only_model.AccOnlyModel

dataset_args:
  mode: 'sliding_window'
  max_length: 64
  window_stride: 10
  task: 'fd'
  modalities: ['accelerometer']
  age_group: ['young']
  sensors: ['watch']
EOL
    elif [[ "$model_type" == "acc_quat" ]]; then
        cat >> $config_file << EOL
model: Models.acc_quat_model.AccQuatModel

model_args:
  feature_dim: 64  # For acc + quat

dataset_args:
  mode: 'sliding_window'
  max_length: 64
  window_stride: 10
  task: 'fd'
  modalities: ['accelerometer', 'gyroscope']  # Gyro needed to generate quaternions
  age_group: ['young']
  sensors: ['watch']
  fusion_options:
    enabled: true
    filter_type: '$filter_type'
    acc_threshold: 3.0
    gyro_threshold: 1.0
    visualize: false
    save_aligned: true
EOL
    elif [[ "$model_type" == "full_fusion" ]]; then
        cat >> $config_file << EOL
model: Models.fusion_transformer.FusionTransModel

model_args:
  fusion_type: 'concat'
  feature_dim: 96  # For acc + gyro + quat

dataset_args:
  mode: 'sliding_window'
  max_length: 64
  window_stride: 10
  task: 'fd'
  modalities: ['accelerometer', 'gyroscope']
  age_group: ['young']
  sensors: ['watch']
  fusion_options:
    enabled: true
    filter_type: '$filter_type'
    acc_threshold: 3.0
    gyro_threshold: 1.0
    visualize: false
    save_aligned: true
EOL
    fi

    # Common configuration continued
    cat >> $config_file << EOL

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
    - [43, 35, 36]  # Fold 1: ~38.3% falls
    - [44, 34, 32]  # Fold 2: ~39.7% falls
    - [45, 37, 38]  # Fold 3: ~44.8% falls
    - [46, 29, 31]  # Fold 4: ~41.4% falls
    - [30, 39]      # Fold 5: ~43.3% falls
EOL
}

# Train a model with a specific configuration
train_model() {
    local config_file=$1
    local model_name=$2
    local output_dir="$RESULTS_DIR/$model_name"
    
    log "INFO" "Training model: $model_name"
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
        return 1
    fi
    
    # Extract and log key metrics
    if [ -f "$output_dir/cv_summary.json" ]; then
        log "INFO" "Cross-validation results for $model_name:"
        
        # Extract metrics using Python
        python -c "
import json
import sys
with open('$output_dir/cv_summary.json') as f:
    data = json.load(f)
    metrics = data['average_metrics']
    print(f\"Accuracy: {metrics.get('accuracy', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}\")
    print(f\"F1 score: {metrics.get('f1', 0):.4f} ± {metrics.get('f1_std', 0):.4f}\")
    print(f\"Precision: {metrics.get('precision', 0):.4f} ± {metrics.get('precision_std', 0):.4f}\")
    print(f\"Recall: {metrics.get('recall', 0):.4f} ± {metrics.get('recall_std', 0):.4f}\")
    
    # Save to comparison CSV
    with open('$RESULTS_DIR/model_comparison.csv', 'a') as csv:
        model_type = '$model_name'.split('_')[0] if '_' in '$model_name' else '$model_name'
        filter_type = '$model_name'.split('_')[1] if '_' in '$model_name' and len('$model_name'.split('_')) > 1 else 'none'
        csv.write(f\"{model_type},{filter_type},{metrics.get('accuracy', 0):.6f},{metrics.get('f1', 0):.6f},{metrics.get('precision', 0):.6f},{metrics.get('recall', 0):.6f},{metrics.get('balanced_accuracy', 0):.6f}\\n\")
"
    else
        log "ERROR" "No cross-validation summary found for $model_name"
    fi
    
    return 0
}

# Main function to run the model comparison
main() {
    log "INFO" "Starting comprehensive model comparison for fall detection"
    
    # Create comparison CSV header
    echo "model_type,filter_type,accuracy,f1,precision,recall,balanced_accuracy" > "$RESULTS_DIR/model_comparison.csv"
    
    # 1. Accelerometer-only model
    log "INFO" "============= CONFIGURING ACCELEROMETER-ONLY MODEL ============="
    create_config "$CONFIG_DIR/acc_only.yaml" "acc_only"
    
    # 2. Accelerometer + Quaternion models with different filters
    log "INFO" "============= CONFIGURING ACC + QUATERNION MODELS ============="
    create_config "$CONFIG_DIR/acc_quat_madgwick.yaml" "acc_quat" "madgwick"
    create_config "$CONFIG_DIR/acc_quat_kalman.yaml" "acc_quat" "kalman"
    create_config "$CONFIG_DIR/acc_quat_ekf.yaml" "acc_quat" "ekf"
    create_config "$CONFIG_DIR/acc_quat_ukf.yaml" "acc_quat" "ukf"
    
    # 3. Full fusion models with different filters
    log "INFO" "============= CONFIGURING FULL FUSION MODELS ============="
    create_config "$CONFIG_DIR/full_fusion_madgwick.yaml" "full_fusion" "madgwick"
    create_config "$CONFIG_DIR/full_fusion_kalman.yaml" "full_fusion" "kalman"
    create_config "$CONFIG_DIR/full_fusion_ekf.yaml" "full_fusion" "ekf"
    create_config "$CONFIG_DIR/full_fusion_ukf.yaml" "full_fusion" "ukf"
    
    # Train all models
    log "INFO" "============= TRAINING ACCELEROMETER-ONLY MODEL ============="
    train_model "$CONFIG_DIR/acc_only.yaml" "acc_only"
    
    log "INFO" "============= TRAINING ACC + QUATERNION MODELS ============="
    train_model "$CONFIG_DIR/acc_quat_madgwick.yaml" "acc_quat_madgwick"
    train_model "$CONFIG_DIR/acc_quat_kalman.yaml" "acc_quat_kalman"
    train_model "$CONFIG_DIR/acc_quat_ekf.yaml" "acc_quat_ekf"
    train_model "$CONFIG_DIR/acc_quat_ukf.yaml" "acc_quat_ukf"
    
    log "INFO" "============= TRAINING FULL FUSION MODELS ============="
    train_model "$CONFIG_DIR/full_fusion_madgwick.yaml" "full_fusion_madgwick"
    train_model "$CONFIG_DIR/full_fusion_kalman.yaml" "full_fusion_kalman"
    train_model "$CONFIG_DIR/full_fusion_ekf.yaml" "full_fusion_ekf"
    train_model "$CONFIG_DIR/full_fusion_ukf.yaml" "full_fusion_ukf"
    
    # Generate comprehensive comparison results
    log "INFO" "Generating comprehensive model comparison analysis"
    python -c "
from utils.filter_comparison import process_comparison_results
process_comparison_results('$RESULTS_DIR')
"
    
    log "INFO" "Model comparison completed successfully"
    log "INFO" "Results available in $RESULTS_DIR"
    
    # Generate and display model feature importance analysis
    log "INFO" "Analyzing feature importance across models"
    python -c "
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load comparison data
comparison_file = os.path.join('$RESULTS_DIR', 'model_comparison.csv')
if os.path.exists(comparison_file):
    df = pd.read_csv(comparison_file)
    
    # Group models by type and filter
    df['model_category'] = df['model_type'].apply(lambda x: 'Accelerometer Only' if x == 'acc_only' 
                                                 else ('Acc + Quaternion' if x == 'acc_quat' 
                                                      else 'Full Fusion'))
    
    # Find best performers by category
    best_models = df.loc[df.groupby('model_category')['f1'].idxmax()]
    
    # Create bar chart comparing best models from each category
    plt.figure(figsize=(12, 8))
    sns.barplot(x='model_category', y='f1', data=best_models)
    plt.title('Feature Importance: F1 Score by Model Type')
    plt.xlabel('Model Type')
    plt.ylabel('F1 Score')
    plt.ylim(0, 100)
    
    # Add filter types as annotations
    for i, row in enumerate(best_models.itertuples()):
        plt.text(i, row.f1 + 2, f'Filter: {row.filter_type}', 
                ha='center', va='bottom', rotation=0,
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add statistical significance indicators
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('$RESULTS_DIR', 'feature_importance.png'))
    
    # Print summary
    print('\\nFeature Importance Summary:')
    print('============================')
    for i, row in enumerate(best_models.itertuples()):
        print(f'{row.model_category} with {row.filter_type} filter: F1={row.f1:.2f}, Accuracy={row.accuracy:.2f}%')
    
    # Calculate improvements
    if len(best_models) > 1:
        base_model = best_models[best_models['model_category'] == 'Accelerometer Only']
        if not base_model.empty:
            base_f1 = base_model['f1'].values[0]
            for i, row in enumerate(best_models.itertuples()):
                if row.model_category != 'Accelerometer Only':
                    improvement = row.f1 - base_f1
                    pct_improvement = (improvement / base_f1) * 100
                    print(f'  • {row.model_category} improves F1 by {improvement:.2f} points ({pct_improvement:.1f}%)')
    
    # Generate recommendation
    best_overall = best_models.loc[best_models['f1'].idxmax()]
    print('\\nRecommendation:')
    print(f'The best overall model is {best_overall.model_category} with {best_overall.filter_type} filter')
    print(f'F1={best_overall.f1:.2f}, Precision={best_overall.precision:.2f}%, Recall={best_overall.recall:.2f}%')
else:
    print('No comparison data found')
"

    echo -e "\n====================================================="
    echo -e "Model comparison complete. All results have been processed."
    echo -e "See $RESULTS_DIR/filter_comparison_report.md for comprehensive analysis."
    echo -e "====================================================="
}

# Run the main function
main
