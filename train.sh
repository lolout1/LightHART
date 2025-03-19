#!/bin/bash
#
# Comprehensive IMU Filter Comparison Script
# 
# This script trains models with different IMU fusion filters (Madgwick, Kalman, EKF)
# and generates comparative analysis of their performance for fall detection.
#
# Features:
# - Creates separate configurations for each filter type
# - Runs k-fold cross-validation for robust performance assessment
# - Collects and aggregates metrics across folds
# - Generates comparative visualizations and reports
# - Implements robust error handling and logging

# Enable strict error handling
set -e                  # Exit immediately if a command fails
set -o pipefail         # Pipeline fails if any command fails
set -u                  # Treat unset variables as errors

# Configuration variables
DEVICE="0,1"                        # GPU devices
BASE_LR=0.0005                      # Learning rate
WEIGHT_DECAY=0.001                  # Weight decay
NUM_EPOCHS=60                       # Number of epochs
RESULTS_DIR="filter_comparison_results"   # Results directory
CONFIG_DIR="config/filter_comparison"     # Configuration directory
SCRIPTS_DIR="filter_scripts"        # Directory for helper scripts

# Create necessary directories
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/visualizations"
mkdir -p "${CONFIG_DIR}"
mkdir -p "${SCRIPTS_DIR}"

# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------

# Logging function with timestamps
log() {
    local level="$1"
    local msg="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[${timestamp}] [${level}] ${msg}"
    echo "[${timestamp}] [${level}] ${msg}" >> "${RESULTS_DIR}/logs/main.log"
}

# Function to check if previous command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        log "ERROR" "$1"
        return 1
    fi
    return 0
}

# -----------------------------------------
# CREATE PYTHON HELPER SCRIPTS
# -----------------------------------------

# Create script for extracting metrics from CV summary
cat > "${SCRIPTS_DIR}/extract_metrics.py" << 'EOF'
#!/usr/bin/env python3
"""
Extract and log metrics from cross-validation summary files.
"""
import json
import sys
import os
import argparse
from typing import Dict, Any, Optional, List

def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file with error handling."""
    try:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
        
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def extract_metrics(cv_summary_path: str, model_name: str, filter_type: str, output_csv: str) -> bool:
    """Extract metrics from cross-validation summary and write to CSV."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Load the CV summary
    data = load_json(cv_summary_path)
    if not data:
        # Write zeros to output if file can't be loaded
        with open(output_csv, 'a') as f:
            f.write(f"{model_name},{filter_type},0.0,0.0,0.0,0.0,0.0\n")
        return False
    
    try:
        # Extract metrics
        metrics = data.get('average_metrics', {})
        
        # Print metrics to console
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}")
        print(f"F1 score:  {metrics.get('f1', 0):.4f} ± {metrics.get('f1_std', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f} ± {metrics.get('precision_std', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f} ± {metrics.get('recall_std', 0):.4f}")
        
        # Write to CSV file
        with open(output_csv, 'a') as f:
            f.write(f"{model_name},{filter_type},"
                   f"{metrics.get('accuracy', 0):.6f},"
                   f"{metrics.get('f1', 0):.6f},"
                   f"{metrics.get('precision', 0):.6f},"
                   f"{metrics.get('recall', 0):.6f},"
                   f"{metrics.get('balanced_accuracy', 0):.6f}\n")
        return True
    except Exception as e:
        print(f"Error extracting metrics: {str(e)}")
        # Write zeros on error
        with open(output_csv, 'a') as f:
            f.write(f"{model_name},{filter_type},0.0,0.0,0.0,0.0,0.0\n")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract metrics from CV summary')
    parser.add_argument('cv_summary_path', help='Path to CV summary JSON file')
    parser.add_argument('model_name', help='Model name')
    parser.add_argument('filter_type', help='Filter type')
    parser.add_argument('output_csv', help='Output CSV file')
    
    args = parser.parse_args()
    success = extract_metrics(
        args.cv_summary_path, 
        args.model_name,
        args.filter_type,
        args.output_csv
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
EOF

# Create script for generating comparison visualizations
cat > "${SCRIPTS_DIR}/generate_comparison.py" << 'EOF'
#!/usr/bin/env python3
"""
Generate comparison visualizations and reports for filter performance.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import sys
from typing import Dict, List, Optional, Tuple

def load_csv(file_path: str) -> Optional[pd.DataFrame]:
    """Load CSV file with error handling."""
    try:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
        
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {str(e)}")
        return None

def create_bar_chart(data: pd.DataFrame, output_dir: str) -> bool:
    """Create bar chart comparing filter performance."""
    try:
        # Define metrics to visualize
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Set bar positions
        x = np.arange(len(data['filter_type']))
        width = 0.15
        multiplier = 0
        
        # Plot each metric as a group of bars
        for metric in metrics:
            offset = width * multiplier
            plt.bar(x + offset, data[metric], width, label=metric.capitalize())
            multiplier += 1
        
        # Add labels and legend
        plt.xlabel('Filter Type', fontsize=12, fontweight='bold')
        plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
        plt.title('IMU Fusion Filter Comparison', fontsize=16, fontweight='bold')
        plt.xticks(x + width * 2, data['filter_type'], fontsize=10, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels to bars
        for i, metric in enumerate(metrics):
            for j, value in enumerate(data[metric]):
                plt.text(j + width * i, value + 1, f"{value:.1f}", 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'filter_comparison.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved comparison chart to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating bar chart: {str(e)}")
        return False

def create_report(data: pd.DataFrame, output_dir: str) -> bool:
    """Create detailed report of filter comparison."""
    try:
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        output_path = os.path.join(output_dir, 'comparison_report.md')
        
        with open(output_path, 'w') as f:
            # Write header
            f.write('# IMU Fusion Filter Comparison Results\n\n')
            f.write('## Performance Metrics\n\n')
            
            # Convert DataFrame to markdown table
            f.write(data.to_markdown(index=False, floatfmt=".4f"))
            
            # Add best filter for each metric
            f.write('\n\n## Best Performing Filter by Metric\n\n')
            
            for metric in metrics:
                best_idx = data[metric].idxmax()
                best_filter = data.loc[best_idx, 'filter_type']
                best_value = data.loc[best_idx, metric]
                
                f.write(f'- **{metric.capitalize()}**: {best_filter} ({best_value:.4f})\n')
            
            # Add description of each filter
            f.write('\n\n## Filter Descriptions\n\n')
            f.write('- **Madgwick**: A computationally efficient orientation filter that uses '
                   'gradient descent for gyroscope drift correction via accelerometer data.\n')
            f.write('- **Kalman**: A standard Kalman filter implementation that optimally '
                   'combines accelerometer and gyroscope measurements using a linearized model.\n')
            f.write('- **EKF (Extended Kalman Filter)**: An advanced version of the Kalman filter '
                   'that better handles the non-linear relationships in orientation estimation, '
                   'providing improved accuracy during rapid orientation changes.\n')
        
        print(f"Saved comparison report to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating report: {str(e)}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_comparison.py <comparison_csv> <output_dir>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Load data
    data = load_csv(csv_path)
    if data is None:
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations and report
    success1 = create_bar_chart(data, output_dir)
    success2 = create_report(data, output_dir)
    
    sys.exit(0 if success1 and success2 else 1)

if __name__ == '__main__':
    main()
EOF

# Create script for recovering or creating CV summary
cat > "${SCRIPTS_DIR}/create_cv_summary.py" << 'EOF'
#!/usr/bin/env python3
"""
Create or recover cross-validation summary from individual fold results.
"""
import json
import os
import glob
import numpy as np
import sys
from typing import Dict, List, Optional, Any

def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file with error handling."""
    try:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
        
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def create_cv_summary(output_dir: str, filter_type: str) -> bool:
    """
    Create cross-validation summary from individual fold results.
    
    Args:
        output_dir: Directory containing fold results
        filter_type: Type of filter used (madgwick, kalman, ekf)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize metrics
        metrics = {
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'balanced_accuracy': []
        }
        
        # Collect fold metrics
        fold_metrics = []
        fold_dirs = sorted(glob.glob(os.path.join(output_dir, 'fold_*')))
        
        if not fold_dirs:
            print(f"No fold directories found in {output_dir}")
            return False
        
        for i, fold_dir in enumerate(fold_dirs, 1):
            # Check if fold has a test_results.json file
            test_file = os.path.join(fold_dir, 'test_results.json')
            if os.path.exists(test_file):
                data = load_json(test_file)
                if data:
                    fold_result = {
                        'fold': i,
                        'test_subjects': data.get('test_subjects', []),
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
        
        if not fold_metrics:
            print("No valid fold results found")
            return False
        
        # Calculate averages and std deviations
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[key + '_std'] = float(np.std(values))
            else:
                avg_metrics[key] = 0.0
                avg_metrics[key + '_std'] = 0.0
        
        # Create summary
        cv_summary = {
            'fold_metrics': fold_metrics,
            'average_metrics': avg_metrics,
            'filter_type': filter_type
        }
        
        # Save to file
        output_file = os.path.join(output_dir, 'cv_summary.json')
        with open(output_file, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        print(f"Created summary file for {len(fold_metrics)} folds at {output_file}")
        return True
    
    except Exception as e:
        print(f"Error creating CV summary: {str(e)}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python create_cv_summary.py <output_dir> <filter_type>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    filter_type = sys.argv[2]
    
    success = create_cv_summary(output_dir, filter_type)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
EOF

# Make scripts executable
chmod +x "${SCRIPTS_DIR}/extract_metrics.py"
chmod +x "${SCRIPTS_DIR}/generate_comparison.py"
chmod +x "${SCRIPTS_DIR}/create_cv_summary.py"

# -----------------------------------------
# CREATE CONFIGURATION TEMPLATES
# -----------------------------------------

# Function to create filter-specific configuration
create_config() {
    local config_file="$1"
    local filter_type="$2"
    
    log "INFO" "Creating config for ${filter_type} filter: ${config_file}"
    
    cat > "${config_file}" << EOF
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
EOF

    # Check if config file was created successfully
    if [ -f "${config_file}" ]; then
        log "INFO" "Successfully created config file: ${config_file}"
        return 0
    else
        log "ERROR" "Failed to create config file: ${config_file}"
        return 1
    fi
}

# -----------------------------------------
# TRAINING FUNCTION
# -----------------------------------------

# Function to train a model with specific filter
train_model() {
    local config_file="$1"
    local model_name="$2"
    local filter_type="$3"
    local output_dir="${RESULTS_DIR}/${model_name}"
    
    log "INFO" "Training model with ${filter_type} filter: ${model_name}"
    
    # Create output directory
    mkdir -p "${output_dir}"
    mkdir -p "${output_dir}/logs"
    
    # Verify config file exists
    if [ ! -f "${config_file}" ]; then
        log "ERROR" "Config file does not exist: ${config_file}"
        return 1
    fi
    
    # Run training with cross-validation
    log "INFO" "Starting training: ${model_name} with ${filter_type} filter"
    CUDA_VISIBLE_DEVICES=${DEVICE} python main.py \
        --config "${config_file}" \
        --work-dir "${output_dir}" \
        --model-saved-name "${model_name}" \
        --device 0 1 \
        --multi-gpu True \
        --kfold True \
        --parallel-threads 40 \
        --num-epoch "${NUM_EPOCHS}" \
        --patience 15 2>&1 | tee "${output_dir}/logs/training.log"
    
    local train_status=$?
    if [ ${train_status} -ne 0 ]; then
        log "WARNING" "Training process exited with status ${train_status}"
        
        # Check if we might be missing the cv_summary.json
        if [ ! -f "${output_dir}/cv_summary.json" ]; then
            log "INFO" "CV summary not found. Attempting to create from fold results..."
            python "${SCRIPTS_DIR}/create_cv_summary.py" "${output_dir}" "${filter_type}"
        fi
    fi
    
    # Check if we have a valid CV summary
    if [ -f "${output_dir}/cv_summary.json" ]; then
        log "INFO" "Cross-validation results for ${model_name} (${filter_type}):"
        python "${SCRIPTS_DIR}/extract_metrics.py" "${output_dir}/cv_summary.json" "${model_name}" "${filter_type}" "${RESULTS_DIR}/comparison.csv"
    else
        log "ERROR" "No cross-validation summary found for ${model_name}"
        # Create an empty entry in the comparison table
        mkdir -p "$(dirname "${RESULTS_DIR}/comparison.csv")"
        echo "${model_name},${filter_type},0.0,0.0,0.0,0.0,0.0" >> "${RESULTS_DIR}/comparison.csv"
    fi
    
    return 0
}

# -----------------------------------------
# MAIN SCRIPT EXECUTION
# -----------------------------------------

main() {
    # Start time
    SCRIPT_START_TIME=$(date +%s)
    
    log "INFO" "Starting comprehensive filter comparison for IMU fusion"
    log "INFO" "Results will be saved to ${RESULTS_DIR}"
    
    # Create comparison CSV header
    echo "model,filter_type,accuracy,f1,precision,recall,balanced_accuracy" > "${RESULTS_DIR}/comparison.csv"
    
    # Create configurations for each filter type
    create_config "${CONFIG_DIR}/madgwick.yaml" "madgwick" || {
        log "ERROR" "Failed to create madgwick config, aborting"
        exit 1
    }
    
    create_config "${CONFIG_DIR}/kalman.yaml" "kalman" || {
        log "ERROR" "Failed to create kalman config, aborting"
        exit 1
    }
    
    create_config "${CONFIG_DIR}/ekf.yaml" "ekf" || {
        log "ERROR" "Failed to create ekf config, aborting"
        exit 1
    }
    
    # Train models with different filter types
    log "INFO" "============= TRAINING WITH MADGWICK FILTER (BASELINE) ============="
    train_model "${CONFIG_DIR}/madgwick.yaml" "madgwick_model" "madgwick"
    
    log "INFO" "============= TRAINING WITH STANDARD KALMAN FILTER ============="
    train_model "${CONFIG_DIR}/kalman.yaml" "kalman_model" "kalman"
    
    log "INFO" "============= TRAINING WITH EXTENDED KALMAN FILTER ============="
    train_model "${CONFIG_DIR}/ekf.yaml" "ekf_model" "ekf"
    
    # Generate comparison visualizations and report
    log "INFO" "Generating comparative analysis"
    python "${SCRIPTS_DIR}/generate_comparison.py" "${RESULTS_DIR}/comparison.csv" "${RESULTS_DIR}/visualizations"
    
    # Calculate execution time
    SCRIPT_END_TIME=$(date +%s)
    EXECUTION_TIME=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
    HOURS=$((EXECUTION_TIME / 3600))
    MINUTES=$(( (EXECUTION_TIME % 3600) / 60 ))
    SECONDS=$((EXECUTION_TIME % 60))
    
    log "INFO" "Filter comparison completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"
    log "INFO" "Results available in ${RESULTS_DIR}"
    log "INFO" "Visualizations available in ${RESULTS_DIR}/visualizations"
}

# Run the main function
main
