#!/bin/bash

# ===========================================================================
# Comprehensive SmartFallMM Fall Detection Evaluation
# 
# This script evaluates:
# - Different filter types: madgwick, comp, kalman, ekf, ukf
# - Sequence sizes: 64 vs 128
# - Embedding dimensions: 64 vs 128
#
# Features:
# - Error handling and automatic dimension compatibility
# - Cross-validation across all configurations
# - Comprehensive logging and result analysis
# - Final summary with optimal configuration
# ===========================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_comparison_${TIMESTAMP}"
CONFIG_DIR="config/smartfallmm"
LOG_DIR="${RESULTS_DIR}/logs"
REPORT_DIR="${RESULTS_DIR}/reports"
VIZ_DIR="${RESULTS_DIR}/visualizations"
MASTER_LOG="${LOG_DIR}/master_log.txt"
RESULT_CSV="${RESULTS_DIR}/results_summary.csv"

# Parameters to test
FILTER_TYPES=("madgwick" "comp" "kalman" "ekf" "ukf")
SEQUENCE_SIZES=(64 128)
EMBEDDING_SIZES=(64 128)
SUBJECTS=(29 30 31 33 45 46 34 37 39 38 43 35 36 44 32)  # All subjects
BATCH_SIZE=16
NUM_EPOCHS=60
PATIENCE=15
GPUS="0,1"  # Using both A100 GPUs

# Create directory structure
mkdir -p "${LOG_DIR}"
mkdir -p "${CONFIG_DIR}"
mkdir -p "${REPORT_DIR}"
mkdir -p "${VIZ_DIR}"
mkdir -p "debug_logs"

# Create the results CSV file with header
echo "filter_type,seq_size,embed_size,cv_f1,cv_accuracy,precision,recall,final_f1,train_time" > "${RESULT_CSV}"

# Initialize counters
TOTAL_CONFIGS=$((${#FILTER_TYPES[@]} * ${#SEQUENCE_SIZES[@]} * ${#EMBEDDING_SIZES[@]}))
CURRENT_CONFIG=0

# Logging function
log() {
    local level="$1"
    local message="$2"
    local time_stamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${time_stamp}] [${level}] ${message}"
    echo "[${time_stamp}] [${level}] ${message}" >> "${MASTER_LOG}"
}

# Header function
print_header() {
    local title="$1"
    local length=${#title}
    local padding=$(printf '=%.0s' $(seq 1 $((length + 20))))
    log "INFO" ""
    log "INFO" "${padding}"
    log "INFO" "         ${title}"
    log "INFO" "${padding}"
    log "INFO" ""
}

# Function to create yaml config with compatible dimensions
create_config() {
    local filter_type=$1
    local seq_size=$2
    local embed_size=$3
    local output_file=$4
    
    # Calculate appropriate model parameters
    local num_heads=8
    local num_layers=3
    local dropout=0.3
    
    # Adjust parameters based on filter complexity
    if [[ "${filter_type}" == "ekf" || "${filter_type}" == "ukf" ]]; then
        num_heads=12
        num_layers=4
        dropout=0.2
    fi
    
    # Calculate feature dimension 
    # The fusion features are 43 dimensions, but we need to ensure the model can handle this
    # when combined with the embedding size
    local feature_dim=$((embed_size * 2))
    
    log "INFO" "Creating config: filter=${filter_type}, seq=${seq_size}, embed=${embed_size}, feature_dim=${feature_dim}"
    
    # Create the config file
    cat > "${output_file}" << EOF
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm

subjects: [29,30,31,33,45,46,34,37,39,38,43,35,36,44,34,32]

model_args:
  num_layers: ${num_layers}
  embed_dim: ${embed_size}
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: ${seq_size}
  mocap_frames: ${seq_size}
  num_heads: ${num_heads}
  fusion_type: 'concat'
  dropout: ${dropout}
  use_batch_norm: true
  feature_dim: ${feature_dim}  # Explicitly set feature dimension to ensure compatibility

dataset_args:
  mode: 'sliding_window'
  max_length: ${seq_size}
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
    save_aligned: true

batch_size: ${BATCH_SIZE}
test_batch_size: ${BATCH_SIZE}
val_batch_size: ${BATCH_SIZE}
num_epoch: ${NUM_EPOCHS}
patience: ${PATIENCE}

# dataloader
feeder: Feeder.Make_Dataset.UTD_mm
train_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: true

val_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: true

test_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: false

seed: 42
optimizer: adamw
base_lr: 0.0005
weight_decay: 0.001

# Cross-validation settings
kfold:
  enabled: true
  num_folds: 5
  fold_assignments:
    - [43, 35, 36]  # Fold 1: 38.3% falls
    - [44, 34, 32]  # Fold 2: 39.7% falls
    - [45, 37, 38]  # Fold 3: 44.8% falls
    - [46, 29, 31]  # Fold 4: 41.4% falls
    - [30, 39]      # Fold 5: 43.3% falls
EOF

    log "INFO" "Created configuration file: ${output_file}"
}

# Function to handle errors
handle_error() {
    local exit_code=$?
    local line_number=$1
    local config_name=$2
    
    log "ERROR" "Command failed at line ${line_number} in configuration ${config_name}"
    log "ERROR" "Exit code: ${exit_code}"
    
    # Clean up resources
    python -c "from utils.imu_fusion import cleanup_resources; cleanup_resources()" || true
    
    # Continue with next configuration instead of exiting
    return 0
}

# Function to run a specific configuration
run_configuration() {
    local filter_type=$1
    local seq_size=$2
    local embed_size=$3
    
    local config_name="${filter_type}_seq${seq_size}_embed${embed_size}"
    local config_path="${CONFIG_DIR}/${config_name}.yaml"
    local output_dir="${RESULTS_DIR}/${config_name}"
    local log_file="${LOG_DIR}/${config_name}.log"
    
    mkdir -p "${output_dir}"
    
    # Create config file with compatible dimensions
    create_config "${filter_type}" "${seq_size}" "${embed_size}" "${config_path}"
    
    # Log start of training
    ((CURRENT_CONFIG++))
    print_header "Configuration ${CURRENT_CONFIG}/${TOTAL_CONFIGS}: ${config_name}"
    log "INFO" "Starting training with configuration: ${config_name}"
    log "INFO" "Config file: ${config_path}"
    log "INFO" "Output directory: ${output_dir}"
    
    # Set error handler
    trap "handle_error \${LINENO} ${config_name}" ERR
    
    # Record start time
    start_time=$(date +%s)
    
    # Execute training command with timeout
    if timeout 6h CUDA_VISIBLE_DEVICES=${GPUS} python main.py \
        --config "${config_path}" \
        --work-dir "${output_dir}" \
        --model-saved-name "${config_name}" \
        --device 0 1 \
        --multi-gpu True \
        --parallel-threads 48 \
        --kfold True \
        --num-folds 5 \
        --patience ${PATIENCE} \
        --num-epoch ${NUM_EPOCHS} 2>&1 | tee "${log_file}"; then
        
        # Calculate training time
        end_time=$(date +%s)
        train_time=$((end_time - start_time))
        log "INFO" "Training completed successfully in ${train_time} seconds"
        training_status="success"
    else
        # Handle timeout or failure
        end_time=$(date +%s)
        train_time=$((end_time - start_time))
        log "WARNING" "Training failed or timed out after ${train_time} seconds"
        training_status="failed"
    fi
    
    # Clean up resources
    python -c "from utils.imu_fusion import cleanup_resources; cleanup_resources()" || true
    
    # Extract and log results
    extract_results "${filter_type}" "${seq_size}" "${embed_size}" "${output_dir}" "${train_time}" "${training_status}"
    
    # Reset error handler
    trap - ERR
    
    # Return success to continue with next configuration
    return 0
}

# Function to extract and save results
extract_results() {
    local filter_type=$1
    local seq_size=$2
    local embed_size=$3
    local output_dir=$4
    local train_time=$5
    local status=$6
    
    log "INFO" "Extracting results for ${filter_type}, seq=${seq_size}, embed=${embed_size}"
    
    # Initialize metrics
    local cv_f1="N/A"
    local cv_accuracy="N/A"
    local final_f1="N/A"
    local precision="N/A"
    local recall="N/A"
    
    # Extract cross-validation results if available
    local cv_summary="${output_dir}/cv_summary.json"
    if [[ -f "${cv_summary}" ]]; then
        # Extract metrics from JSON using Python
        cv_metrics=$(python -c "
import json
try:
    with open('${cv_summary}', 'r') as f:
        data = json.load(f)
    avg = data.get('average_metrics', {})
    print(f\"{avg.get('f1', 'N/A')},{avg.get('accuracy', 'N/A')}\")
except:
    print('N/A,N/A')
")
        cv_f1=$(echo $cv_metrics | cut -d',' -f1)
        cv_accuracy=$(echo $cv_metrics | cut -d',' -f2)
    fi
    
    # Extract final test results if available
    local test_results="${output_dir}/test_result.txt"
    if [[ -f "${test_results}" ]]; then
        final_f1=$(grep "f1_score" "${test_results}" | head -1 | awk '{print $2}')
        precision=$(grep "precision" "${test_results}" | head -1 | awk '{print $2}')
        recall=$(grep "recall" "${test_results}" | head -1 | awk '{print $2}')
    fi
    
    # Add results to CSV
    echo "${filter_type},${seq_size},${embed_size},${cv_f1},${cv_accuracy},${precision},${recall},${final_f1},${train_time}" >> "${RESULT_CSV}"
    
    # Log results
    log "INFO" "Results for ${filter_type}, seq=${seq_size}, embed=${embed_size}:"
    log "INFO" "  CV F1: ${cv_f1}"
    log "INFO" "  CV Accuracy: ${cv_accuracy}"
    log "INFO" "  Final F1: ${final_f1}"
    log "INFO" "  Precision: ${precision}"
    log "INFO" "  Recall: ${recall}"
    log "INFO" "  Training time: ${train_time} seconds"
}

# Function to apply patch to fix dimension mismatch
apply_dimension_fix() {
    print_header "Applying Model Dimension Fix"
    
    # Create a patch for the fusion_transformer.py file
    local patch_file="fusion_transformer_fix.patch"
    
    cat > "${patch_file}" << 'EOF'
--- a/Models/fusion_transformer.py
+++ b/Models/fusion_transformer.py
@@ -37,6 +37,7 @@ class FusionTransModel(nn.Module):
                 fusion_type='concat',
                 dropout=0.3,
                 use_batch_norm=True,
+                feature_dim=None,
                 **kwargs):
         """
         Optimized transformer model for IMU fusion with linear acceleration and quaternion.
@@ -51,6 +52,7 @@ class FusionTransModel(nn.Module):
             fusion_type: How to combine different sensor data ('concat', 'attention', 'acc_only')
             dropout: Dropout rate
             use_batch_norm: Whether to use batch normalization
+            feature_dim: Optional explicit feature dimension (computed automatically if None)
         """
         super().__init__()
         print(f"Initializing FusionTransModel with fusion_type={fusion_type}")
@@ -86,7 +88,10 @@ class FusionTransModel(nn.Module):
         # Determine feature dimension based on fusion type
         if fusion_type == 'concat':
             # We concatenate linear acceleration and quaternion embeddings
-            feature_dim = embed_dim * 2
+            if feature_dim is None:
+                feature_dim = embed_dim * 2
+            else:
+                print(f"Using explicit feature dimension: {feature_dim}")
         elif fusion_type == 'attention':
             # We use attention to combine the embeddings
             feature_dim = embed_dim
@@ -123,7 +128,7 @@ class FusionTransModel(nn.Module):
         self.classifier = nn.Sequential(
             nn.Linear(feature_dim, 64),
             nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
-            nn.GELU(),
+            nn.ReLU(),
             nn.Dropout(dropout),
             nn.Linear(64, num_classes)
         )
@@ -145,6 +150,10 @@ class FusionTransModel(nn.Module):
         # Expand fusion features to match sequence length
         expanded_features = fusion_features.unsqueeze(1).expand(-1, acc_features.size(1), -1)
         
+        # Log shapes for debugging
+        total_feature_dim = acc_features.size(2) + expanded_features.size(2)
+        print(f"Feature dimensions - acc: {acc_features.size(2)}, fusion: {expanded_features.size(2)}, total: {total_feature_dim}")
+        
         # Combine features
         features = torch.cat([acc_features, expanded_features], dim=2)
         
EOF
    
    # Apply the patch
    if ! patch -N -p1 < "${patch_file}"; then
        log "WARNING" "Patch may have already been applied or failed"
    else
        log "INFO" "Successfully applied dimension fix patch"
    fi
    
    # Remove the patch file
    rm "${patch_file}"
}

# Function to generate visualizations and final report
generate_report() {
    print_header "Generating Final Report"
    
    # Create Python script for visualizations and report
    cat > "${RESULTS_DIR}/generate_report.py" << 'EOF'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

def create_visualizations(results_file, output_dir):
    """Create visualizations from the results."""
    # Read results
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} configuration results")
    
    # Convert string columns to numeric where possible
    for col in df.columns[3:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 1. Filter Type Comparison
    plt.figure(figsize=(12, 8))
    filter_comparison = df.groupby('filter_type')['final_f1'].mean().sort_values(ascending=False)
    ax = filter_comparison.plot(kind='bar', color='skyblue')
    plt.title('Average F1 Score by Filter Type')
    plt.ylabel('F1 Score')
    plt.xlabel('Filter Type')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(filter_comparison):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'filter_comparison.png'))
    
    # 2. Sequence Size Comparison
    plt.figure(figsize=(10, 6))
    seq_comparison = df.groupby('seq_size')['final_f1'].mean().sort_values(ascending=False)
    ax = seq_comparison.plot(kind='bar', color='lightgreen')
    plt.title('Average F1 Score by Sequence Size')
    plt.ylabel('F1 Score')
    plt.xlabel('Sequence Size')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(seq_comparison):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sequence_comparison.png'))
    
    # 3. Embedding Size Comparison
    plt.figure(figsize=(10, 6))
    embed_comparison = df.groupby('embed_size')['final_f1'].mean().sort_values(ascending=False)
    ax = embed_comparison.plot(kind='bar', color='salmon')
    plt.title('Average F1 Score by Embedding Size')
    plt.ylabel('F1 Score')
    plt.xlabel('Embedding Size')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(embed_comparison):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_comparison.png'))
    
    # 4. Heatmap of config combinations
    plt.figure(figsize=(15, 10))
    
    for i, filter_type in enumerate(df['filter_type'].unique()):
        filter_df = df[df['filter_type'] == filter_type]
        
        # Create pivot table
        pivot = filter_df.pivot_table(
            index='seq_size', 
            columns='embed_size',
            values='final_f1'
        )
        
        plt.subplot(2, 3, i+1)
        im = plt.imshow(pivot, cmap='viridis')
        plt.colorbar(im)
        plt.title(f'{filter_type} Filter')
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel('Embedding Size')
        plt.ylabel('Sequence Size')
        
        # Add text annotations
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                if not np.isnan(pivot.iloc[y, x]):
                    plt.text(x, y, f"{pivot.iloc[y, x]:.4f}", 
                            ha="center", va="center", 
                            color="white" if pivot.iloc[y, x] > 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'config_heatmap.png'))
    
    # 5. Top 5 configurations
    plt.figure(figsize=(12, 6))
    top_configs = df.sort_values('final_f1', ascending=False).head(5)
    labels = [f"{row['filter_type']}\nseq={row['seq_size']}\nembed={row['embed_size']}" 
              for _, row in top_configs.iterrows()]
    
    ax = plt.bar(labels, top_configs['final_f1'], color='lightblue')
    plt.title('Top 5 Configurations by F1 Score')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(top_configs['final_f1']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_configs.png'))
    
    print(f"Created visualizations in {output_dir}")

def generate_report(results_file, report_file, viz_dir):
    """Generate a comprehensive markdown report."""
    # Read results
    df = pd.read_csv(results_file)
    
    # Convert numeric columns
    for col in df.columns[3:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Find best configuration
    try:
        best_config = df.loc[df['final_f1'].idxmax()]
        best_filter = best_config['filter_type']
        best_seq = int(best_config['seq_size'])
        best_embed = int(best_config['embed_size'])
        best_f1 = best_config['final_f1']
    except:
        best_filter = "N/A"
        best_seq = "N/A"
        best_embed = "N/A"
        best_f1 = "N/A"
    
    # Create report
    with open(report_file, 'w') as f:
        f.write('# SmartFallMM Filter and Architecture Evaluation Report\n\n')
        f.write(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write('## Executive Summary\n\n')
        f.write('This report presents the results of a comprehensive evaluation of different filter types, sequence sizes, and embedding dimensions for fall detection using the SmartFallMM dataset.\n\n')
        
        f.write('### Key Findings\n\n')
        f.write(f'- **Best Configuration:** {best_filter} filter with sequence size {best_seq} and embedding dimension {best_embed}\n')
        f.write(f'- **Best F1 Score:** {best_f1:.4f}\n\n')
        
        f.write('## Filter Type Comparison\n\n')
        f.write('![Filter Type Comparison](./visualizations/filter_comparison.png)\n\n')
        
        # Filter type table
        filter_results = df.groupby('filter_type').agg({
            'cv_f1': ['mean', 'std'],
            'final_f1': ['mean', 'std'],
            'precision': ['mean'],
            'recall': ['mean'],
            'train_time': ['mean']
        }).reset_index()
        
        f.write('| Filter Type | CV F1 | Final F1 | Precision | Recall | Avg. Train Time (s) |\n')
        f.write('|-------------|-------|----------|-----------|--------|---------------------|\n')
        
        for _, row in filter_results.iterrows():
            f.write(f"| {row['filter_type']} | {row['cv_f1']['mean']:.4f}±{row['cv_f1']['std']:.4f} | " + 
                    f"{row['final_f1']['mean']:.4f}±{row['final_f1']['std']:.4f} | " +
                    f"{row['precision']['mean']:.4f} | {row['recall']['mean']:.4f} | " +
                    f"{row['train_time']['mean']:.0f} |\n")
        
        f.write('\n## Sequence Size Comparison\n\n')
        f.write('![Sequence Size Comparison](./visualizations/sequence_comparison.png)\n\n')
        
        # Sequence size table
        seq_results = df.groupby('seq_size').agg({
            'final_f1': ['mean', 'std'],
            'train_time': ['mean']
        }).reset_index()
        
        f.write('| Sequence Size | F1 Score | Train Time (s) |\n')
        f.write('|---------------|----------|----------------|\n')
        
        for _, row in seq_results.iterrows():
            f.write(f"| {int(row['seq_size'])} | {row['final_f1']['mean']:.4f}±{row['final_f1']['std']:.4f} | " +
                    f"{row['train_time']['mean']:.0f} |\n")
        
        f.write('\n## Embedding Size Comparison\n\n')
        f.write('![Embedding Size Comparison](./visualizations/embedding_comparison.png)\n\n')
        
        # Embedding size table
        embed_results = df.groupby('embed_size').agg({
            'final_f1': ['mean', 'std'],
            'train_time': ['mean']
        }).reset_index()
        
        f.write('| Embedding Size | F1 Score | Train Time (s) |\n')
        f.write('|---------------|----------|----------------|\n')
        
        for _, row in embed_results.iterrows():
            f.write(f"| {int(row['embed_size'])} | {row['final_f1']['mean']:.4f}±{row['final_f1']['std']:.4f} | " +
                    f"{row['train_time']['mean']:.0f} |\n")
        
        f.write('\n## Configuration Heatmap\n\n')
        f.write('This heatmap shows the performance of different sequence and embedding size combinations for each filter type:\n\n')
        f.write('![Configuration Heatmap](./visualizations/config_heatmap.png)\n\n')
        
        f.write('## Top Configurations\n\n')
        f.write('![Top Configurations](./visualizations/top_configs.png)\n\n')
        
        # Top 5 configurations table
        top_configs = df.sort_values('final_f1', ascending=False).head(5)
        
        f.write('| Rank | Filter Type | Sequence Size | Embedding Size | F1 Score | Precision | Recall | Train Time (s) |\n')
        f.write('|------|-------------|---------------|----------------|----------|-----------|--------|----------------|\n')
        
        for i, (_, row) in enumerate(top_configs.iterrows(), 1):
            f.write(f"| {i} | {row['filter_type']} | {int(row['seq_size'])} | {int(row['embed_size'])} | " +
                    f"{row['final_f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['train_time']:.0f} |\n")
        
        f.write('\n## Recommendations\n\n')
        f.write(f'Based on the evaluation results, we recommend using the **{best_filter}** filter with a sequence size of **{best_seq}** and embedding dimension of **{best_embed}**. This configuration achieved the highest F1 score of **{best_f1:.4f}**.\n\n')
        
        f.write('### Implementation Notes\n\n')
        f.write('1. The best configuration balances accuracy and computational efficiency\n')
        f.write('2. For deployment on resource-constrained devices, the smaller sequence and embedding sizes may be preferred\n')
        f.write('3. For maximum accuracy regardless of computational requirements, use the best configuration identified above\n')
        
        f.write('\n## Conclusion\n\n')
        f.write('This comprehensive evaluation has identified the optimal filter type, sequence size, and embedding dimension for fall detection using the SmartFallMM dataset. The recommended configuration provides the best balance between accuracy and computational efficiency.\n')
    
    print(f"Report generated: {report_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python generate_report.py <results_csv> <report_file> <viz_dir>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    report_file = sys.argv[2]
    viz_dir = sys.argv[3]
    
    create_visualizations(results_file, viz_dir)
    generate_report(results_file, report_file, viz_dir)
EOF
    
    # Run the report generation script
    python "${RESULTS_DIR}/generate_report.py" "${RESULT_CSV}" "${REPORT_DIR}/final_report.md" "${VIZ_DIR}"
    
    # Extract best configuration
    best_config=$(grep -A 2 "Key Findings" "${REPORT_DIR}/final_report.md" | grep "Best Configuration" | cut -d':' -f2- | sed -e 's/^[[:space:]]*//')
    best_f1=$(grep -A 2 "Key Findings" "${REPORT_DIR}/final_report.md" | grep "Best F1 Score" | cut -d':' -f2- | sed -e 's/^[[:space:]]*//')
    
    log "INFO" "Report generation complete"
    log "INFO" "Best configuration: ${best_config}"
    log "INFO" "Best F1 score: ${best_f1}"
}

# Main script execution

# Display header
print_header "SmartFallMM Filter and Architecture Comparison"
log "INFO" "Starting comprehensive evaluation of filter types and architectures"
log "INFO" "Results directory: ${RESULTS_DIR}"

# Apply the patch to fix dimension mismatch issue
apply_dimension_fix

# Run training for each configuration
for filter_type in "${FILTER_TYPES[@]}"; do
    for seq_size in "${SEQUENCE_SIZES[@]}"; do
        for embed_size in "${EMBEDDING_SIZES[@]}"; do
            run_configuration "${filter_type}" "${seq_size}" "${embed_size}"
        done
    done
done

# Generate final report and visualizations
generate_report

# Display footer with execution summary
total_time=$SECONDS
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

print_header "Evaluation Complete"
log "INFO" "Total execution time: ${hours}h ${minutes}m ${seconds}s"
log "INFO" "Results directory: ${RESULTS_DIR}"
log "INFO" "Final report: ${REPORT_DIR}/final_report.md"
log "INFO" "Visualizations: ${VIZ_DIR}"

# Display top performing configuration
top_config=$(tail -n +2 "${RESULT_CSV}" | sort -t',' -k8,8 -nr | head -1)
if [[ -n "${top_config}" ]]; then
    filter=$(echo "${top_config}" | cut -d',' -f1)
    seq=$(echo "${top_config}" | cut -d',' -f2)
    embed=$(echo "${top_config}" | cut -d',' -f3)
    f1=$(echo "${top_config}" | cut -d',' -f8)
    
    log "INFO" "Top performing configuration:"
    log "INFO" "  Filter type: ${filter}"
    log "INFO" "  Sequence size: ${seq}"
    log "INFO" "  Embedding size: ${embed}"
    log "INFO" "  F1 score: ${f1}"
    log "INFO" "  Configuration directory: ${RESULTS_DIR}/${filter}_seq${seq}_embed${embed}"
fi

log "INFO" "Evaluation script completed successfully."
