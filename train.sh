#!/bin/bash

# Fall Detection Filter Comparison Training Script
# Systematically compares different sensor fusion filters for fall detection

set -e  # Exit on error

# Configuration
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="filter_comparison_${timestamp}"
base_config="config/smartfallmm/fusion_madgwick.yaml"
log_file="${results_dir}/log.txt"

# GPU configuration - use both available GPUs
gpus="0,1"
  
# Create results directory
mkdir -p ${results_dir}

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" | tee -a ${log_file}
}

# Print header
log ""
log "=========================================================="
log "         FALL DETECTION WITH FILTER COMPARISON"
log "=========================================================="
log ""

log "Starting filter comparison"
log "Results directory: ${results_dir}"

# Create config directory
config_dir="${results_dir}/configs"
mkdir -p ${config_dir}

# Create configurations for each filter type
log "Generating filter configurations..."

for filter_type in madgwick comp kalman ekf ukf; do
    # Copy the base config
    filter_config="${config_dir}/${filter_type}_filter.yaml"
    cp ${base_config} ${filter_config}
    
    # Update the filter type
    sed -i "s/filter_type:.*$/filter_type: '${filter_type}'/" ${filter_config}
    
    # Adjust model parameters for more complex filters
    if [[ "${filter_type}" == "ekf" || "${filter_type}" == "ukf" ]]; then
        # Increase model capacity for more complex filters
        sed -i "s/embed_dim:.*$/embed_dim: 64/" ${filter_config}
        sed -i "s/num_heads:.*$/num_heads: 4/" ${filter_config}
        sed -i "s/max_length:.*$/max_length: 128/" ${filter_config}
        sed -i "s/acc_frames:.*$/acc_frames: 128/" ${filter_config}
    fi
    
    log "Created configuration for ${filter_type} filter"
done

# Run training for each filter type
for filter_type in madgwick comp kalman ekf ukf; do
    filter_dir="${results_dir}/${filter_type}"
    filter_config="${config_dir}/${filter_type}_filter.yaml"
    
    log ""
    log "======================================================"
    log "         TRAINING WITH ${filter_type^^} FILTER"
    log "======================================================"
    log ""
    
    log "Starting training with ${filter_type} filter"
    log "Configuration: ${filter_config}"
    log "Output directory: ${filter_dir}"
    
    # Run the training command
    CUDA_VISIBLE_DEVICES=${gpus} python main.py \
        --config ${filter_config} \
        --work-dir ${filter_dir} \
        --model-saved-name "${filter_type}_model" \
        --device 0 1 \
        --multi-gpu True \
        --kfold True \
        --num-folds 5 \
        --patience 15 \
        --parallel-threads 48
    
    # Check if training was successful
    if [ $? -ne 0 ]; then
        log "Warning: Training with ${filter_type} filter failed"
    else
        log "Training with ${filter_type} filter completed successfully"
    fi
done

# Create comparison table and visualization
log ""
log "=================================================="
log "         COMPARING FILTER PERFORMANCE"
log "=================================================="
log ""

# Create a CSV file for comparison
echo "filter_type,accuracy,f1_score,precision,recall" > "${results_dir}/comparison.csv"

# Extract metrics from each filter's test results
for filter_type in madgwick comp kalman ekf ukf; do
    test_result="${results_dir}/${filter_type}/test_result.txt"
    if [ -f "$test_result" ]; then
        accuracy=$(grep "accuracy" "$test_result" | head -1 | awk '{print $2}')
        f1=$(grep "f1_score" "$test_result" | head -1 | awk '{print $2}')
        precision=$(grep "precision" "$test_result" | head -1 | awk '{print $2}')
        recall=$(grep "recall" "$test_result" | head -1 | awk '{print $2}')
        
        echo "${filter_type},${accuracy},${f1},${precision},${recall}" >> "${results_dir}/comparison.csv"
    else
        log "Warning: No test results found for ${filter_type} filter"
    fi
done

# Run the filter comparison report generator
log "Generating comparison report and visualizations..."
python -c "
from utils.filter_comparison import process_comparison_results
process_comparison_results('${results_dir}')
"

# Display the results table
if command -v column > /dev/null 2>&1; then
    log "Filter comparison results:"
    column -t -s ',' "${results_dir}/comparison.csv"
else
    log "Filter comparison results:"
    cat "${results_dir}/comparison.csv"
fi

# Identify the best filter (by F1 score)
best_filter=$(sort -t',' -k3,3 -nr "${results_dir}/comparison.csv" | head -2 | tail -1 | cut -d',' -f1)
best_f1=$(sort -t',' -k3,3 -nr "${results_dir}/comparison.csv" | head -2 | tail -1 | cut -d',' -f3)

if [ -n "$best_filter" ]; then
    log ""
    log "========================================================"
    log "Best performing filter: ${best_filter^^} with F1 score ${best_f1}"
    log "Complete results saved in: ${results_dir}/comparison.csv"
    log "Visualizations available in: ${results_dir}/visualizations/"
    log "========================================================"
else
    log ""
    log "========================================================"
    log "Could not determine best filter. Check results manually."
    log "Complete results saved in: ${results_dir}/comparison.csv"
    log "========================================================"
fi
