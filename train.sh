#!/bin/bash
set -e

# Configuration
DEVICE="0,1"
FILTER_TYPES=("madgwick" "kalman" "ekf")
BASE_LR=0.0005
WEIGHT_DECAY=0.001
NUM_EPOCHS=60
PATIENCE=15
SEED=42
BATCH_SIZE=16
RESULTS_DIR="filter_comparison_results"
CONFIG_DIR="config/smartfallmm"
VISUALIZE="false"
SAVE_ALIGNED="true"
MAX_PARALLEL=2

# Create required directories
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/visualizations"

# Print configuration
echo "=== IMU Fusion Filter Training ==="
echo "Filters: ${FILTER_TYPES[*]}"
echo "Results directory: ${RESULTS_DIR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${BASE_LR}"
echo "Device: ${DEVICE}"
echo "============================="

# Function to train a model with a specific filter
train_filter_model() {
    local filter_type="$1"
    local config_file="${CONFIG_DIR}/${filter_type}_fusion.yaml"
    local output_dir="${RESULTS_DIR}/${filter_type}_model"
    
    mkdir -p "${output_dir}/logs"
    
    echo "$(date) - Starting training for ${filter_type} filter"
    
    # Ensure config exists
    if [ ! -f "${config_file}" ]; then
        echo "ERROR: Config file not found: ${config_file}"
        return 1
    fi
    
    # Train the model
    CUDA_VISIBLE_DEVICES=${DEVICE} python main.py \
        --config "${config_file}" \
        --work-dir "${output_dir}" \
        --model-saved-name "${filter_type}_model" \
        --device 0 1 \
        --multi-gpu True \
        --patience ${PATIENCE} \
        --parallel-threads 30 \
        --num-epoch ${NUM_EPOCHS} 2>&1 | tee "${output_dir}/logs/training.log"
    
    # Check training status
    if [ $? -ne 0 ]; then
        echo "WARNING: Training process exited with error for ${filter_type}"
        if [ ! -f "${output_dir}/cv_summary.json" ]; then
            echo "Creating empty summary for ${filter_type}"
            echo "{\"filter_type\":\"${filter_type}\",\"average_metrics\":{\"accuracy\":0,\"f1\":0,\"precision\":0,\"recall\":0,\"balanced_accuracy\":0},\"fold_metrics\":[]}" > "${output_dir}/cv_summary.json"
        fi
    fi
    
    echo "$(date) - Completed training for ${filter_type} filter"
    return 0
}

# Function to generate comparison report
generate_comparison() {
    echo "$(date) - Generating comparison report"
    
    # Create summary CSV
    echo "filter_type,accuracy,accuracy_std,f1,f1_std,precision,precision_std,recall,recall_std,balanced_accuracy,balanced_accuracy_std" > "${RESULTS_DIR}/metrics_summary.csv"
    
    # Extract metrics from each filter
    for filter_type in "${FILTER_TYPES[@]}"; do
        if [ -f "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" ]; then
            # Extract metrics using jq if available, otherwise use grep/sed
            if command -v jq &> /dev/null; then
                avg_metrics=$(jq -r '.average_metrics' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                accuracy=$(jq -r '.average_metrics.accuracy' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                accuracy_std=$(jq -r '.average_metrics.accuracy_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                f1=$(jq -r '.average_metrics.f1' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                f1_std=$(jq -r '.average_metrics.f1_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                precision=$(jq -r '.average_metrics.precision' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                precision_std=$(jq -r '.average_metrics.precision_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                recall=$(jq -r '.average_metrics.recall' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                recall_std=$(jq -r '.average_metrics.recall_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                bal_acc=$(jq -r '.average_metrics.balanced_accuracy' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                bal_acc_std=$(jq -r '.average_metrics.balanced_accuracy_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
            else
                # Fallback if jq is not available
                accuracy=$(grep -o '"accuracy":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                accuracy_std=$(grep -o '"accuracy_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                f1=$(grep -o '"f1":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                f1_std=$(grep -o '"f1_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                precision=$(grep -o '"precision":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                precision_std=$(grep -o '"precision_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                recall=$(grep -o '"recall":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                recall_std=$(grep -o '"recall_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                bal_acc=$(grep -o '"balanced_accuracy":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                bal_acc_std=$(grep -o '"balanced_accuracy_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
            fi
            
            echo "${filter_type},${accuracy},${accuracy_std},${f1},${f1_std},${precision},${precision_std},${recall},${recall_std},${bal_acc},${bal_acc_std}" >> "${RESULTS_DIR}/metrics_summary.csv"
        else
            echo "${filter_type},0,0,0,0,0,0,0,0,0,0" >> "${RESULTS_DIR}/metrics_summary.csv"
        fi
    done
    
    # Generate comparison report
    echo "# IMU Fusion Filter Comparison Report" > "${RESULTS_DIR}/comparison_report.md"
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    echo "## Performance Summary" >> "${RESULTS_DIR}/comparison_report.md"
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    echo "| Filter | Accuracy | F1 Score | Precision | Recall | Balanced Accuracy |" >> "${RESULTS_DIR}/comparison_report.md"
    echo "|--------|----------|----------|-----------|--------|-------------------|" >> "${RESULTS_DIR}/comparison_report.md"
    
    for filter_type in "${FILTER_TYPES[@]}"; do
        if [ -f "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" ]; then
            # Extract metrics using jq if available, otherwise use grep/sed
            if command -v jq &> /dev/null; then
                accuracy=$(jq -r '.average_metrics.accuracy' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                accuracy_std=$(jq -r '.average_metrics.accuracy_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                f1=$(jq -r '.average_metrics.f1' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                f1_std=$(jq -r '.average_metrics.f1_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                precision=$(jq -r '.average_metrics.precision' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                precision_std=$(jq -r '.average_metrics.precision_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                recall=$(jq -r '.average_metrics.recall' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                recall_std=$(jq -r '.average_metrics.recall_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                bal_acc=$(jq -r '.average_metrics.balanced_accuracy' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
                bal_acc_std=$(jq -r '.average_metrics.balanced_accuracy_std' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json")
            else
                # Fallback if jq is not available
                accuracy=$(grep -o '"accuracy":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                accuracy_std=$(grep -o '"accuracy_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                f1=$(grep -o '"f1":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                f1_std=$(grep -o '"f1_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                precision=$(grep -o '"precision":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                precision_std=$(grep -o '"precision_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                recall=$(grep -o '"recall":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                recall_std=$(grep -o '"recall_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                bal_acc=$(grep -o '"balanced_accuracy":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
                bal_acc_std=$(grep -o '"balanced_accuracy_std":[0-9.]*' "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" | cut -d: -f2)
            fi
            
            echo "| ${filter_type^} | ${accuracy}% ± ${accuracy_std}% | ${f1} ± ${f1_std} | ${precision}% ± ${precision_std}% | ${recall}% ± ${recall_std}% | ${bal_acc}% ± ${bal_acc_std}% |" >> "${RESULTS_DIR}/comparison_report.md"
        fi
    done
    
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    echo "## Filter Characteristics" >> "${RESULTS_DIR}/comparison_report.md"
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    echo "### Madgwick Filter" >> "${RESULTS_DIR}/comparison_report.md"
    echo "- Computationally efficient quaternion-based filter" >> "${RESULTS_DIR}/comparison_report.md"
    echo "- Uses gradient descent algorithm for orientation estimation" >> "${RESULTS_DIR}/comparison_report.md"
    echo "- Good stability with linear acceleration data" >> "${RESULTS_DIR}/comparison_report.md"
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    echo "### Kalman Filter" >> "${RESULTS_DIR}/comparison_report.md"
    echo "- Classical state estimation algorithm" >> "${RESULTS_DIR}/comparison_report.md"
    echo "- Combines gyroscope integration with accelerometer measurements" >> "${RESULTS_DIR}/comparison_report.md"
    echo "- Effective at handling noise in varying sampling rates" >> "${RESULTS_DIR}/comparison_report.md"
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    echo "### Extended Kalman Filter (EKF)" >> "${RESULTS_DIR}/comparison_report.md"
    echo "- Non-linear extension of Kalman filter" >> "${RESULTS_DIR}/comparison_report.md"
    echo "- Better handles quaternion orientation state" >> "${RESULTS_DIR}/comparison_report.md"
    echo "- More accurate for orientation tracking with linear acceleration" >> "${RESULTS_DIR}/comparison_report.md"
    
    echo "$(date) - Comparison report generated: ${RESULTS_DIR}/comparison_report.md"
}

# Main execution
echo "$(date) - Starting IMU fusion filter training and comparison"

# Train models in parallel or sequentially
if [ "$MAX_PARALLEL" -gt 1 ]; then
    echo "Running training in parallel (max: $MAX_PARALLEL jobs)"
    parallel_jobs=()
    
    for filter_type in "${FILTER_TYPES[@]}"; do
        # Check if we need to wait for a job to finish
        while [ ${#parallel_jobs[@]} -ge "$MAX_PARALLEL" ]; do
            for i in "${!parallel_jobs[@]}"; do
                if ! kill -0 ${parallel_jobs[$i]} 2>/dev/null; then
                    unset 'parallel_jobs[$i]'
                fi
            done
            parallel_jobs=("${parallel_jobs[@]}")
            sleep 1
        done
        
        # Start new training job
        echo "Starting training for ${filter_type} filter"
        train_filter_model "${filter_type}" &
        parallel_jobs+=($!)
    done
    
    # Wait for all jobs to finish
    for job in "${parallel_jobs[@]}"; do
        wait $job
    done
else
    echo "Running training sequentially"
    for filter_type in "${FILTER_TYPES[@]}"; do
        train_filter_model "${filter_type}"
    done
fi

# Generate comparison report
generate_comparison

echo "$(date) - IMU fusion filter training and comparison complete"
echo "Results available in: ${RESULTS_DIR}/comparison_report.md"
