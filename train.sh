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
NUM_THREADS=30

# Create required directories
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/visualizations"

# Function to log messages
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" | tee -a "${RESULTS_DIR}/logs/main.log"
}

# Print configuration
log "=== IMU Fusion Filter Training ==="
log "Filters to evaluate: ${FILTER_TYPES[*]}"
log "Results directory: ${RESULTS_DIR}"
log "Epochs: ${NUM_EPOCHS}"
log "Batch size: ${BATCH_SIZE}"
log "Learning rate: ${BASE_LR}"
log "Device: ${DEVICE}"
log "============================="

# Function to train a model with a specific filter
train_filter_model() {
    local filter_type="$1"
    local config_file="${CONFIG_DIR}/${filter_type}_fusion.yaml"
    local output_dir="${RESULTS_DIR}/${filter_type}_model"
    
    mkdir -p "${output_dir}/logs"
    
    log "Starting training for ${filter_type} filter"
    
    # Ensure config exists
    if [ ! -f "${config_file}" ]; then
        log "ERROR: Config file not found: ${config_file}"
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
        --parallel-threads ${NUM_THREADS} \
        --num-epoch ${NUM_EPOCHS} \
        --seed ${SEED} 2>&1 | tee "${output_dir}/logs/training.log"
    
    # Check training status
    if [ $? -ne 0 ]; then
        log "WARNING: Training process exited with error for ${filter_type}"
        # Try to recover fold results if available
        python -c "
import os, json, glob
try:
    fold_dirs = sorted(glob.glob('${output_dir}/fold_*'))
    if fold_dirs:
        metrics = []
        for fold_dir in fold_dirs:
            results_file = os.path.join(fold_dir, 'validation_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    metrics.append(json.load(f))
        
        if metrics:
            import numpy as np
            avg_metrics = {}
            for key in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']:
                values = [m.get(key, 0) for m in metrics]
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[key+'_std'] = float(np.std(values))
            
            summary = {
                'filter_type': '${filter_type}',
                'average_metrics': avg_metrics,
                'fold_metrics': metrics
            }
            
            with open('${output_dir}/cv_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print('Created summary from available fold results')
        else:
            raise ValueError('No fold results available')
    else:
        raise ValueError('No fold directories found')
except Exception as e:
    print(f'Error recovering results: {e}')
    with open('${output_dir}/cv_summary.json', 'w') as f:
        json.dump({
            'filter_type': '${filter_type}',
            'average_metrics': {
                'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'balanced_accuracy': 0
            },
            'fold_metrics': []
        }, f, indent=2)
    print('Created empty summary')
"
    fi
    
    log "Completed training for ${filter_type} filter"
    return 0
}

# Function to generate comparison visualization
generate_comparison_plot() {
    log "Generating comparison visualization"
    
    python -c "
import matplotlib.pyplot as plt
import numpy as np
import json
import os

filter_types = ['${FILTER_TYPES[0]}', '${FILTER_TYPES[1]}', '${FILTER_TYPES[2]}']
metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
labels = ['Accuracy (%)', 'F1 Score', 'Precision (%)', 'Recall (%)', 'Balanced Acc (%)']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

# Collect metrics
data = []
for filter_type in filter_types:
    summary_path = '${RESULTS_DIR}/' + filter_type + '_model/cv_summary.json'
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            avg_metrics = summary.get('average_metrics', {})
            values = []
            errors = []
            for metric in metrics:
                values.append(avg_metrics.get(metric, 0))
                errors.append(avg_metrics.get(metric + '_std', 0))
            data.append((filter_type.capitalize(), values, errors))
    except Exception as e:
        print(f'Error loading {summary_path}: {e}')
        data.append((filter_type.capitalize(), [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]))

# Plot
plt.figure(figsize=(12, 8))
width = 0.2
x = np.arange(len(metrics))

for i, (filter_name, values, errors) in enumerate(data):
    offset = (i - len(data)/2 + 0.5) * width
    plt.bar(x + offset, values, width, label=filter_name, yerr=errors, capsize=5)

plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Performance Comparison of IMU Fusion Filters', fontsize=14, fontweight='bold')
plt.xticks(x, labels, fontsize=10)
plt.ylim(0, 100)
plt.legend(loc='best')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('${RESULTS_DIR}/visualizations/filter_comparison.png', dpi=300)
plt.close()

print('Comparison plot saved to ${RESULTS_DIR}/visualizations/filter_comparison.png')
"
}

# Function to generate comparison report
generate_comparison_report() {
    log "Generating comparison report"
    
    # Create comparison table
    echo "# IMU Fusion Filter Comparison Report" > "${RESULTS_DIR}/comparison_report.md"
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    echo "## Performance Summary" >> "${RESULTS_DIR}/comparison_report.md"
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    echo "| Filter | Accuracy | F1 Score | Precision | Recall | Balanced Accuracy |" >> "${RESULTS_DIR}/comparison_report.md"
    echo "|--------|----------|----------|-----------|--------|-------------------|" >> "${RESULTS_DIR}/comparison_report.md"
    
    # Find best filter
    best_f1=0
    best_filter=""
    
    for filter_type in "${FILTER_TYPES[@]}"; do
        if [ -f "${RESULTS_DIR}/${filter_type}_model/cv_summary.json" ]; then
            # Extract metrics using Python
            metrics=$(python -c "
import json
try:
    with open('${RESULTS_DIR}/${filter_type}_model/cv_summary.json', 'r') as f:
        data = json.load(f)
        avg = data.get('average_metrics', {})
        acc = avg.get('accuracy', 0)
        acc_std = avg.get('accuracy_std', 0)
        f1 = avg.get('f1', 0)
        f1_std = avg.get('f1_std', 0)
        prec = avg.get('precision', 0)
        prec_std = avg.get('precision_std', 0)
        rec = avg.get('recall', 0)
        rec_std = avg.get('recall_std', 0)
        bal_acc = avg.get('balanced_accuracy', 0)
        bal_acc_std = avg.get('balanced_accuracy_std', 0)
        print(f'{acc:.2f},{acc_std:.2f},{f1:.2f},{f1_std:.2f},{prec:.2f},{prec_std:.2f},{rec:.2f},{rec_std:.2f},{bal_acc:.2f},{bal_acc_std:.2f}')
except:
    print('0,0,0,0,0,0,0,0,0,0')
")
            
            IFS=',' read -r acc acc_std f1 f1_std prec prec_std rec rec_std bal_acc bal_acc_std <<< "$metrics"
            
            # Update best filter
            if (( $(echo "$f1 > $best_f1" | bc -l) )); then
                best_f1=$f1
                best_filter=$filter_type
            fi
            
            # Add row to table
            echo "| ${filter_type^} | ${acc}% ± ${acc_std}% | ${f1} ± ${f1_std} | ${prec}% ± ${prec_std}% | ${rec}% ± ${rec_std}% | ${bal_acc}% ± ${bal_acc_std}% |" >> "${RESULTS_DIR}/comparison_report.md"
        fi
    done
    
    # Add filter characteristics
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
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    
    # Add conclusion
    echo "## Conclusion" >> "${RESULTS_DIR}/comparison_report.md"
    echo "" >> "${RESULTS_DIR}/comparison_report.md"
    echo "Based on the cross-validation results, the **${best_filter^}** filter provides the best performance for fall detection with an F1 score of ${best_f1}." >> "${RESULTS_DIR}/comparison_report.md"
    echo "This filter should be preferred for real-time implementation on wearable devices." >> "${RESULTS_DIR}/comparison_report.md"
    
    log "Comparison report generated: ${RESULTS_DIR}/comparison_report.md"
}

# Main execution
log "Starting IMU fusion filter evaluation"

# Train models sequentially
for filter_type in "${FILTER_TYPES[@]}"; do
    train_filter_model "${filter_type}"
    
    # Clear GPU memory between runs
    if command -v nvidia-smi &> /dev/null; then
        log "Clearing GPU memory after ${filter_type} training"
        nvidia-smi --gpu-reset 2>/dev/null || true
    fi
    
    # Brief pause between training runs
    sleep 5
done

# Generate comparison visualization and report
generate_comparison_plot
generate_comparison_report

log "IMU fusion filter evaluation complete"
log "Results available in: ${RESULTS_DIR}/comparison_report.md"
log "Visualization available in: ${RESULTS_DIR}/visualizations/filter_comparison.png"
