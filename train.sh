#!/bin/bash

# Set up paths and configurations
DATE=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="madgwick_results_${DATE}"
CONFIG_FILE="config/smartfallmm/madgwick_fusion.yaml"
MODEL_NAME="madgwick_model"
DEVICE="0,1"  # Use both GPUs if available

# Create working directory
mkdir -p ${WORK_DIR}

# Copy config file for reference
cp ${CONFIG_FILE} ${WORK_DIR}/

echo "Starting Madgwick filter training at $(date)"
echo "Working directory: ${WORK_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "======================================================================================"

# Run training with cross-validation
CUDA_VISIBLE_DEVICES=${DEVICE} python main.py \
  --config ${CONFIG_FILE} \
  --work-dir ${WORK_DIR} \
  --model-saved-name ${MODEL_NAME} \
  --device 0 1 \
  --multi-gpu True \
  --kfold True \
  --parallel-threads 48 \
  --num-epoch 60 \
  --patience 15

# Check if training completed successfully
if [ $? -eq 0 ]; then
  echo "======================================================================================"
  echo "Training completed successfully at $(date)"
  
  # Extract key metrics from the results
  if [ -f "${WORK_DIR}/cv_summary.json" ]; then
    echo "Cross-validation results:"
    python -c "
import json, sys
with open('${WORK_DIR}/cv_summary.json') as f:
    data = json.load(f)
    metrics = data['average_metrics']
    print(f\"Average accuracy: {metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}\")
    print(f\"Average F1 score: {metrics['f1']:.4f} ± {metrics['f1_std']:.4f}\")
    print(f\"Average precision: {metrics['precision']:.4f} ± {metrics['precision_std']:.4f}\")
    print(f\"Average recall: {metrics['recall']:.4f} ± {metrics['recall_std']:.4f}\")
    print(f\"Average balanced accuracy: {metrics['balanced_accuracy']:.4f} ± {metrics['balanced_accuracy_std']:.4f}\")
    "
  else
    echo "Cross-validation summary not found"
  fi
else
  echo "======================================================================================"
  echo "Training failed with exit code $?"
fi

echo "Results saved to ${WORK_DIR}"
