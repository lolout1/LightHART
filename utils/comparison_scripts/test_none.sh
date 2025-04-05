#!/bin/bash
set -e
set -o pipefail

CONFIG_FILE="config/filter_comparison/none.yaml"
OUTPUT_DIR="filter_comparison_results/none_model"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Config file not found: ${CONFIG_FILE}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"

echo "Starting training with none filter"
CUDA_VISIBLE_DEVICES=0,1 python main2.py \
    --config "${CONFIG_FILE}" \
    --work-dir "${OUTPUT_DIR}" \
    --model-saved-name "none_model" \
    --device 0 1 \
    --multi-gpu True \
    --patience 15 \
    --filter-type "none" \
    --parallel-threads 8 \
    --num-epoch 100 2>&1 | tee "${OUTPUT_DIR}/logs/training.log"

if [ ! -f "${OUTPUT_DIR}/test_summary.json" ]; then
    echo "Recovering summary from fold results"
    python "utils/comparison_scripts/recover_cv_summary.py" \
           --output-dir "${OUTPUT_DIR}" \
           --filter-type "none"
fi

echo "Training complete"
