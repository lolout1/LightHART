#!/bin/bash
# run_training.sh

# Find available GPU
nvidia-smi -L 2>/dev/null || echo "No NVIDIA GPUs detected"
GPU_ID=$(nvidia-smi -L 2>/dev/null | grep -n "GPU" | head -1 | cut -d: -f1)
GPU_ID=$((GPU_ID - 1))
GPU_ID=${GPU_ID:-0}  # Default to 0 if detection fails

# Create data directory if it doesn't exist
DATA_DIR="data/smartfallmm"
mkdir -p ${DATA_DIR}/{young}/accelerometer/watch

# Run training with mock data option and proper paths
python train.py \
    --work-dir "results" \
    --use-gpu true \
    --device ${GPU_ID} \
    --batch-size 16 \
    --test-batch-size 16 \
    --max-epoch 100 \
    --patience 20 \
    --base-lr 0.0001 \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --embed-dim 32 \
    --num-heads 4 \
    --num-layers 2 \
    --data-dir "${DATA_DIR}" \
    --num-workers 0
