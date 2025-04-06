#!/bin/bash

# Load necessary modules

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1  # First GPU for training, second for data loading

# Define filters to use (only using madgwick, kalman, and ekf)
FILTERS="madgwick,kalman,ekf"

# Set working directory for results
WORK_DIR="results"
mkdir -p $WORK_DIR

echo "Starting training with filters: $FILTERS"
python main.py \
    --work-dir $WORK_DIR \
    --phase train \
    --filters $FILTERS \
    --use-gpu True \
    --device 0 \
    --seed 42 \
    --feeder Feeder.Make_Dataset.UTD_mm \
    --batch-size 16 \
    --test-batch-size 32 \
    --num-worker 8 \
    --fuse True \
    --model Models.fusion_transformer.FusionTransModel \
    --optimizer AdamW \
    --base-lr 0.0005 \
    --weight-decay 0.001 \
    --loss bce \
    --max-epoch 100 \
    --patience 20 \
    --use_features False  # Set to False for inference on watch data only

echo "Cross-validation complete"

