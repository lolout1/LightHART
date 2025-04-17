#!/bin/bash
set -e

# Environment setup
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create timestamped working directory
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="results_${RUN_TIMESTAMP}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p $WORK_DIR $LOG_DIR

# Set up logging
exec > >(tee -a "${LOG_DIR}/training_run.log") 2>&1

echo "================================================================"
echo "SmartFallMM Edge Model Training - Run $RUN_TIMESTAMP"
echo "================================================================"

# Define subject groups for cross-validation
VAL_SUBJECTS="38,46"
TRAIN_CORE_SUBJECTS="45,36,29"
TEST_SUBJECTS="32,39,30,31,33,34,35,37,43,44"
ALL_SUBJECTS="${TEST_SUBJECTS},${TRAIN_CORE_SUBJECTS},${VAL_SUBJECTS}"

# Create directory structure
mkdir -p Models utils Feeder
touch Models/__init__.py utils/__init__.py Feeder/__init__.py

# Start training
echo "Starting model training..."

python main.py \
    --work-dir $WORK_DIR \
    --use-gpu True \
    --device 0 \
    --seed 42 \
    --batch-size 16 \
    --test-batch-size 16 \
    --num-worker 8 \
    --embed-dim 32 \
    --num-heads 2 \
    --num-layer 2 \
    --base-lr 0.0001 \
    --weight-decay 0.001 \
    --grad-clip 1.0 \
    --max-epoch 100 \
    --patience 20 \
    --subjects $ALL_SUBJECTS \
    --fold -1

echo "================================================================"
echo "Training complete."
echo "Models and checkpoints saved to: $WORK_DIR"
echo "================================================================"
