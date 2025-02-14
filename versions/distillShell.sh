#!/bin/bash
# run_distiller.sh
# This script runs the distillation process using distiller.py.
# Ensure that the config file and teacher weight directory are correctly set.

# Configuration file (must follow your repo's strict YAML format)
CONFIG="./config/smartfallmm/distill.yaml"

# Hyperparameters and settings (adjust as needed)
BATCH_SIZE=16
NUM_EPOCH=75
DEVICE="0"              # Use device 0 (or list multiple devices separated by spaces)
SEED=2
WORK_DIR="exps/distilled_student"
TEACHER_WEIGHT_DIR="exps/teacher_var_time"

# Optional: other arguments can be added as needed.

echo "Running distiller with the following parameters:"
echo "Config file: $CONFIG"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCH"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo "Work directory: $WORK_DIR"
echo "Teacher weight directory: $TEACHER_WEIGHT_DIR"

# Run the distiller script with the specified arguments.
python distiller.py \
    --config $CONFIG \
    --batch-size $BATCH_SIZE \
    --num-epoch $NUM_EPOCH \
    --device $DEVICE \
    --seed $SEED \
    --work-dir $WORK_DIR \
    #--student-model Models.fall_time2vec_transformer.FallTime2VecTransformer \
    #--teacher-model Models.t3.TransformerTeacher
    #--teacher-weight-dir $TEACHER_WEIGHT_DIR
