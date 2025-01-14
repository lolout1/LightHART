#!/bin/bash

# Set base directories
BASE_DIR="exps/smartfall_har"
TEACHER_DIR="${BASE_DIR}/teacher/multimodal"
STUDENT_DIR="${BASE_DIR}/student/watch_only"

# Set model weights names
TEACHER_WEIGHTS="teacher_model"
STUDENT_WEIGHTS="student_model"

# Create directories if they don't exist
mkdir -p "$TEACHER_DIR"
mkdir -p "$STUDENT_DIR"

# Training teacher model
echo "Training teacher model..."
python3 main4.py \
    --config ./config/smartfallmm/teacher.yaml \
    --work-dir "${TEACHER_DIR}" \
    --model-saved-name "${TEACHER_WEIGHTS}" \
    --device 0 \
    --phase train \
    --base-lr 0.0001 \
    --include-val True

# After teacher training, train student model
echo "Training student model..."
python3 main4.py \
    --config ./config/smartfallmm/student.yaml \
    --work-dir "${STUDENT_DIR}" \
    --model-saved-name "${STUDENT_WEIGHTS}" \
    --teacher-weight "\\wsl.localhost\Ubuntu\home\abheekp\Fall_Detection_KD_Multimodal\single_sensor\LightHART\exps\smartfall_har\kd\student\fold_1\TeacherModel_best_weights_f1_0.9320_loss_0.1896.pt" \
    --device 0 \
    --phase train \
    --base-lr 0.0005 \
    --include-val True
