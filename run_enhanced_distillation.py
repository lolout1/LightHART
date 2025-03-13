#!/bin/bash
# run_enhanced_distillation.sh
# Script to run enhanced cross-modal distillation for quaternion-based fall detection

# Set device and seed
DEVICE=0
SEED=42

# Create log directory
LOG_DIR="logs/enhanced_distillation"
mkdir -p $LOG_DIR

echo "=== Starting Enhanced Cross-Modal Distillation ==="
echo "Device: $DEVICE, Seed: $SEED"
date

# Train teacher with EKF if not already trained
if [ ! -f "exps/teacher_quat/teacher_quat_best.pth" ]; then
    echo -e "\n=== Training Teacher with EKF ==="
    python train_teacher_quat.py \
      --config config/smartfallmm/teacher_quat.yaml \
      --device $DEVICE \
      --seed $SEED \
      > $LOG_DIR/teacher_ekf.log 2>&1
fi

# Run enhanced distillation with various configurations
for LOSS_TYPE in "enhanced" "adversarial"; do
    echo -e "\n=== Running Enhanced Distillation with $LOSS_TYPE Loss ==="
    python distill_student_enhanced.py \
      --config config/smartfallmm/distill_student_enhanced.yaml \
      --device $DEVICE \
      --seed $SEED \
      --loss-type $LOSS_TYPE \
      --visualize \
      > $LOG_DIR/distill_${LOSS_TYPE}.log 2>&1
done

# Run 5-fold cross-validation with the best approach
echo -e "\n=== Running 5-Fold Cross-Validation with Enhanced Distillation ==="
python distill_student_enhanced.py \
  --config config/smartfallmm/distill_student_enhanced.yaml \
  --device $DEVICE \
  --seed $SEED \
  --loss-type enhanced \
  --cross-val \
  > $LOG_DIR/distill_cv.log 2>&1

# Compare all approaches
echo -e "\n=== Comparing All Approaches ==="
python compare_enhanced.py \
  --config config/smartfallmm/distill_student_enhanced.yaml \
  --output-dir comparison_results_enhanced \
  --device $DEVICE \
  > $LOG_DIR/comparison.log 2>&1

echo -e "\n=== Enhanced Distillation Pipeline Completed ==="
date
