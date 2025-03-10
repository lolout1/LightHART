#!/bin/bash
# distill_student_fixed.sh
# Example usage: bash distill_student_fixed.sh
TEACHER_CKPT="exps/teacher_tt4/teacher_tt4_best.pth"
python distill_student.py \
  --config config/smartfallmm/distill_student_fixed.yaml \
  --teacher-weights $TEACHER_CKPT \
  --phase train \
  --device 0 \
  --seed 2 \
  --print-log True

