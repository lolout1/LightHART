#!/bin/bash

python tt.py \
  --config ./config/smartfallmm/t3.yaml \
  --work-dir ./exps/teacher_var_time \
  --model-saved-name teacher_best \
  --device 0 \
  --base-lr 1e-4 \
  --include-val True
