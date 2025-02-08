#!/bin/bash

python tt2.py \
  --config ./config/smartfallmm/master_t3.yaml \
  --work-dir ./exps/teacher_var_time2 \
  --model-saved-name teacher_best \
  --device 0 \
  --base-lr 1e-4 \
  --include-val True
