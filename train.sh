#!/bin/bash

teacher_weights="kdTransformerWatch33.pth"
student_dir="exps/smartfall_har/student/watchgyro_divid3bw20/accel"
work_dir="exps/smartfall_har/kd/student/"
student_weights="ttfStudent"
teacher_dir="exps/smartfall_har/kd/student"
result_file="result.txt"

# Execute training using main4.py
python3 main4.py --config ./config/smartfallmm/StudentTrans.yaml \
    --model Models.StudentTrans.LightTransformerStudent \
    --work-dir $work_dir \
    --model-saved-name $student_weights \
    --device 0 \
    --base-lr .0005 \
    --include-val True
