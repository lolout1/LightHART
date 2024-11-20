teacher_weights="kdTransformerWatch33.pth"
student_dir="exps/smartfall_har/student/watchgyro_divid3bw20/accel"
work_dir="exps/smartfall_har/kd/student/"
student_weights="ttfStudent"
teacher_dir="exps/smartfall_har/kd/student"
result_file="result.txt"
# weights="berkley_best.pt"
# work_dir="exps/bmhad_woKD/late_fusion_epoch150_alldrop0.4"
# #Utd student without KD
#python3 main.py --config ./config/utd/student.yaml --model-saved-name $student_weights --work-dir $student_dir --device 7 --base-lr 2.5e-3 --include-val True
#python3 main.py --config ./config/utd/student.yaml --work-dir $work_dir  --weights "$work_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'
#Utd teacher
#python3 main.py --config ./config/utd/teacher.yaml --work-dir $work_dir --model-saved-name $teacher_weights  --device 7  --base-lr 2.5e-3 --phase 'train' --result-file $work_dir/$result_file  --include-val True
#python3 main.py --config ./config/utd/teacher.yaml --work-dir $work_dir  --weights "$work_dir/$teacher_weights" --device 7 --base-lr 2.5e-3 --phase 'test'
#berkley_student
#python3 main.py --config ./config/berkley/student.yaml --work-dir "$student_dir"  --model-saved-name "$student_weights" --device 7 --base-lr 2.5e-3 --include-val True
#python3 main.py --config ./config/berkley/student.yaml --work-dir "$student_dir"  --weights "$student_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'
#utd student
#python3 main.py --config ./config/czu/student.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True
#czu
#python3 main.py --config ./config/czu/student.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True
#python3 distiller.py --config ./config/czu/distill.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True
#smartfallmm
#skelton_only experiment
#python3 main.py --config ./config/smartfallmm/SKteacher.yaml --work-dir $work_dir --model-saved-name $teacher_weights  --device 0  --base-lr 2.5e-3 --phase 'train' --result-file $work_dir/$result_file  --include-val True
#multimodal experiment
python3 main4.py --config ./config/smartfallmm/t3.yaml --weights "exps/smartfall_har/student/watchgyro_divid3bw20/accel/ttfStudent.pth" --work-dir $teacher_dir --model-saved-name $teacher_weights  --device 0 --include-val True
#accelerometer only experiment
#python3 main.py --config ./config/smartfallmm/tr.yaml --work-dir $student_dir --model-saved-name $student_weights --device 0 --include-val True #--base-lr 1e-3
#python3 main.py --config ./config/smartfallmm/transformer.yaml --phase 'train' --work-dir $teacher_dir --model-saved-name $teacher_weights --device 0 --base-lr 1e-3 --include-val True
#distillation
#python3 distiller.py --config ./config/smartfallmm/distill.yaml --work-dir $work_dir  --teacher-weight "$teacher_dir/$teacher_weights" --model-saved-name "$student_weights" --device 0 --include-val True --base-lr 2.5e-3 --include-val True
