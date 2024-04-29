#!/bin/bash
teacher_weights="spctTransformer.pth"
teacher_dir="exps/smfall_woKD/spatial_transformer"
student_dir="exps/smfall_woKD/inertial_transformer"
student_weights="ttfstudent.pth"
distil_dir="exps/smfall_wKD/inertail_transformer"
#teacher_dir="exps/UTD_wKD/ttf4"
result_file="result.txt"


# weights="berkley_best.pt"
# work_dir="exps/bmhad_woKD/late_fusion_epoch150_alldrop0.4"

# #Utd student without KD
# python3 main.py --config ./config/utd/student.yaml --model-saved-name $student_weights --work-dir $work_dir --device 7 --base-lr 2.5e-2 --include-val True
# python3 main.py --config ./config/utd/student.yaml --work-dir $work_dir  --weights "$work_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'


#Utd teacher 
#python3 main.py --config ./config/utd/teacher.yaml --work-dir $work_dir --model-saved-name $teacher_weights  --device 7  --base-lr 2.5e-3 --phase 'train' --result-file $work_dir/$result_file  --include-val True
#python3 main.py --config ./config/utd/teacher.yaml --work-dir $work_dir  --weights "$work_dir/$teacher_weights" --device 7 --base-lr 2.5e-3 --phase 'test'

#berkley_student
#python3 main.py --config ./config/berkley/student.yaml --work-dir "$work_dir" --weights "$work_dir/$weights" --model-saved-name "$weights" --device 3 --base-lr 2.5e-3 --phase test
#python3 main.py --config ./config/berkley/student.yaml --work-dir "$work_dir" --model-saved-name "$weights" --weights "$work_dir/$weights" --device 3  --base-lr 2.5e-3 --include-val True
#python3 distiller.py --config ./config/berkley/distill.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True

#utd student
#python3 main.py --config ./config/czu/student.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True

#distillation 
#python3 distiller.py --config ./config/utd/distill.yaml --work-dir $work_dir --device 7  --teacher-weight "$work_dir/$teacher_weights" --model-saved-name "$student_weights" --device 7 --base-lr 2.5e-3 --include-val True
#python3 distiller.py --config ./config/utd/distill.yaml --weights "$work_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'

#czu 
#python3 main.py --config ./config/czu/student.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True
#python3 distiller.py --config ./config/czu/distill.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True

#smartfallmm
#python3 main.py --config ./config/smartfallmm/teacher.yaml --work-dir $work_dir --model-saved-name $teacher_weights  --weights $work_dir/$teacher_weights --device 1 --base-lr 2.5e-3 --include-val True
python main.py --config ./config/smartfallmm/student.yaml --work-dir $student_dir --model-saved-name $student_weights --device 1 --base-lr 2.5e-3 --include-val True
python main.py --config ./config/smartfallmm/student.yaml --work-dir $student_dir --weights $student_dir/$student_dir --device 1 --base-lr 2.5e-3 --phase test
#python3 distiller.py --config ./config/smartfallmm/distill.yaml --work-dir $work_dir --model-saved-name $student_weights  --teacher-weights $work_dir/$teacher_weights --device 6 --base-lr 2.5e-3 --include-val True
#python3 distiller.py --config ./config/smartfallmm/distill.yaml --work-dir $student_dir --device 7  --teacher-weight "$teacher_dir/$teacher_weights" --model-saved-name "$student_weights" --device 7 --base-lr 2.5e-3 --include-val True