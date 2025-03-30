#!/bin/bash
set -e

DEVICE="0,1"
FILTER_TYPES=("madgwick" "kalman" "ekf")
BASE_LR=0.0005
WEIGHT_DECAY=0.001
NUM_EPOCHS=60
PATIENCE=15
SEED=42
BATCH_SIZE=16
RESULTS_DIR="filter_training_results"
CONFIG_DIR="config/smartfallmm"
LOG_LEVEL="DEBUG"
NUM_THREADS=32

mkdir -p "${RESULTS_DIR}/logs"

log() {
  echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" | tee -a "${RESULTS_DIR}/logs/train.log"
}

log "Training with filters: ${FILTER_TYPES[*]}"
log "Log level: ${LOG_LEVEL}"

for filter_type in "${FILTER_TYPES[@]}"; do
  output_dir="${RESULTS_DIR}/${filter_type}"
  mkdir -p "${output_dir}/logs"
  
  log "Starting training for ${filter_type} filter"
  
  cat > "${RESULTS_DIR}/${filter_type}_config.yaml" << EOF
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm
subjects: [29, 30, 31, 33, 45, 46, 34, 37, 39, 38, 43, 35, 36, 44, 32]
model_args:
  num_layers: 3
  embed_dim: 48
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: 64
  mocap_frames: 64
  num_heads: 8
  fusion_type: 'concat'
  dropout: 0.3
  use_batch_norm: true
  feature_dim: 144
dataset_args:
  mode: 'sliding_window'
  max_length: 64
  task: 'fd'
  modalities: ['accelerometer', 'gyroscope']
  age_group: ['young']
  sensors: ['watch']
  fusion_options:
    enabled: true
    filter_type: '${filter_type}'
    process_per_window: true 
    preserve_filter_state: true
    acc_threshold: 3.0
    gyro_threshold: 1.0
    visualize: false
    save_aligned: true
    window_stride: 32
    is_linear_acc: true
    debug: true
    use_cache: false
EOF

  if [[ "${filter_type}" == "madgwick" ]]; then
    echo "    beta: 0.15" >> "${RESULTS_DIR}/${filter_type}_config.yaml"
  elif [[ "${filter_type}" == "kalman" ]]; then
    echo "    process_noise: 5e-5" >> "${RESULTS_DIR}/${filter_type}_config.yaml"
    echo "    measurement_noise: 0.1" >> "${RESULTS_DIR}/${filter_type}_config.yaml"
  elif [[ "${filter_type}" == "ekf" ]]; then
    echo "    process_noise: 1e-5" >> "${RESULTS_DIR}/${filter_type}_config.yaml"
    echo "    measurement_noise: 0.05" >> "${RESULTS_DIR}/${filter_type}_config.yaml"
  fi

  cat >> "${RESULTS_DIR}/${filter_type}_config.yaml" << EOF
batch_size: ${BATCH_SIZE}
test_batch_size: ${BATCH_SIZE}
val_batch_size: ${BATCH_SIZE}
num_epoch: ${NUM_EPOCHS}
feeder: Feeder.Make_Dataset.UTD_mm
train_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: true
val_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: true
test_feeder_args:
  batch_size: ${BATCH_SIZE}
  drop_last: false
seed: ${SEED}
optimizer: adamw
base_lr: ${BASE_LR}
weight_decay: ${WEIGHT_DECAY}
patience: ${PATIENCE}
kfold:
  enabled: true
  num_folds: 5
  fold_assignments:
    - [43, 35, 36]
    - [44, 34, 32]
    - [45, 37, 38]
    - [46, 29, 31]
    - [30, 39]
EOF

  start_time=$(date +%s)
  
  # Force debug output to stdout
  export PYTHONIOENCODING=utf-8
  export PYTHONUNBUFFERED=1
  
  log "Running training with filter type: ${filter_type}"
  
  CUDA_VISIBLE_DEVICES=${DEVICE} python -u main.py \
    --config "${RESULTS_DIR}/${filter_type}_config.yaml" \
    --work-dir "${output_dir}" \
    --model-saved-name "${filter_type}_model" \
    --device 0 1 \
    --multi-gpu True \
    --patience ${PATIENCE} \
    --parallel-threads ${NUM_THREADS} \
    --num-epoch ${NUM_EPOCHS} \
    --seed ${SEED} 2>&1 | tee -a "${output_dir}/logs/training.log"

  end_time=$(date +%s)
  duration=$((end_time - start_time))
  hours=$((duration / 3600))
  minutes=$(( (duration % 3600) / 60 ))
  seconds=$((duration % 60))
  
  if [ $? -ne 0 ]; then
    log "WARNING: Training process exited with error for ${filter_type}"
  else
    log "Completed training for ${filter_type} filter in ${hours}h ${minutes}m ${seconds}s"
  fi
  
  if [ -f "${output_dir}/cv_summary.json" ]; then
    accuracy=$(grep -o '"accuracy": [0-9.]*' "${output_dir}/cv_summary.json" | head -1 | cut -d' ' -f2)
    f1=$(grep -o '"f1": [0-9.]*' "${output_dir}/cv_summary.json" | head -1 | cut -d' ' -f2)
    precision=$(grep -o '"precision": [0-9.]*' "${output_dir}/cv_summary.json" | head -1 | cut -d' ' -f2)
    recall=$(grep -o '"recall": [0-9.]*' "${output_dir}/cv_summary.json" | head -1 | cut -d' ' -f2)
    log "Results for ${filter_type}: acc=${accuracy}%, f1=${f1}, precision=${precision}%, recall=${recall}%"
  fi
  
  if command -v nvidia-smi &> /dev/null; then
    log "Clearing GPU memory after ${filter_type} training"
    nvidia-smi --gpu-reset 2>/dev/null || true
  fi
  
  log "Waiting before next filter training"
  sleep 5
done

log "Creating summary report"
summary_file="${RESULTS_DIR}/filter_comparison_summary.md"

cat > "${summary_file}" << EOF
# IMU Filter Comparison Summary

| Filter Type | Accuracy (%) | F1 Score | Precision (%) | Recall (%) |
|-------------|--------------|----------|---------------|------------|
EOF

for filter_type in "${FILTER_TYPES[@]}"; do
  filter_dir="${RESULTS_DIR}/${filter_type}"
  if [ -f "${filter_dir}/cv_summary.json" ]; then
    accuracy=$(grep -o '"accuracy": [0-9.]*' "${filter_dir}/cv_summary.json" | head -1 | cut -d' ' -f2)
    f1=$(grep -o '"f1": [0-9.]*' "${filter_dir}/cv_summary.json" | head -1 | cut -d' ' -f2)
    precision=$(grep -o '"precision": [0-9.]*' "${filter_dir}/cv_summary.json" | head -1 | cut -d' ' -f2)
    recall=$(grep -o '"recall": [0-9.]*' "${filter_dir}/cv_summary.json" | head -1 | cut -d' ' -f2)
    echo "| ${filter_type} | ${accuracy} | ${f1} | ${precision} | ${recall} |" >> "${summary_file}"
  else
    echo "| ${filter_type} | - | - | - | - |" >> "${summary_file}"
  fi
done

log "All filter training completed"
log "Summary available at: ${summary_file}"
