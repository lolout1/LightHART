#!/bin/bash

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/cv_${TIMESTAMP}"
mkdir -p $RESULTS_DIR
echo "Results directory: $RESULTS_DIR"

# Define filters
FILTERS=("madgwick" "kalman" "ekf")

# Run training with proper fold config
run_fold() {
    filter=$1
    config_file="${RESULTS_DIR}/${filter}_config.yaml"
    
    cat > "$config_file" <<EOF
work_dir: ./${RESULTS_DIR}/${filter}
seed: 2023
dataset: smartfallmm
phase: train
kfold: true
num_folds: 5

# Mandatory fields
model: Models.fusion_transformer.FusionTransModel
model_args:
  num_layers: 3
  embed_dim: 32
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: 64
  mocap_frames: 64
  num_heads: 4
  fusion_type: concat
  dropout: 0.3
  use_batch_norm: true
  feature_dim: 64

# Dataset config
dataset_args:
  sensors: ['watch']
  modalities: ['accelerometer', 'gyroscope']
  age_group: ['young']
  mode: sliding_window
  max_length: 64
  task: fd
  fusion_options:
    enabled: true
    filter_type: ${filter}

# Critical - must specify feeder
feeder: Feeder.Make_Dataset.UTD_mm
train_feeder_args:
  batch_size: 32
  num_workers: 4
test_feeder_args:
  batch_size: 64
  num_workers: 4

# Subject list - 38,46 for validation, 45,36,29 always in training
subjects: [32,39,30,31,33,34,35,37,43,44,45,36,29,38,46]

# Training parameters
optimizer: adamw
base_lr: 0.0005
weight_decay: 0.001
scheduler: ReduceLROnPlateau
num_epoch: 60
patience: 15
EOF

    echo "Running ${filter} cross-validation"
    python main.py --config "$config_file"
}

# Run each filter
for filter in "${FILTERS[@]}"; do
    echo "Processing filter: ${filter}"
    run_fold "${filter}"
    
    # Quick results summary
    metrics_file="${RESULTS_DIR}/${filter}/test_metrics.json"
    if [ -f "$metrics_file" ]; then
        echo "Results for ${filter}:"
        cat "$metrics_file" | sed -e 's/{//g' -e 's/}//g' -e 's/"//g' -e 's/,//g'
    fi
done

# Compare filters
echo "Generating comparison"
python - <<EOF
import os, json, matplotlib.pyplot as plt
results_dir = "${RESULTS_DIR}"
filters = [${FILTERS[@]/#/\"}${FILTERS[@]/%/\",}]
results = {}

for f in filters:
    metrics_file = os.path.join(results_dir, f, "test_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file) as file:
            results[f] = json.load(file)

if results:
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, metric in enumerate(metrics):
        ax = axs[i//2, i%2]
        values = [results[f].get(metric, 0) for f in results]
        ax.bar(list(results.keys()), values)
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/comparison.png")
    print(f"Comparison saved to {results_dir}/comparison.png")
    
    with open(f"{results_dir}/comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
EOF

echo "Cross-validation complete"
