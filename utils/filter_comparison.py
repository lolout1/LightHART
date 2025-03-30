# config/filter_comparison/experiment.yaml
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
    filter_type: 'madgwick'
    acc_threshold: 3.0
    gyro_threshold: 1.0
    visualize: true
    save_aligned: true
    preserve_filter_state: true

filter_configs:
  madgwick:
    beta: 0.15
  kalman:
    process_noise: 5e-5
    measurement_noise: 0.1
  ekf:
    process_noise: 1e-5
    measurement_noise: 0.05
  ukf:
    alpha: 0.001
    beta: 2.0
    kappa: 0.0
    process_noise: 1e-4

batch_size: 16
test_batch_size: 16
val_batch_size: 16
num_epoch: 60

feeder: Feeder.Make_Dataset.UTD_mm
train_feeder_args:
  batch_size: 16
  drop_last: true

val_feeder_args:
  batch_size: 16
  drop_last: true

test_feeder_args:
  batch_size: 16
  drop_last: false

seed: 42
optimizer: adamw
base_lr: 0.0005
weight_decay: 0.001

kfold:
  enabled: true
  num_folds: 5
  fold_assignments:
    - [43, 35, 36]
    - [44, 34, 32]
    - [45, 37, 38]
    - [46, 29, 31]
    - [30, 39]
