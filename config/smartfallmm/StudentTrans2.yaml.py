# Basic configuration
model: "Models.LightTransformer.PyTorchTransformer"
dataset: "smartfallmm"

# Model parameters - these are passed directly to the model
model_args:
  seq_len: 128       # same as n_timesteps
  channels: 3        # watch accelerometer (x, y, z)
  num_layers: 4
  embed_dim: 128
  mlp_dim: 16
  num_heads: 4
  dropout_rate: 0.25
  attn_drop_rate: 0.25

# Dataset arguments
dataset_args:
  mode: "avg_pool"
  max_length: 128
  task: "fd"
  modalities: ["accelerometer"]
  age_group: ["older"]
  sensors: ["watch"]

# Subject IDs as a flat list for the argument parser
subjects: [1, 2, 3, 4, 5, 6]

# Training parameters
batch_size: 32
test_batch_size: 32
val_batch_size: 32
num_epoch: 100
start_epoch: 0

# Optimizer settings
optimizer: "adamw"
base_lr: 0.001
weight_decay: 0.01

# Device configuration
device: [0]

# Feeder settings
feeder: "Feeder.Make_Dataset.UTD_mm"
train_feeder_args:
  batch_size: 32
val_feeder_args:
  batch_size: 32
test_feeder_args:
  batch_size: 32

# Additional training settings
num_worker: 4
seed: 42
include_val: true
print_log: true
phase: "train"

# Loss function config
loss: "torch.nn.BCEWithLogitsLoss"
loss_args: {}

# Scheduler settings
scheduler: "cosine"
scheduler_args:
  warmup_epochs: 5
