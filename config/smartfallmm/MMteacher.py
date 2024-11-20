model: Models.multi_modal_teacher.MultiModalTeacher
dataset: 'smartfallmm'
subjects:
  - 29
  - 30
  - 31
  - 32
  - 33
  - 34
  - 35
  - 36
  - 37
  - 38
  - 39
  - 43
  - 44
  - 45
  - 46

model_args:
  device: 'cuda'
  mocap_frames: 128
  acc_frames: 128
  num_joints: 32
  in_chans: 3
  num_patch: 4
  acc_coords: 4
  spatial_embed: 64
  sdepth: 2
  adepth: 2
  tdepth: 2
  num_heads: 2
  mlp_ratio: 2
  qkv_bias: True
  qk_scale: null
  op_type: 'all'
  embed_type: 'lin'
  drop_rate: 0.2
  attn_drop_rate: 0.2
  drop_path_rate: 0.2
  norm_layer: null
  num_classes: 2

# Loss function
loss: 'torch.nn.CrossEntropyLoss'
loss_args: {}

# Dataset arguments
dataset_args: 
  mode: 'avg_pool'
  max_length: 128
  task: 'fd'
  modalities: ['skeleton', 'accelerometer']
  age_group: ['young','old']
  sensors: ['watch', 'phone']

# Training parameters
batch_size: 32
test_batch_size: 32
val_batch_size: 32
num_epoch: 120

# Data loader configuration
feeder: Feeder.Make_Dataset.UTD_mm

train_feeder_args:
  batch_size: 32

val_feeder_args:
  batch_size: 32

test_feeder_args:
  batch_size: 32

seed: 2
optimizer: 'adamw'  # Changed to AdamW for better convergence
base_lr: 0.001
weight_decay: 0.01

# Additional settings
work_dir: 'exps/smartfall_har/teacher/multimodal'
model_saved_name: 'multimodal_teacher'
print_log: True
phase: 'train'
num_worker: 4
result_file: null
