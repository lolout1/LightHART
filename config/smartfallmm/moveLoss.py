model: "Models.aMaxOp.FallDetectionModel"

# Model arguments
model_args:
  seq_len: 128          
  num_channels: 3       
  num_filters: 64       # Reduced to match new model architecture
  num_classes: 1        # Binary classification for falls

# Dataset configuration
dataset: "smartfallmm"
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

dataset_args:
  root_dir: "data/smartfallmm"
  age_groups:
    - "young"
    - "old"
  modalities:
    - "accelerometer"
  sensors:
    accelerometer:
      - "phone"
      - "watch"
  mode: "avg_pool"
  max_length: 128
  task: "fd"            
  use_smv: false        

# Rest remains the same...