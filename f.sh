#!/bin/bash

# ================================================================
# IMU FUSION FILTER COMPARISON FOR FALL DETECTION
# ================================================================
# This script runs a comprehensive comparison of different
# IMU sensor fusion filters for fall detection:
# - Madgwick filter (baseline)
# - Complementary filter
# - Kalman filter
# - Extended Kalman filter (EKF)
# - Unscented Kalman filter (UKF)
#
# The script handles the entire pipeline:
# 1. Filter implementation testing
# 2. Configuration generation
# 3. Dataset preparation
# 4. Model training with each filter
# 5. Results analysis and visualization
# ================================================================

# Set strict error handling
set -e
set -o pipefail

# ================================================================
# CONFIGURATION
# ================================================================
DEVICE="0,1"                        # GPU devices to use
BASE_LR=0.0005                      # Base learning rate
WEIGHT_DECAY=0.001                  # Weight decay factor
NUM_EPOCHS=60                       # Number of training epochs
PATIENCE=15                         # Early stopping patience
RESULTS_DIR="filter_comparison_results" # Results directory
CONFIG_DIR="config/filter_comparison"  # Config directory
LOG_DIR="logs"                      # Log directory
VISUALIZATION_DIR="visualizations"  # Visualization directory
PARALLEL_THREADS=48                 # Number of parallel threads
KFOLD=true                          # Use k-fold cross-validation
NUM_FOLDS=5                         # Number of folds
DATE=$(date +"%Y%m%d_%H%M%S")       # Current date/time for naming

# Filter types to compare
FILTER_TYPES=("madgwick" "comp" "kalman" "ekf" "ukf")

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${CONFIG_DIR}"
mkdir -p "${RESULTS_DIR}/${LOG_DIR}"
mkdir -p "${RESULTS_DIR}/${VISUALIZATION_DIR}"
mkdir -p "debug_logs"

# ================================================================
# LOGGING SETUP
# ================================================================
MAIN_LOG="${RESULTS_DIR}/${LOG_DIR}/main_${DATE}.log"
FILTER_TEST_LOG="${RESULTS_DIR}/${LOG_DIR}/filter_test_${DATE}.log"

# Log function with timestamp
log() {
    local level=$1
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $2"
    echo "$msg"
    echo "$msg" >> "${MAIN_LOG}"
}

# ================================================================
# UTILITY FUNCTIONS
# ================================================================
# Function to handle errors
handle_error() {
    local exit_code=$?
    local line_number=$1
    log "ERROR" "Command failed at line $line_number with exit code $exit_code"
    
    # Log Python-specific error information if available
    if [ -f "error_traceback.txt" ]; then
        log "ERROR" "Python error details:"
        cat "error_traceback.txt" >> "${MAIN_LOG}"
        rm "error_traceback.txt"
    fi
    
    log "ERROR" "Exiting due to error"
    exit $exit_code
}
trap 'handle_error $LINENO' ERR

# Function to run diagnostics and test filter implementations
test_filters() {
    log "INFO" "Testing filter implementations"
    
    # Create a Python script to test the filter implementations
    cat > test_filters.py << 'EOL'
import numpy as np
import sys
import traceback
import time
import logging
from utils.imu_fusion import (
    MadgwickFilter, 
    ComplementaryFilter, 
    KalmanFilter, 
    ExtendedKalmanFilter, 
    UnscentedKalmanFilter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("filter_test")

def test_filter(filter_type, iterations=1000):
    """Test a specific filter with synthetic data"""
    # Create test data
    acc_data = np.array([0, 0, 9.81])  # Gravity in z direction
    gyro_data = np.array([0.1, 0.2, 0.3])  # Small rotation
    
    # Create filter instance
    if filter_type == 'madgwick':
        filter_obj = MadgwickFilter()
    elif filter_type == 'comp':
        filter_obj = ComplementaryFilter()
    elif filter_type == 'kalman':
        filter_obj = KalmanFilter()
    elif filter_type == 'ekf':
        filter_obj = ExtendedKalmanFilter()
    elif filter_type == 'ukf':
        filter_obj = UnscentedKalmanFilter()
    else:
        logger.error(f"Unknown filter type: {filter_type}")
        return False, None, None
    
    # Test filter with basic data
    try:
        # Initial test
        q = filter_obj.update(acc_data, gyro_data)
        if q is None or len(q) != 4:
            logger.error(f"{filter_type} filter failed initial test")
            return False, None, None
            
        logger.info(f"{filter_type} filter passed initial test: {q}")
        
        # Benchmark performance
        start_time = time.time()
        for i in range(iterations):
            # Slightly modify data in each iteration to test dynamic response
            angle = i * 0.01
            acc = np.array([np.sin(angle) * 0.1, np.cos(angle) * 0.1, 9.81])
            gyro = np.array([0.1 + np.sin(angle) * 0.05, 
                             0.2 + np.cos(angle) * 0.05, 
                             0.3])
            q = filter_obj.update(acc, gyro)
            
        elapsed_time = time.time() - start_time
        rate = iterations / elapsed_time
        
        logger.info(f"{filter_type} filter performance: {elapsed_time:.4f}s for {iterations} iterations ({rate:.1f} updates/sec)")
        
        return True, q, elapsed_time
    except Exception as e:
        logger.error(f"{filter_type} filter test failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None, None

def main():
    """Test all filter implementations"""
    filter_types = ['madgwick', 'comp', 'kalman', 'ekf', 'ukf']
    results = {}
    
    logger.info("Starting filter tests")
    
    for filter_type in filter_types:
        logger.info(f"Testing {filter_type} filter...")
        success, final_q, elapsed_time = test_filter(filter_type)
        
        results[filter_type] = {
            'success': success,
            'quaternion': final_q.tolist() if final_q is not None else None,
            'time': elapsed_time
        }
        
        if not success:
            logger.warning(f"{filter_type} filter test failed")
    
    # Output results as JSON
    import json
    with open("filter_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Check if all tests passed
    all_passed = all(results[f]['success'] for f in filter_types)
    
    logger.info(f"Filter tests completed. All passed: {all_passed}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
EOL
    
    # Run the filter test script
    log "INFO" "Running filter tests..."
    python test_filters.py 2>&1 | tee "${FILTER_TEST_LOG}"
    
    # Check if tests succeeded
    if [ $? -ne 0 ]; then
        log "ERROR" "Filter tests failed. See ${FILTER_TEST_LOG} for details."
        return 1
    fi
    
    # Create visualizations of filter differences
    log "INFO" "Creating filter comparison visualizations..."
    
    # Python script to visualize filter differences
    cat > visualize_filters.py << 'EOL'
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from utils.imu_fusion import (
    MadgwickFilter, 
    ComplementaryFilter, 
    KalmanFilter, 
    ExtendedKalmanFilter, 
    UnscentedKalmanFilter
)

# Load test results
with open("filter_test_results.json", "r") as f:
    results = json.load(f)

# Create path for visualizations
viz_dir = os.path.join(os.environ.get("RESULTS_DIR", "filter_comparison_results"), 
                       os.environ.get("VISUALIZATION_DIR", "visualizations"))
os.makedirs(viz_dir, exist_ok=True)

# Create performance comparison chart
filter_types = list(results.keys())
elapsed_times = [results[f]['time'] for f in filter_types if results[f]['success']]
success_filters = [f for f in filter_types if results[f]['success']]

plt.figure(figsize=(10, 6))
plt.bar(success_filters, elapsed_times, color='skyblue')
plt.title('Filter Processing Time Comparison (1000 iterations)')
plt.xlabel('Filter Type')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, "filter_performance_comparison.png"))
plt.close()

# Generate test data for a simulated fall
def generate_simulated_fall(samples=500):
    # Time vector
    t = np.linspace(0, 5, samples)
    
    # Accelerometer data
    acc = np.zeros((samples, 3))
    acc[:, 2] = 9.81  # Gravity in z-direction
    
    # Add a fall at t=2.5s
    fall_start = int(samples * 0.5)
    fall_duration = int(samples * 0.1)
    
    # During fall, accelerometer shows sharp changes
    for i in range(fall_duration):
        progress = i / fall_duration
        acc[fall_start + i, 0] = 5 * np.sin(progress * np.pi)
        acc[fall_start + i, 1] = 3 * np.cos(progress * np.pi)
        acc[fall_start + i, 2] = 9.81 * (1 - progress) + 2 * progress
    
    # After fall, values stabilize to zero
    acc[fall_start + fall_duration:, 0] = 0
    acc[fall_start + fall_duration:, 1] = 0
    acc[fall_start + fall_duration:, 2] = 2  # Low gravity after fall
    
    # Gyroscope data
    gyro = np.zeros((samples, 3))
    
    # Add rotation during fall
    for i in range(fall_duration):
        progress = i / fall_duration
        gyro[fall_start + i, 0] = 2 * np.sin(progress * np.pi * 2)
        gyro[fall_start + i, 1] = 3 * np.cos(progress * np.pi * 2)
        gyro[fall_start + i, 2] = 1 * np.sin(progress * np.pi * 4)
    
    return t, acc, gyro

# Create filters
filters = {
    'madgwick': MadgwickFilter(),
    'comp': ComplementaryFilter(),
    'kalman': KalmanFilter(),
    'ekf': ExtendedKalmanFilter(),
    'ukf': UnscentedKalmanFilter()
}

# Generate simulated fall data
t, acc, gyro = generate_simulated_fall(samples=500)

# Process with each filter
quaternions = {}
linear_accs = {}

for name, filter_obj in filters.items():
    if not results.get(name, {}).get('success', False):
        continue
        
    quaternions[name] = []
    linear_accs[name] = []
    
    # Reset filter
    filter_obj.reset()
    
    for i in range(len(t)):
        # Update filter
        q = filter_obj.update(acc[i], gyro[i])
        quaternions[name].append(q)
        
        # Calculate linear acceleration
        from scipy.spatial.transform import Rotation
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # Convert to scipy format
        gravity = r.apply([0, 0, 9.81])  # Rotate standard gravity vector
        lin_acc = acc[i] - gravity  # Remove gravity component
        linear_accs[name].append(lin_acc)

# Convert to arrays
for name in quaternions:
    quaternions[name] = np.array(quaternions[name])
    linear_accs[name] = np.array(linear_accs[name])

# Plot quaternion components for each filter
component_names = ['w', 'x', 'y', 'z']
for i, component in enumerate(component_names):
    plt.figure(figsize=(12, 6))
    for name in quaternions:
        plt.plot(t, quaternions[name][:, i], label=f'{name}')
    plt.title(f'Quaternion {component} Component')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"quaternion_{component}_comparison.png"))
    plt.close()

# Plot linear acceleration magnitude
plt.figure(figsize=(12, 6))
for name in linear_accs:
    mag = np.linalg.norm(linear_accs[name], axis=1)
    plt.plot(t, mag, label=f'{name}')
plt.title('Linear Acceleration Magnitude')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True, alpha=0.3)
plt.axhline(y=3.0, color='r', linestyle='--', label='Fall threshold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, f"linear_acceleration_comparison.png"))
plt.close()

# Convert quaternions to Euler angles
euler_angles = {}
for name in quaternions:
    from scipy.spatial.transform import Rotation
    euler_angles[name] = np.zeros((len(quaternions[name]), 3))
    for i, q in enumerate(quaternions[name]):
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # Convert to scipy format
        euler_angles[name][i] = r.as_euler('xyz', degrees=True)

# Plot Euler angles
angle_names = ['Roll', 'Pitch', 'Yaw']
for i, angle in enumerate(angle_names):
    plt.figure(figsize=(12, 6))
    for name in euler_angles:
        plt.plot(t, euler_angles[name][:, i], label=f'{name}')
    plt.title(f'{angle} Angle')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"euler_{angle.lower()}_comparison.png"))
    plt.close()

print(f"Visualizations saved to {viz_dir}")
EOL
    
    # Run the visualization script
    export RESULTS_DIR="${RESULTS_DIR}"
    export VISUALIZATION_DIR="${VISUALIZATION_DIR}"
    python visualize_filters.py 2>&1 | tee -a "${FILTER_TEST_LOG}"
    
    # Check if visualization succeeded
    if [ $? -ne 0 ]; then
        log "ERROR" "Filter visualization failed. See ${FILTER_TEST_LOG} for details."
        return 1
    fi
    
    log "INFO" "Filter tests and visualizations completed successfully"
    return 0
}

# Function to analyze preprocessing efficiency
analyze_preprocessing() {
    log "INFO" "Analyzing preprocessing efficiency"
    
    # Create Python script to analyze preprocessing
    cat > analyze_preprocessing.py << 'EOL'
import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils.imu_fusion import (
    process_imu_data,
    MadgwickFilter,
    ComplementaryFilter,
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("preprocessing_analysis")

# Load sample data (you'll need to provide path to your test data)
def load_sample_data(subject_id=29, action_id=10, trial_id=1):
    """Load sample data for analysis"""
    try:
        # Define paths based on your data structure
        root_dir = os.path.join(os.getcwd(), 'data/smartfallmm/young')
        acc_path = os.path.join(root_dir, 'accelerometer/watch', f'S{subject_id:02d}A{action_id:02d}T{trial_id:02d}.csv')
        gyro_path = os.path.join(root_dir, 'gyroscope/watch', f'S{subject_id:02d}A{action_id:02d}T{trial_id:02d}.csv')
        
        logger.info(f"Loading data from: {acc_path} and {gyro_path}")
        
        # Check if files exist
        if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
            logger.error(f"Data files not found. Please check paths: {acc_path}, {gyro_path}")
            return None, None
        
        # Load data
        import pandas as pd
        acc_data = pd.read_csv(acc_path, header=None).values[:, 1:4]  # Assuming timestamp in col 0
        gyro_data = pd.read_csv(gyro_path, header=None).values[:, 1:4]  # Assuming timestamp in col 0
        
        logger.info(f"Loaded data shapes: acc={acc_data.shape}, gyro={gyro_data.shape}")
        
        # Synchronize lengths if needed
        min_len = min(len(acc_data), len(gyro_data))
        acc_data = acc_data[:min_len]
        gyro_data = gyro_data[:min_len]
        
        return acc_data, gyro_data
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return None, None

def benchmark_filters():
    """Benchmark different filters on the same data"""
    # Load sample data
    acc_data, gyro_data = load_sample_data()
    if acc_data is None or gyro_data is None:
        logger.error("Failed to load sample data")
        return False
    
    # Create filter instances
    filters = {
        'madgwick': MadgwickFilter(),
        'comp': ComplementaryFilter(),
        'kalman': KalmanFilter(),
        'ekf': ExtendedKalmanFilter(),
        'ukf': UnscentedKalmanFilter()
    }
    
    # Benchmark results
    results = {}
    
    for name, filter_obj in filters.items():
        try:
            logger.info(f"Testing {name} filter on real data")
            
            # Reset filter
            filter_obj.reset()
            
            # Process with timing
            start_time = time.time()
            quaternions = []
            
            for i in range(len(acc_data)):
                q = filter_obj.update(acc_data[i], gyro_data[i])
                quaternions.append(q)
            
            elapsed_time = time.time() - start_time
            rate = len(acc_data) / elapsed_time
            
            # Store results
            results[name] = {
                'time': elapsed_time,
                'rate': rate,
                'samples': len(acc_data)
            }
            
            logger.info(f"{name} filter: {elapsed_time:.4f}s for {len(acc_data)} samples ({rate:.1f} updates/sec)")
            
        except Exception as e:
            logger.error(f"Error testing {name} filter: {e}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [results[k]['rate'] for k in results], color='skyblue')
    plt.title('Filter Processing Rate Comparison (Real Data)')
    plt.xlabel('Filter Type')
    plt.ylabel('Processing Rate (samples/sec)')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (name, data) in enumerate(results.items()):
        plt.text(i, data['rate']+50, f"{data['rate']:.1f}", ha='center')
    
    # Save visualization
    viz_dir = os.path.join(os.environ.get("RESULTS_DIR", "filter_comparison_results"), 
                         os.environ.get("VISUALIZATION_DIR", "visualizations"))
    os.makedirs(viz_dir, exist_ok=True)
    plt.savefig(os.path.join(viz_dir, "filter_processing_rate.png"))
    plt.close()
    
    # Save results to file
    import json
    with open(os.path.join(viz_dir, "filter_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return True

def main():
    """Main analysis function"""
    success = benchmark_filters()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
EOL
    
    # Run the analysis script
    export RESULTS_DIR="${RESULTS_DIR}"
    export VISUALIZATION_DIR="${VISUALIZATION_DIR}"
    
    python analyze_preprocessing.py 2>&1 | tee "${RESULTS_DIR}/${LOG_DIR}/preprocessing_analysis.log"
    
    # Check if analysis succeeded
    if [ $? -ne 0 ]; then
        log "WARNING" "Preprocessing analysis failed. Continuing anyway."
    else
        log "INFO" "Preprocessing analysis completed successfully"
    fi
}

# ================================================================
# CONFIGURATION FILE GENERATION
# ================================================================
# Function to create configuration file for a specific filter
create_config() {
    local config_file=$1
    local filter_type=$2
    
    log "INFO" "Creating config for $filter_type filter: $config_file"
    
    # Set filter-specific parameters
    local embed_dim=48
    local num_heads=8
    local feature_dim=$(($embed_dim * 3))  # For concat fusion
    local num_layers=3
    local dropout=0.3
    
    # Adjust parameters for each filter type
    case $filter_type in
        "madgwick")
            # Baseline parameters
            ;;
        "comp")
            # Complementary filter typically needs less complex model
            num_heads=4
            ;;
        "kalman")
            # Standard Kalman filter
            ;;
        "ekf")
            # EKF may need more capacity
            embed_dim=64
            feature_dim=$(($embed_dim * 3))
            ;;
        "ukf")
            # UKF is most complex, increase capacity
            embed_dim=64
            num_heads=8
            feature_dim=$(($embed_dim * 3))
            num_layers=4
            ;;
    esac
    
    # Create the configuration file
    cat > $config_file << EOL
model: Models.fusion_transformer.FusionTransModel
dataset: smartfallmm

subjects: [29, 30, 31, 33, 45, 46, 34, 37, 39, 38, 43, 35, 36, 44, 32]

model_args:
  num_layers: ${num_layers}
  embed_dim: ${embed_dim}
  acc_coords: 3
  quat_coords: 4
  num_classes: 2
  acc_frames: 64
  mocap_frames: 64
  num_heads: ${num_heads}
  fusion_type: 'concat'
  dropout: ${dropout}
  use_batch_norm: true
  feature_dim: ${feature_dim}  # For concat fusion

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
    acc_threshold: 3.0
    gyro_threshold: 1.0
    visualize: false  # Disable during training for speed
    save_aligned: true

batch_size: 16
test_batch_size: 16
val_batch_size: 16
num_epoch: ${NUM_EPOCHS}

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
base_lr: ${BASE_LR}
weight_decay: ${WEIGHT_DECAY}

kfold:
  enabled: true
  num_folds: 5
  fold_assignments:
    - [43, 35, 36]  # Fold 1: ~38.3% falls
    - [44, 34, 32]  # Fold 2: ~39.7% falls
    - [45, 37, 38]  # Fold 3: ~44.8% falls
    - [46, 29, 31]  # Fold 4: ~41.4% falls
    - [30, 39]      # Fold 5: ~43.3% falls
EOL
}

# ================================================================
# MODEL TRAINING WITH DIFFERENT FILTERS
# ================================================================
# Function to train a model with a specific filter
train_model() {
    local config_file=$1
    local model_name=$2
    local filter_type=$3
    local output_dir="${RESULTS_DIR}/${model_name}"
    
    log "INFO" "Training model with ${filter_type} filter: ${model_name}"
    mkdir -p "${output_dir}"
    
    # Create model-specific log file
    local train_log="${RESULTS_DIR}/${LOG_DIR}/${model_name}_train.log"
    
    # Prepare Python error capture
    local python_err="${RESULTS_DIR}/${LOG_DIR}/${model_name}_error.log"
    
    # Configure environment variables for logging
    export PYTHONUNBUFFERED=1
    export FILTER_TYPE="${filter_type}"
    
    # Run training with cross-validation
    log "INFO" "Starting training process for ${filter_type} filter"
    CUDA_VISIBLE_DEVICES=${DEVICE} python -u main.py \
        --config ${config_file} \
        --work-dir ${output_dir} \
        --model-saved-name ${model_name} \
        --device 0 1 \
        --multi-gpu True \
        --kfold ${KFOLD} \
        --parallel-threads ${PARALLEL_THREADS} \
        --num-epoch ${NUM_EPOCHS} \
        --patience ${PATIENCE} 2>${python_err} | tee "${train_log}"
    
    # Check training status
    local status=$?
    if [ $status -ne 0 ]; then
        log "ERROR" "Training failed with exit code ${status}"
        
        # Capture Python traceback if available
        if [ -s "${python_err}" ]; then
            log "ERROR" "Python error details:"
            cat "${python_err}" >> "${MAIN_LOG}"
        fi
        
        return 1
    fi
    
    # Extract and log key metrics
    if [ -f "${output_dir}/cv_summary.json" ]; then
        log "INFO" "Cross-validation results for ${model_name} (${filter_type}):"
        
        # Extract metrics using Python and save to both log and comparison file
        python -c "
import json
import sys
try:
    with open('${output_dir}/cv_summary.json') as f:
        data = json.load(f)
        metrics = data['average_metrics']
        
        # Log results
        print(f\"Accuracy: {metrics.get('accuracy', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}\")
        print(f\"F1 score: {metrics.get('f1', 0):.4f} ± {metrics.get('f1_std', 0):.4f}\")
        print(f\"Precision: {metrics.get('precision', 0):.4f} ± {metrics.get('precision_std', 0):.4f}\")
        print(f\"Recall: {metrics.get('recall', 0):.4f} ± {metrics.get('recall_std', 0):.4f}\")
        print(f\"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f} ± {metrics.get('balanced_accuracy_std', 0):.4f}\")
        
        # Save to comparison CSV
        with open('${RESULTS_DIR}/comparison.csv', 'a') as csv:
            csv.write(f\"${model_name},${filter_type},{metrics.get('accuracy', 0):.6f},{metrics.get('f1', 0):.6f},{metrics.get('precision', 0):.6f},{metrics.get('recall', 0):.6f},{metrics.get('balanced_accuracy', 0):.6f}\\n\")
except Exception as e:
    print(f\"Error extracting metrics: {str(e)}\", file=sys.stderr)
    sys.exit(1)
" 2>>"${python_err}" | tee -a "${train_log}" "${MAIN_LOG}"
    else
        log "ERROR" "No cross-validation summary found for ${model_name}"
    fi
    
    log "INFO" "Training completed for ${model_name} (${filter_type})"
    return 0
}

# ================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# ================================================================
# Function to generate comparison visualizations and report
generate_report() {
    log "INFO" "Generating comparative analysis and report"
    
    # Check if we have results to analyze
    if [ ! -f "${RESULTS_DIR}/comparison.csv" ]; then
        log "ERROR" "No comparison data found at ${RESULTS_DIR}/comparison.csv"
        return 1
    fi
    
    # Create Python script for generating visualizations and report
    cat > generate_report.py << 'EOL'
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Configure paths
results_dir = os.environ.get("RESULTS_DIR", "filter_comparison_results")
viz_dir = os.path.join(results_dir, os.environ.get("VISUALIZATION_DIR", "visualizations"))
filter_types = os.environ.get("FILTER_TYPES", "madgwick,comp,kalman,ekf,ukf").split(",")

# Load comparison data
try:
    df = pd.read_csv(os.path.join(results_dir, "comparison.csv"))
    print(f"Loaded comparison data with {len(df)} entries")
except Exception as e:
    print(f"Error loading comparison data: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Create directory for visualizations
os.makedirs(viz_dir, exist_ok=True)

# Create accuracy comparison chart
plt.figure(figsize=(12, 7))
x = np.arange(len(df['filter_type']))
width = 0.15
metrics = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, metric in enumerate(metrics):
    plt.bar(x + width*(i-2), df[metric], width, label=metric.capitalize(), color=colors[i])

plt.xlabel('Filter Type')
plt.ylabel('Score')
plt.title('Performance Metrics by Filter Type')
plt.xticks(x, df['filter_type'])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'))
plt.close()

# Create individual metric comparisons
for metric in metrics:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['filter_type'], df[metric], color='skyblue')
    plt.xlabel('Filter Type')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} by Filter Type')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{metric}_comparison.png'))
    plt.close()

# Find best filter
best_filter = df.loc[df['f1'].idxmax()]
best_f1_filter = best_filter['filter_type']
best_f1_score = best_filter['f1']

# Generate HTML report
report_file = os.path.join(viz_dir, "report.html")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>IMU Filter Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .highlight {{
            background-color: #e8f4f8;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .summary-box {{
            background-color: #f0f7fb;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>IMU Filter Comparison for Fall Detection</h1>
        <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>This report compares different IMU sensor fusion filters for fall detection:</p>
            <ul>
                <li><strong>Madgwick Filter:</strong> Gradient descent-based orientation filter</li>
                <li><strong>Complementary Filter:</strong> Simple frequency-domain fusion</li>
                <li><strong>Kalman Filter:</strong> Basic linear Kalman filter</li>
                <li><strong>Extended Kalman Filter (EKF):</strong> Nonlinear state estimation</li>
                <li><strong>Unscented Kalman Filter (UKF):</strong> Sigma point-based nonlinear estimation</li>
            </ul>
            <p><strong>Best performing filter:</strong> {best_f1_filter.upper()} with F1 score of {best_f1_score:.4f}</p>
        </div>
        
        <h2>Performance Comparison</h2>
        <img src="performance_comparison.png" alt="Performance Comparison">
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Filter Type</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Balanced Accuracy</th>
            </tr>
"""

# Add table rows
for i, row in df.iterrows():
    highlight = 'class="highlight"' if row['filter_type'] == best_f1_filter else ''
    html_content += f"""
            <tr {highlight}>
                <td>{row['filter_type']}</td>
                <td>{row['accuracy']:.4f}</td>
                <td>{row['f1']:.4f}</td>
                <td>{row['precision']:.4f}</td>
                <td>{row['recall']:.4f}</td>
                <td>{row['balanced_accuracy']:.4f}</td>
            </tr>"""

html_content += """
        </table>
        
        <h2>Individual Metric Comparisons</h2>
"""

# Add individual metric visualizations
for metric in metrics:
    html_content += f"""
        <h3>{metric.capitalize()}</h3>
        <img src="{metric}_comparison.png" alt="{metric.capitalize()} Comparison">
"""

html_content += """
        <h2>Filter Characteristics</h2>
        
        <h3>Madgwick Filter</h3>
        <ul>
            <li>Fast and efficient orientation tracking</li>
            <li>Uses gradient descent optimization</li>
            <li>Good for real-time applications on wearable devices</li>
            <li>Moderate computational requirements</li>
        </ul>
        
        <h3>Complementary Filter</h3>
        <ul>
            <li>Simple and computationally efficient</li>
            <li>Combines accelerometer and gyroscope data in frequency domain</li>
            <li>Lowest computational requirements</li>
            <li>May struggle with complex motions</li>
        </ul>
        
        <h3>Kalman Filter</h3>
        <ul>
            <li>Optimal for linear systems with Gaussian noise</li>
            <li>Provides good tracking with proper tuning</li>
            <li>Moderate computational requirements</li>
            <li>Limited performance for highly nonlinear motion</li>
        </ul>
        
        <h3>Extended Kalman Filter (EKF)</h3>
        <ul>
            <li>Handles non-linear systems through linearization</li>
            <li>More accurate than basic Kalman for complex motions</li>
            <li>Higher computational cost than basic Kalman</li>
            <li>May struggle with highly nonlinear dynamics</li>
        </ul>
        <h3>Unscented Kalman Filter (UKF)</h3>
        <ul>
            <li>Best handling of non-linearities without derivatives</li>
            <li>Most accurate for highly dynamic motions like falls</li>
            <li>Uses sigma points to capture state distribution</li>
            <li>Highest computational cost but often worth it for fall detection</li>
        </ul>
        
        <h2>Recommendations</h2>
        <div class="summary-box">
            <p>Based on our comprehensive analysis, we recommend using the <strong>{best_f1_filter.upper()} filter</strong> for fall detection applications on wearable devices.</p>
            
            <p>Key considerations:</p>
            <ul>
                <li><strong>Accuracy vs. Efficiency:</strong> For resource-constrained devices like smartwatches, consider the computational requirements alongside accuracy.</li>
                <li><strong>Fall Characteristics:</strong> Falls involve sudden, nonlinear motion patterns that benefit from advanced filtering techniques.</li>
                <li><strong>Sensor Noise:</strong> Consumer-grade IMUs in wearable devices have significant noise that must be properly filtered.</li>
            </ul>
        </div>
        
        <h2>Next Steps</h2>
        <ul>
            <li>Implement the recommended filter in the production pipeline</li>
            <li>Consider hardware-specific optimizations for the selected filter</li>
            <li>Further tune filter parameters for specific target devices</li>
            <li>Evaluate power consumption impact on wearable devices</li>
        </ul>
    </div>
</body>
</html>
"""

# Write HTML report
with open(report_file, "w") as f:
    f.write(html_content)

print(f"Report generated at: {report_file}")
print(f"Best performing filter: {best_f1_filter} with F1 score of {best_f1_score:.4f}")

# Create JSON summary for programmatic use
summary = {
    "best_filter": best_f1_filter,
    "best_f1_score": float(best_f1_score),
    "results": df.to_dict(orient='records'),
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(results_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Analysis completed successfully")
EOL
    
    # Run the report generation script
    export RESULTS_DIR="${RESULTS_DIR}"
    export VISUALIZATION_DIR="${VISUALIZATION_DIR}"
    export FILTER_TYPES=$(IFS=, ; echo "${FILTER_TYPES[*]}")
    
    python generate_report.py 2>&1 | tee "${RESULTS_DIR}/${LOG_DIR}/report_generation.log"
    
    # Check status
    if [ $? -ne 0 ]; then
        log "ERROR" "Report generation failed"
        return 1
    fi
    
    log "INFO" "Report generated successfully"
    log "INFO" "Full results available at: ${RESULTS_DIR}/${VISUALIZATION_DIR}/report.html"
    
    return 0
}

# ================================================================
# MAIN FUNCTION
# ================================================================
main() {
    log "INFO" "Starting IMU filter comparison for fall detection"
    log "INFO" "Results will be saved to: ${RESULTS_DIR}"
    
    # Display configuration
    log "INFO" "Configuration:"
    log "INFO" "  - GPU devices: ${DEVICE}"
    log "INFO" "  - Number of epochs: ${NUM_EPOCHS}"
    log "INFO" "  - Early stopping patience: ${PATIENCE}"
    log "INFO" "  - Parallel threads: ${PARALLEL_THREADS}"
    log "INFO" "  - Cross-validation: ${KFOLD} (${NUM_FOLDS} folds)"
    
    # Initialize results directory and comparison file
    echo "model,filter_type,accuracy,f1,precision,recall,balanced_accuracy" > "${RESULTS_DIR}/comparison.csv"
    
    # Test filter implementations
    log "INFO" "================ TESTING FILTER IMPLEMENTATIONS ================"
    test_filters
    
    # Analyze preprocessing efficiency
    log "INFO" "================ ANALYZING PREPROCESSING EFFICIENCY ================"
    analyze_preprocessing
    
    # Create configurations for each filter type
    log "INFO" "================ CREATING CONFIGURATIONS ================"
    for filter_type in "${FILTER_TYPES[@]}"; do
        create_config "${CONFIG_DIR}/${filter_type}.yaml" "${filter_type}"
    done
    
    # Train models with different filter types
    for filter_type in "${FILTER_TYPES[@]}"; do
        log "INFO" "================ TRAINING WITH ${filter_type^^} FILTER ================"
        train_model "${CONFIG_DIR}/${filter_type}.yaml" "${filter_type}_model" "${filter_type}"
    done
    
    # Generate comparison visualizations and report
    log "INFO" "================ GENERATING COMPARATIVE ANALYSIS ================"
    generate_report
    
    # Final message
    log "INFO" "Filter comparison completed successfully"
    log "INFO" "Results available in ${RESULTS_DIR}"
    log "INFO" "See ${RESULTS_DIR}/${VISUALIZATION_DIR}/report.html for detailed analysis"
    
    # Print best filter from summary
    if [ -f "${RESULTS_DIR}/summary.json" ]; then
        best_filter=$(grep '"best_filter"' "${RESULTS_DIR}/summary.json" | cut -d '"' -f 4)
        best_score=$(grep '"best_f1_score"' "${RESULTS_DIR}/summary.json" | cut -d ':' -f 2 | cut -d ',' -f 1)
        log "INFO" "================ RESULTS SUMMARY ================"
        log "INFO" "Best performing filter: ${best_filter^^} with F1 score of ${best_score}"
    fi
}

# Run the main function
main
