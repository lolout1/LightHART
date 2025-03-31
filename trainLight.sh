#!/bin/bash
# train_lightweight_model.sh - Train the lightweight transformer model

set -e  # Exit on any error
set -o pipefail

# ===== CONFIGURATION =====
MODEL_NAME="lightweightTransformer"
MODEL_FILE="Models/${MODEL_NAME}.py"
CONFIG_FILE="config/${MODEL_NAME}.yaml"
WORK_DIR="work_dir/${MODEL_NAME}"
MODEL_OUTPUT="${WORK_DIR}/${MODEL_NAME}.pt"
WEIGHTS_OUTPUT="${WORK_DIR}/${MODEL_NAME}_weights_only.pt"
LOG_FILE="${WORK_DIR}/training.log"

# Training parameters
DEVICE="0"  # GPU device ID (comma-separated list for multiple GPUs)
NUM_EPOCHS=60
BATCH_SIZE=16
BASE_LR=0.0005
WEIGHT_DECAY=0.0005
PATIENCE=15
SEED=42

# ===== FUNCTIONS =====
log() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $1"
    echo "[$timestamp] $1" >> "${LOG_FILE}"
}

check_command() {
    command -v $1 >/dev/null 2>&1 || { 
        log "ERROR: $1 is required but not installed. Aborting."
        exit 1
    }
}

check_file() {
    if [ ! -f "$1" ]; then
        log "ERROR: Required file not found: $1"
        exit 1
    else
        log "Found required file: $1"
    fi
}

# ===== PRELIMINARY CHECKS =====
check_command python
check_command nvidia-smi

# Create working directory and log file
mkdir -p "${WORK_DIR}"
touch "${LOG_FILE}"

log "Starting lightweight fall detection model training"

# ===== VERIFY REQUIRED FILES EXIST =====
check_file "${MODEL_FILE}"
check_file "${CONFIG_FILE}"

# ===== START TRAINING =====
log "Starting training with configuration: ${CONFIG_FILE}"
log "Using device: ${DEVICE}"
log "Batch size: ${BATCH_SIZE}, Learning rate: ${BASE_LR}, Epochs: ${NUM_EPOCHS}"

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=${DEVICE}

# Run training
python main.py \
    --config "${CONFIG_FILE}" \
    --work-dir "${WORK_DIR}" \
    --model-saved-name "${MODEL_NAME}" \
    --device ${DEVICE} \
    --num-epoch ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --base-lr ${BASE_LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --patience ${PATIENCE} \
    --seed ${SEED} 2>&1 | tee -a "${LOG_FILE}"

TRAINING_STATUS=$?

if [ ${TRAINING_STATUS} -eq 0 ]; then
    log "Training completed successfully"
    
    # Create conversion script
    log "Creating TFLite conversion script"
    CONVERT_SCRIPT="${WORK_DIR}/convert_to_tflite.py"
    
    cat > "${CONVERT_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
import torch
import numpy as np
import os
import argparse
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

def convert_to_tflite(model_path, output_path, input_shape=(64, 3)):
    """
    Convert PyTorch model to TFLite
    
    Args:
        model_path: Path to PyTorch model file
        output_path: Path to save TFLite model
        input_shape: Input shape (sequence_length, features)
    """
    print(f"Loading PyTorch model from {model_path}")
    
    # Load PyTorch model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    
    # Create example inputs
    batch_size = 1
    acc_data = torch.randn(batch_size, *input_shape)
    gyro_data = torch.randn(batch_size, *input_shape)
    
    # ONNX export
    onnx_path = f"{os.path.splitext(output_path)[0]}.onnx"
    print(f"Exporting to ONNX: {onnx_path}")
    
    # Export with input names that match your TFLite implementation
    torch.onnx.export(
        model,
        (acc_data, gyro_data),
        onnx_path,
        input_names=["accelerometer_input", "gyroscope_input"],
        output_names=["output"],
        dynamic_axes={
            'accelerometer_input': {0: 'batch_size'},
            'gyroscope_input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=12,
        verbose=False
    )
    
    # Load ONNX model
    print("Converting ONNX model to TensorFlow")
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    
    # Save as TensorFlow SavedModel
    tf_model_dir = f"{os.path.splitext(output_path)[0]}_tf_model"
    os.makedirs(tf_model_dir, exist_ok=True)
    print(f"Saving TensorFlow model to {tf_model_dir}")
    tf_rep.export_graph(tf_model_dir)
    
    # Convert to TFLite
    print("Converting TensorFlow model to TFLite")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
    
    # Set optimization options
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save TFLite model
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    
    # Clean up temporary files
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    if os.path.exists(tf_model_dir):
        import shutil
        shutil.rmtree(tf_model_dir)
    
    print("Conversion complete!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TFLite")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model file")
    parser.add_argument("--output", type=str, required=True, help="Path to save TFLite model")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length")
    parser.add_argument("--features", type=int, default=3, help="Number of features per modality")
    
    args = parser.parse_args()
    
    convert_to_tflite(
        model_path=args.model, 
        output_path=args.output,
        input_shape=(args.seq_length, args.features)
    )
EOF
    
    chmod +x "${CONVERT_SCRIPT}"
    
    # Create Android assets directory
    ANDROID_ASSETS_DIR="MyFallDetectionAppLiteRTPro/MyFallDetectionAppLiteRTPro/app/src/main/assets"
    mkdir -p "${ANDROID_ASSETS_DIR}"
    
    # Convert model to TFLite if available
    if [ -f "${MODEL_OUTPUT}" ]; then
        log "Converting model to TFLite format"
        TFLITE_OUTPUT="${ANDROID_ASSETS_DIR}/${MODEL_NAME}.tflite"
        
        # Install required packages if needed
        pip install onnx onnx-tf tensorflow>=2.4.0 --user
        
        python "${CONVERT_SCRIPT}" \
               --model "${MODEL_OUTPUT}" \
               --output "${TFLITE_OUTPUT}" \
               --seq_length 64 \
               --features 3
        
        CONVERT_STATUS=$?
        if [ ${CONVERT_STATUS} -eq 0 ]; then
            log "Conversion successful. TFLite model saved to: ${TFLITE_OUTPUT}"
            
            # Update PrefsHelper.kt to use the new model
            PREFS_HELPER_FILE="MyFallDetectionAppLiteRTPro/MyFallDetectionAppLiteRTPro/app/src/main/java/com/example/myfalldetectionapplitertpro/PrefsHelper.kt"
            if [ -f "${PREFS_HELPER_FILE}" ]; then
                log "Updating default model file in PrefsHelper.kt"
                sed -i "s/const val DEFAULT_MODEL_FILE = \".*\"/const val DEFAULT_MODEL_FILE = \"${MODEL_NAME}.tflite\"/" "${PREFS_HELPER_FILE}"
            fi
        else
            log "Error: TFLite conversion failed with status ${CONVERT_STATUS}"
        fi
    else
        log "Warning: Model file not found at ${MODEL_OUTPUT}. Skipping TFLite conversion."
    fi
    
    # Generate evaluation report
    log "Generating evaluation summary"
    python -c "
import json, os
report_file = '${WORK_DIR}/test_summary.json'
if os.path.exists(report_file):
    with open(report_file, 'r') as f:
        data = json.load(f)
        metrics = data.get('average_metrics', {})
        print('\n===== MODEL EVALUATION SUMMARY =====')
        print(f'Accuracy: {metrics.get(\"accuracy\", 0):.2f}% ± {metrics.get(\"accuracy_std\", 0):.2f}%')
        print(f'F1 Score: {metrics.get(\"f1\", 0):.2f} ± {metrics.get(\"f1_std\", 0):.2f}')
        print(f'Precision: {metrics.get(\"precision\", 0):.2f}% ± {metrics.get(\"precision_std\", 0):.2f}%')
        print(f'Recall: {metrics.get(\"recall\", 0):.2f}% ± {metrics.get(\"recall_std\", 0):.2f}%')
        print(f'Balanced Accuracy: {metrics.get(\"balanced_accuracy\", 0):.2f}% ± {metrics.get(\"balanced_accuracy_std\", 0):.2f}%')
        print('====================================\n')
"
else
    log "Error: Training failed with status ${TRAINING_STATUS}"
    exit ${TRAINING_STATUS}
fi

log "Script completed"
