#!/bin/bash
# Robust training script with proper GPU handling and error recovery

# Exit if any command fails
set -e

# Set up logging
LOGDIR="logs"
mkdir -p $LOGDIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/training_$TIMESTAMP.log"

echo "Starting training script at $(date)" | tee -a $LOGFILE

# Find best available GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver not found, using CPU" | tee -a $LOGFILE
    USE_GPU="false"
    GPU_ID=0
else
    echo "Detecting available GPUs..." | tee -a $LOGFILE
    nvidia-smi -L | tee -a $LOGFILE
    
    # Get GPU with most free memory
    GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    GPU_ID=$(echo "$GPU_INFO" | sort -k2 -nr | head -n1 | awk '{print $1}' | tr -d '[:space:],')
    
    # Validate GPU ID
    if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID, defaulting to GPU 0" | tee -a $LOGFILE
        GPU_ID=0
    fi
    
    echo "Selected GPU: $GPU_ID" | tee -a $LOGFILE
    USE_GPU="true"
fi

# Create data directories
DATA_DIR="data/smartfallmm"
mkdir -p "${DATA_DIR}/young/accelerometer/watch"
mkdir -p "${DATA_DIR}/old/accelerometer/watch"

# Environment variables for better performance
export OMP_NUM_THREADS=4
export PYTHONFAULTHANDLER=1

echo "Starting training with GPU $GPU_ID at $(date)" | tee -a $LOGFILE

# Run training with optimized parameters for stability
python train.py \
    --work-dir "results_$TIMESTAMP" \
    --use-gpu $USE_GPU \
    --device $GPU_ID \
    --batch-size 16 \
    --test-batch-size 32 \
    --max-epoch 100 \
    --patience 20 \
    --base-lr 0.0001 \
    --min-lr 0.000001 \
    --weight-decay 0.001 \
    --grad-clip 0.5 \
    --embed-dim 32 \
    --num-heads 4 \
    --num-layers 2 \
    --data-dir "$DATA_DIR" \
    --focal-loss \
    --debug 2>&1 | tee -a $LOGFILE

# Check if training completed successfully
TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully at $(date)" | tee -a $LOGFILE
    
    # Convert best model to TFLite
    BEST_MODEL=$(find "results_$TIMESTAMP" -name "best_model.pth" | head -1)
    if [ -n "$BEST_MODEL" ]; then
        echo "Converting best model to TFLite..." | tee -a $LOGFILE
        
        # Create a simplified conversion script
        cat > convert_model_simple.py << EOF
import os
import torch
import numpy as np
import ai_edge_torch
import logging
from models.EdgeFallModel import EdgeFallTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert(model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = EdgeFallTransformer(
        acc_frames=128,
        num_classes=1,
        num_heads=4,
        acc_coords=4,
        num_layer=2,
        embed_dim=32
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 128, 4)
    
    # Normalize dummy input
    mean = dummy_input.mean(dim=(1, 2), keepdim=True)
    std = dummy_input.std(dim=(1, 2), keepdim=True) + 1e-6
    dummy_input = (dummy_input - mean) / std
    
    # Convert to TFLite
    try:
        logger.info(f"Converting model from {model_path}")
        edge_model = ai_edge_torch.convert(model.eval(), (dummy_input,))
        tflite_path = os.path.join(output_dir, 'model.tflite')
        edge_model.export(tflite_path)
        logger.info(f"Model saved to {tflite_path}")
        return True
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert_model_simple.py <model_path> <output_dir>")
        sys.exit(1)
    
    success = convert(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
EOF
        
        # Run conversion
        CONVERT_DIR="results_$TIMESTAMP/tflite"
        python convert_model_simple.py "$BEST_MODEL" "$CONVERT_DIR" 2>&1 | tee -a $LOGFILE
        CONVERT_EXIT_CODE=$?
        
        if [ $CONVERT_EXIT_CODE -eq 0 ]; then
            echo "Conversion completed successfully at $(date)" | tee -a $LOGFILE
        else
            echo "Conversion failed with exit code $CONVERT_EXIT_CODE" | tee -a $LOGFILE
        fi
    else
        echo "No best model found to convert" | tee -a $LOGFILE
    fi
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE" | tee -a $LOGFILE
    exit 1
fi
