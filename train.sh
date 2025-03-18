#!/bin/bash
#
# Comprehensive training script for SmartFallMM dataset with Kalman filter fusion
# Supports multiple filter types and configuration options
#

# Set error handling
set -e

# Default parameters
CONFIG="config/smartfallmm/fusion_madgwick.yaml"
WORK_DIR="results"
MODEL_NAME="smartfallmm_model"
FILTER_TYPE="madgwick"
NUM_EPOCHS=60
DEVICE=0
SAVE_ALIGNED=true
BATCH_SIZE=16

# Function for displaying usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -c, --config CONFIG      Configuration file (default: $CONFIG)"
    echo "  -w, --work-dir DIR       Working directory (default: $WORK_DIR)"
    echo "  -m, --model NAME         Model name (default: $MODEL_NAME)"
    echo "  -f, --filter TYPE        Filter type: madgwick, comp, kalman, ekf, ukf (default: $FILTER_TYPE)"
    echo "  -e, --epochs NUM         Number of epochs (default: $NUM_EPOCHS)"
    echo "  -d, --device ID          GPU device ID (default: $DEVICE)"
    echo "  -s, --save-aligned       Save aligned sensor data (default: $SAVE_ALIGNED)"
    echo "  -b, --batch-size SIZE    Batch size (default: $BATCH_SIZE)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --filter ukf --device 1 --epochs 100"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -w|--work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -f|--filter)
            FILTER_TYPE="$2"
            shift 2
            ;;
        -e|--epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -s|--save-aligned)
            SAVE_ALIGNED=true
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Create directory structure
mkdir -p "$WORK_DIR/logs"
mkdir -p "data/aligned/accelerometer" "data/aligned/gyroscope" "data/aligned/skeleton"
mkdir -p "debug_logs"

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$WORK_DIR/logs/${MODEL_NAME}_${FILTER_TYPE}_${TIMESTAMP}.log"

# Log header
echo "==================================================" | tee -a "$LOG_FILE"
echo "SmartFallMM Training - $(date)" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Config file:    $CONFIG" | tee -a "$LOG_FILE"
echo "  Working dir:    $WORK_DIR" | tee -a "$LOG_FILE"
echo "  Model name:     $MODEL_NAME" | tee -a "$LOG_FILE"
echo "  Filter type:    $FILTER_TYPE" | tee -a "$LOG_FILE"
echo "  Number of epochs: $NUM_EPOCHS" | tee -a "$LOG_FILE"
echo "  Device:         $DEVICE" | tee -a "$LOG_FILE"
echo "  Save aligned:   $SAVE_ALIGNED" | tee -a "$LOG_FILE"
echo "  Batch size:     $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# Verify filter type is valid
case $FILTER_TYPE in
    madgwick|comp|kalman|ekf|ukf)
        # Valid filter
        ;;
    *)
        echo "Error: Invalid filter type '$FILTER_TYPE'" | tee -a "$LOG_FILE"
        echo "Valid options: madgwick, comp, kalman, ekf, ukf" | tee -a "$LOG_FILE"
        exit 1
        ;;
esac

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG" | tee -a "$LOG_FILE"
    exit 1
fi

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$DEVICE
echo "Setting CUDA_VISIBLE_DEVICES=$DEVICE" | tee -a "$LOG_FILE"

# Dynamically update the config file with the selected filter
TEMP_CONFIG="${WORK_DIR}/${MODEL_NAME}_${FILTER_TYPE}_config.yaml"
cp "$CONFIG" "$TEMP_CONFIG"

# Update filter type in the config file
sed -i "s/filter_type: '.*'/filter_type: '$FILTER_TYPE'/g" "$TEMP_CONFIG"

# Update the batch size if different from default
if [ "$BATCH_SIZE" != "16" ]; then
    sed -i "s/batch_size: [0-9]*/batch_size: $BATCH_SIZE/g" "$TEMP_CONFIG"
    sed -i "s/test_batch_size: [0-9]*/test_batch_size: $BATCH_SIZE/g" "$TEMP_CONFIG"
    sed -i "s/val_batch_size: [0-9]*/val_batch_size: $BATCH_SIZE/g" "$TEMP_CONFIG"
fi

# Enable save_aligned if requested
if [ "$SAVE_ALIGNED" = true ]; then
    # Check if fusion_options already exists in the config
    if grep -q "fusion_options:" "$TEMP_CONFIG"; then
        # Add save_aligned under fusion_options if it doesn't exist
        if ! grep -q "save_aligned:" "$TEMP_CONFIG"; then
            sed -i "/fusion_options:/a\ \ \ \ save_aligned: true" "$TEMP_CONFIG"
        else
            # Otherwise update the existing value
            sed -i "s/save_aligned: .*/save_aligned: true/g" "$TEMP_CONFIG"
        fi
    else
        # Add complete fusion_options section if it doesn't exist
        echo "  fusion_options:" >> "$TEMP_CONFIG"
        echo "    enabled: true" >> "$TEMP_CONFIG"
        echo "    filter_type: '$FILTER_TYPE'" >> "$TEMP_CONFIG"
        echo "    save_aligned: true" >> "$TEMP_CONFIG"
    fi
fi

echo "Updated configuration saved to $TEMP_CONFIG" | tee -a "$LOG_FILE"

# Set up model directory
MODEL_DIR="${WORK_DIR}/${MODEL_NAME}_${FILTER_TYPE}"
mkdir -p "$MODEL_DIR"

# Run training
echo "Starting training..." | tee -a "$LOG_FILE"
python main.py \
  --config "$TEMP_CONFIG" \
  --work-dir "$MODEL_DIR" \
  --model-saved-name "$MODEL_NAME" \
  --device "$DEVICE" \
  --num-epoch "$NUM_EPOCHS" 2>&1 | tee -a "$LOG_FILE"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!" | tee -a "$LOG_FILE"
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE" | tee -a "$LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi

# Run evaluation
echo "Running evaluation..." | tee -a "$LOG_FILE"
python main.py \
  --config "$TEMP_CONFIG" \
  --work-dir "$MODEL_DIR" \
  --weights "${MODEL_DIR}/${MODEL_NAME}.pt" \
  --device "$DEVICE" \
  --phase test 2>&1 | tee -a "$LOG_FILE"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!" | tee -a "$LOG_FILE"
    
    # Extract and display test results
    if [ -f "${MODEL_DIR}/test_result.txt" ]; then
        echo "==================================================" | tee -a "$LOG_FILE"
        echo "Test Results:" | tee -a "$LOG_FILE"
        echo "==================================================" | tee -a "$LOG_FILE"
        cat "${MODEL_DIR}/test_result.txt" | tee -a "$LOG_FILE"
    else
        echo "Warning: Test results file not found" | tee -a "$LOG_FILE"
    fi
else
    echo "Evaluation failed with exit code $EVAL_EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EVAL_EXIT_CODE
fi

echo "==================================================" | tee -a "$LOG_FILE"
echo "Training and evaluation completed" | tee -a "$LOG_FILE"
echo "Model saved to ${MODEL_DIR}/${MODEL_NAME}.pt" | tee -a "$LOG_FILE"
echo "Log saved to $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
