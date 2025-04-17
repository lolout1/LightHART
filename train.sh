#!/bin/bash
set -e

# ================================================================
# Fall Detection Model Training Script for AI Edge Torch
# ================================================================

# Configuration parameters - edit these as needed
WORK_DIR="results_$(date +"%Y%m%d_%H%M%S")"
GPU_ID=0
SEED=42
BATCH_SIZE=16
TEST_BATCH_SIZE=16
NUM_WORKERS=4
MAX_EPOCHS=100
PATIENCE=20
BASE_LR=0.0001
WEIGHT_DECAY=0.01
GRAD_CLIP=1.0
EMBED_DIM=32
NUM_HEADS=4
NUM_LAYERS=2
DATA_DIR="data/smartfallmm"
PYTHON_ENV_PATH="$HOME/venv/bin/activate"

# Subjects for cross-validation
VAL_SUBJECTS="38,46"
ALWAYS_TRAIN_SUBJECTS="45,36,29"
TEST_SUBJECTS="32,39,30,31,33,34,35,37,43,44"
ALL_SUBJECTS="${TEST_SUBJECTS},${ALWAYS_TRAIN_SUBJECTS},${VAL_SUBJECTS}"

# Create output directories
mkdir -p $WORK_DIR/models $WORK_DIR/logs $WORK_DIR/tflite

# Setup logging
LOG_FILE="$WORK_DIR/logs/training_$(date +"%Y%m%d_%H%M%S").log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================================"
echo "Fall Detection Model Training with AI Edge Torch"
echo "Started at: $(date)"
echo "Working directory: $WORK_DIR"
echo "================================================================"

# Activate Python environment if it exists
if [ -f "$PYTHON_ENV_PATH" ]; then
    echo "Activating Python environment from $PYTHON_ENV_PATH"
    source "$PYTHON_ENV_PATH"
else
    echo "No Python environment found at $PYTHON_ENV_PATH, using system Python"
fi

# Check for required packages
echo "Checking for required packages..."
python -c "
import sys
required_packages = ['torch', 'numpy', 'ai_edge_torch', 'sklearn']
missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        missing.append(package)
        print(f'✗ {package} not found')
if missing:
    print('Missing required packages. Please install them with:')
    print(f'pip install {\" \".join(missing)}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Missing required packages. Please install them and try again."
    exit 1
fi

# Create falldetection_model.py file
echo "Creating model definition file..."
cat > $WORK_DIR/falldetection_model.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import model_config as cfg
from ai_edge_torch.generative.layers.attention import TransformerBlock

class FallDetectionModel(nn.Module):
    def __init__(self, 
                 acc_frames=128,
                 acc_coords=4,  # x, y, z, smv
                 embed_dim=32,
                 num_heads=4,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()
        
        # Configure attention
        attn_config = cfg.AttentionConfig(
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
            num_query_groups=num_heads,
            qkv_use_bias=True,
            output_proj_use_bias=True
        )
        
        # Configure feed forward network
        ff_activation = cfg.ActivationConfig(
            type=cfg.ActivationType.GELU
        )
        
        ff_config = cfg.FeedForwardConfig(
            type=cfg.FeedForwardType.SEQUENTIAL,
            activation=ff_activation,
            intermediate_size=embed_dim * 2,
            use_bias=True
        )
        
        # Configure normalization
        norm_config = cfg.NormalizationConfig(
            type=cfg.NormalizationType.LAYER_NORM,
            epsilon=1e-5
        )
        
        # Create transformer block config
        transformer_config = cfg.TransformerBlockConfig(
            attn_config=attn_config,
            ff_config=ff_config,
            pre_attention_norm_config=norm_config,
            post_attention_norm_config=norm_config,
            parallel_residual=False
        )
        
        # Create full model config
        self.model_config = cfg.ModelConfig(
            vocab_size=1,  # Not used for this model but required
            num_layers=num_layers,
            max_seq_len=acc_frames,
            embedding_dim=embed_dim,
            block_configs=transformer_config,
            final_norm_config=norm_config,
            enable_hlfb=False  # Set to True for high-performance inference
        )
        
        # Input projection - convert accelerometer data to embeddings
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, acc_frames, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        # Create transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TransformerBlock(transformer_config, self.model_config)
            )
        
        # Output projection
        self.norm = builder.build_norm(embed_dim, norm_config)
        self.output = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Prepare input for 1D convolution
        x = x.transpose(1, 2)  # [batch_size, features, seq_len]
        
        # Apply input projection
        x = self.input_proj(x)  # [batch_size, embed_dim, seq_len]
        
        # Reshape for transformer
        x = x.transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, embed_dim]
        
        # Final classification
        x = self.output(x)  # [batch_size, 1]
        
        return x
EOF

# Create training script
echo "Creating training script..."
cat > $WORK_DIR/train.py << 'EOF'
import os
import torch
import argparse
import numpy as np
import json
import logging
import time
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from falldetection_model import FallDetectionModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FallDetectionDataset(Dataset):
    def __init__(self, data):
        self.acc_data = torch.tensor(data['accelerometer'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.acc_data[idx], self.labels[idx]

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return {
        'accuracy': 100 * accuracy_score(y_true, y_pred),
        'f1': 100 * f1_score(y_true, y_pred, average='binary', zero_division=0),
        'precision': 100 * precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': 100 * recall_score(y_true, y_pred, average='binary', zero_division=0)
    }

def train_epoch(model, loader, criterion, optimizer, device, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for acc_data, labels in loader:
        acc_data = acc_data.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(acc_data)
        
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss):
            continue
            
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        running_loss += loss.item()
        
        preds = (torch.sigmoid(outputs) > 0.5).int()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for acc_data, labels in loader:
            acc_data = acc_data.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(acc_data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return val_loss, metrics

def export_to_edge_torch(model, input_shape, output_path):
    try:
        import ai_edge_torch
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # Convert to Edge Torch model
        edge_model = ai_edge_torch.convert(model, (dummy_input,))
        
        # Export to TFLite
        edge_model.export(output_path)
        logger.info(f"Model successfully exported to {output_path}")
        
        # Test the model with sample input
        test_output = edge_model(dummy_input)
        logger.info(f"Test output shape: {test_output.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return False

def process_fold(args, fold_idx, train_subjects, val_subjects, test_subjects):
    from utils.dataset import split_by_subjects, prepare_smartfallmm
    
    logger.info(f"Fold {fold_idx+1} - Train: {train_subjects}, Val: {val_subjects}, Test: {test_subjects}")
    
    # Prepare datasets
    fold_dir = os.path.join(args.work_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    
    builder = prepare_smartfallmm(args)
    
    train_data = split_by_subjects(builder, train_subjects, False)
    val_data = split_by_subjects(builder, val_subjects, False)
    test_data = split_by_subjects(builder, test_subjects, False)
    
    train_dataset = FallDetectionDataset(train_data)
    val_dataset = FallDetectionDataset(val_data)
    test_dataset = FallDetectionDataset(test_data)
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Set device
    device = torch.device(f'cuda:{args.device}' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = FallDetectionModel(
        acc_frames=128,
        acc_coords=4,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.1
    )
    model = model.to(device)
    
    # Loss function, optimizer, and scheduler
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-7, verbose=True)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    best_val_f1 = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(args.max_epoch):
        logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}/{args.max_epoch}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.2f}%, Accuracy: {train_metrics['accuracy']:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.2f}%, Accuracy: {val_metrics['accuracy']:.2f}%")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
            
            # Save model checkpoint
            model_path = os.path.join(fold_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved with val F1: {val_metrics['f1']:.2f}%")
            
            # Try export to TFLite
            if args.export_during_training:
                output_path = os.path.join(fold_dir, 'best_model.tflite')
                export_to_edge_torch(model, (1, 128, 4), output_path)
        
        if early_stopping(val_loss):
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test model
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    logger.info(f"Test Loss: {test_loss:.4f}, F1: {test_metrics['f1']:.2f}%, Accuracy: {test_metrics['accuracy']:.2f}%")
    
    # Export best model to TFLite
    model_path = os.path.join(fold_dir, 'best_model.pth')
    tflite_path = os.path.join(fold_dir, 'best_model.tflite')
    export_success = export_to_edge_torch(model, (1, 128, 4), tflite_path)
    
    # Save model metadata
    metadata = {
        'fold': fold_idx,
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'test_metrics': test_metrics,
        'model_config': {
            'acc_frames': 128,
            'acc_coords': 4,
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'dropout': 0.1
        },
        'export_success': export_success
    }
    
    with open(os.path.join(fold_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'fold': fold_idx,
        'test_metrics': test_metrics,
        'export_success': export_success
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=str, default='./results')
    parser.add_argument('--use-gpu', type=bool, default=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--base-lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--export-during-training', type=bool, default=False)
    parser.add_argument('--all-subjects', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='data/smartfallmm')
    parser.add_argument('--dataset-args', type=dict, default={
        'age_group': ['young', 'old'],
        'modalities': ['accelerometer'],
        'sensors': ['watch'],
        'mode': 'sliding_window',
        'max_length': 128,
        'task': 'fd'
    })
    parser.add_argument('--fold', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed_all(args.seed)
    
    # Parse subjects
    if isinstance(args.all_subjects, str):
        args.all_subjects = [int(s) for s in args.all_subjects.split(',')]
    
    # Create folds
    val_subjects = [38, 46]
    always_train_subjects = [45, 36, 29]
    eligible_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
    
    folds = []
    for test_subject in eligible_subjects:
        test_subjects = [test_subject]
        train_subjects = always_train_subjects + [s for s in eligible_subjects if s != test_subject]
        folds.append({
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        })
    
    # Process folds
    results = []
    
    if args.fold == -1:
        fold_indices = range(len(folds))
    else:
        fold_indices = [args.fold]
    
    for fold_idx in fold_indices:
        if fold_idx >= len(folds):
            logger.error(f"Fold index {fold_idx} is out of range (max: {len(folds)-1})")
            continue
        
        fold = folds[fold_idx]
        fold_result = process_fold(
            args, 
            fold_idx, 
            fold['train'], 
            fold['val'], 
            fold['test']
        )
        results.append(fold_result)
    
    # Save overall results
    if results:
        overall_f1 = np.mean([r['test_metrics']['f1'] for r in results])
        overall_accuracy = np.mean([r['test_metrics']['accuracy'] for r in results])
        
        logger.info(f"Overall F1: {overall_f1:.2f}%, Accuracy: {overall_accuracy:.2f}%")
        
        with open(os.path.join(args.work_dir, 'overall_results.json'), 'w') as f:
            json.dump({
                'overall_f1': float(overall_f1),
                'overall_accuracy': float(overall_accuracy),
                'folds': results
            }, f, indent=2)

if __name__ == "__main__":
    main()
EOF

# Create conversion script
echo "Creating model conversion script..."
cat > $WORK_DIR/convert.py << 'EOF'
import os
import torch
import argparse
import logging
import json
import ai_edge_torch
from falldetection_model import FallDetectionModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_model(model_path, output_path, model_config=None):
    """
    Convert a PyTorch model to TFLite using AI Edge Torch
    
    Args:
        model_path: Path to the trained PyTorch model (.pth)
        output_path: Path to save the TFLite model
        model_config: Dictionary with model configuration parameters
    """
    # Default config if not provided
    if model_config is None:
        model_config = {
            'acc_frames': 128,
            'acc_coords': 4,
            'embed_dim': 32,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1
        }
    
    # Load model
    model = FallDetectionModel(**model_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create sample input with the right shape
    sample_input = torch.randn(1, 128, 4)
    
    # Try simple inference before conversion
    logger.info("Testing model before conversion...")
    with torch.no_grad():
        try:
            output = model(sample_input)
            logger.info(f"Model output shape: {output.shape}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return False
    
    # Convert the model with specific flags
    try:
        logger.info(f"Converting model from {model_path}")
        
        # Basic conversion
        edge_model = ai_edge_torch.convert(model, (sample_input,))
        
        # Test converted model
        logger.info("Testing converted model...")
        edge_output = edge_model(sample_input)
        logger.info(f"Edge model output shape: {edge_output.shape}")
        
        # Export to TFLite
        edge_model.export(output_path)
        logger.info(f"Model successfully exported to {output_path}")
        
        # Save metadata
        metadata = {
            'input_shape': [1, 128, 4],
            'model_config': model_config,
            'input_details': {
                'shape': [1, 128, 4],
                'dtype': 'float32',
                'name': 'input'
            },
            'output_details': {
                'shape': [1, 1],
                'dtype': 'float32',
                'name': 'output'
            }
        }
        
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        logger.error("Detailed error information:")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained PyTorch model')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the TFLite model')
    parser.add_argument('--embed-dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of transformer layers')
    
    args = parser.parse_args()
    
    model_config = {
        'acc_frames': 128,
        'acc_coords': 4,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': 0.1
    }
    
    success = convert_model(args.model_path, args.output_path, model_config)
    
    if success:
        logger.info("Conversion completed successfully")
    else:
        logger.error("Conversion failed")

if __name__ == "__main__":
    main()
EOF

# Run training
echo "Starting training..."
export CUDA_VISIBLE_DEVICES=$GPU_ID
python train.py \
    --work-dir $WORK_DIR \
    --use-gpu True \
    --device 0 \
    --seed $SEED \
    --batch-size $BATCH_SIZE \
    --test-batch-size $TEST_BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --max-epoch $MAX_EPOCHS \
    --patience $PATIENCE \
    --base-lr $BASE_LR \
    --weight-decay $WEIGHT_DECAY \
    --grad-clip $GRAD_CLIP \
    --embed-dim $EMBED_DIM \
    --num-heads $NUM_HEADS \
    --num-layers $NUM_LAYERS \
    --data-dir $DATA_DIR \
    --all-subjects $ALL_SUBJECTS \
    --fold -1

# Convert best model from each fold to TFLite
echo "Converting best models to TFLite..."
for fold_dir in $WORK_DIR/fold_*/; do
    if [ -d "$fold_dir" ]; then
        fold_name=$(basename $fold_dir)
        model_path="$fold_dir/best_model.pth"
        output_path="$WORK_DIR/tflite/${fold_name}_model.tflite"
        
        if [ -f "$model_path" ]; then
            echo "Converting model from $model_path to $output_path"
            python $WORK_DIR/convert.py \
                --model-path $model_path \
                --output-path $output_path \
                --embed-dim $EMBED_DIM \
                --num-heads $NUM_HEADS \
                --num-layers $NUM_LAYERS
        else
            echo "No model found at $model_path"
        fi
    fi
done

# Create Android-friendly metadata file
echo "Creating Android metadata file..."
cat > $WORK_DIR/tflite/model_info.json << EOF
{
  "model_name": "fall_detection_transformer",
  "version": "1.0.0",
  "timestamp": "$(date +"%Y-%m-%d %H:%M:%S")",
  "input_details": {
    "shape": [1, 128, 4],
    "dtype": "float32",
    "name": "input"
  },
  "output_details": {
    "shape": [1, 1],
    "dtype": "float32",
    "name": "output"
  },
  "preprocessing": {
    "normalization": true
  },
  "labels": ["not_fall", "fall"],
  "threshold": 0.5,
  "configuration": {
    "embed_dim": $EMBED_DIM,
    "num_heads": $NUM_HEADS,
    "num_layers": $NUM_LAYERS
  }
}
EOF

# Summarize results
echo "================================================================"
echo "Training and conversion completed"
echo "Results are saved in $WORK_DIR"
echo "TFLite models are saved in $WORK_DIR/tflite"
echo "================================================================"
echo "Checking TFLite models..."
for tflite_file in $WORK_DIR/tflite/*.tflite; do
    if [ -f "$tflite_file" ]; then
        file_size=$(du -h "$tflite_file" | cut -f1)
        echo "- $(basename $tflite_file) ($file_size)"
    fi
done
echo "================================================================"
echo "Training complete at: $(date)"
echo "================================================================"
