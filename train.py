#!/usr/bin/env python
# train.py
import os
import sys
import torch
import argparse
import numpy as np
import json
import logging
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.data_loader import create_subject_folds, prepare_datasets, create_data_loaders
    from Models.fall_detection import FallDetectionTransformer as FallDetectionModel
except ImportError:
    logger.error("Could not import required modules. Make sure you're running from the project root.")
    sys.exit(1)

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if not np.isfinite(val_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
            
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
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    try:
        metrics = {
            'accuracy': 100 * accuracy_score(y_true, y_pred),
            'f1': 100 * f1_score(y_true, y_pred, average='binary', zero_division=0),
            'precision': 100 * precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': 100 * recall_score(y_true, y_pred, average='binary', zero_division=0)
        }
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

def train_epoch(model, loader, criterion, optimizer, device, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    valid_batches = 0
    
    for batch_idx, (acc_data, labels, _) in enumerate(loader):
        try:
            acc_data = acc_data.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            outputs, _ = model(acc_data)  # Model returns (outputs, features)
            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels)
            
            if not torch.isfinite(loss):
                logger.warning(f"Skipping batch {batch_idx} due to non-finite loss")
                continue
                
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            running_loss += loss.item()
            valid_batches += 1
            
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    epoch_loss = running_loss / max(valid_batches, 1)
    metrics = calculate_metrics(all_preds, all_labels)
    
    return epoch_loss, all_preds, all_labels, metrics

def validate(model, loader, criterion, device, prefix="val"):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    valid_batches = 0
    
    with torch.no_grad():
        for acc_data, labels, _ in loader:
            try:
                acc_data = acc_data.to(device)
                labels = labels.to(device).float()
                
                outputs, _ = model(acc_data)  # Model returns (outputs, features)
                outputs = outputs.squeeze(1)
                
                loss = criterion(outputs, labels)
                
                if not torch.isfinite(loss):
                    continue
                    
                running_loss += loss.item()
                valid_batches += 1
                
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error in validation: {e}")
                continue
    
    epoch_loss = running_loss / max(valid_batches, 1)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return epoch_loss, all_preds, all_labels, metrics

def save_model(model, fold_work_dir, epoch, metrics, device, is_best=False):
    save_path = os.path.join(fold_work_dir, 'best_model.pth' if is_best else f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    
    try:
        dummy_input = torch.randn(1, 128, 4).to(device)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        script_path = os.path.join(fold_work_dir, 'best_model_scripted.pt' if is_best else f'model_epoch_{epoch}_scripted.pt')
        torch.jit.save(traced_model, script_path)
        logger.info(f"Saved TorchScript model to {script_path}")
    except Exception as e:
        logger.warning(f"Could not save TorchScript model: {e}")

def export_to_edge_torch(model, input_shape, output_path):
    try:
        import ai_edge_torch
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # Basic conversion
        logger.info("Converting model with AI Edge Torch...")
        edge_model = ai_edge_torch.convert(model, (dummy_input,))
        
        # Test converted model
        test_output = edge_model(dummy_input)
        logger.info(f"Edge model output shape: {test_output.shape}")
        
        # Export to TFLite
        edge_model.export(output_path)
        logger.info(f"Model successfully exported to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def process_fold(args, fold_idx, train_subjects, val_subjects, test_subjects):
    fold_work_dir = os.path.join(args.work_dir, f"fold_{fold_idx}")
    os.makedirs(fold_work_dir, exist_ok=True)
    
    try:
        logger.info(f"Fold {fold_idx+1} - Train: {train_subjects}, Val: {val_subjects}, Test: {test_subjects}")
        
        # Prepare datasets
        train_set, val_set, test_set = prepare_datasets(args, fold_idx)
        if train_set is None or val_set is None or test_set is None:
            logger.error(f"Failed to prepare datasets for fold {fold_idx}")
            return {"fold": fold_idx, "error": "Dataset preparation failed"}
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_set, val_set, test_set, 
            args.batch_size, args.test_batch_size, args.num_workers
        )
        
        logger.info(f"Loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        # Set device
        device = torch.device(f'cuda:{args.device}' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create model
        model = FallDetectionModel(
            acc_frames=128,
            num_classes=1,
            num_heads=args.num_heads,
            acc_coords=4,
            num_layer=args.num_layers,
            embed_dim=args.embed_dim
        ).to(device)
        
        # Loss function, optimizer, and scheduler
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            patience=5,
            factor=0.5,
            min_lr=1e-7,
            verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=args.patience)
        best_val_f1 = 0
        best_epoch = 0
        best_val_metrics = {}
        
        # Training loop
        for epoch in range(args.max_epoch):
            logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}/{args.max_epoch}")
            
            # Train
            train_loss, _, _, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, args.grad_clip
            )
            
            # Validate
            val_loss, _, _, val_metrics = validate(
                model, val_loader, criterion, device, prefix="val"
            )
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.2f}%, Accuracy: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.2f}%, Accuracy: {val_metrics['accuracy']:.2f}%")
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
                best_val_metrics = val_metrics
                save_model(model, fold_work_dir, epoch, val_metrics, device, is_best=True)
                logger.info(f"New best model saved with val F1: {val_metrics['f1']:.2f}%")
            
            # Early stopping
            if early_stopping(val_loss):
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model for testing
        best_model_path = os.path.join(fold_work_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            model = FallDetectionModel(
                acc_frames=128,
                num_classes=1,
                num_heads=args.num_heads,
                acc_coords=4,
                num_layer=args.num_layers,
                embed_dim=args.embed_dim
            ).to(device)
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            
            # Test model
            test_loss, _, _, test_metrics = validate(
                model, test_loader, criterion, device, prefix="test"
            )
            
            logger.info(f"Test Loss: {test_loss:.4f}, F1: {test_metrics['f1']:.2f}%, Accuracy: {test_metrics['accuracy']:.2f}%")
            
            # Export to TFLite
            tflite_path = os.path.join(fold_work_dir, 'model.tflite')
            export_success = export_to_edge_torch(model, (1, 128, 4), tflite_path)
            
            # Save metadata
            metadata = {
                'fold': fold_idx,
                'best_epoch': best_epoch + 1,
                'best_val_metrics': best_val_metrics,
                'test_metrics': test_metrics,
                'model_config': {
                    'acc_frames': 128,
                    'num_classes': 1,
                    'num_heads': args.num_heads,
                    'acc_coords': 4,
                    'num_layers': args.num_layers,
                    'embed_dim': args.embed_dim
                },
                'export_success': export_success
            }
            
            with open(os.path.join(fold_work_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'fold': fold_idx,
                'best_val_f1': float(best_val_f1),
                'test_metrics': test_metrics,
                'export_success': export_success
            }
        else:
            logger.error(f"Best model not found at {best_model_path}")
            return {'fold': fold_idx, 'error': 'Best model not found'}
        
    except Exception as e:
        logger.error(f"Error in fold {fold_idx}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'fold': fold_idx, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=str, default='./results')
    parser.add_argument('--use-gpu', type=lambda x: x.lower() == 'true', default=True)
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
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--all-subjects', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default='data/smartfallmm')
    parser.add_argument('--dataset-args', type=dict, default={
        'age_group': ['young', 'old'],
        'modalities': ['accelerometer'],
        'sensors': ['watch'],
        'mode': 'sliding_window',
        'max_length': 128,
        'task': 'fd'
    })
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.work_dir = f"{args.work_dir}_{timestamp}"
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.work_dir, 'config.json'), 'w') as f:
        config = vars(args).copy()
        # Convert non-serializable items
        for k, v in config.items():
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                config[k] = str(v)
        json.dump(config, f, indent=2)
    
    # Parse subjects if provided
    if args.all_subjects:
        args.all_subjects = [int(s) for s in args.all_subjects.split(',')]
    
    # Create folds
    folds = create_subject_folds()
    
    # Process folds
    results = []
    
    if args.fold == -1:
        fold_indices = range(len(folds))
        logger.info(f"Processing all {len(folds)} folds")
    else:
        fold_indices = [args.fold]
        logger.info(f"Processing fold {args.fold+1}/{len(folds)}")
    
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
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        # Calculate overall metrics
        try:
            overall_f1 = np.mean([r.get('test_metrics', {}).get('f1', 0) for r in valid_results])
            overall_accuracy = np.mean([r.get('test_metrics', {}).get('accuracy', 0) for r in valid_results])
            
            logger.info(f"Overall F1: {overall_f1:.2f}%, Accuracy: {overall_accuracy:.2f}%")
            
            with open(os.path.join(args.work_dir, 'overall_results.json'), 'w') as f:
                json.dump({
                    'overall_f1': float(overall_f1),
                    'overall_accuracy': float(overall_accuracy),
                    'folds': results
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
    else:
        logger.warning("No valid results to compute overall metrics")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
