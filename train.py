import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from utils.processor.base import csvloader, matloader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
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
    """Calculate binary classification metrics safely"""
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    try:
        # Handle special cases first
        if np.sum(y_pred) == 0 and np.sum(y_true) == 0:
            return {'accuracy': 100.0, 'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
        
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

def robust_normalize_batch(batch, eps=1e-6):
    """Normalize batch with robust handling of edge cases"""
    # Calculate mean and std across time and feature dimensions
    mean = batch.mean(dim=(1, 2), keepdim=True)
    std = batch.std(dim=(1, 2), keepdim=True) + eps
    
    # Ensure std is not too small
    std = torch.clamp(std, min=eps)
    
    # Normalize and clip to avoid extreme values
    normalized = (batch - mean) / std
    normalized = torch.clamp(normalized, min=-5.0, max=5.0)
    
    return normalized

def train_epoch(model, loader, criterion, optimizer, device, clip_grad=0.5, debug=False):
    """Train for one epoch with improved stability"""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    valid_batches = 0
    
    for batch_idx, (acc_data, labels) in enumerate(tqdm(loader, desc="Training")):
        try:
            # Apply robust normalization
            acc_data = robust_normalize_batch(acc_data.to(device))
            labels = labels.to(device).float().unsqueeze(1)
            
            # Debug first batch
            if debug and batch_idx == 0:
                logger.info(f"Batch {batch_idx} - Input shape: {acc_data.shape}, Labels shape: {labels.shape}")
                logger.info(f"First sample SMV values: {acc_data[0, :5, 0].detach().cpu().numpy()}")
                logger.info(f"First sample label: {labels[0].item()}")
            
            optimizer.zero_grad()
            
            # Use autocast for mixed precision training - FIXED: adding device_type
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                outputs, features = model(acc_data)
                loss = criterion(outputs, labels)
            
            # Skip batches with non-finite loss
            if not torch.isfinite(loss):
                logger.warning(f"Skipping batch {batch_idx} due to non-finite loss")
                continue
                
            # Backward pass with gradient clipping
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            valid_batches += 1
            
            # Calculate predictions
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # Debug first and last batch
            if debug and (batch_idx == 0 or batch_idx == len(loader) - 1):
                logger.info(f"Batch {batch_idx} - Loss: {loss.item():.4f}")
                pred_counts = np.bincount(preds.detach().cpu().numpy().flatten())
                logger.info(f"Prediction counts: 0={pred_counts[0]}, 1={pred_counts[1] if len(pred_counts)>1 else 0}")
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Calculate epoch statistics
    epoch_loss = running_loss / max(valid_batches, 1)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return epoch_loss, metrics

def validate(model, loader, criterion, device, prefix="val", debug=False):
    """Validate model with improved stability"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    valid_batches = 0
    
    with torch.no_grad():
        for batch_idx, (acc_data, labels) in enumerate(tqdm(loader, desc=f"{prefix.capitalize()}")):
            try:
                # Apply robust normalization
                acc_data = robust_normalize_batch(acc_data.to(device))
                labels = labels.to(device).float().unsqueeze(1)
                
                # Debug first batch
                if debug and batch_idx == 0:
                    logger.info(f"{prefix} batch {batch_idx} - Input shape: {acc_data.shape}")
                
                # FIXED: Adding device_type parameter
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                    outputs, features = model(acc_data)
                    loss = criterion(outputs, labels)
                
                # Skip non-finite loss
                if not torch.isfinite(loss):
                    continue
                    
                running_loss += loss.item()
                valid_batches += 1
                
                # Calculate predictions
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Debug first batch
                if debug and batch_idx == 0:
                    logger.info(f"{prefix} batch {batch_idx} - Loss: {loss.item():.4f}")
                    logger.info(f"Sample outputs: {outputs[:3].detach().cpu().numpy()}")
                
            except Exception as e:
                logger.error(f"Error in validation: {e}")
                continue
    
    # Calculate validation statistics
    val_loss = running_loss / max(valid_batches, 1)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return val_loss, metrics

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with imbalanced data
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
        
    def forward(self, inputs, targets):
        # Get probabilities
        inputs = torch.clamp(torch.sigmoid(inputs), min=self.eps, max=1-self.eps)
        
        # For binary classification
        targets = targets.float()
        
        # Calculate focal weight
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Apply focal weight
        loss = -alpha * (1 - pt) ** self.gamma * torch.log(pt)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FallDetectionDataset(torch.utils.data.Dataset):
    """Fall detection dataset that properly handles missing data"""
    def __init__(self, dataset, batch_size=32):
        self.acc_data = []
        self.labels = []
        
        # Process accelerometer data if available
        if isinstance(dataset, dict) and 'accelerometer' in dataset and 'labels' in dataset:
            acc_data = dataset['accelerometer']
            labels = dataset['labels']
            
            if len(acc_data) > 0 and len(labels) > 0:
                # Check shapes
                if len(acc_data.shape) == 3:
                    # Calculate SMV if not present (shape should be [samples, frames, 3])
                    if acc_data.shape[2] == 3:
                        # Calculate SMV
                        x, y, z = acc_data[:, :, 0], acc_data[:, :, 1], acc_data[:, :, 2]
                        smv = np.sqrt(x**2 + y**2 + z**2).reshape(acc_data.shape[0], acc_data.shape[1], 1)
                        acc_data = np.concatenate([smv, acc_data], axis=2)
                    
                    self.acc_data = acc_data
                    self.labels = labels
        
        self.num_samples = len(self.acc_data)
        logger.info(f"Dataset initialized with {self.num_samples} samples")
        if self.num_samples > 0:
            logger.info(f"Data shape: {self.acc_data.shape}, Label shape: {self.labels.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if index >= self.num_samples:
            raise IndexError(f"Index {index} out of range")
        
        # Get accelerometer data and ensure it's a tensor
        acc_data = torch.tensor(self.acc_data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        
        return acc_data, label

def main():
    parser = argparse.ArgumentParser(description='Robust Fall Detection Training')
    parser.add_argument('--work-dir', type=str, default='./results')
    parser.add_argument('--use-gpu', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--data-dir', type=str, default='data/smartfallmm')
    parser.add_argument('--debug', action='store_true', help='Enable detailed logging')
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create work directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.work_dir = os.path.join(args.work_dir, f"run_{timestamp}")
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Configure logging to file
    log_file = os.path.join(args.work_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Save configuration
    with open(os.path.join(args.work_dir, 'config.json'), 'w') as f:
        config = vars(args)
        json.dump(config, f, indent=2)
    
    # Log system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device: {torch.cuda.get_device_name(args.device)}")
    
    # Get cross-validation folds
    val_subjects = [38, 46]
    always_train_subjects = [45, 36, 29]
    eligible_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
    folds = []
    for test_subject in eligible_subjects:
        test_subjects = [test_subject]
        train_subjects = always_train_subjects + [s for s in eligible_subjects if s != test_subject]
        folds.append({'train': train_subjects, 'val': val_subjects, 'test': test_subjects})
    
    # Determine which folds to process
    if args.fold == -1:
        fold_indices = range(len(folds))
        logger.info(f"Processing all {len(folds)} folds")
    else:
        fold_indices = [args.fold]
        logger.info(f"Processing fold {args.fold+1}/{len(folds)}")
    
    # Track results across folds
    results = []
    
    for fold_idx in fold_indices:
        if fold_idx >= len(folds):
            logger.error(f"Fold index {fold_idx} is out of range (max: {len(folds)-1})")
            continue
        
        # Setup for current fold
        fold = folds[fold_idx]
        fold_work_dir = os.path.join(args.work_dir, f"fold_{fold_idx}")
        os.makedirs(fold_work_dir, exist_ok=True)
        
        logger.info(f"Fold {fold_idx+1} - Train: {fold['train']}, Val: {fold['val']}, Test: {fold['test']}")
        
        # Prepare dataset
        args.dataset_args = {
            'age_group': ['young', 'old'],
            'modalities': ['accelerometer'],
            'sensors': ['watch'],
            'mode': 'sliding_window',
            'max_length': 128,
            'task': 'fd'
        }
        
        try:
            # Import dataset modules here to avoid circular imports
            from utils.dataset import split_by_subjects, prepare_smartfallmm
            
            logger.info("Preparing datasets...")
            builder = prepare_smartfallmm(args)
            train_data = split_by_subjects(builder, fold['train'], False)
            val_data = split_by_subjects(builder, fold['val'], False)
            test_data = split_by_subjects(builder, fold['test'], False)
            
            train_dataset = FallDetectionDataset(train_data)
            val_dataset = FallDetectionDataset(val_data)
            test_dataset = FallDetectionDataset(test_data)
            
            logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
            
            # Log class distribution
            if len(train_dataset) > 0 and hasattr(train_dataset, 'labels'):
                logger.info(f"Train class distribution: {np.bincount(train_dataset.labels.astype(int))}")
            if len(val_dataset) > 0 and hasattr(val_dataset, 'labels'):
                logger.info(f"Val class distribution: {np.bincount(val_dataset.labels.astype(int))}")
            if len(test_dataset) > 0 and hasattr(test_dataset, 'labels'):
                logger.info(f"Test class distribution: {np.bincount(test_dataset.labels.astype(int))}")
            
            # Check for empty datasets
            if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
                logger.warning(f"Skipping fold {fold_idx} due to empty dataset split")
                continue
                
        except Exception as e:
            logger.error(f"Error preparing datasets: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
        
        # Create data loaders with appropriate batch sizes
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.test_batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.test_batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Set device
        if args.use_gpu and torch.cuda.is_available():
            device_id = min(args.device, torch.cuda.device_count() - 1)
            device = torch.device(f'cuda:{device_id}')
            logger.info(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Import model here to avoid circular imports
        from Models.EdgeFallModel import EdgeFallTransformer
        
        # Create model
        model = EdgeFallTransformer(
            acc_frames=128,
            num_classes=1,
            num_heads=args.num_heads,
            acc_coords=4,  # SMV + x,y,z
            num_layer=args.num_layers,
            embed_dim=args.embed_dim,
            dropout=0.1,
            debug=args.debug
        ).to(device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params}, Trainable: {trainable_params}")
        
        # Define loss function 
        if args.focal_loss:
            # Focal loss for better handling of class imbalance
            criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
            logger.info("Using Focal Loss")
        else:
            # Standard BCE loss with positive weighting
            pos_weight = torch.tensor([2.0]).to(device)
            criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
            logger.info("Using standard BCE Loss with pos_weight=2.0")
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay,
            eps=1e-5
        )
        
        # Learning rate scheduler with warmup
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=args.min_lr
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=args.patience)
        
        # Training loop
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(args.max_epoch):
            logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}/{args.max_epoch}")
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.8f}")
            
            # Train
            train_loss, train_metrics = train_epoch(
                model, 
                train_loader, 
                criterion, 
                optimizer,
                device, 
                args.grad_clip,
                args.debug
            )
            
            # Validate
            val_loss, val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                prefix="val",
                debug=args.debug
            )
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.2f}%, Accuracy: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.2f}%, Accuracy: {val_metrics['accuracy']:.2f}%")
            
            # Save best model based on F1 score
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = model.state_dict()
                
                model_path = os.path.join(fold_work_dir, 'best_model.pth')
                torch.save(model.state_dict(), model_path)
                logger.info(f"New best model saved with val F1: {val_metrics['f1']:.2f}%")
            
            # Early stopping check
            if early_stopping(val_loss):
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model for testing
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test best model
        test_loss, test_metrics = validate(
            model,
            test_loader,
            criterion,
            device,
            prefix="test",
            debug=True  # Always use detailed logging for test set
        )
        
        logger.info(f"Test Loss: {test_loss:.4f}, F1: {test_metrics['f1']:.2f}%, Accuracy: {test_metrics['accuracy']:.2f}%")
        
        # Save metadata
        metadata = {
            'fold': fold_idx,
            'train_subjects': fold['train'],
            'val_subjects': fold['val'],
            'test_subjects': fold['test'],
            'test_metrics': test_metrics,
            'model_config': {
                'acc_frames': 128,
                'acc_coords': 4,
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers
            }
        }
        
        with open(os.path.join(fold_work_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Track results
        results.append({
            'fold': fold_idx,
            'test_metrics': test_metrics
        })
    
    # Calculate overall metrics
    if results:
        overall_f1 = np.mean([r['test_metrics']['f1'] for r in results])
        overall_accuracy = np.mean([r['test_metrics']['accuracy'] for r in results])
        overall_precision = np.mean([r['test_metrics']['precision'] for r in results])
        overall_recall = np.mean([r['test_metrics']['recall'] for r in results])
        
        logger.info(f"Overall F1: {overall_f1:.2f}%, Accuracy: {overall_accuracy:.2f}%, Precision: {overall_precision:.2f}%, Recall: {overall_recall:.2f}%")
        
        with open(os.path.join(args.work_dir, 'overall_results.json'), 'w') as f:
            json.dump({
                'overall_f1': float(overall_f1),
                'overall_accuracy': float(overall_accuracy),
                'overall_precision': float(overall_precision),
                'overall_recall': float(overall_recall),
                'folds': results
            }, f, indent=2)

if __name__ == "__main__":
    main()
