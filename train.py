import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import torch
import argparse
import numpy as np
import json
import logging
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from Models.EdgeFallTransformer import EdgeFallTransformer
    from utils.dataset import FallDetectionDataset, prepare_smartfallmm, split_by_subjects
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

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
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    try:
        return {
            'accuracy': 100 * accuracy_score(y_true, y_pred),
            'f1': 100 * f1_score(y_true, y_pred, average='binary', zero_division=0),
            'precision': 100 * precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': 100 * recall_score(y_true, y_pred, average='binary', zero_division=0)
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

def normalize_batch(batch):
    mean = batch.mean(dim=(1, 2), keepdim=True)
    std = batch.std(dim=(1, 2), keepdim=True) + 1e-5
    return (batch - mean) / std

def train_epoch(model, loader, criterion, optimizer, device, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    valid_batches = 0
    for batch_idx, (acc_data, labels) in enumerate(loader):
        try:
            acc_data = normalize_batch(acc_data.to(device))
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs, _ = model(acc_data)
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
            continue
            
    epoch_loss = running_loss / max(valid_batches, 1)
    metrics = calculate_metrics(all_labels, all_preds)
    return epoch_loss, metrics

def validate(model, loader, criterion, device, prefix="val"):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    valid_batches = 0
    with torch.no_grad():
        for acc_data, labels in loader:
            try:
                acc_data = normalize_batch(acc_data.to(device))
                labels = labels.to(device).unsqueeze(1)
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs, _ = model(acc_data)
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
                
    val_loss = running_loss / max(valid_batches, 1)
    metrics = calculate_metrics(all_preds, all_labels)
    return val_loss, metrics

def create_mock_dataset(num_samples=10, seq_len=128, features=4):
    logger.info(f"Creating mock dataset with {num_samples} samples")
    X = torch.randn(num_samples, seq_len, features)
    y = torch.randint(0, 2, (num_samples,)).float()
    dataset = TensorDataset(X, y)
    return dataset

def create_subject_folds():
    val_subjects = [38, 46]
    always_train_subjects = [45, 36, 29]
    eligible_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
    folds = []
    for test_subject in eligible_subjects:
        test_subjects = [test_subject]
        train_subjects = always_train_subjects + [s for s in eligible_subjects if s != test_subject]
        folds.append({'train': train_subjects, 'val': val_subjects, 'test': test_subjects})
    return folds

def check_data_paths(args):
    data_paths = [
        os.path.join(args.data_dir, "young", "accelerometer", "watch"),
    ]
    for path in data_paths:
        if os.path.exists(path):
            logger.info(f"Path exists: {path}")
            files = [f for f in os.listdir(path) if f.endswith('.csv')]
            logger.info(f"Found {len(files)} CSV files in {path}")
        else:
            logger.warning(f"Path does not exist: {path}")
            try:
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created directory: {path}")
            except Exception as e:
                logger.error(f"Could not create directory {path}: {e}")

def modify_transformer(model):
    for param in model.parameters():
        if param.requires_grad:
            param.data.mul_(0.1)  # Scale down initial weights
    return model

def process_fold(args, fold_idx, train_subjects, val_subjects, test_subjects):
    fold_work_dir = os.path.join(args.work_dir, f"fold_{fold_idx}")
    os.makedirs(fold_work_dir, exist_ok=True)
    logger.info(f"Fold {fold_idx+1} - Train: {train_subjects}, Val: {val_subjects}, Test: {test_subjects}")
    
    check_data_paths(args)
    
    try:
        args.dataset_args = {
            'age_group': ['young', 'old'],
            'modalities': ['accelerometer'],
            'sensors': ['watch'],
            'mode': 'sliding_window',
            'max_length': 128,
            'task': 'fd'
        }
        
        builder = prepare_smartfallmm(args)
        train_data = split_by_subjects(builder, train_subjects, False)
        val_data = split_by_subjects(builder, val_subjects, False)
        test_data = split_by_subjects(builder, test_subjects, False)
        
        train_dataset = FallDetectionDataset(train_data)
        val_dataset = FallDetectionDataset(val_data)
        test_dataset = FallDetectionDataset(test_data)
        
        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
        
        use_mock_data = False
        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
            logger.warning("One or more datasets are empty, using mock data for testing the model")
            train_dataset = create_mock_dataset(20)
            val_dataset = create_mock_dataset(10)
            test_dataset = create_mock_dataset(10)
            use_mock_data = True
    except Exception as e:
        logger.error(f"Error preparing datasets: {e}")
        logger.warning("Using mock data due to dataset preparation error")
        train_dataset = create_mock_dataset(20)
        val_dataset = create_mock_dataset(10)
        test_dataset = create_mock_dataset(10)
        use_mock_data = True

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=False)

    if args.use_gpu and torch.cuda.is_available():
        device_id = min(args.device, torch.cuda.device_count() - 1)
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # Create the model with modified architecture for stability
    model = EdgeFallTransformer(
        acc_frames=128,
        num_classes=1,
        num_heads=args.num_heads,
        acc_coords=4,
        num_layer=args.num_layers,
        embed_dim=args.embed_dim,
        dropout=0.1  # Reduced dropout for stability
    ).to(device)
    
    # Scale down initial weights for more stability
    model = modify_transformer(model)

    # Use a more robust loss function with label smoothing
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([2.0]).to(device))
    
    # Lower learning rate and use gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.base_lr*0.1,  # Reduced learning rate
        weight_decay=args.weight_decay,
        eps=1e-5  # Higher eps for numerical stability
    )
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.2, min_lr=1e-7, verbose=False)

    early_stopping = EarlyStopping(patience=args.patience)
    best_val_f1 = 0
    best_model_state = None

    # Scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(args.max_epoch):
        logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}/{args.max_epoch}")
        
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip
        )
        
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.2f}%, Accuracy: {train_metrics['accuracy']:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.2f}%, Accuracy: {val_metrics['accuracy']:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
            
            model_path = os.path.join(fold_work_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved with val F1: {val_metrics['f1']:.2f}%")
        
        if early_stopping(val_loss):
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_loss, test_metrics = validate(model, test_loader, criterion, device, prefix="test")
    
    logger.info(f"Test Loss: {test_loss:.4f}, F1: {test_metrics['f1']:.2f}%, Accuracy: {test_metrics['accuracy']:.2f}%")
    
    metadata = {
        'fold': fold_idx,
        'mock_data_used': use_mock_data,
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
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
    
    return {
        'fold': fold_idx,
        'mock_data_used': use_mock_data,
        'test_metrics': test_metrics
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=str, default='./results')
    parser.add_argument('--use-gpu', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)  # Smaller batch size
    parser.add_argument('--test-batch-size', type=int, default=8)  # Smaller batch size
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--base-lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=0.5)  # Lower grad clip
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--data-dir', type=str, default='data/smartfallmm')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.work_dir = os.path.join(args.work_dir, f"run_{timestamp}")
    os.makedirs(args.work_dir, exist_ok=True)
    
    with open(os.path.join(args.work_dir, 'config.json'), 'w') as f:
        config = vars(args).copy()
        for k, v in config.items():
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                config[k] = str(v)
        json.dump(config, f, indent=2)
    
    folds = create_subject_folds()
    
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
    
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        overall_f1 = np.mean([r['test_metrics']['f1'] for r in valid_results])
        overall_accuracy = np.mean([r['test_metrics']['accuracy'] for r in valid_results])
        
        logger.info(f"Overall F1: {overall_f1:.2f}%, Accuracy: {overall_accuracy:.2f}%")
        logger.info(f"Mock data was used in {sum(1 for r in valid_results if r.get('mock_data_used', False))} folds")
        
        with open(os.path.join(args.work_dir, 'overall_results.json'), 'w') as f:
            json.dump({
                'overall_f1': float(overall_f1),
                'overall_accuracy': float(overall_accuracy),
                'folds': results
            }, f, indent=2)

if __name__ == "__main__":
    main()
