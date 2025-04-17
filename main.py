import os
import torch
import argparse
import numpy as np
import json
import logging
import time
from datetime import datetime
from collections import defaultdict
from Models.fall_detection import FallDetectionTransformer
from utils.data_loader import prepare_datasets, create_data_loaders, create_subject_folds
from utils.metrics import calculate_metrics, EarlyStopping
from utils.training import train_epoch, validate, save_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=-1, help='Fold index (-1 means all folds)')
    parser.add_argument('--subjects', type=str, default='30,31,32,33,34,35,37,39,43,44,45,36,29,38,46')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--base-lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--work-dir', type=str, default='./results/')
    parser.add_argument('--use-gpu', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layer', type=int, default=2)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--dataset-args', type=dict, default={
        'mode': 'sliding_window',
        'max_length': 128,
        'task': 'fd',
        'modalities': ['accelerometer'],
        'age_group': ['young', 'old'],
        'sensors': ['watch']
    })
    return parser.parse_args()

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_fold(args, fold_idx, fold_work_dir, device):
    train_set, val_set, test_set = prepare_datasets(args, fold_idx)
    if train_set is None or val_set is None or test_set is None:
        logger.warning(f"Skipping fold {fold_idx} due to missing datasets")
        return None
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_set, val_set, test_set, 
        args.batch_size, args.test_batch_size, args.num_worker
    )
    
    logger.info(f"Loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    model = FallDetectionTransformer(
        acc_frames=128,
        num_classes=1,
        num_heads=args.num_heads,
        acc_coords=3,
        num_layer=args.num_layer,
        embed_dim=args.embed_dim
    ).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-7, verbose=True)
    
    early_stopping = EarlyStopping(patience=args.patience)
    best_val_f1, best_epoch = 0, 0
    best_val_metrics, best_test_metrics = {}, {}
    metrics_history = defaultdict(list)
    
    logger.info(f"Starting training for fold {fold_idx+1}")
    
    for epoch in range(args.max_epoch):
        epoch_start = time.time()
        logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}/{args.max_epoch}")
        
        try:
            train_loss, train_preds, train_labels, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, args.grad_clip
            )
            
            val_loss, val_preds, val_labels, val_metrics = validate(
                model, val_loader, criterion, device, prefix="val"
            )
            
            is_valid_epoch = np.isfinite(train_loss) and np.isfinite(val_loss)
            
            if is_valid_epoch:
                scheduler.step(val_loss)
                
                metrics_history['train_loss'].append(train_loss)
                metrics_history['val_loss'].append(val_loss)
                
                for key, value in train_metrics.items():
                    metrics_history[f'train_{key}'].append(value)
                    
                for key, value in val_metrics.items():
                    metrics_history[f'val_{key}'].append(value)
                
                current_lr = optimizer.param_groups[0]['lr']
                epoch_time = time.time() - epoch_start
                
                logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Train F1: {train_metrics.get('f1', 0):.2f}%, "
                          f"Val F1: {val_metrics.get('f1', 0):.2f}%, "
                          f"LR: {current_lr:.6f}")
                
                val_f1 = val_metrics.get('f1', 0)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    best_val_metrics = val_metrics
                    save_model(model, fold_work_dir, epoch, val_metrics, device, is_best=True)
                    logger.info(f"New best model saved with validation F1: {val_f1:.2f}%")
            else:
                logger.warning(f"Skipping epoch due to non-finite loss - Train: {train_loss}, Val: {val_loss}")
            
            if early_stopping(val_loss):
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        except Exception as e:
            logger.error(f"Error in epoch {epoch+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info("Training complete - Loading best model for test evaluation")
    
    best_model_path = os.path.join(fold_work_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model = FallDetectionTransformer(
            acc_frames=128,
            num_classes=1,
            num_heads=args.num_heads,
            acc_coords=3,
            num_layer=args.num_layer,
            embed_dim=args.embed_dim
        ).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        test_loss, test_preds, test_labels, test_metrics = validate(
            model, test_loader, criterion, device, prefix="test"
        )
        
        logger.info(f"Test evaluation with best model (epoch {best_epoch+1}): "
                  f"F1: {test_metrics.get('f1', 0):.2f}%, "
                  f"Accuracy: {test_metrics.get('accuracy', 0):.2f}%, "
                  f"Precision: {test_metrics.get('precision', 0):.2f}%, "
                  f"Recall: {test_metrics.get('recall', 0):.2f}%")
        
        best_test_metrics = test_metrics
        
        with torch.no_grad():
            sample_input = torch.randn(1, 128, 4).to(device)
            sample_output, _ = model(sample_input)
        
        metadata = {
            "input_shape": [1, 128, 4],
            "output_shape": list(sample_output.shape),
            "model_parameters": {
                "num_classes": 1,
                "num_heads": args.num_heads,
                "num_layer": args.num_layer,
                "embed_dim": args.embed_dim
            }
        }
        
        with open(os.path.join(fold_work_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    else:
        logger.error(f"Best model file not found at {best_model_path}")
        best_test_metrics = {}
    
    fold_summary = {
        'fold': fold_idx,
        'best_epoch': best_epoch + 1,
        'best_val_metrics': best_val_metrics,
        'test_metrics': best_test_metrics
    }
    
    with open(os.path.join(fold_work_dir, 'fold_summary.json'), 'w') as f:
        json.dump(fold_summary, f, indent=2)
    
    return fold_summary

def main():
    args = parse_args()
    init_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.work_dir = f"{args.work_dir}_{timestamp}"
    os.makedirs(args.work_dir, exist_ok=True)
    
    config = vars(args)
    with open(os.path.join(args.work_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    folds = create_subject_folds()
    
    if args.fold == -1:
        fold_indices = list(range(len(folds)))
        logger.info(f"Processing all {len(folds)} folds")
    else:
        fold_indices = [args.fold]
        logger.info(f"Processing single fold: {args.fold}")
    
    if isinstance(args.subjects, str):
        args.subjects = [int(s) for s in args.subjects.split(',')]
    
    results = []
    
    for fold_idx in fold_indices:
        fold_start_time = time.time()
        logger.info("="*80)
        logger.info(f"Processing fold {fold_idx+1}/{len(folds)}")
        logger.info("="*80)
        
        fold_work_dir = os.path.join(args.work_dir, f'fold_{fold_idx}')
        os.makedirs(fold_work_dir, exist_ok=True)
        
        fold_result = process_fold(args, fold_idx, fold_work_dir, device)
        if fold_result:
            fold_result['training_time'] = time.time() - fold_start_time
            results.append(fold_result)
    
    if results:
        val_f1_values = [result['best_val_metrics'].get('f1', 0) for result in results]
        test_metrics_keys = results[0]['test_metrics'].keys() if results[0]['test_metrics'] else []
        
        avg_results = {
            'num_folds': len(results),
            'avg_val_f1': float(np.mean(val_f1_values)),
        }
        
        for metric in test_metrics_keys:
            values = [result['test_metrics'].get(metric, 0) for result in results]
            avg_results[f'avg_test_{metric}'] = float(np.mean(values))
        
        logger.info("Average results across folds:")
        for key, value in avg_results.items():
            if key.startswith('avg_'):
                logger.info(f"{key}: {value:.2f}%")
        
        with open(os.path.join(args.work_dir, 'overall_results.json'), 'w') as f:
            json.dump(avg_results, f, indent=2)
    
    logger.info("="*80)
    logger.info(f"Training completed")
    logger.info("="*80)

if __name__ == "__main__":
    main()
