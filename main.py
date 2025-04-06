#!/usr/bin/env python
"""
main.py

Main training script for LightHART fall detection with subject‐based cross‐validation.
Subject splits:
  - Full training/test pool: [32,39,30,31,33,34,35,37,43,44,45,36,29]
  - Fixed validation subjects: [38,46]
  - Fixed training subjects (always in training): [45,36,29]
  - The remaining 10 subjects (from the pool) are partitioned into 5 folds (2 per fold) with no overlap.
  
For each fold:
  - Test subjects: one disjoint pair from the available pool.
  - Training subjects: fixed training subjects plus remaining available subjects (excluding the current test pair).
  - Validation set: fixed subjects [38,46].
  
Assumes that the dataset (via UTD_mm and DatasetBuilder) returns a dictionary that includes a key "subject".
"""

import argparse
import os
import random
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from Feeder.Make_Dataset import UTD_mm
from Models.fusion_transformer import FusionTransModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="LightHART Fall Detection Training with Subject CV")
    parser.add_argument("--work-dir", type=str, default="results", help="Directory for saving results")
    parser.add_argument("--phase", type=str, choices=["train", "test"], default="train", help="Training phase")
    parser.add_argument("--filters", type=str, default="madgwick,kalman,ekf", help="IMU filter types (comma separated)")
    parser.add_argument("--use-gpu", type=str2bool, default=True, help="Whether to use GPU")
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--test-batch-size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--num-worker", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--fuse", type=str2bool, default=True, help="Flag for sensor fusion usage")
    parser.add_argument("--model", type=str, default="Models.fusion_transformer.FusionTransModel", help="Model module path")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")
    parser.add_argument("--base-lr", type=float, default=0.0005, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--loss", type=str, default="bce", help="Loss type")
    parser.add_argument("--max-epoch", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--use_features", type=str2bool, default=True,
                        help="Use extra fusion features (teacher mode) or not (student/inference mode)")
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model(args):
    model_args = {
        "num_layers": 3,
        "embed_dim": 32,
        "acc_coords": 3,
        "quat_coords": 4,
        "num_classes": 2,
        "acc_frames": 64,
        "mocap_frames": 64,
        "num_heads": 4,
        "fusion_type": "concat",
        "dropout": 0.3,
        "use_batch_norm": True,
        "feature_dim": 64,
        "use_features": args.use_features
    }
    model = FusionTransModel(**model_args)
    return model

def get_optimizer(model, args):
    if args.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    return optimizer

# Simple Dataset wrapper for numpy data
class NumpyDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
        self.length = self.data["labels"].shape[0]
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        sample = {k: self.data[k][idx] for k in self.data if k != "labels"}
        label = self.data["labels"][idx]
        return sample, label

# Split the normalized data dictionary by subject IDs
def split_data_by_subjects(data, train_subjects, test_subjects, val_subjects):
    subject_ids = data["subject"]
    train_idx = [i for i, s in enumerate(subject_ids) if s in train_subjects]
    test_idx = [i for i, s in enumerate(subject_ids) if s in test_subjects]
    val_idx = [i for i, s in enumerate(subject_ids) if s in val_subjects]
    def subset(data, indices):
        return {k: data[k][indices] for k in data if isinstance(data[k], np.ndarray)}
    return subset(data, train_idx), subset(data, test_idx), subset(data, val_idx)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, label) in enumerate(dataloader):
        for key in data:
            data[key] = torch.tensor(data[key]).float().to(device)
        label = torch.tensor(label).long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in dataloader:
            for key in data:
                data[key] = torch.tensor(data[key]).float().to(device)
            label = torch.tensor(label).long().to(device)
            output = model(data)
            loss = criterion(output, label)
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = 100.0 * correct / total if total > 0 else 0
    return running_loss / len(dataloader), accuracy

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Define subject splits:
    # Full pool (for training/test): [32,39,30,31,33,34,35,37,43,44,45,36,29]
    full_pool = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44, 45, 36, 29]
    # Fixed validation subjects:
    val_subjects = [38, 46]
    # Fixed training subjects (always training):
    fixed_train = [45, 36, 29]
    # Available for test: the remaining from full_pool (excluding fixed training)
    available_for_test = sorted([s for s in full_pool if s not in fixed_train])
    # From available_for_test, choose the ones that can serve as test candidates.
    # Here we assume available_for_test has 10 subjects.
    # Create 5 folds (2 subjects per fold) with disjoint pairs:
    folds = [
        [available_for_test[0], available_for_test[1]],
        [available_for_test[2], available_for_test[3]],
        [available_for_test[4], available_for_test[5]],
        [available_for_test[6], available_for_test[7]],
        [available_for_test[8], available_for_test[9]]
    ]
    logger.info(f"Test folds: {folds}")
    
    # Create feeder with all subjects in the full pool.
    feeder = UTD_mm(data_path="data/smartfallmm", fold=1, subjects=full_pool, fuse=args.fuse,
                    fusion_options={"enabled": True, "filter_type": args.filters.split(",")[0]})
    
    if len(feeder) == 0:
        logger.error("No samples loaded. Exiting.")
        return
    
    # feeder.prepared_data is assumed to be a dictionary with a "subject" key.
    data = feeder.prepared_data
    if "subject" not in data:
        logger.error("Subject information not found in dataset. Ensure DatasetBuilder adds 'subject' to each sample.")
        return

    all_fold_metrics = []
    # Loop over the folds
    for fold_idx, test_subjects in enumerate(folds, start=1):
        # Training subjects: fixed_train + available subjects not in test_subjects
        train_subjects = fixed_train + [s for s in available_for_test if s not in test_subjects]
        logger.info(f"Fold {fold_idx} split:")
        logger.info(f"  Train subjects: {train_subjects}")
        logger.info(f"  Test subjects: {test_subjects}")
        logger.info(f"  Val subjects: {val_subjects}")
        
        train_data, test_data, val_data = split_data_by_subjects(data, train_subjects, test_subjects, val_subjects)
        if train_data["labels"].shape[0] == 0 or test_data["labels"].shape[0] == 0 or val_data["labels"].shape[0] == 0:
            logger.error(f"One of the splits is empty in fold {fold_idx}. Skipping this fold.")
            continue
        
        train_dataset = NumpyDataset(train_data)
        test_dataset = NumpyDataset(test_data)
        val_dataset = NumpyDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_worker)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_worker)
        
        model = get_model(args).to(device)
        optimizer = get_optimizer(model, args)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float("inf")
        epochs_no_improve = 0
        fold_metrics = {"fold": fold_idx, "train_loss": [], "val_loss": [], "val_acc": []}
        
        logger.info(f"Starting training for fold {fold_idx}...")
        for epoch in range(1, args.max_epoch + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            fold_metrics["train_loss"].append(train_loss)
            fold_metrics["val_loss"].append(val_loss)
            fold_metrics["val_acc"].append(val_acc)
            logger.info(f"Fold {fold_idx} Epoch {epoch}: Train Loss {train_loss:.4f}  Val Loss {val_loss:.4f}  Val Acc {val_acc:.2f}%")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(args.work_dir, f"fold_{fold_idx}_best_model.pt"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logger.info(f"Early stopping triggered for fold {fold_idx}.")
                    break
        
        test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
        fold_metrics["test_loss"] = test_loss
        fold_metrics["test_acc"] = test_acc
        logger.info(f"Fold {fold_idx} completed: Test Loss {test_loss:.4f}  Test Acc {test_acc:.2f}%")
        all_fold_metrics.append(fold_metrics)
    
    if all_fold_metrics:
        avg_test_acc = np.mean([m["test_acc"] for m in all_fold_metrics])
        logger.info(f"Cross-validation complete. Average Test Accuracy: {avg_test_acc:.2f}%")
    else:
        logger.error("No valid folds completed.")

if __name__ == "__main__":
    main()

