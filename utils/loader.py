# utils/loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger("data_loader")

class FallDetectionDataset(Dataset):
    def __init__(self, acc_data, labels, subjects=None):
        self.acc_data = acc_data
        self.labels = labels
        self.subjects = subjects if subjects is not None else np.zeros_like(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        acc = torch.tensor(self.acc_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        subject = torch.tensor(self.subjects[idx], dtype=torch.long)

        return acc, label, subject

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

def prepare_datasets(args, fold_idx=0):
    # Import here to avoid circular imports
    from utils.dataset import split_by_subjects, prepare_smartfallmm
    from Feeder.Make_Dataset import FallDataset

    folds = create_subject_folds()
    if fold_idx >= len(folds):
        logger.error(f"Fold index {fold_idx} out of range. Max fold: {len(folds)-1}")
        return None, None, None

    current_fold = folds[fold_idx]
    logger.info(f"Fold {fold_idx+1}/{len(folds)}: Train subjects={current_fold['train']}, Val subjects={current_fold['val']}, Test subjects={current_fold['test']}")

    try:
        logger.info("Loading and preprocessing data")
        import time
        start_time = time.time()
        builder = prepare_smartfallmm(args)

        train_data = split_by_subjects(builder, current_fold['train'], False)
        val_data = split_by_subjects(builder, current_fold['val'], False)
        test_data = split_by_subjects(builder, current_fold['test'], False)

        logger.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")

        # Log data shapes
        for key, value in train_data.items():
            if isinstance(value, np.ndarray):
                logger.info(f"Train {key} shape: {value.shape}")

        # Create datasets
        train_set = FallDataset(train_data)
        val_set = FallDataset(val_data)
        test_set = FallDataset(test_data)

        if len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0:
            logger.warning(f"Skipping fold {fold_idx} due to empty dataset split")
            return None, None, None

        return train_set, val_set, test_set

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

def create_data_loaders(train_set, val_set, test_set, batch_size, test_batch_size, num_workers):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
