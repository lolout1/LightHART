import traceback
from typing import List
import random
import sys
import os
import time
import shutil
import argparse
import yaml

import numpy as np
import pandas as pd
import torch

import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from collections import Counter  # Newly imported to count labels

# Local imports (unchanged)
from utils.dataset import prepare_smartfallmm
from utils.loader import DatasetBuilder
from Feeder.time2vec_varlen import Time2VecVarLenFeeder, time2vec_varlen_collate_fn

###########################################################################
# Model: FallTime2VecTransformer (unchanged)
###########################################################################
class FallTime2VecTransformer(torch.nn.Module):
    def __init__(self, feat_dim=11, d_model=32, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.input_proj = torch.nn.Linear(feat_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        x_proj = self.input_proj(x)      # (B, N, d_model)
        x_proj = x_proj.transpose(0, 1)    # (N, B, d_model)
        out = self.transformer(x_proj, src_key_padding_mask=mask)
        out = out.transpose(0, 1)          # (B, N, d_model)
        if mask is not None:
            lengths = (~mask).sum(dim=-1, keepdim=True)
            out = out * (~mask).unsqueeze(-1)
            out = out.sum(dim=1) / torch.clamp(lengths, min=1e-9)
        else:
            out = out.mean(dim=1)
        logits = self.classifier(out)
        return logits

###########################################################################
# Argument parsing (unchanged except added subjects and dataset_args)
###########################################################################
def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported boolean string')

def get_args():
    parser = argparse.ArgumentParser(description='Fall detection with variable_time + Time2Vec (Cross-Validation)')
    parser.add_argument('--config', default='./config/smartfallmm/time2vec_fall.yaml')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=16)
    parser.add_argument('--num-epoch', type=int, default=30)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--base-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0004)
    parser.add_argument('--model', default='Models.fall_time2vec_transformer.FallTime2VecTransformer')
    parser.add_argument('--model-saved-name', type=str, default='ttfstudent')
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--loss', default='CE', help='Loss function type')
    parser.add_argument('--loss-args', default="{}", type=str)
    parser.add_argument('--dataset_args', default={}, type=dict)
    parser.add_argument('--feeder', default=None, help='Feeder class path')
    parser.add_argument('--train_feeder_args', default={}, type=dict)
    parser.add_argument('--val_feeder_args', default={}, type=dict)
    parser.add_argument('--test_feeder_args', default={}, type=dict)
    parser.add_argument('--include-val', type=str2bool, default=True)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--work-dir', type=str, default='exps', help='Working directory')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--seed', type=int, default=2)
    # Additional parameters for compatibility
    parser.add_argument('--dataset', type=str, default='smartfallmm')
    parser.add_argument('--subjects', nargs='+', type=int, default=[29,30,31,32,33,34,35,36,37,38,39,43,44,45,46])
    parser.add_argument('--model_args', default={}, type=dict)
    return parser

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

###########################################################################
# Trainer Class (with 5-fold cross-validation and early stopping)
###########################################################################
class Trainer():
    def __init__(self, arg):
        self.arg = arg
        self.work_dir = arg.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.print_log(f"Args: {vars(arg)}")
        if arg.config and os.path.exists(arg.config):
            shutil.copy(arg.config, os.path.join(self.work_dir, os.path.basename(arg.config)))
        self.fold_results = []  # list to hold fold-level metrics

    def print_log(self, msg):
        print(msg)
        if self.arg.print_log:
            log_path = os.path.join(self.work_dir, 'log.txt')
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

    def load_data(self, val_subjects: List[int]):
        # Compute training subjects as the complement of val_subjects
        full_subjects = self.arg.subjects
        train_subjects = [s for s in full_subjects if s not in val_subjects]
        self.print_log(f"Train Subjects={train_subjects}, Validation Subjects={val_subjects}")
        
        builder_train = prepare_smartfallmm(self.arg)
        builder_train.make_dataset(train_subjects)
        train_data = builder_train.processed_data
        
        all_arrays = train_data.get('accelerometer', [])
        if len(all_arrays) > 0 and isinstance(all_arrays, list) and isinstance(all_arrays[0], np.ndarray):
            stacked = np.concatenate(all_arrays, axis=0)
            scaler = StandardScaler()
            scaler.fit(stacked)
            def transform_fn(x):
                return scaler.transform(x)
        else:
            def transform_fn(x): return x
        
        builder_test = prepare_smartfallmm(self.arg)
        builder_test.make_dataset(val_subjects)
        test_data = builder_test.processed_data
        
        return train_data, test_data, transform_fn

    def load_model(self):
        feat_dim    = self.arg.model_args.get('feat_dim', 11)
        d_model     = self.arg.model_args.get('d_model', 32)
        nhead       = self.arg.model_args.get('nhead', 4)
        num_layers  = self.arg.model_args.get('num_layers', 2)
        num_classes = self.arg.model_args.get('num_classes', 2)
        model = FallTime2VecTransformer(
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes
        )
        device = f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model

    def train_fold(self, fold_num: int, val_subjects: List[int], fold_info: str):
        self.print_log(f"--- Starting Fold {fold_num}: {fold_info} ---")
        train_data, test_data, transform_fn = self.load_data(val_subjects)
        train_ds = Time2VecVarLenFeeder(train_data, transform=transform_fn)
        test_ds  = Time2VecVarLenFeeder(test_data, transform=transform_fn)
        
        # --- New: Count and output the number of labels ---
        # Assuming each sample is a tuple: (data, label, mask)
        train_labels = [label for _, label, _ in train_ds]
        val_labels = [label for _, label, _ in test_ds]
        train_label_counts = Counter(train_labels)
        val_label_counts = Counter(val_labels)
        
        self.print_log(f"Fold {fold_num}: Train Label Counts: {dict(train_label_counts)}")
        self.print_log(f"Fold {fold_num}: Validation Label Counts: {dict(val_label_counts)}")
        
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.arg.batch_size, shuffle=True,
            collate_fn=time2vec_varlen_collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.arg.batch_size, shuffle=False,
            collate_fn=time2vec_varlen_collate_fn
        )
        model = self.load_model()
        device = f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu"
        if self.arg.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        
        # Suppose label 0 is non-falls and label 1 is falls.
        # You might set weights inversely proportional to their frequencies.
        weight = torch.tensor([0.35, 0.65], dtype=torch.float32).to(device)  # adjust as needed
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        
        best_val_loss = float('inf')
        best_epoch = -1
        patience = 15
        patience_counter = 0
        best_metrics = None

        for epoch in range(self.arg.num_epoch):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for data_batch, labels_batch, mask_batch in tqdm(train_loader, ncols=80, desc=f"Fold {fold_num} Epoch {epoch}"):
                data_batch = data_batch.to(device)
                labels_batch = labels_batch.to(device)
                mask_batch = mask_batch.to(device)
                optimizer.zero_grad()
                logits = model(data_batch, mask_batch)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels_batch)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels_batch).sum().item()
                total += len(labels_batch)
            train_loss = total_loss / total if total > 0 else 0
            train_acc = correct / total if total > 0 else 0
            self.print_log(f"Fold {fold_num} Epoch {epoch}: Train Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")

            # --- Evaluate on validation set ---
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for data_batch, labels_batch, mask_batch in test_loader:
                    data_batch = data_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    logits = model(data_batch, mask_batch)
                    loss = criterion(logits, labels_batch)
                    val_loss += loss.item() * len(labels_batch)
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels_batch).sum().item()
                    val_total += len(labels_batch)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels_batch.cpu().numpy())
                    probs = torch.softmax(logits, dim=-1)  # probabilities
                    all_probs.extend(probs.cpu().numpy()[:, 1])  # probability of positive class
            val_loss = val_loss / val_total if val_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            try:
                val_roc_auc = roc_auc_score(all_labels, all_probs)
            except Exception as e:
                val_roc_auc = 0.0

            self.print_log(f"Fold {fold_num} Epoch {epoch} Validation Metrics:")
            self.print_log(f"    Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            self.print_log(f"    Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, ROC AUC: {val_roc_auc:.4f}")

            # Early stopping: monitor validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model for this fold
                best_model_state = model.state_dict()
                best_metrics = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_roc_auc": val_roc_auc
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.print_log(f"Fold {fold_num}: Early stopping triggered at epoch {epoch}")
                    break

        # Save best model weights for this fold
        save_path = os.path.join(self.work_dir, f"Fold{fold_num}_{self.arg.model_saved_name}.pth")
        torch.save(best_model_state, save_path)
        self.print_log(f"Fold {fold_num}: Best Epoch {best_epoch}, Val Loss={best_metrics['val_loss']:.4f}, "
                       f"Acc={best_metrics['val_acc']:.4f}, Precision={best_metrics['val_precision']:.4f}, "
                       f"Recall={best_metrics['val_recall']:.4f}, F1={best_metrics['val_f1']:.4f}, ROC AUC={best_metrics['val_roc_auc']:.4f}")
        self.fold_results.append(best_metrics)

    def run(self):
        # Define the five folds with validation subjects and fold info
        folds = [
            ([43, 35, 36], "Fold 1: 38.3% falls"),
            ([44, 34, 32], "Fold 2: 39.7% falls"),
            ([45, 37, 38], "Fold 3: 44.8% falls"),
            ([46, 29, 31], "Fold 4: 41.4% falls"),
            ([30, 33, 39], "Fold 5")
        ]
        for i, (val_subjects, fold_info) in enumerate(folds, start=1):
            self.train_fold(i, val_subjects, fold_info)
        
        # Average metrics over folds
        avg_loss = np.mean([fold["val_loss"] for fold in self.fold_results])
        avg_acc = np.mean([fold["val_acc"] for fold in self.fold_results])
        avg_precision = np.mean([fold["val_precision"] for fold in self.fold_results])
        avg_recall = np.mean([fold["val_recall"] for fold in self.fold_results])
        avg_f1 = np.mean([fold["val_f1"] for fold in self.fold_results])
        avg_roc_auc = np.mean([fold["val_roc_auc"] for fold in self.fold_results])
        self.print_log(f"--- Cross-Validation Summary (5 Folds) ---")
        self.print_log(f"Average Val Loss: {avg_loss:.4f}")
        self.print_log(f"Average Val Accuracy: {avg_acc:.4f}")
        self.print_log(f"Average Precision: {avg_precision:.4f}")
        self.print_log(f"Average Recall: {avg_recall:.4f}")
        self.print_log(f"Average F1: {avg_f1:.4f}")
        self.print_log(f"Average ROC AUC: {avg_roc_auc:.4f}")

###########################################################################
# Main entry
###########################################################################
def main():
    parser = get_args()
    p = parser.parse_args()
    if p.config is not None and os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        parser.set_defaults(**default_arg)
        p = parser.parse_args()
    init_seed(p.seed)
    trainer = Trainer(p)
    trainer.run()

if __name__ == "__main__":
    main()

