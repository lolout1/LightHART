#!/usr/bin/env python
"""
Train single-input student model with x,y,z,time in shape (B,T,4).

Cross validation approach is similar to 'tt4.py' but strictly for the new single-input pipeline.
"""

import argparse
import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import importlib

# Local or custom imports
from utils.dataset import prepare_smartfallmm  # your existing function
from Feeder.single_input_feeder import SingleInputFeeder, single_input_collate_fn

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def str2bool(v):
    return v.lower() in ('true', '1', 'yes', 'y', 't')

class StudentSingleTrainer:
    def __init__(self, args):
        self.args = args
        self.work_dir = args.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_file = os.path.join(self.work_dir, 'log.txt')
        self.fold_results = []

        self.print_log(f"[INFO] Using config={args.config}")
        self.print_log(f"[INFO] Model => {args.model}")
        self.print_log(f"[INFO] Model args => {args.model_args}")

    def print_log(self, msg):
        print(msg)
        if self.args.print_log:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

    def build_model(self):
        """
        Dynamically import the single-input student model from self.args.model,
        with init kwargs from self.args.model_args.
        """
        model_path_str = self.args.model
        module_name, class_name = model_path_str.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        ModelClass = getattr(mod, class_name)

        model = ModelClass(**self.args.model_args)
        device_str = f"cuda:{self.args.device[0]}" if torch.cuda.is_available() else "cpu"
        model.to(device_str)
        return model

    def load_data(self, val_subj):
        """
        Use your existing code to build train_data from all but val_subj,
        build val_data from val_subj.
        Each data => { 'accelerometer': [...(128,4)...], 'labels': [...], ... }
        """
        all_subj = self.args.subjects
        train_subj = [s for s in all_subj if s not in val_subj]

        self.print_log(f"[DEBUG] train_subj={train_subj}, val_subj={val_subj}")

        # Build train set
        builder_train = prepare_smartfallmm(self.args)
        builder_train.make_dataset(train_subj)
        train_data = builder_train.processed_data

        # Build val set
        builder_val = prepare_smartfallmm(self.args)
        builder_val.make_dataset(val_subj)
        val_data = builder_val.processed_data

        return train_data, val_data

    def train_fold(self, fold_idx, val_subj):
        """
        Single fold => build data => SingleInputFeeder => train => val => early stop
        """
        self.print_log(f"\n--- Starting Fold {fold_idx}: val_subj={val_subj} ---")

        # Build data
        train_data, val_data = self.load_data(val_subj)
        # Feeder
        train_ds = SingleInputFeeder(train_data)
        val_ds   = SingleInputFeeder(val_data)

        # Debug label distribution
        train_labels = train_ds.labels
        val_labels   = val_ds.labels
        self.print_log(f"[Fold{fold_idx}] train label dist: {Counter(train_labels)}")
        self.print_log(f"[Fold{fold_idx}] val label dist: {Counter(val_labels)}")

        # Dataloader
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=single_input_collate_fn,
            num_workers=self.args.num_worker
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            collate_fn=single_input_collate_fn,
            num_workers=self.args.num_worker
        )

        # Build model fresh
        model = self.build_model()
        device_str = f"cuda:{self.args.device[0]}" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        # Optimizer
        opt_name = self.args.optimizer.lower()
        if opt_name == 'adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=self.args.base_lr,
                                   weight_decay=self.args.weight_decay)
        elif opt_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(),
                                    lr=self.args.base_lr,
                                    weight_decay=self.args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.args.base_lr,
                                  weight_decay=self.args.weight_decay)

        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_epoch = -1
        patience = 10
        patience_cnt = 0
        best_model_state = None
        best_metrics = None

        # Main training loop
        for epoch in range(self.args.num_epoch):
            # ---- Train
            model.train()
            total_loss = 0.0
            correct = 0
            total_samples = 0

            for data_tensor, labels_tensor, mask_tensor in tqdm(train_loader, ncols=80, desc=f"Fold{fold_idx}Ep{epoch}"):
                data_tensor = data_tensor.to(device)
                labels_tensor = labels_tensor.to(device)
                mask_tensor = mask_tensor.to(device)

                optimizer.zero_grad()
                logits = model(data_tensor, mask_tensor)
                loss = criterion(logits, labels_tensor)
                loss.backward()
                optimizer.step()

                bs = len(labels_tensor)
                total_loss += loss.item() * bs
                preds = logits.argmax(dim=-1)
                correct += (preds == labels_tensor).sum().item()
                total_samples += bs

            train_loss = total_loss / total_samples
            train_acc = correct / total_samples
            self.print_log(f"[Train] Fold{fold_idx} Ep{epoch} => loss={train_loss:.4f}, acc={train_acc:.4f}")

            # ---- Validate
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_cnt = 0

            all_labels = []
            all_preds = []
            all_probs = []

            with torch.no_grad():
                for data_tensor, labels_tensor, mask_tensor in val_loader:
                    data_tensor = data_tensor.to(device)
                    labels_tensor = labels_tensor.to(device)
                    mask_tensor = mask_tensor.to(device)

                    logits = model(data_tensor, mask_tensor)
                    loss = criterion(logits, labels_tensor)

                    bs = len(labels_tensor)
                    val_loss_sum += loss.item() * bs
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels_tensor).sum().item()
                    val_cnt += bs

                    all_labels.extend(labels_tensor.cpu().tolist())
                    all_preds.extend(preds.cpu().tolist())
                    # If you want ROC => get class1 probability
                    probs = torch.softmax(logits, dim=-1)
                    all_probs.extend(probs.cpu().numpy()[:, 1])

            val_loss = val_loss_sum / val_cnt if val_cnt>0 else 0
            val_acc = val_correct / val_cnt if val_cnt>0 else 0
            val_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            val_rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            val_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            try:
                val_roc = roc_auc_score(all_labels, all_probs)
            except:
                val_roc = 0.0

            self.print_log(f"[Val] Fold{fold_idx} Ep{epoch}: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}, roc={val_roc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_cnt = 0
                best_model_state = model.state_dict().copy()
                best_metrics = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_roc': val_roc
                }
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    self.print_log(f"[EarlyStop] at epoch={epoch}")
                    break

        # Save best weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            saved_weights = os.path.join(self.work_dir, f"Fold{fold_idx}_{self.args.model_saved_name}_weights.pth")
            torch.save(model.state_dict(), saved_weights)
            self.print_log(f"[Fold{fold_idx}] best epoch={best_metrics['epoch']}, val_loss={best_metrics['val_loss']:.4f}, acc={best_metrics['val_acc']:.4f}")
            self.print_log(f"[INFO] Saved => {saved_weights}")

        # record fold results
        self.fold_results.append(best_metrics)

    def run(self):
        """
        5-fold cross validation, or adapt as needed for your subject splits.
        """
        folds = [
            ([43, 35, 36], "Fold1"),
            ([44, 34, 32], "Fold2"),
            ([45, 37, 38], "Fold3"),
            ([46, 29, 31], "Fold4"),
            ([30, 33, 39], "Fold5")
        ]

        for i, (val_subs, foldname) in enumerate(folds, start=1):
            self.train_fold(i, val_subs)

        # Summarize
        val_losses = [f['val_loss'] for f in self.fold_results]
        val_accs   = [f['val_acc']  for f in self.fold_results]
        val_f1s    = [f['val_f1']   for f in self.fold_results]
        val_rocs   = [f['val_roc']  for f in self.fold_results]

        avg_loss = np.mean(val_losses)
        avg_acc  = np.mean(val_accs)
        avg_f1   = np.mean(val_f1s)
        avg_roc  = np.mean(val_rocs)
        self.print_log("\n===== Cross-Validation Summary =====")
        self.print_log(f"Avg val_loss={avg_loss:.4f}, Acc={avg_acc:.4f}, F1={avg_f1:.4f}, ROC={avg_roc:.4f}")
        for i, f in enumerate(self.fold_results, start=1):
            self.print_log(f"Fold{i}: epoch={f['epoch']}, loss={f['val_loss']:.4f}, acc={f['val_acc']:.4f}, f1={f['val_f1']:.4f}, roc={f['val_roc']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train single-input student model (x,y,z,time).")
    parser.add_argument('--config', type=str, default='config/smartfallmm/student_single.yaml')
    parser.add_argument('--device', nargs='+', type=int, default=[0])
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--print-log', type=str2bool, default=True)
    args = parser.parse_args()

    # load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    # init random seed
    init_seed(2)

    # run
    trainer = StudentSingleTrainer(args)
    trainer.run()

if __name__ == "__main__":
    main()
