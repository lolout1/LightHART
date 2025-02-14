#!/usr/bin/env python
# tt2.py

import argparse
import yaml
import shutil
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import importlib  # <-- for dynamic import

# Optional import for model summary
try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

# Local imports
from utils.dataset import prepare_smartfallmm
from Feeder.teacher_varlen import TeacherVarLenFeeder, teacher_varlen_collate_fn


##############################################################################
# 1) CLI ARGS
##############################################################################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported boolean string')


def get_args():
    parser = argparse.ArgumentParser(description="Train Teacher Model (Skeleton + Accelerometer)")
    parser.add_argument('--config', default='./config/smartfallmm/teach.yaml')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epoch', type=int, default=70)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0004)
    parser.add_argument('--work-dir', type=str, default='exps/teacher_var_time')
    parser.add_argument('--device', nargs='+', type=int, default=[0])
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--print-log', type=str2bool, default=True)

    # Additional arguments to avoid "unrecognized argument" errors:
    parser.add_argument('--model-saved-name', type=str, default='teacher_best')
    parser.add_argument('--include-val', type=str2bool, default=False,
                        help="(Unused in this script, but included to match your CLI)")

    return parser


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


##############################################################################
# 2) TEACHER TRAINER CLASS
##############################################################################
class TeacherTrainer:
    def __init__(self, args):
        self.args = args
        self.work_dir = args.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_file = os.path.join(self.work_dir, 'log.txt')
        self.fold_results = []

    def print_log(self, msg):
        """Logs both to stdout and to a file if print_log is True."""
        print(msg)
        if self.args.print_log:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')

    def load_data(self, val_subj):
        """
        Build train_data from all subjects minus val_subj,
        build val_data from val_subj, using SmartFallMM pipeline.
        """
        all_subj = self.args.subjects
        train_subj = [s for s in all_subj if s not in val_subj]

        self.print_log(f"[DEBUG] Train subjects: {train_subj}, Val subjects: {val_subj}")

        # Build train
        builder_train = prepare_smartfallmm(self.args)
        builder_train.make_dataset(train_subj)
        train_data = builder_train.processed_data

        # Build val
        builder_val = prepare_smartfallmm(self.args)
        builder_val.make_dataset(val_subj)
        val_data = builder_val.processed_data

        return train_data, val_data

    def build_feeders(self, train_data, val_data):
        """
        Convert raw dictionary into TeacherVarLenFeeder from your Feeder folder.
        """
        num_joints = self.args.model_args.get('num_joints', 32)
        train_ds = TeacherVarLenFeeder(train_data, num_joints=num_joints)
        val_ds = TeacherVarLenFeeder(val_data, num_joints=num_joints)
        return train_ds, val_ds

    def build_loaders(self, train_ds, val_ds):
        """
        Create DataLoaders with teacher_varlen_collate_fn.
        """
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=teacher_varlen_collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=teacher_varlen_collate_fn
        )
        return train_loader, val_loader

    def build_model(self):
        """
        Dynamically import and instantiate the model specified by self.args.model
        with parameters in self.args.model_args.
        """
        model_path_str = self.args.model  # e.g. 'Models.master_t3.TransformerTeacher'
        module_name, class_name = model_path_str.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        ModelClass = getattr(mod, class_name)

        # Print model info
        self.print_log(f"[INFO] Using model from config: {model_path_str}")
        self.print_log(f"[INFO] Model args: {self.args.model_args}")

        # Instantiate model
        model = ModelClass(**self.args.model_args)

        device_str = f"cuda:{self.args.device[0]}" if torch.cuda.is_available() else "cpu"
        model.to(device_str)
        return model

    def train_fold(self, fold_num, val_subj, fold_info):
        """
        Single fold cross-validation training + early stopping.
        """
        self.print_log(f"--- Starting Fold {fold_num}: {fold_info} ---")

        # 1) Data
        train_data, val_data = self.load_data(val_subj)
        train_ds, val_ds = self.build_feeders(train_data, val_data)
        train_loader, val_loader = self.build_loaders(train_ds, val_ds)

        # 2) Debug label distribution
        train_labels = [lab for _, _, _, lab in train_ds]
        val_labels = [lab for _, _, _, lab in val_ds]
        self.print_log(f"Fold {fold_num} train labels: {dict(Counter(train_labels))}")
        self.print_log(f"Fold {fold_num} val labels: {dict(Counter(val_labels))}")

        # 3) Build the teacher model (dynamic import)
        model = self.build_model()
        device_str = f"cuda:{self.args.device[0]}" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        # 4) Optimizer & Criterion
        if self.args.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=self.args.base_lr,
                                   weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == 'adamw':
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

        # 5) Training loop
        for epoch in range(self.args.num_epoch):
            # ============ TRAIN =============
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels in tqdm(
                    train_loader, ncols=80, desc=f"Fold{fold_num} Ep{epoch}"
            ):
                skel_pad = skel_pad.to(device)
                accel_pad = accel_pad.to(device)
                time_pad = time_pad.to(device)
                accel_mask = accel_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(
                    skel_seq=skel_pad,
                    accel_seq=accel_pad,
                    accel_time=time_pad,
                    accel_mask=accel_mask
                )

                # Handle both dictionary returns (e.g. {'logits', 'accel_feat'}) or raw logits
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

            train_loss = total_loss / total if total > 0 else 0
            train_acc = correct / total if total > 0 else 0
            self.print_log(
                f"[Train] Fold{fold_num} Ep{epoch} => Loss={train_loss:.4f}, Acc={train_acc:.4f}"
            )

            # ============ VALIDATE ============
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels in val_loader:
                    skel_pad = skel_pad.to(device)
                    accel_pad = accel_pad.to(device)
                    time_pad = time_pad.to(device)
                    accel_mask = accel_mask.to(device)
                    labels = labels.to(device)

                    outputs = model(
                        skel_seq=skel_pad,
                        accel_seq=accel_pad,
                        accel_time=time_pad,
                        accel_mask=accel_mask
                    )

                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs

                    loss = criterion(logits, labels)
                    val_loss += loss.item() * len(labels)
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += len(labels)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    probs = torch.softmax(logits, dim=-1)
                    all_probs.extend(probs.cpu().numpy()[:, 1])

            val_loss = val_loss / val_total if val_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            val_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            try:
                val_roc = roc_auc_score(all_labels, all_probs)
            except:
                val_roc = 0.0

            self.print_log(
                f"[Val] Fold{fold_num} Ep{epoch}: "
                f"Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, ROC={val_roc:.4f}"
            )

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
                    self.print_log(f"[EarlyStop] Fold{fold_num} at epoch={epoch}")
                    break

        # 6) Save best model for this fold
        if best_model_state is not None:
            self.print_log(
                f"[Fold{fold_num}] Best epoch={best_metrics['epoch']}, "
                f"ValLoss={best_metrics['val_loss']:.4f}, Acc={best_metrics['val_acc']:.4f}"
            )

            # Re-load best weights to model (so that we have the best for saving the full model)
            model.load_state_dict(best_model_state)

            # Save weights only
            model_path_weights = os.path.join(
                self.work_dir, f"Fold{fold_num}_{self.args.model_saved_name}_weights.pth"
            )
            torch.save(model.state_dict(), model_path_weights)
            self.print_log(f"[INFO] Saved best weights => {model_path_weights}")

            # Save full model (state + architecture)
            model_path_full = os.path.join(
                self.work_dir, f"Fold{fold_num}_{self.args.model_saved_name}_full.pth"
            )
            torch.save(model, model_path_full)
            self.print_log(f"[INFO] Saved full model => {model_path_full}")

        self.fold_results.append(best_metrics)

    def run(self):
        """
        Kick off cross validation folds, print final summary,
        and optionally print model summary at the end.
        """
        # Example 5-Fold subject splits
        folds = [
            ([43, 35, 36], "Fold 1"),
            ([44, 34, 32], "Fold 2"),
            ([45, 37, 38], "Fold 3"),
            ([46, 29, 31], "Fold 4"),
            ([30, 33, 39], "Fold 5")
        ]

        for i, (val_subj, fold_info) in enumerate(folds, start=1):
            self.train_fold(i, val_subj, fold_info)

        # Summarize results
        avg_loss = np.mean([f['val_loss'] for f in self.fold_results])
        avg_acc = np.mean([f['val_acc'] for f in self.fold_results])
        avg_f1 = np.mean([f['val_f1'] for f in self.fold_results])
        avg_roc = np.mean([f['val_roc'] for f in self.fold_results])

        self.print_log("===== Cross-Validation Summary =====")
        self.print_log(
            f"Avg Val Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, F1={avg_f1:.4f}, ROC={avg_roc:.4f}"
        )

        self.print_log("===== Best Metrics per Fold =====")
        for i, metrics in enumerate(self.fold_results, start=1):
            self.print_log(
                f"Fold{i}: Best Epoch={metrics['epoch']}, "
                f"Val Loss={metrics['val_loss']:.4f}, Acc={metrics['val_acc']:.4f}, "
                f"F1={metrics['val_f1']:.4f}, ROC={metrics['val_roc']:.4f}"
            )

        # Model summary (pick the last model built: just for demonstration)
        model = self.build_model()
        total_params = sum(p.numel() for p in model.parameters())
        self.print_log("===== Model Summary (Param Count) =====")
        self.print_log(f"Model: {model.__class__.__name__}")
        self.print_log(f"Total Parameters: {total_params}")
        self.print_log(f"Saved Model Name: {self.args.model_saved_name}")

        # If torchinfo is available, print a more detailed summary
        if HAS_TORCHINFO:
            self.print_log("[INFO] Printing torchinfo.summary() for a dummy input:")
            dummy_skel = torch.randn(1, 64, self.args.model_args['num_joints'] * self.args.model_args['joint_dim'])
            dummy_accel = torch.randn(1, 128, self.args.model_args['accel_dim'])
            dummy_time = torch.randn(1, 128)
            try:
                info = summary(
                    model,
                    input_data=(dummy_skel, dummy_accel, dummy_time),
                    verbose=0
                )
                self.print_log(str(info))
            except Exception as e:
                self.print_log(f"[WARN] Could not run detailed summary: {e}")
        else:
            self.print_log("[WARN] torchinfo not installed; skipping model.summary().")


##############################################################################
# 3) main()
##############################################################################
def main():
    parser = get_args()
    args = parser.parse_args()

    # Merge config from YAML
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    # Print confirmation of config used
    print(f"[INFO] Using config file: {args.config}")
    print(f"[INFO] Model path from config: {getattr(args, 'model', None)}")
    print(f"[INFO] Model args from config: {getattr(args, 'model_args', None)}")

    # init random seed
    init_seed(args.seed)

    # run training
    trainer = TeacherTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()

