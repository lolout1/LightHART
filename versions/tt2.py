#!/usr/bin/env python
# tt.py

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
# 2) MODEL (TransformerTeacher) - with dimension fix
##############################################################################
class TransformerTeacher(nn.Module):
    """
    Transformer-based model for fall detection using skeleton + accelerometer data.
    This version includes a fix for dimension mismatches by truncating sequences
    longer than the position-embedding length (default of 64 frames).
    """
    def __init__(
        self,
        num_joints=32,
        joint_dim=3,
        hidden_skel=128,
        accel_dim=3,
        time2vec_dim=16,
        hidden_accel=128,
        accel_heads=4,
        accel_layers=3,
        skeleton_heads=4,
        skeleton_layers=2,
        fusion_hidden=256,
        num_classes=2,
        dropout=0.3,
        dim_feedforward=256,
        **kwargs
    ):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.hidden_skel = hidden_skel

        # 1) Skeleton Transformer Branch
        self.skel_embed = nn.Linear(num_joints * joint_dim, hidden_skel)
        # Learnable positional encoding (assume max skeleton length = 64)
        self.skel_pos = nn.Parameter(torch.randn(1, 64, hidden_skel))

        skel_layer = nn.TransformerEncoderLayer(
            d_model=hidden_skel,
            nhead=skeleton_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.skel_transformer = nn.TransformerEncoder(skel_layer, num_layers=skeleton_layers)

        # 2) Accelerometer Branch
        from utils.processor.base import Time2Vec  # your existing implementation
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        self.accel_proj = nn.Linear(accel_dim + time2vec_dim, hidden_accel)
        accel_layer = nn.TransformerEncoderLayer(
            d_model=hidden_accel,
            nhead=accel_heads,
            dim_feedforward=dim_feedforward * 2,  # more capacity
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.accel_transformer = nn.TransformerEncoder(accel_layer, num_layers=accel_layers)

        # 3) Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_skel + hidden_accel, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, skel_seq, accel_seq, accel_time, accel_mask=None):
        """
        Forward pass:
          skel_seq: (B, T_s, num_joints*joint_dim)
          accel_seq: (B, T_a, accel_dim)
          accel_time: (B, T_a) time stamps for each accelerometer sample
          accel_mask: (B, T_a) boolean mask for padded positions (True=padded)

        Dimension-fix: if T_s > 64, we truncate to 64 frames to match self.skel_pos.
        """
        B, Ts, _ = skel_seq.shape

        # ------------- Debug prints -------------
        # You may comment these out as needed
        # print(f"[DEBUG] Skeleton seq shape before clamp = {skel_seq.shape}")

        # Truncate skeleton sequence if needed
        max_skel_len = self.skel_pos.shape[1]  # 64 by default
        if Ts > max_skel_len:
            skel_seq = skel_seq[:, :max_skel_len, :]
            Ts = max_skel_len
            # print(f"[DEBUG] Truncated skeleton seq to shape = {skel_seq.shape}")

        # Skeleton branch
        skel_emb = self.skel_embed(skel_seq) + self.skel_pos[:, :Ts, :]
        skel_feat = self.skel_transformer(skel_emb).mean(dim=1)

        # Accelerometer branch
        B, Ta, _ = accel_seq.shape
        # print(f"[DEBUG] Accelerometer seq shape = {accel_seq.shape}")
        # Flatten times to pass into Time2Vec
        t_emb = self.time2vec(accel_time.view(B * Ta, 1)).view(B, Ta, -1)
        accel_in = torch.nn.functional.gelu(
            self.accel_proj(torch.cat([accel_seq, t_emb], dim=-1))
        )
        accel_feat_seq = self.accel_transformer(accel_in, src_key_padding_mask=accel_mask)

        # compute masked mean along time dimension
        accel_feat = masked_mean(accel_feat_seq, accel_mask)

        # Fusion
        fused = self.fusion(torch.cat([skel_feat, accel_feat], dim=-1))
        logits = self.classifier(fused)
        return logits


def masked_mean(features, mask):
    """
    Computes mean over time dimension for each sample, ignoring padded positions.
    mask: boolean tensor with True for padded positions => these are excluded.
    """
    if mask is not None:
        valid = ~mask  # valid = where mask is False
        features = features * valid.unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return features.sum(dim=1) / denom
    else:
        return features.mean(dim=1)


##############################################################################
# 3) TEACHER TRAINER CLASS
##############################################################################
class TeacherTrainer:
    def __init__(self, args):
        self.args = args
        self.work_dir = args.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_file = os.path.join(self.work_dir, 'log.txt')
        # Clear log if you prefer fresh each run:
        # open(self.log_file, 'w').close()
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
        Instantiates TransformerTeacher with config parameters in self.args.model_args.
        """
        from Models.master_t32 import TransformerTeacher  # or use the local class directly
        model = TransformerTeacher(**self.args.model_args)
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

        # 3) Build the teacher model
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
                logits = model(
                    skel_seq=skel_pad,
                    accel_seq=accel_pad,
                    accel_time=time_pad,
                    accel_mask=accel_mask
                )
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

                    logits = model(
                        skel_seq=skel_pad,
                        accel_seq=accel_pad,
                        accel_time=time_pad,
                        accel_mask=accel_mask
                    )
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
        # If you want the summary of the best model in each fold, you'd do it inside train_fold.
        model = self.build_model()
        total_params = sum(p.numel() for p in model.parameters())
        self.print_log("===== Model Summary (Param Count) =====")
        self.print_log(f"Model: {model.__class__.__name__}")
        self.print_log(f"Total Parameters: {total_params}")
        self.print_log(f"Saved Model Name: {self.args.model_saved_name}")

        # If torchinfo is available, print a more detailed summary
        if HAS_TORCHINFO:
            self.print_log("[INFO] Printing torchinfo.summary() for a dummy input:")
            # Suppose the default max skeleton length is 64, and let's guess
            # an accelerometer length ~128, purely for example:
            dummy_skel = torch.randn(1, 64, self.args.model_args['num_joints'] * self.args.model_args['joint_dim'])
            dummy_accel = torch.randn(1, 128, self.args.model_args['accel_dim'])
            dummy_time = torch.randn(1, 128)  # sample timestamps
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
# 4) main()
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

    # init random seed
    init_seed(args.seed)

    # run training
    trainer = TeacherTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
