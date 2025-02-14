#!/usr/bin/env python
# File: distill_master_feat.py
"""
This script now does a two-step approach:
  1) Train the student alone (no distillation).
  2) Re-initialize and train with teacher distillation.

We keep all existing logic for distillation, 5-fold cross validation, etc.
We only add:
  - No-distill pass
  - More descriptive log prints
  - A final comparison of no-distill vs. distill results
  - Additional output of all metrics (val loss, val acc, val f1, val recall, val precision)
    after each fold and averaged across folds.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# Local repo imports
from main import Trainer, str2bool, init_seed, import_class
from utils.dataset import prepare_smartfallmm
from Feeder.teacher_varlen import TeacherVarLenFeeder, teacher_varlen_collate_fn
from utils.loss import ExtendedDistillationLoss  # KD with feature alignment


def get_args():
    parser = argparse.ArgumentParser(
        description="Distillation script that first trains student alone, then distills from teacher."
    )

    # --- YAML config path ---
    parser.add_argument("--config", type=str, default="./config/smartfallmm/distill_student.yaml")

    # --- Basic data info ---
    parser.add_argument("--dataset", type=str, default="smartfallmm")
    parser.add_argument("--subjects", nargs="+", type=int, help="List of subject IDs to use.")
    parser.add_argument("--dataset-args", type=str, default="{}",
                        help='JSON for dataset args, e.g. \'{"mode":"variable_time"}\'')

    # --- Training Hyperparams ---
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-epoch", type=int, default=50)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--base-lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0004)

    # --- Teacher / Student ---
    parser.add_argument("--teacher-model", type=str, default="Models.master_t3_feat.TransformerTeacher")
    parser.add_argument("--teacher-args", type=str, default="{}")
    parser.add_argument("--teacher-weight-dir", type=str, default="exps/teacher_var_time3",
                        help="Directory with Fold{i}_teacher_best_weights.pth or _full.pth")
    parser.add_argument("--student-model", type=str, default="Models.fall_time2vec_transformer_feat.FallTime2VecTransformer")
    parser.add_argument("--student-args", type=str, default="{}")

    # --- Parent Trainer model fallback ---
    parser.add_argument("--model", default=None,
                        help="We override this with a dummy so parent's Trainer won't error")
    parser.add_argument("--model-args", default=None)

    # --- Distillation / Loss ---
    parser.add_argument("--distill-args", type=str, default="{}",
                        help='JSON for ExtendedDistillationLoss, e.g. \'{"temperature":3,"alpha":0.5,"beta":1,"teacher_feat_dim":128,"student_feat_dim":64}\'')
    # --- Phase / device / etc. ---
    parser.add_argument("--include-val", type=str2bool, default=True)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--device", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--work-dir", type=str, default="exps/distilled_student")
    parser.add_argument("--print-log", type=str2bool, default=True)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--result-file", type=str, default=None)

    # --- Save name for student model ---
    parser.add_argument("--model-saved-name", type=str, default="student_distilled")

    return parser


class Distiller(Trainer):
    """
    Original distillation logic from your script, now also includes a 'no-distill' pass.
    We do:

      1) Student-only training (5-fold).
      2) Student-with-distillation training (5-fold).
      3) Final comparison.

    The teacher uses skeleton+acc, the student only uses watch (acc).
    """

    def __init__(self, arg):
        # 1) If arg.model is None, set a dummy so parent's Trainer won't blow up
        if not arg.model:
            arg.model = "torch.nn.Module"
        if not arg.model_args:
            arg.model_args = {}

        super().__init__(arg)
        self.arg = arg
        self.data_loader = {}
        self.model = {}

        # Override create_df to ensure we have our expected columns
        self.results_df_columns = ["phase", "fold", "val_subjects", "best_epoch",
                                   "val_loss", "val_acc", "val_f1", "val_prec", "val_rec",
                                   "lr", "batch_size", "teacher_args", "student_args"]

        # parse possible JSON strings
        self._parse_json_args()

        # Build the distillation loss
        T = self.arg.distill_args.get("temperature", 3.0)
        alpha = self.arg.distill_args.get("alpha", 0.5)
        beta = self.arg.distill_args.get("beta", 0.5)
        t_dim = self.arg.distill_args.get("teacher_feat_dim", 128)
        s_dim = self.arg.distill_args.get("student_feat_dim", 48)
        self.kd_criterion = ExtendedDistillationLoss(
            temperature=T,
            alpha=alpha,
            beta=beta,
            teacher_feat_dim=t_dim,
            student_feat_dim=s_dim
        )
        device = torch.device("cuda:" + str(self.arg.device[0]) if torch.cuda.is_available() else "cpu")
        self.kd_criterion = self.kd_criterion.to(device)

        # Some placeholders
        self.no_distill_results = None
        self.distill_results = None

    def create_df(self):
        """
        Create an empty DataFrame with the expected column names.
        """
        return pd.DataFrame(columns=self.results_df_columns)

    def _parse_json_args(self):
        """Helper to parse the JSON fields in self.arg (teacher_args, student_args, distill_args)."""
        # dataset_args
        if isinstance(self.arg.dataset_args, str):
            try:
                self.arg.dataset_args = json.loads(self.arg.dataset_args)
            except:
                self.arg.dataset_args = {}
        # distill_args
        if isinstance(self.arg.distill_args, str):
            try:
                self.arg.distill_args = json.loads(self.arg.distill_args)
            except:
                self.arg.distill_args = {}
        # teacher_args
        if isinstance(self.arg.teacher_args, str):
            try:
                self.arg.teacher_args = json.loads(self.arg.teacher_args)
            except:
                pass
        # student_args
        if isinstance(self.arg.student_args, str):
            try:
                self.arg.student_args = json.loads(self.arg.student_args)
            except:
                pass

    ########################################################################
    # PART A: Student-Only Training (No Distillation)
    ########################################################################
    def train_student_only_5fold(self):
        """
        5-fold cross validation, but NO teacher usage, no KD.
        We'll store the final results in self.no_distill_results (DataFrame).
        """
        print("\n\n========== PHASE A: Student-Only Training (No Distillation) ==========")
        device = torch.device(f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu")

        folds = [
            ([43, 35, 36], "Fold1"),
            ([44, 34, 32], "Fold2"),
            ([45, 37, 38], "Fold3"),
            ([46, 29, 31], "Fold4"),
            ([30, 33, 39], "Fold5")
        ]

        # We'll store final results in a DataFrame
        results = self.create_df()

        # actual cross-validation
        for i, (val_subs, foldname) in enumerate(folds, start=1):
            print(f"\n--- Student-Only {foldname}, val_subs={val_subs} ---")

            # We'll track train/val losses for plotting, but not do KD
            self.train_loss_history = []
            self.val_loss_history = []
            self.distill_loss_history = []  # not used here

            # Build student fresh
            student_cls = import_class(self.arg.student_model)
            student_model = student_cls(**self.arg.student_args).to(device)
            self.model["student"] = student_model

            print(f"[NoDistill] Building student model => {student_cls.__name__}\n{student_model}")

            # figure out train subs from all minus val
            all_subs = self.arg.subjects
            train_subs = [s for s in all_subs if s not in val_subs]

            # load data => ignoring teacher or skeleton usage
            builder = prepare_smartfallmm(self.arg)
            builder.make_dataset(train_subs)
            train_data = builder.processed_data
            train_ds = TeacherVarLenFeeder(train_data, num_joints=32)  # or your default
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                collate_fn=teacher_varlen_collate_fn
            )
            self.data_loader["train"] = train_loader

            if self.arg.include_val:
                builder.make_dataset(val_subs)
                val_data = builder.processed_data
                val_ds = TeacherVarLenFeeder(val_data, num_joints=32)
                val_loader = torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=self.arg.val_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    collate_fn=teacher_varlen_collate_fn
                )
                self.data_loader["val"] = val_loader

            # load optimizer
            self.load_optimizer_no_distill()

            best_acc = 0.0
            best_metrics_for_fold = {"val_loss": 999, "val_acc": 0, "val_f1": 0, "val_prec": 0, "val_rec": 0, "epoch": 0}

            # training
            for ep in range(self.arg.num_epoch):
                self.train_epoch_no_distill(ep)
                val_out = self.eval_epoch_no_distill(ep)
                if val_out is not None:
                    val_loss, val_acc, val_f1, val_prec, val_rec = val_out
                    # check if improved
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_metrics_for_fold = {
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_f1": val_f1,
                            "val_prec": val_prec,
                            "val_rec": val_rec,
                            "epoch": ep+1
                        }
            print(f"[NoDistill:Fold {i}] Best Metrics => Epoch: {best_metrics_for_fold['epoch']}, Loss: {best_metrics_for_fold['val_loss']:.4f}, Acc: {best_metrics_for_fold['val_acc']:.2f}%, F1: {best_metrics_for_fold['val_f1']:.2f}%, Precision: {best_metrics_for_fold['val_prec']:.2f}%, Recall: {best_metrics_for_fold['val_rec']:.2f}%")

            # store in results
            row = pd.Series({
                "phase": "NoDistill",
                "fold": foldname,
                "val_subjects": val_subs,
                "best_epoch": best_metrics_for_fold["epoch"],
                "val_loss": round(best_metrics_for_fold["val_loss"], 4),
                "val_acc": round(best_metrics_for_fold["val_acc"], 2),
                "val_f1": round(best_metrics_for_fold["val_f1"], 2),
                "val_prec": round(best_metrics_for_fold["val_prec"], 2),
                "val_rec": round(best_metrics_for_fold["val_rec"], 2),
                "lr": self.arg.base_lr,
                "batch_size": self.arg.batch_size,
                "student_args": str(self.arg.student_args),
                "teacher_args": ""  # not applicable for no-distill
            })
            results.loc[len(results)] = row

        self.no_distill_results = results

    def load_optimizer_no_distill(self):
        """
        Just an optimizer for student only (like normal).
        """
        opt_name = self.arg.optimizer.lower()
        device = torch.device("cuda:" + str(self.arg.device[0]) if torch.cuda.is_available() else "cpu")

        if opt_name == "adam":
            self.optimizer_no_distill = optim.Adam(
                self.model["student"].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif opt_name == "adamw":
            self.optimizer_no_distill = optim.AdamW(
                self.model["student"].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def train_epoch_no_distill(self, epoch):
        """
        Single epoch of *student only* training (no teacher, no KD).
        """
        device = torch.device(f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu")
        self.model["student"].train()
        loader = self.data_loader["train"]

        ce_loss = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct_student = 0
        total_cnt = 0

        for batch in tqdm(loader, ncols=80, desc=f"[NoDistill] Ep{epoch}"):
            skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels = batch
            # Student only uses accel
            accel_pad = accel_pad.to(device).float()
            accel_mask = accel_mask.to(device) if accel_mask is not None else None
            labels = labels.to(device)

            self.optimizer_no_distill.zero_grad()

            s_out = self.model["student"](
                accel_seq=accel_pad,
                accel_mask=accel_mask,
                accel_time=time_pad.to(device).float()
            )
            logits = s_out["logits"]  # student output

            loss = ce_loss(logits, labels)
            loss.backward()
            self.optimizer_no_distill.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            correct_student += (preds == labels).sum().item()
            total_cnt += len(labels)

        avg_loss = total_loss / total_cnt if total_cnt > 0 else 0
        stu_acc = 100.0 * correct_student / total_cnt if total_cnt > 0 else 0

        self.train_loss_history.append(avg_loss)
        print(f"[NoDistill] Epoch {epoch+1} => StudentLoss={avg_loss:.4f}, StudentAcc={stu_acc:.2f}%")

    def eval_epoch_no_distill(self, epoch):
        if "val" not in self.data_loader:
            return None

        device = torch.device(f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu")
        loader = self.data_loader["val"]
        self.model["student"].eval()

        ce_loss = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct_stu = 0
        cnt = 0

        all_labels = []
        all_stu_preds = []

        with torch.no_grad():
            for batch in loader:
                skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels = batch
                accel_pad = accel_pad.to(device).float()
                accel_mask = accel_mask.to(device) if accel_mask is not None else None
                labels = labels.to(device)

                s_out = self.model["student"](
                    accel_seq=accel_pad,
                    accel_mask=accel_mask,
                    accel_time=time_pad.to(device).float()
                )
                logits = s_out["logits"]
                loss = ce_loss(logits, labels)
                total_loss += loss.item() * len(labels)

                preds = logits.argmax(dim=-1)
                correct_stu += (preds == labels).sum().item()
                cnt += len(labels)

                all_labels.extend(labels.cpu().tolist())
                all_stu_preds.extend(preds.cpu().tolist())

        if cnt > 0:
            avg_loss = total_loss / cnt
            stu_acc = 100.0 * correct_stu / cnt
        else:
            avg_loss = 0
            stu_acc = 0

        stu_f1 = f1_score(all_labels, all_stu_preds, average="macro") * 100 if cnt > 0 else 0
        stu_prec = precision_score(all_labels, all_stu_preds, average="macro", zero_division=0)*100 if cnt > 0 else 0
        stu_rec  = recall_score(all_labels, all_stu_preds, average="macro", zero_division=0)*100 if cnt > 0 else 0

        self.val_loss_history.append(avg_loss)
        print(f"[NoDistill:Val] Epoch {epoch+1} => StudentLoss={avg_loss:.4f}, StudentAcc={stu_acc:.2f}%, F1={stu_f1:.2f}%, Precision={stu_prec:.2f}%, Recall={stu_rec:.2f}%")
        return avg_loss, stu_acc, stu_f1, stu_prec, stu_rec

    ########################################################################
    # PART B: Distillation Training (Original Code)
    ########################################################################
    def load_teacher(self, fold_idx):
        """
        Overriding parent's stub. We do exactly what the existing script does:
        Load teacher from e.g. Fold{fold_idx}_teacher_best_weights.pth
        """
        device = torch.device("cuda:" + str(self.arg.device[0]) if torch.cuda.is_available() else "cpu")
        teacher_cls = import_class(self.arg.teacher_model)
        teacher = teacher_cls(**self.arg.teacher_args).to(device)

        teacher_ckpt = os.path.join(
            self.arg.teacher_weight_dir,
            f"Fold{fold_idx}_teacher_best_weights.pth"
        )
        if not os.path.exists(teacher_ckpt):
            print(f"[ERROR] Could not find teacher weights => {teacher_ckpt}")
            return None
        print(f"[FOLD {fold_idx}] Load teacher => {teacher_ckpt}")

        state_dict = torch.load(teacher_ckpt, map_location=device)
        teacher.load_state_dict(state_dict, strict=True)

        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        return teacher

    def load_data_distill(self, fold_idx, train_subjs, val_subjs):
        """
        Same approach as original, but for KD. We do builder + teacher_varlen_feeder.
        """
        builder = prepare_smartfallmm(self.arg)

        # train
        builder.make_dataset(train_subjs)
        train_data = builder.processed_data
        train_ds = TeacherVarLenFeeder(train_data, num_joints=self.arg.teacher_args.get("num_joints", 32))
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            collate_fn=teacher_varlen_collate_fn
        )
        self.data_loader["train"] = train_loader

        # val
        if self.arg.include_val:
            builder.make_dataset(val_subjs)
            val_data = builder.processed_data
            val_ds = TeacherVarLenFeeder(val_data, num_joints=self.arg.teacher_args.get("num_joints", 32))
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=self.arg.val_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                collate_fn=teacher_varlen_collate_fn
            )
            self.data_loader["val"] = val_loader

    def load_optimizer(self):
        """
        For distillation approach (original script).
        """
        opt_name = self.arg.optimizer.lower()
        if opt_name == "adam":
            self.optimizer = optim.Adam(
                self.model["student"].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif opt_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model["student"].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def train_epoch(self, epoch):
        """
        Distillation epoch => teacher is used, kd_criterion is used.
        (We keep the name 'train_epoch' to keep code unchanged from the original.)
        """
        device = torch.device(f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu")
        self.model["teacher"].eval()
        self.model["student"].train()

        loader = self.data_loader["train"]
        total_loss = 0.0
        correct_student = 0
        correct_teacher = 0
        total_cnt = 0

        for batch in tqdm(loader, ncols=80):
            skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels = batch
            skel_pad   = skel_pad.to(device).float()
            accel_pad  = accel_pad.to(device).float()
            time_pad   = time_pad.to(device).float()
            if skel_mask is not None:
                skel_mask  = skel_mask.to(device)
            if accel_mask is not None:
                accel_mask = accel_mask.to(device)
            labels     = labels.to(device)

            # Teacher forward
            with torch.no_grad():
                t_out = self.model["teacher"](
                    skel_seq=skel_pad,
                    accel_seq=accel_pad,
                    accel_time=time_pad,
                    accel_mask=accel_mask
                )

            # Student forward
            s_out = self.model["student"](
                accel_seq=accel_pad,
                accel_mask=accel_mask,
                accel_time=time_pad
            )

            # Distillation loss
            loss = self.kd_criterion(
                student_logits=s_out["logits"],
                teacher_logits=t_out["logits"],
                student_feat=s_out["feat"],
                teacher_feat=t_out["accel_feat"],
                labels=labels
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(labels)

            # track accuracy
            stu_pred = torch.argmax(s_out["logits"], dim=-1)
            tea_pred = torch.argmax(t_out["logits"], dim=-1)
            correct_student += (stu_pred == labels).sum().item()
            correct_teacher += (tea_pred == labels).sum().item()
            total_cnt += len(labels)

        avg_loss = total_loss / total_cnt
        stu_acc  = 100.0 * correct_student / total_cnt
        tea_acc  = 100.0 * correct_teacher / total_cnt

        self.train_loss_history.append(avg_loss)
        self.distill_loss_history.append(avg_loss)
        print(f"[Distill] Epoch {epoch+1} => DistillLoss={avg_loss:.4f}, StudentAcc={stu_acc:.2f}%, TeacherAcc={tea_acc:.2f}%")

    def eval_epoch(self, epoch):
        """
        Distillation's eval, matching your original script's approach.
        """
        if "val" not in self.data_loader:
            return None

        device = torch.device(f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu")
        loader = self.data_loader["val"]
        self.model["teacher"].eval()
        self.model["student"].eval()

        ce_loss = nn.CrossEntropyLoss()
        total_loss_stu = 0.0
        correct_stu = 0
        correct_tea = 0
        cnt = 0

        all_labels = []
        all_stu_preds = []
        all_tea_preds = []

        with torch.no_grad():
            for batch in loader:
                skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels = batch
                skel_pad   = skel_pad.to(device).float()
                accel_pad  = accel_pad.to(device).float()
                time_pad   = time_pad.to(device).float()
                if skel_mask is not None:
                    skel_mask  = skel_mask.to(device)
                if accel_mask is not None:
                    accel_mask = accel_mask.to(device)
                labels     = labels.to(device)

                t_out = self.model["teacher"](
                    skel_seq=skel_pad,
                    accel_seq=accel_pad,
                    accel_time=time_pad,
                    accel_mask=accel_mask
                )
                s_out = self.model["student"](
                    accel_seq=accel_pad,
                    accel_mask=accel_mask,
                    accel_time=time_pad
                )

                stu_logits = s_out["logits"]
                teacher_logits = t_out["logits"]
                loss_stu = ce_loss(stu_logits, labels)
                total_loss_stu += loss_stu.item() * len(labels)

                stu_pred = stu_logits.argmax(dim=-1)
                tea_pred = teacher_logits.argmax(dim=-1)
                correct_stu += (stu_pred == labels).sum().item()
                correct_tea += (tea_pred == labels).sum().item()
                cnt += len(labels)

                all_labels.extend(labels.cpu().tolist())
                all_stu_preds.extend(stu_pred.cpu().tolist())
                all_tea_preds.extend(tea_pred.cpu().tolist())

        if cnt > 0:
            avg_stu_loss = total_loss_stu / cnt
            stu_acc = 100.0 * correct_stu / cnt
            tea_acc = 100.0 * correct_tea / cnt
        else:
            avg_stu_loss = 0
            stu_acc = 0
            tea_acc = 0

        stu_f1 = f1_score(all_labels, all_stu_preds, average="macro") * 100 if cnt>0 else 0
        stu_prec = precision_score(all_labels, all_stu_preds, average="macro", zero_division=0)*100 if cnt>0 else 0
        stu_rec  = recall_score(all_labels, all_stu_preds, average="macro", zero_division=0)*100 if cnt>0 else 0

        self.val_loss_history.append(avg_stu_loss)
        print(f"[Distill:Val] Epoch {epoch+1} => StudentLoss={avg_stu_loss:.4f}, StudentAcc={stu_acc:.2f}%," +
              f" F1={stu_f1:.2f}%, Precision={stu_prec:.2f}%, Recall={stu_rec:.2f}%, TeacherAcc={tea_acc:.2f}%")

        return avg_stu_loss, stu_acc, stu_f1, stu_prec, stu_rec

    def start_distillation_5fold(self):
        """
        The original 5-fold procedure for distillation. We'll store results in self.distill_results.
        """
        print("\n\n========== PHASE B: Distillation Training (Using Teacher) ==========")
        folds = [
            ([43, 35, 36], "Fold1"),
            ([44, 34, 32], "Fold2"),
            ([45, 37, 38], "Fold3"),
            ([46, 29, 31], "Fold4"),
            ([30, 33, 39], "Fold5")
        ]
        device = torch.device(f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu")

        results = self.create_df()

        for i, (val_subs, foldname) in enumerate(folds, start=1):
            print(f"\n--- Distillation {foldname}, val_subs={val_subs} ---")
            self.train_loss_history = []
            self.val_loss_history = []
            self.distill_loss_history = []
            self.fold_best_metrics = {}

            # Load teacher
            teacher = self.load_teacher(i)
            if teacher is None:
                print(f"[ERROR] Could not load teacher for fold{i}, skipping.")
                continue
            self.model["teacher"] = teacher

            # Build student from scratch
            student_cls = import_class(self.arg.student_model)
            student_model = student_cls(**self.arg.student_args).to(device)
            self.model["student"] = student_model
            print(f"[Distill] Building teacher+student =>\nTeacher:\n{teacher}\n\nStudent:\n{student_model}")

            # figure out train subs from all minus val
            all_subs = self.arg.subjects
            train_subs = [s for s in all_subs if s not in val_subs]

            # load data
            self.load_data_distill(i, train_subs, val_subs)
            self.load_optimizer()

            best_acc = 0.0
            best_metrics_for_fold = {"val_loss": 999, "val_acc": 0, "val_f1": 0, "val_prec": 0, "val_rec": 0, "epoch": 0}

            # training loop
            for ep in range(self.arg.num_epoch):
                self.train_epoch(ep)
                val_out = self.eval_epoch(ep)
                if val_out is not None:
                    val_loss, val_acc, val_f1, val_prec, val_rec = val_out
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_metrics_for_fold = {
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_f1": val_f1,
                            "val_prec": val_prec,
                            "val_rec": val_rec,
                            "epoch": ep+1
                        }
            print(f"[Distill:Fold {i}] Best Metrics => Epoch: {best_metrics_for_fold['epoch']}, Loss: {best_metrics_for_fold['val_loss']:.4f}, Acc: {best_metrics_for_fold['val_acc']:.2f}%, F1: {best_metrics_for_fold['val_f1']:.2f}%, Precision: {best_metrics_for_fold['val_prec']:.2f}%, Recall: {best_metrics_for_fold['val_rec']:.2f}%")

            row = pd.Series({
                "phase": "Distill",
                "fold": foldname,
                "val_subjects": val_subs,
                "best_epoch": best_metrics_for_fold["epoch"],
                "val_loss": round(best_metrics_for_fold["val_loss"], 4),
                "val_acc": round(best_metrics_for_fold["val_acc"], 2),
                "val_f1": round(best_metrics_for_fold["val_f1"], 2),
                "val_prec": round(best_metrics_for_fold["val_prec"], 2),
                "val_rec": round(best_metrics_for_fold["val_rec"], 2),
                "lr": self.arg.base_lr,
                "batch_size": self.arg.batch_size,
                "teacher_args": str(self.arg.teacher_args),
                "student_args": str(self.arg.student_args)
            })
            results.loc[len(results)] = row

        self.distill_results = results

    ########################################################################
    # MAIN START: Combine both passes + final comparison
    ########################################################################
    def start(self):
        print("Parameters:\n", vars(self.arg))

        if self.arg.phase == "test":
            print("[INFO] Test phase not implemented.")
            return

        # 1) Student-Only (no distill) training
        self.train_student_only_5fold()

        # 2) Distillation training (teacher+student)
        self.start_distillation_5fold()

        # 3) Final summary => compare no-distill vs. distill
        print("\n\n===== FINAL COMPARISON: NO-DISTILL vs. DISTILL =====\n")
        if self.no_distill_results is None:
            print("[WARN] No-distill results missing!")
            return
        if self.distill_results is None:
            print("[WARN] Distill results missing!")
            return

        # Merge them or print side by side
        noDist_df = self.no_distill_results.copy()
        dist_df   = self.distill_results.copy()

        # For each fold, let's print them side by side:
        print("No-Distill Results:\n", noDist_df)
        print("Distill Results:\n", dist_df)

        # compute average across folds for all metrics
        nd_mean_loss = noDist_df["val_loss"].mean()
        nd_mean_acc = noDist_df["val_acc"].mean()
        nd_mean_f1  = noDist_df["val_f1"].mean()
        nd_mean_prec = noDist_df["val_prec"].mean()
        nd_mean_rec  = noDist_df["val_rec"].mean()

        dt_mean_loss = dist_df["val_loss"].mean()
        dt_mean_acc = dist_df["val_acc"].mean()
        dt_mean_f1  = dist_df["val_f1"].mean()
        dt_mean_prec = dist_df["val_prec"].mean()
        dt_mean_rec  = dist_df["val_rec"].mean()

        print(f"\n--- 5-Fold Averages ---")
        print(f"No-Distill => Loss={nd_mean_loss:.4f}, Acc={nd_mean_acc:.2f}%, F1={nd_mean_f1:.2f}%, Precision={nd_mean_prec:.2f}%, Recall={nd_mean_rec:.2f}%")
        print(f"Distill   => Loss={dt_mean_loss:.4f}, Acc={dt_mean_acc:.2f}%, F1={dt_mean_f1:.2f}%, Precision={dt_mean_prec:.2f}%, Recall={dt_mean_rec:.2f}%")

        # If you want to save them
        os.makedirs(self.arg.work_dir, exist_ok=True)
        no_dist_path = os.path.join(self.arg.work_dir, "scores_no_distill.csv")
        dist_path    = os.path.join(self.arg.work_dir, "scores_distill.csv")
        noDist_df.to_csv(no_dist_path, index=False)
        dist_df.to_csv(dist_path, index=False)

        print(f"\n[INFO] Saved no-distill results => {no_dist_path}")
        print(f"[INFO] Saved distill results => {dist_path}")
        print("\nDone! :)\n")


def main():
    parser = get_args()
    p = parser.parse_args()

    # 1) If there's a config file, load it into defaults
    if p.config and os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG:", k)
                # if you want to skip unknown keys, remove or adapt this
                assert (k in key), f"Unknown YAML key: {k}"
        parser.set_defaults(**default_arg)

    # re-parse with updated defaults
    arg = parser.parse_args()

    # 2) init seed
    init_seed(arg.seed)

    # 3) run
    trainer = Distiller(arg)
    trainer.start()


if __name__ == "__main__":
    main()

