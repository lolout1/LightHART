
#!/usr/bin/env python
# File: distill_master_feat.py

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import yaml

# Local repo imports
from main import Trainer, str2bool, init_seed, import_class
from utils.dataset import prepare_smartfallmm
from Feeder.teacher_varlen import TeacherVarLenFeeder, teacher_varlen_collate_fn
from utils.loss import ExtendedDistillationLoss  # KD with feature alignment


def get_args():
    parser = argparse.ArgumentParser(description="Distillation with teacher & student feature alignment")

    # --- YAML config path ---
    parser.add_argument("--config", type=str, default="./config/smartfallmm/distill_student.yaml")

    # --- Basic data info ---
    parser.add_argument("--dataset", type=str, default="smartfallmm")
    parser.add_argument("--subjects", nargs="+", type=int, help="List of subject IDs to use.")
    parser.add_argument("--dataset-args", type=str, default="{}",
                        help="JSON for dataset args, e.g. '{\"mode\":\"variable_time\"}'")

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
    parser.add_argument("--teacher-weight-dir", type=str, default="exps/teacher_var_time",
                        help="Directory with Fold{i}_teacher_best_full.pth files")
    parser.add_argument("--student-model", type=str, default="Models.fall_time2vec_transformer_feat.FallTime2VecTransformer")
    parser.add_argument("--student-args", type=str, default="{}")

    # --- Parent Trainer model fallback ---
    parser.add_argument("--model", default=None,
                        help="We override this with a dummy so parent's Trainer won't error")
    parser.add_argument("--model-args", default=None)

    # --- Distillation / Loss ---
    parser.add_argument("--distill-args", type=str, default="{}",
                        help="JSON for ExtendedDistillationLoss, e.g. '{\"temperature\":3,\"alpha\":0.5,\"beta\":1,\"teacher_feat_dim\":128,\"student_feat_dim\":64}'")

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
    Distiller that uses ExtendedDistillationLoss for both logit KD and feature alignment.
    Teacher => returns {"logits", "accel_feat"}
    Student => returns {"logits", "feat"}
    """

    def __init__(self, arg):
        # 1) If arg.model is None, set a dummy so parent's Trainer won't blow up
        if not arg.model:
            arg.model = "torch.nn.Module"
        if not arg.model_args:
            arg.model_args = {}

        # 2) Call parent's Trainer constructor
        super().__init__(arg)
        self.arg = arg
        self.data_loader = {}
        self.model = {}

        # Histories
        self.train_loss_history = []
        self.val_loss_history = []
        self.distill_loss_history = []
        self.fold_best_metrics = {}
        self.fold_results = []

        # parse dataset_args if needed
        if isinstance(self.arg.dataset_args, str):
            try:
                self.arg.dataset_args = json.loads(self.arg.dataset_args)
            except:
                self.arg.dataset_args = {}

        # parse distill_args
        if isinstance(self.arg.distill_args, str):
            try:
                self.arg.distill_args = json.loads(self.arg.distill_args)
            except:
                self.arg.distill_args = {}

        # parse teacher_args, student_args
        try:
            self.arg.teacher_args = json.loads(self.arg.teacher_args)
        except:
            pass
        try:
            self.arg.student_args = json.loads(self.arg.student_args)
        except:
            pass

        # Build distillation loss
        # ExtendedDistillationLoss(temperature=3.0, alpha=0.5, beta=1.0, teacher_feat_dim=128, student_feat_dim=48)
        T = self.arg.distill_args.get("temperature", 3.0)
        alpha = self.arg.distill_args.get("alpha", 0.7)
        beta = self.arg.distill_args.get("beta", 0.3)
        t_dim = self.arg.distill_args.get("teacher_feat_dim", 128)
        s_dim = self.arg.distill_args.get("student_feat_dim", 48)

        self.kd_criterion = ExtendedDistillationLoss(
            temperature=T,
            alpha=alpha,
            beta=beta,
            teacher_feat_dim=t_dim,
            student_feat_dim=s_dim
        )
        # Ensure the loss module is on the same device as the models.
        device = torch.device("cuda:" + str(self.arg.device[0]) if torch.cuda.is_available() else "cpu")
        self.kd_criterion = self.kd_criterion.to(device)

    def load_teacher(self, fold_idx):
        """
        Loads teacher from e.g. Fold{fold_idx}_teacher_best_weights.pth
        NOT from the full pickled model, to avoid the TypeError.
        """
        # 1) Build teacher model
        device = torch.device("cuda:" + str(self.arg.device[0]) if torch.cuda.is_available() else "cpu")
        teacher_cls = import_class(self.arg.teacher_model)
        teacher = teacher_cls(**self.arg.teacher_args).to(device)

        # 2) Construct the path to the teacher weights
        teacher_ckpt = os.path.join(
            self.arg.teacher_weight_dir,
            f"Fold{fold_idx}_teacher_best_weights.pth"
        )
        if not os.path.exists(teacher_ckpt):
            print(f"[ERROR] Could not find teacher weights => {teacher_ckpt}")
            return None
        print(f"[FOLD {fold_idx}] Load teacher => {teacher_ckpt}")

        # 3) Load teacher state_dict
        state_dict = torch.load(teacher_ckpt, map_location=device)
        teacher.load_state_dict(state_dict, strict=True)

        # freeze teacher
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        return teacher

    def load_data(self, fold_idx, train_subjs, val_subjs):
        """
        Prepare the dataset for train & val using teacher_varlen_feeder & teacher_varlen_collate_fn
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
        A single epoch: teacher eval, student train, forward -> KD -> backward
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
                # t_out["logits"], t_out["accel_feat"]

            # Student forward
            s_out = self.model["student"](
                accel_seq=accel_pad,
                accel_mask=accel_mask,
                accel_time=time_pad
            )
            # s_out["logits"], s_out["feat"]

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
        print(f"Epoch {epoch+1} => DistillLoss={avg_loss:.4f}, StudentAcc={stu_acc:.2f}%, TeacherAcc={tea_acc:.2f}%")

    def eval_epoch(self, epoch):
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

                # teacher
                t_out = self.model["teacher"](
                    skel_seq=skel_pad,
                    accel_seq=accel_pad,
                    accel_time=time_pad,
                    accel_mask=accel_mask
                )
                # student
                s_out = self.model["student"](
                    accel_seq=accel_pad,
                    accel_mask=accel_mask,
                    accel_time=time_pad
                )

                stu_logits = s_out["logits"]
                teacher_logits = t_out["logits"]

                # CE for student's final
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

        # F1
        stu_f1 = f1_score(all_labels, all_stu_preds, average="macro") * 100 if cnt>0 else 0
        stu_prec = precision_score(all_labels, all_stu_preds, average="macro", zero_division=0)*100 if cnt>0 else 0
        stu_rec  = recall_score(all_labels, all_stu_preds, average="macro", zero_division=0)*100 if cnt>0 else 0

        self.val_loss_history.append(avg_stu_loss)
        print(f"[Val] Epoch {epoch+1} => StudentLoss={avg_stu_loss:.4f}, StudentAcc={stu_acc:.2f}%,"
              f" F1={stu_f1:.2f}%, TeacherAcc={tea_acc:.2f}%")

        return avg_stu_loss, stu_acc, stu_f1, stu_prec, stu_rec

    def generate_graph(self, fold_dir, best_metrics):
        """
        A simple function to plot training vs. val loss, saving to fold_dir.
        """
        os.makedirs(fold_dir, exist_ok=True)
        epochs = range(1, len(self.train_loss_history)+1)
        plt.figure()
        plt.plot(epochs, self.train_loss_history, label="Train (KD) Loss", marker="o")
        plt.plot(epochs, self.val_loss_history, label="Val Loss", marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Distillation Loss vs. Epoch")
        plt.legend()
        fname = (f"{self.arg.model_saved_name}_loss_{best_metrics['val_loss']:.4f}"
                 f"_acc_{best_metrics['val_acc']:.2f}_f1_{best_metrics['val_f1']:.2f}.png")
        save_path = os.path.join(fold_dir, fname)
        plt.savefig(save_path)
        plt.close()
        print(f"Graph saved to {save_path}")

    def start(self):
        print("Parameters:\n", vars(self.arg))

        # If test only
        if self.arg.phase == "test":
            # Not implemented
            print("[INFO] Test phase not implemented.")
            return

        # multi-fold
        folds = [
            ([43, 35, 36], "Fold1"),
            ([44, 34, 32], "Fold2"),
            ([45, 37, 38], "Fold3"),
            ([46, 29, 31], "Fold4"),
            ([30, 33, 39], "Fold5")
        ]
        device = torch.device(f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu")

        # We'll store final results in a DataFrame
        results = self.create_df()

        # actual cross-validation
        for i, (val_subs, foldname) in enumerate(folds, start=1):
            print(f"\n--- Distillation {foldname}, val_subs={val_subs} ---")
            self.train_loss_history = []
            self.val_loss_history = []
            self.distill_loss_history = []
            self.fold_best_metrics = {}

            # Build teacher model & load weights
            teacher_cls = import_class(self.arg.teacher_model)
            teacher_model = teacher_cls(**self.arg.teacher_args).to(device)

            teacher_ckpt = os.path.join(self.arg.teacher_weight_dir, f"Fold{i}_teacher_best_full.pth")
            print(f"[FOLD {i}] Load teacher => {teacher_ckpt}")
            teacher = self.load_teacher(i)
            if teacher is None:
                print(f"[ERROR] Could not load teacher for fold{i}, skipping.")
                continue

            self.model["teacher"] = teacher_model

            # Build student
            student_cls = import_class(self.arg.student_model)
            student_model = student_cls(**self.arg.student_args).to(device)
            self.model["student"] = student_model

            print(f"Teacher model:\n{teacher_model}")
            print(f"Student model:\n{student_model}")

            # figure out train subs from all minus val
            all_subs = self.arg.subjects
            train_subs = [s for s in all_subs if s not in val_subs]

            # load data
            self.load_data(i, train_subs, val_subs)
            self.load_optimizer()

            best_acc = 0.0
            best_metrics_for_fold = {"val_loss": 999, "val_acc": 0, "val_f1": 0, "epoch": 0}

            # training
            for ep in range(self.arg.num_epoch):
                self.train_epoch(ep)
                val_out = self.eval_epoch(ep)
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
            print(f"[FOLD {i}] best val acc={best_acc:.2f}%")

            # generate graph
            fold_dir = os.path.join(self.arg.work_dir, f"Fold{i}")
            self.generate_graph(fold_dir, best_metrics_for_fold)

            # store in results
            row = pd.Series({
                "fold": foldname,
                "val_subjects": val_subs,
                "best_epoch": best_metrics_for_fold["epoch"],
                "val_loss": round(best_metrics_for_fold["val_loss"], 4),
                "val_acc": round(best_metrics_for_fold["val_acc"], 2),
                "val_f1": round(best_metrics_for_fold["val_f1"], 2),
                "teacher_loaded": f"Fold{i}_teacher_best_full.pth",
                "lr": self.arg.base_lr,
                "batch_size": self.arg.batch_size,
                "teacher_args": str(self.arg.teacher_args),
                "student_args": str(self.arg.student_args),
            })
            results.loc[len(results)] = row

        # final CV summary
        avg_acc = results["val_acc"].mean()
        avg_f1  = results["val_f1"].mean()
        print("\n===== Cross-Validation Summary =====")
        print(f"Avg Val Acc: {avg_acc:.2f}")
        print(f"Avg Val F1 : {avg_f1:.2f}")
        results_path = os.path.join(self.arg.work_dir, "scores.csv")
        results.to_csv(results_path, index=False)
        print(f"Saved cross-validation results => {results_path}")


def main():
    parser = get_args()
    p = parser.parse_args()

    # --- 1) If there's a config file, load it into defaults ---
    if p.config and os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)

        # check keys
        key = vars(p).keys()
        for k in default_arg.keys():
            # This ensures we skip any unrecognized top-level YAML key
            if k not in key:
                print("WRONG ARG:", k)
                # If you want to allow unknown keys, remove the assert
                assert (k in key), f"Unknown YAML key: {k}"
        parser.set_defaults(**default_arg)

    # re-parse with updated defaults
    arg = parser.parse_args()

    # 2) init seed
    init_seed(arg.seed)

    # 3) run distiller
    trainer = Distiller(arg)
    trainer.start()


if __name__ == "__main__":
    main()

