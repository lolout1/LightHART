#!/usr/bin/env python3
from typing import List
import argparse
import os
import random
import shutil
import sys
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Local imports
from utils.dataset import prepare_smartfallmm
from utils.loader import DatasetBuilder, align_sequence
from Feeder.time2vec_varlen import Time2VecVarLenFeeder, time2vec_varlen_collate_fn

# Import teacher and student models from your repository.
from Models.teach import TeacherModel
from Models.s import StudentModel

###########################################################################
# Argument Parsing
###########################################################################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported boolean string')

def get_args():
    parser = argparse.ArgumentParser(description='Fall detection with variable_time + Time2Vec (Cross-Validation)')
    parser.add_argument('--config', default='./config/smartfallmm/teach.yaml')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epoch', type=int, default=70)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0004)
    parser.add_argument('--phase', type=str, default='train_teacher', help="Options: train_teacher, train_student")
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--seed', type=int, default=42)
    # Additional parameters
    parser.add_argument('--subjects', nargs='+', type=int, default=[29,30,31,32,33,44,45,46,34,35,36,37,38,39,43])
    parser.add_argument('--work-dir', type=str, default='exps/teacher_time2vec_fall')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--dataset_args', default={}, type=dict)
    parser.add_argument('--model_args', default={}, type=dict)
    parser.add_argument('--student_model_args', default={}, type=dict)
    parser.add_argument('--kd_temperature', type=float, default=4.0)
    parser.add_argument('--kd_alpha', type=float, default=0.5)
    return parser

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

###########################################################################
# Trainer Class (5-Fold Cross-Validation)
###########################################################################
class Trainer():
    def __init__(self, arg):
        self.arg = arg
        self.work_dir = arg.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.print_log(f"Args: {vars(arg)}")
        if arg.config and os.path.exists(arg.config):
            shutil.copy(arg.config, os.path.join(self.work_dir, os.path.basename(arg.config)))
        self.fold_results = []  # to hold fold-level metrics

    def print_log(self, msg):
        print(msg)
        if self.arg.print_log:
            log_path = os.path.join(self.work_dir, 'log.txt')
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

    def load_data(self, val_subjects: List[int]):
        full_subjects = self.arg.subjects
        train_subjects = [s for s in full_subjects if s not in val_subjects]
        self.print_log(f"Train Subjects={train_subjects}, Validation Subjects={val_subjects}")

        builder_train = prepare_smartfallmm(self.arg)
        builder_train.make_dataset(train_subjects)
        train_data = builder_train.processed_data

        all_arrays = train_data.get('accelerometer', [])
        if len(all_arrays) > 0 and isinstance(all_arrays[0], np.ndarray):
            stacked = np.concatenate(all_arrays, axis=0)
            scaler = StandardScaler()
            scaler.fit(stacked)
            def transform_fn(x):
                return scaler.transform(x)
        else:
            def transform_fn(x):
                return x

        builder_val = prepare_smartfallmm(self.arg)
        builder_val.make_dataset(val_subjects)
        val_data = builder_val.processed_data

        return train_data, val_data, transform_fn

    def load_teacher_model(self):
        teacher_model = TeacherModel(**self.arg.model_args).to(self.device)
        return teacher_model

    def load_student_model(self):
        student_model = StudentModel(**self.arg.student_model_args).to(self.device)
        return student_model

    def train_fold_teacher(self, fold_num: int, val_subjects: List[int], fold_info: str):
        self.print_log(f"--- Starting Teacher Fold {fold_num}: {fold_info} ---")
        train_data, val_data, transform_fn = self.load_data(val_subjects)
        train_ds = Time2VecVarLenFeeder(train_data, transform=transform_fn)
        val_ds = Time2VecVarLenFeeder(val_data, transform=transform_fn)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.arg.batch_size, shuffle=True,
            collate_fn=time2vec_varlen_collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.arg.batch_size, shuffle=False,
            collate_fn=time2vec_varlen_collate_fn
        )
        model = self.load_teacher_model()
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        best_epoch = -1
        patience = 15
        patience_counter = 0
        fold_metrics = {}

        for epoch in range(self.arg.num_epoch):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            for skel, accel, acc_time, mask, labels in tqdm(train_loader, desc=f"Teacher Fold {fold_num} Epoch {epoch+1}"):
                skel = skel.to(device)
                accel = accel.to(device)
                acc_time = acc_time.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(skel, accel, acc_time, mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            train_loss = total_loss / total
            train_acc = correct / total
            self.print_log(f"Teacher Fold {fold_num} Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}")

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_preds, all_labels, all_probs = [], [], []
            with torch.no_grad():
                for skel, accel, acc_time, mask, labels in val_loader:
                    skel = skel.to(device)
                    accel = accel.to(device)
                    acc_time = acc_time.to(device)
                    mask = mask.to(device)
                    labels = labels.to(device)
                    logits = model(skel, accel, acc_time, mask)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    probs = torch.softmax(logits, dim=-1)
                    all_probs.extend(probs.cpu().numpy()[:, 1])
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            try:
                val_roc_auc = roc_auc_score(all_labels, all_probs)
            except Exception as e:
                val_roc_auc = 0.0
            self.print_log(f"Teacher Fold {fold_num} Epoch {epoch} Validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}, Prec={val_precision:.4f}, Rec={val_recall:.4f}, F1={val_f1:.4f}, ROC AUC={val_roc_auc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = model.state_dict()
                fold_metrics = {
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
                    self.print_log(f"Teacher Fold {fold_num}: Early stopping at epoch {epoch}")
                    break
        teacher_save_path = os.path.join(self.work_dir, f"Fold{fold_num}_teacher.pth")
        torch.save(best_model_state, teacher_save_path)
        self.print_log(f"Teacher Fold {fold_num}: Best Epoch {best_epoch} -- {fold_metrics}")
        self.fold_results.append(fold_metrics)

    def train_fold_student(self, fold_num: int, val_subjects: List[int], fold_info: str):
        self.print_log(f"--- Starting Student Fold {fold_num}: {fold_info} ---")
        builder = prepare_smartfallmm(self.arg)
        builder.make_dataset(self.arg.subjects)
        full_data = builder.processed_data
        student_data = {
            'accelerometer': full_data['accelerometer'],
            'accel_time': full_data['accel_time'],
            'labels': full_data['labels']
        }
        student_ds = Time2VecVarLenFeeder(student_data, transform=None)
        student_loader = torch.utils.data.DataLoader(
            student_ds, batch_size=self.arg.batch_size, shuffle=True,
            collate_fn=time2vec_varlen_collate_fn
        )
        # Load teacher model from saved weights
        teacher_model = TeacherModel(**self.arg.model_args).to(self.device)
        teacher_path = os.path.join(self.work_dir, f"Fold{fold_num}_teacher.pth")
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=self.device))
        teacher_model.eval()
        student_model = StudentModel(**self.arg.student_model_args).to(self.device)
        optimizer = optim.Adam(student_model.parameters(), lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        ce_loss = torch.nn.CrossEntropyLoss()
        kd_loss = torch.nn.KLDivLoss(reduction='batchmean')
        temperature = self.arg.kd_temperature
        alpha = self.arg.kd_alpha
        best_val_loss = float('inf')
        best_epoch = -1
        patience = 15
        patience_counter = 0
        fold_metrics = {}

        for epoch in range(self.arg.num_epoch):
            student_model.train()
            total_loss = 0
            correct = 0
            total = 0
            for _, accel, acc_time, mask, labels in tqdm(student_loader, desc=f"Student Fold {fold_num} Epoch {epoch+1}"):
                accel = accel.to(self.device)
                acc_time = acc_time.to(self.device)
                mask = mask.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                student_logits = student_model(accel, acc_time, mask)
                loss_ce = ce_loss(student_logits, labels)
                # Teacher expects skeleton and accel; supply a dummy skeleton.
                dummy_skel = torch.zeros((accel.size(0), 1, self.arg.model_args['num_joints'] * self.arg.model_args['joint_dim']),
                                           device=self.device)
                with torch.no_grad():
                    teacher_logits = teacher_model(dummy_skel, accel, acc_time, mask)
                loss_kd = kd_loss(torch.log_softmax(student_logits/temperature, dim=-1),
                                  torch.softmax(teacher_logits/temperature, dim=-1)) * (temperature ** 2)
                loss = alpha * loss_kd + (1 - alpha) * loss_ce
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                preds = student_logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            epoch_loss = total_loss / total
            epoch_acc = correct / total
            self.print_log(f"Student Fold {fold_num} Epoch {epoch}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = student_model.state_dict()
                fold_metrics = {
                    "epoch": epoch,
                    "val_loss": epoch_loss,
                    "val_acc": epoch_acc
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.print_log(f"Student Fold {fold_num}: Early stopping at epoch {epoch}")
                    break
        student_save_path = os.path.join(self.work_dir, f"Fold{fold_num}_student.pth")
        torch.save(best_model_state, student_save_path)
        self.print_log(f"Student Fold {fold_num}: Best Epoch {best_epoch} -- {fold_metrics}")
        self.fold_results.append(fold_metrics)

    def run(self):
        folds = [
            ([43, 35, 36], "Fold 1: 38.3% falls"),
            ([44, 34, 32], "Fold 2: 39.7% falls"),
            ([45, 37, 38], "Fold 3: 44.8% falls"),
            ([46, 29, 31], "Fold 4: 41.4% falls"),
            ([30, 33, 39], "Fold 5")
        ]
        self.device = torch.device(f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu")
        if self.arg.phase == 'train_teacher':
            for i, (val_subjects, fold_info) in enumerate(folds, start=1):
                self.train_fold_teacher(i, val_subjects, fold_info)
        elif self.arg.phase == 'train_student':
            for i, (val_subjects, fold_info) in enumerate(folds, start=1):
                self.train_fold_student(i, val_subjects, fold_info)
        else:
            self.print_log("Phase not recognized. Use 'train_teacher' or 'train_student'.")
            return

        avg_loss = np.mean([fold["val_loss"] for fold in self.fold_results])
        avg_acc = np.mean([fold["val_acc"] for fold in self.fold_results])
        self.print_log(f"--- Cross-Validation Summary (5 Folds) ---")
        self.print_log(f"Average Val Loss: {avg_loss:.4f}")
        self.print_log(f"Average Val Accuracy: {avg_acc:.4f}")

###########################################################################
# Main entry
###########################################################################
def main():
    parser = get_args()
    arg = parser.parse_args()
    if arg.config is not None and os.path.exists(arg.config):
        with open(arg.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        parser.set_defaults(**default_arg)
        arg = parser.parse_args()
    init_seed(arg.seed)
    trainer = Trainer(arg)
    trainer.run()

if __name__ == "__main__":
    main()
