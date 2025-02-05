import argparse
import yaml
import traceback
import random
import sys
import os
import time

# environment
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import warnings
import json
import torch.nn.functional as F

# local
from utils.dataset import prepare_smartfallmm
from Feeder.teacher_varlen import TeacherVarLenFeeder, teacher_varlen_collate_fn
from main import Trainer, str2bool, init_seed, import_class
from utils.loss import DistillationLoss
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def get_args():
    parser = argparse.ArgumentParser(description='Distillation with real data loader')

    parser.add_argument('--config', default='./config/smartfallmm/distill.yaml')
    parser.add_argument('--dataset', type=str, default='smartfallmm')

    # Training hyperparams
    parser.add_argument('--batch-size', type=int, default=16, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N')
    parser.add_argument('--num-epoch', type=int, default=70, metavar='N')
    parser.add_argument('--start-epoch', type=int, default=0)

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--weight-decay', type=float, default=0.0004)

    # Data & subjects
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--dataset-args', default=None, type=str)

    # Teacher / Student
    parser.add_argument('--teacher-model', default=None)
    parser.add_argument('--teacher-args', default="{}")
    parser.add_argument('--teacher-weight-dir', type=str, default='exps/teacher_var_time',
                        help="Directory with Fold{i}_teacher_best_full.pth files")
    parser.add_argument('--student-model', default=None)
    parser.add_argument('--student-args', default="{}")

    # Model fallback for parent's Trainer
    parser.add_argument('--model', default=None)
    parser.add_argument('--model-args', default=None)

    parser.add_argument('--weights', type=str, help='Student model weight file for test only')
    parser.add_argument('--model-saved-name', type=str, default="student_distilled")

    # Distillation & loss
    parser.add_argument('--distill-loss', default='loss.BCE')
    parser.add_argument('--distill-args', default="{}")
    parser.add_argument('--student-loss', default='loss.BCE')
    parser.add_argument('--loss-args', default="{}")

    # Dataloaders
    parser.add_argument('--feeder', default='Feeder.teacher_varlen.TeacherVarLenFeeder')
    parser.add_argument('--train-feeder-args', default="{}")
    parser.add_argument('--val-feeder-args', default="{}")
    parser.add_argument('--test_feeder_args', default="{}")
    parser.add_argument('--include-val', type=str2bool, default=True)

    # etc.
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--work-dir', type=str, default='exps/distilled_student')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--device', nargs='+', default=[0], type=int)

    return parser


class Distiller(Trainer):
    """
    Distiller that:
      1) Loads real data via prepare_smartfallmm().
      2) Splits train vs. val by subjects.
      3) Creates DataLoaders using TeacherVarLenFeeder and teacher_varlen_collate_fn.
      4) Runs multi-fold training by loading teacher from Fold{i}_teacher_best_full.pth.
      5) Passes the correct inputs to the teacher (skel, accel, time, mask) and student (accel, mask, time).
      6) Logs detailed metrics (loss, accuracy, F1, precision, recall) for both teacher and student.
    """
    def __init__(self, arg):
        # fallback for parent's Trainer
        if not getattr(arg, 'model', None):
            arg.model = "torch.nn.Module"
        if not getattr(arg, 'model_args', None):
            arg.model_args = {}
        super().__init__(arg)

        self.arg = arg
        self.data_loader = {}
        self.model = {}

        # Add fold_results attribute for storing per-fold metrics
        self.fold_results = []

        # parse dataset_args
        if isinstance(self.arg.dataset_args, str):
            try:
                self.arg.dataset_args = json.loads(self.arg.dataset_args)
            except:
                self.arg.dataset_args = {}

        if self.arg.phase == "train":
            try:
                self.arg.teacher_args = json.loads(self.arg.teacher_args)
            except:
                self.arg.teacher_args = {}
            try:
                self.arg.student_args = json.loads(self.arg.student_args)
            except:
                self.arg.student_args = {}
            self._init_losses(test_mode=False)
        elif self.arg.phase == 'test':
            use_cuda = torch.cuda.is_available()
            self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
            self.model['student'] = torch.load(
                self.arg.weights,
                map_location=f'cuda:{self.output_device}' if use_cuda else 'cpu'
            )
            self._init_losses(test_mode=True)
            try:
                self.arg.model_args = json.loads(self.arg.student_args)
            except:
                self.arg.model_args = {}
            if 'acc_embed' in self.arg.model_args:
                self.arg.model_args['spatial_embed'] = self.arg.model_args['acc_embed']
        self.include_val = self.arg.include_val

    def _init_losses(self, test_mode=False):
        self.mse = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()
        if not test_mode:
            self.distillation_loss = DistillationLoss(temperature=3)

    def load_data(self, fold_idx=1, train_subjects=None, val_subjects=None):
        """
        Build the dataset using prepare_smartfallmm() and your feeder.
        This function:
          1) Calls builder.make_dataset() with train_subjects and then with val_subjects.
          2) Wraps the processed data in TeacherVarLenFeeder.
          3) Creates DataLoaders using teacher_varlen_collate_fn.
        """
        builder = prepare_smartfallmm(self.arg)

        # Build training data
        builder.make_dataset(train_subjects)
        train_data = builder.processed_data  # expected keys: 'skeleton', 'accelerometer', 'labels'
        train_ds = TeacherVarLenFeeder(train_data, num_joints=self.arg.teacher_args.get('num_joints', 32))
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            collate_fn=teacher_varlen_collate_fn
        )
        self.data_loader['train'] = train_loader

        # Build validation data
        if self.include_val:
            builder.make_dataset(val_subjects)
            val_data = builder.processed_data
            val_ds = TeacherVarLenFeeder(val_data, num_joints=self.arg.teacher_args.get('num_joints', 32))
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=self.arg.val_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                collate_fn=teacher_varlen_collate_fn
            )
            self.data_loader['val'] = val_loader

    def load_optimizer(self, name='student'):
        opt_name = self.arg.optimizer.lower()
        if opt_name == "adam":
            self.optimizer = optim.Adam(
                self.model[name].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif opt_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model[name].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif opt_name == "sgd":
            self.optimizer = optim.SGD(
                self.model[name].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")

    def train(self, epoch):
        self.model['student'].train()
        self.model['teacher'].eval()

        if 'train' not in self.data_loader:
            raise KeyError("No 'train' loader in self.data_loader. Call load_data() first.")
        loader = self.data_loader['train']

        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device

        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, stats=0.001)
        train_loss = 0
        correct_student = 0
        correct_teacher = 0
        cnt = 0

        process = tqdm(loader, ncols=80)
        for batch_idx, (skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels) in enumerate(process):
            skel_pad   = skel_pad.to(self.output_device)
            accel_pad  = accel_pad.to(self.output_device)
            time_pad   = time_pad.to(self.output_device)
            skel_mask  = skel_mask.to(self.output_device)
            accel_mask = accel_mask.to(self.output_device)
            labels     = labels.to(self.output_device)

            timer['dataloader'] += self.split_time()
            self.optimizer.zero_grad()

            with torch.no_grad():
                # Teacher forward: (skel_seq, accel_seq, accel_time, accel_mask)
                teacher_logits = self.model['teacher'](
                    skel_pad.float(),
                    accel_pad.float(),
                    time_pad.float(),
                    accel_mask
                )
            # Student forward: (accel_seq, accel_mask, accel_time)
            student_logits = self.model['student'](
                accel_pad.float(),
                accel_mask,
                time_pad.float()
            )

            loss = self.distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels
            )
            loss.backward()
            self.optimizer.step()

            timer['model'] += self.split_time()

            with torch.no_grad():
                train_loss += loss.item()
                stu_preds = torch.argmax(F.log_softmax(student_logits, dim=1), dim=1)
                correct_student += (stu_preds == labels).sum().item()
                tea_preds = torch.argmax(F.log_softmax(teacher_logits, dim=1), dim=1)
                correct_teacher += (tea_preds == labels).sum().item()

            cnt += len(labels)
            timer['stats'] += self.split_time()

        train_loss /= cnt
        stu_acc = 100.0 * correct_student / cnt
        tea_acc = 100.0 * correct_teacher / cnt

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            f"\tEpoch {epoch+1} Distill Loss: {train_loss:.4f}. Stu Acc: {stu_acc:.2f}%, Tea Acc: {tea_acc:.2f}%"
        )
        self.print_log("\tTime usage: [Data]{dataloader}, [Network]{model}".format(**proportion))

        if not self.include_val:
            if stu_acc > getattr(self, 'best_accuracy', float('-inf')):
                self.best_accuracy = stu_acc
                torch.save(
                    self.model['student'].state_dict(),
                    os.path.join(self.arg.work_dir, f"{self.arg.model_saved_name}_{epoch}.pth")
                )
                self.print_log("Weights saved (no validation).")
        else:
            self.eval(epoch, loader_name='val')

    def eval(self, epoch, loader_name='val', result_file=None):
        if loader_name not in self.data_loader:
            self.print_log(f"[WARN] No loader named {loader_name}.")
            return
        loader = self.data_loader[loader_name]
        self.model['student'].eval()
        self.model['teacher'].eval()

        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device

        self.print_log(f"Eval epoch: {epoch+1} on {loader_name}")

        # Initialize accumulators for student metrics
        total_loss_student = 0
        correct_student = 0
        cnt = 0
        student_labels = []
        student_preds = []

        # Initialize accumulators for teacher metrics
        total_loss_teacher = 0
        correct_teacher = 0
        teacher_preds = []

        for batch_idx, (skel_pad, accel_pad, time_pad, skel_mask, accel_mask, labels) in enumerate(tqdm(loader, ncols=80)):
            skel_pad   = skel_pad.to(self.output_device)
            accel_pad  = accel_pad.to(self.output_device)
            time_pad   = time_pad.to(self.output_device)
            skel_mask  = skel_mask.to(self.output_device)
            accel_mask = accel_mask.to(self.output_device)
            labels     = labels.to(self.output_device)

            # Student forward
            student_logits = self.model['student'](
                accel_pad.float(),
                accel_mask,
                time_pad.float()
            )
            stu_loss = self.criterion(student_logits, labels)
            total_loss_student += stu_loss.item()

            stu_batch_preds = torch.argmax(F.log_softmax(student_logits, dim=1), dim=1)
            correct_student += (stu_batch_preds == labels).sum().item()
            student_labels.extend(labels.cpu().tolist())
            student_preds.extend(stu_batch_preds.cpu().tolist())

            # Teacher forward
            teacher_logits = self.model['teacher'](
                skel_pad.float(),
                accel_pad.float(),
                time_pad.float(),
                accel_mask
            )
            tea_loss = self.criterion(teacher_logits, labels)
            total_loss_teacher += tea_loss.item()

            tea_batch_preds = torch.argmax(F.log_softmax(teacher_logits, dim=1), dim=1)
            correct_teacher += (tea_batch_preds == labels).sum().item()
            teacher_preds.extend(tea_batch_preds.cpu().tolist())

            cnt += len(labels)

        if cnt > 0:
            avg_loss_student = total_loss_student / cnt
            stu_acc = 100.0 * correct_student / cnt
            avg_loss_teacher = total_loss_teacher / cnt
            tea_acc = 100.0 * correct_teacher / cnt
        else:
            avg_loss_student = avg_loss_teacher = stu_acc = tea_acc = 0

        stu_f1 = f1_score(student_labels, student_preds, average='macro') * 100 if cnt > 0 else 0
        stu_prec = precision_score(student_labels, student_preds, average='macro', zero_division=0) * 100 if cnt > 0 else 0
        stu_rec = recall_score(student_labels, student_preds, average='macro', zero_division=0) * 100 if cnt > 0 else 0

        tea_f1 = f1_score(student_labels, teacher_preds, average='macro') * 100 if cnt > 0 else 0
        tea_prec = precision_score(student_labels, teacher_preds, average='macro', zero_division=0) * 100 if cnt > 0 else 0
        tea_rec = recall_score(student_labels, teacher_preds, average='macro', zero_division=0) * 100 if cnt > 0 else 0

        self.print_log(f"[Val - Student] Loss={avg_loss_student:.4f}, Acc={stu_acc:.2f}%, F1={stu_f1:.2f}%, Prec={stu_prec:.2f}%, Rec={stu_rec:.2f}%")
        self.print_log(f"[Val - Teacher] Loss={avg_loss_teacher:.4f}, Acc={tea_acc:.2f}%, F1={tea_f1:.2f}%, Prec={tea_prec:.2f}%, Rec={tea_rec:.2f}%")

        if self.arg.phase == 'train':
            if stu_acc > getattr(self, 'best_accuracy', float('-inf')):
                self.best_accuracy = stu_acc
                self.best_f1 = stu_f1
                save_path = os.path.join(self.arg.work_dir, f"{self.arg.model_saved-name}_best.pth")
                torch.save(self.model['student'], save_path)
                self.print_log(f"Improved {loader_name} => saved to {save_path}")

    def start(self):
        self.print_log("Parameters:\n{}".format(vars(self.arg)))

        if self.arg.phase == 'train':
            self.best_accuracy = float('-inf')
            self.best_f1 = float('-inf')

            folds = [
                ([43,35,36], "Fold1"),
                ([44,34,32], "Fold2"),
                ([45,37,38], "Fold3"),
                ([46,29,31], "Fold4"),
                ([30,33,39], "Fold5")
            ]
            results = self.create_df()

            for i, (val_subs, foldname) in enumerate(folds, start=1):
                self.print_log(f"\n--- Distillation {foldname}, val_subs={val_subs} ---")

                # Load teacher from Fold{i}_teacher_best_full.pth
                teacher_path = os.path.join(
                    self.arg.teacher_weight_dir,
                    f"Fold{i}_teacher_best_full.pth"
                )
                use_cuda = torch.cuda.is_available()
                self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device

                self.model['teacher'] = torch.load(
                    teacher_path,
                    map_location=f"cuda:{self.output_device}" if use_cuda else 'cpu'
                )
                self.model['teacher'].to(f'cuda:{self.output_device}')

                # Build student and move to device
                student_cls = import_class(self.arg.student_model)
                self.model['student'] = student_cls(**self.arg.student_args)
                self.model['student'].to(f'cuda:{self.output_device}')

                self.print_log(f"[FOLD {i}] Teacher => {teacher_path}")
                self.print_log(f"Teacher model: {self.model['teacher']}")
                self.print_log(f"Student model: {self.model['student']}")

                all_subs = self.arg.subjects
                train_subs = [s for s in all_subs if s not in val_subs]
                self.train_subjects = train_subs
                self.test_subject = val_subs

                # Load data using the real dataset pipeline
                self.load_data(fold_idx=i, train_subjects=train_subs, val_subjects=val_subs)
                self.load_optimizer('student')

                self.best_accuracy = float('-inf')
                self.best_f1 = float('-inf')

                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    self.train(epoch)

                # Record final best metrics for this fold
                best_metrics = {
                    'epoch': self.best_accuracy,  # Adjust if you want to store epoch number
                    'val_acc': self.best_accuracy,
                    'val_f1': self.best_f1
                }
                subject_result = pd.Series({
                    'fold': foldname,
                    'val_subjects': str(val_subs),
                    'train_subjects': str(train_subs),
                    'accuracy': round(self.best_accuracy, 2),
                    'f1_score': round(self.best_f1, 2),
                    'teacher_loaded': f"Fold{i}_teacher_best_full.pth"
                })
                results.loc[len(results)] = subject_result
                self.fold_results.append(best_metrics)

            # Final Cross-Validation Summary
            self.print_log("\n===== Cross-Validation Summary =====")
            avg_acc = np.mean([m['val_acc'] for m in self.fold_results if m is not None])
            avg_f1 = np.mean([m['val_f1'] for m in self.fold_results if m is not None])
            self.print_log(f"Average Val Accuracy: {avg_acc:.4f}")
            self.print_log(f"Average Val F1: {avg_f1:.4f}")
            self.print_log("----- Best Metrics per Fold -----")
            for i, metrics in enumerate(self.fold_results, start=1):
                if metrics is not None:
                    self.print_log(
                        f"Fold{i}: Best Acc={metrics['val_acc']:.4f}, Best F1={metrics['val_f1']:.4f}"
                    )
                else:
                    self.print_log(f"Fold{i}: No metrics recorded.")
            results.to_csv(os.path.join(self.arg.work_dir, "scores.csv"), index=False)

        elif self.arg.phase == 'test':
            if not self.arg.weights:
                raise ValueError("Please provide --weights for test phase.")
            self.print_log("[INFO] Test phase not fully implemented.")
        else:
            raise ValueError(f"Unknown phase: {self.arg.phase}")


if __name__ == "__main__":
    parser = get_args()
    p = parser.parse_args()
    if p.config and os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG:", k)
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    init_seed(arg.seed)
    trainer = Distiller(arg)
    trainer.start()

