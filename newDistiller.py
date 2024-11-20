import argparse
import yaml
import traceback
import random
import sys
import os
import time

# Environmental imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

# Local imports
from utils.dataset import prepare_smartfallmm, filter_subjects
from main import str2bool, init_seed, import_class
from utils.loss import DistillationLoss
from sklearn.metrics import f1_score


def get_args():
    parser = argparse.ArgumentParser(description='Distillation')
    parser.add_argument('--config', default='./config/smartfallmm/distill.yaml')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--val-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for validation (default: 8)')
    parser.add_argument('--num-epoch', type=int, default=70, metavar='N',
                        help='number of epochs to train (default: 70)')
    parser.add_argument('--start-epoch', type=int, default=0)

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0004)

    # Learning Rate Scheduler parameters
    parser.add_argument('--lr-scheduler', default=None, type=str, help='Learning rate scheduler configuration')

    # Data parameters
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--dataset-args', default=None, type=str)

    # Teacher model parameters
    parser.add_argument('--teacher-model', default=None, help='Name of teacher model to load')
    parser.add_argument('--teacher-args', default=str, help='A dictionary for teacher args')
    parser.add_argument('--teacher-weight', type=str, default="", help='Path to teacher weights')

    # Student model parameters
    parser.add_argument('--student-model', default=None, help='Name of the student model to load')
    parser.add_argument('--student-args', default=str, help='A dictionary for student args')

    # Device and weights
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--weights', type=str, help='Location of student weight file')
    parser.add_argument('--model-saved-name', type=str, default="student_model", help='Model save name')

    # Loss functions
    parser.add_argument('--distill-loss', default='utils.loss.DistillationLoss', help='Name of distillation loss function')
    parser.add_argument('--distill-args', default="{}", type=str, help='Arguments for distillation loss')
    parser.add_argument('--student-loss', default='torch.nn.CrossEntropyLoss', help='Name of student loss function')
    parser.add_argument('--loss-args', default="{}", type=str, help='Arguments for student loss')

    # DataLoader
    parser.add_argument('--feeder', default=None, help='Dataloader class')
    parser.add_argument('--train-feeder-args', default=str, help='Arguments for train DataLoader')
    parser.add_argument('--val-feeder-args', default=str, help='Arguments for validation DataLoader')
    parser.add_argument('--test_feeder_args', default=str, help='Arguments for test DataLoader')
    parser.add_argument('--include-val', type=str2bool, default=True, help='Include validation during training')

    # Initialization
    parser.add_argument('--seed', type=int, default=2, help='Random seed (default: 2)')

    # Logging
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    parser.add_argument('--work-dir', type=str, default='exps/test', metavar='F', help="Working Directory")
    parser.add_argument('--print-log', type=str2bool, default=True, help='Print logging or not')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str, help='Name of result file')

    return parser


class Distiller():
    def __init__(self, arg):
        self.arg = arg
        self.data_loader = {}
        self.model = {}
        self.best_accuracy = 0
        self.best_f1 = 0

        # Initialize device
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device

        # Initialize teacher and student models
        self.load_models()

        # Load loss functions
        self.load_loss()

        # Initialize optimizer
        if self.arg.phase == 'train':
            self.load_optimizer()

        # Load learning rate scheduler
        self.lr_scheduler = None
        if self.arg.phase == 'train' and self.arg.lr_scheduler is not None:
            self.load_lr_scheduler()

        self.include_val = arg.include_val
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

    def load_models(self):
        # Load teacher model directly
        self.model['teacher'] = torch.load(
            self.arg.teacher_weight,
            map_location=f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu'
        )
        self.model['teacher'].to(f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu')
        self.model['teacher'].eval()  # Teacher model is in evaluation mode

        # Load student model
        StudentModel = import_class(self.arg.student_model)
        self.model['student'] = StudentModel(**self.arg.student_args)
        if self.arg.weights:
            student_state_dict = torch.load(
                self.arg.weights,
                map_location=f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu'
            )
            self.model['student'].load_state_dict(student_state_dict)
        self.model['student'].to(f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu')

    def load_loss(self):
        # Distillation loss
        DistillLossClass = import_class(self.arg.distill_loss)
        distill_args = self.arg.distill_args  # Already parsed as a dictionary
        self.distillation_loss = DistillLossClass(**distill_args)

        # Student loss
        StudentLossClass = import_class(self.arg.student_loss)
        loss_args = self.arg.loss_args
        self.student_loss = StudentLossClass(**loss_args)

    def load_optimizer(self):
        optimizer_config = self.arg.optimizer
        optimizer_type = optimizer_config['type'].lower()

        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model['student'].parameters(),
                lr=optimizer_config['base_lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model['student'].parameters(),
                lr=optimizer_config['base_lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model['student'].parameters(),
                lr=optimizer_config['base_lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")

    def load_lr_scheduler(self):
        scheduler_config = self.arg.lr_scheduler
        scheduler_type = scheduler_config['type'].lower()

        if scheduler_type == 'reducelronplateau':
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.1),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6),
                verbose=True
            )
        elif scheduler_type == 'steplr':
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1),
                verbose=True
            )
        elif scheduler_type == 'multisteplr':
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=scheduler_config.get('milestones', [30, 60, 90]),
                gamma=scheduler_config.get('gamma', 0.1),
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported LR scheduler type: {scheduler_config['type']}")

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        # Prepare the dataset
        builder = prepare_smartfallmm(self.arg)
        norm_train = filter_subjects(builder, self.train_subjects)
        norm_val = filter_subjects(builder, self.test_subject)

        # DataLoader for training
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.train_feeder_args, dataset=norm_train),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker
        )

        # DataLoader for validation
        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.val_feeder_args, dataset=norm_val),
            batch_size=self.arg.val_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker
        )

    def print_log(self, string, print_time=True):
        if print_time:
            time_str = time.strftime("[%Y-%m-%d %H:%M:%S] ", time.localtime())
            string = time_str + string
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)

    def train(self, epoch):
        self.model['student'].train()
        self.model['teacher'].eval()
        use_cuda = torch.cuda.is_available()

        loader = self.data_loader['train']
        process = tqdm(loader, ncols=80)
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, (inputs, targets, idx) in enumerate(process):
            acc_data = inputs['accelerometer'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
            skl_data = inputs['skeleton'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
            targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')

            # Teacher uses both accelerometer and skeleton data
            with torch.no_grad():
                teacher_logits = self.model['teacher'](acc_data.float(), skl_data.float())

            # Student only uses accelerometer data
            student_logits = self.model['student'](acc_data.float())

            # Compute distillation loss
            loss = self.distillation_loss(student_logits=student_logits, teacher_logits=teacher_logits, labels=targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(student_logits.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            process.set_description(f'Epoch [{epoch+1}/{self.arg.num_epoch}] Loss: {total_loss/total_samples:.4f} Acc: {100.0*total_correct/total_samples:.2f}%')

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        self.print_log(f'Epoch [{epoch+1}/{self.arg.num_epoch}] Training Loss: {avg_loss:.4f}, Training Acc: {avg_acc:.2f}%')

        # Validation
        if self.include_val:
            val_loss = self.eval(epoch, loader_name='val')

        # Step the learning rate scheduler
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_loss)
            else:
                self.lr_scheduler.step()

    def eval(self, epoch, loader_name='val'):
        self.model['student'].eval()
        use_cuda = torch.cuda.is_available()

        loader = self.data_loader[loader_name]
        total_loss = 0
        total_correct = 0
        total_samples = 0
        label_list = []
        pred_list = []

        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(loader):
                acc_data = inputs['accelerometer'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')

                student_logits = self.model['student'](acc_data.float())
                loss = self.student_loss(student_logits, targets)

                total_loss += loss.item() * targets.size(0)
                _, predicted = torch.max(student_logits.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                label_list.extend(targets.cpu().numpy())
                pred_list.extend(predicted.cpu().numpy())

        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        f1 = f1_score(label_list, pred_list, average='macro') * 100

        self.print_log(f'Epoch [{epoch+1}/{self.arg.num_epoch}] {loader_name.capitalize()} Loss: {avg_loss:.4f}, {loader_name.capitalize()} Acc: {avg_acc:.2f}%, F1 Score: {f1:.2f}%')

        # Save the best model
        if avg_acc > self.best_accuracy:
            self.best_accuracy = avg_acc
            self.best_f1 = f1
            torch.save(self.model['student'].state_dict(), f'{self.arg.work_dir}/{self.arg.model_saved_name}.pth')
            self.print_log('Best model saved.')

        return avg_loss  # Return validation loss for LR scheduler

    def start(self):
        self.print_log(f'Parameters:\n{vars(self.arg)}\n')

        # Split subjects into training and testing
        test_subject = self.arg.subjects[-3:]
        train_subjects = [x for x in self.arg.subjects if x not in test_subject]
        self.test_subject = test_subject
        self.train_subjects = train_subjects

        self.load_data()

        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            self.train(epoch)

        # Final evaluation on validation set
        if self.include_val:
            self.eval(self.arg.num_epoch - 1, loader_name='val')

        self.print_log(f'Best Validation Accuracy: {self.best_accuracy:.2f}%')
        self.print_log(f'Best Validation F1 Score: {self.best_f1:.2f}%')


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    # Load configuration from YAML
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

            # Corrected recursive_update function
            def recursive_update(d, u):
                if d is None:
                    d = {}
                for k, v in u.items():
                    if isinstance(v, dict):
                        d_value = d.get(k)
                        if not isinstance(d_value, dict):
                            d_value = {}
                        d[k] = recursive_update(d_value, v)
                    else:
                        d[k] = v
                return d

            args_dict = vars(args)
            args_dict = recursive_update(args_dict, config)

    # Ensure types are correct
    args.optimizer = args_dict.get('optimizer', {})
    args.lr_scheduler = args_dict.get('lr_scheduler', {})
    args.student_args = args_dict.get('student_args', {})
    args.teacher_args = args_dict.get('teacher_args', {})
    args.dataset_args = args_dict.get('dataset_args', {})
    args.train_feeder_args = args_dict.get('train_feeder_args', {})
    args.val_feeder_args = args_dict.get('val_feeder_args', {})
    args.test_feeder_args = args_dict.get('test_feeder_args', {})
    args.distill_args = args_dict.get('distill_args', {})
    args.loss_args = args_dict.get('loss_args', {})

    init_seed(args.seed)
    trainer = Distiller(args)
    trainer.start()

