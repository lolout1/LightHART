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
    parser.add_argument('--dataset', type=str, default='utd')

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
    parser.add_argument('--distill-loss', 
                       default='utils.loss.DistillationLoss', 
                       help='Name of distillation loss function')
    parser.add_argument('--distill-args', 
                       default={}, 
                       type=lambda x: {} if x == "{}" else x,
                       help='Arguments for distillation loss')
    parser.add_argument('--student-loss', 
                       default='torch.nn.CrossEntropyLoss', 
                       help='Name of student loss function')
    parser.add_argument('--loss-args', 
                       default={}, 
                       type=lambda x: {} if x == "{}" else x,
                       help='Arguments for student loss')
    


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

        self.include_val = arg.include_val
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

    def load_models(self):
        # Load teacher model
        TeacherModel = import_class(self.arg.teacher_model)
        self.model['teacher'] = TeacherModel(**self.arg.teacher_args)
        teacher_weights = torch.load(self.arg.teacher_weight, map_location=f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu')
        self.model['teacher'].load_state_dict(teacher_weights)
        self.model['teacher'].to(f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu')
        self.model['teacher'].eval()  # Teacher model is in evaluation mode

        # Load student model
        StudentModel = import_class(self.arg.student_model)
        self.model['student'] = StudentModel(**self.arg.student_args)
        if self.arg.weights:
            student_weights = torch.load(self.arg.weights, map_location=f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu')
            self.model['student'].load_state_dict(student_weights)
        self.model['student'].to(f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu')

    def load_loss(self):
        # Distillation loss
        DistillLossClass = import_class(self.arg.distill_loss)
        # Handle the case where distill_args is already a dictionary
        if isinstance(self.arg.distill_args, dict):
            distill_args = self.arg.distill_args
        else:
            # Try to evaluate if it's a string
            try:
                distill_args = eval(self.arg.distill_args)
            except:
                distill_args = {}
        
        self.distillation_loss = DistillLossClass(**distill_args)

        # Student loss
        StudentLossClass = import_class(self.arg.student_loss)
        # Handle the case where loss_args is already a dictionary
        if isinstance(self.arg.loss_args, dict):
            loss_args = self.arg.loss_args
        else:
            # Try to evaluate if it's a string
            try:
                loss_args = eval(self.arg.loss_args)
            except:
                loss_args = {}
        
        self.student_loss = StudentLossClass(**loss_args)
    def load_optimizer(self):
        if self.arg.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model['student'].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model['student'].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model['student'].parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.arg.optimizer}")

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

            # Get teacher output (includes both logits and features)
            with torch.no_grad():
                teacher_output = self.model['teacher'](acc_data.float(), skl_data.float())
                # Unpack if it's a tuple (logits, features)
                if isinstance(teacher_output, tuple):
                    teacher_logits, teacher_features = teacher_output
                else:
                    teacher_logits = teacher_output
                    teacher_features = None

            # Get student output
            student_output = self.model['student'](acc_data.float())
            # Unpack if it's a tuple (logits, features)
            if isinstance(student_output, tuple):
                student_logits, student_features = student_output
            else:
                student_logits = student_output
                student_features = None

            # Prepare outputs for loss calculation
            if teacher_features is not None and student_features is not None:
                teacher_output = (teacher_logits, teacher_features)
                student_output = (student_logits, student_features)
            else:
                teacher_output = teacher_logits
                student_output = student_logits

            # Compute distillation loss
            loss = self.distillation_loss(student_output, teacher_output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Use student logits for accuracy calculation
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
            self.eval(epoch, loader_name='val')

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

                # Get student output
                student_output = self.model['student'](acc_data.float())
                # Unpack if it's a tuple (logits, features)
                if isinstance(student_output, tuple):
                    student_logits, _ = student_output
                else:
                    student_logits = student_output

                # During evaluation, use standard cross-entropy loss
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
def process_args(args_dict):
    """Process loaded YAML arguments to ensure proper types."""
    if 'distill_args' in args_dict:
        if isinstance(args_dict['distill_args'], str):  
            try:
                args_dict['distill_args'] = eval(args_dict['distill_args'])
            except:
                args_dict['distill_args'] = {}
    if 'loss_args' in args_dict:
        if isinstance(args_dict['loss_args'], str):
            try:
                args_dict['loss_args'] = eval(args_dict['loss_args'])
            except:
                args_dict['loss_args'] = {}
    return args_dict
if __name__ == "__main__":
    parser = get_args()

    # Load arguments from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
            default_arg = yaml.safe_load(f)
        default_arg = process_args(default_arg)  # Process the loaded arguments
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    trainer = Distiller(arg)
    trainer.start()