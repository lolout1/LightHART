import traceback
from typing import List
import random
import sys
import os
import time
import shutil
import argparse
import yaml
import logging
from pathlib import Path

# Environmental imports
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from utils.loss import DistillationLoss

from Models.st_cvtransformer import MMTransformer
from Feeder.Make_Dataset import UTD_mm

# Local imports
from utils.dataset import prepare_smartfallmm, filter_subjects

def setup_logger(save_dir):
    """Set up logger with file and console handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(save_dir / 'train.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

class Trainer:
    def __init__(self, arg):
        self.arg = arg
        # Set default values for missing arguments
        if not hasattr(self.arg, 'start_epoch'):
            self.arg.start_epoch = 0

        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_f1 = 0
        self.best_loss = 0
        self.best_accuracy = 0
        self.train_subjects = []
        self.test_subject = []
        self.optimizer = None
        self.data_loader = dict()
        
        # Initialize device first
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.device = torch.device(f'cuda:{self.output_device}' if use_cuda else 'cpu')
        
        # Fix the inertial modality initialization
        self.intertial_modality = (lambda x: next(
            (modality for modality in x if modality != 'skeleton'), None)
        )(arg.dataset_args['modalities'])
        
        # Setup directories
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            self.save_config(arg.config, arg.work_dir)
            
        # Initialize model
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            self.model = torch.load(self.arg.weights)
            
        # Setup loss and validation flag
        self.load_loss()
        self.include_val = arg.include_val
        
        # Initialize timing variables
        self.cur_time = 0
        self.timer = {'dataloader': 0.001, 'model': 0.001, 'stats': 0.001}
        
        # Log model parameters
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size : {num_params / (1024 ** 2):.2f} MB')

    def save_config(self, src_path, dest_path):
        shutil.copy(src_path, f'{dest_path}/{os.path.basename(src_path)}')

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def load_model(self, model_name, model_args):
        Model = self._import_class(model_name)
        model = Model(**model_args).to(self.device)
        return model

    def load_optimizer(self):
        """Enhanced optimizer with parameter-specific weight decay"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.arg.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.arg.base_lr,
            eps=1e-8
        )
        
        # Enhanced scheduler settings
        if hasattr(self, 'data_loader') and 'train' in self.data_loader:
            steps_per_epoch = len(self.data_loader['train'])
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.arg.base_lr,
                epochs=self.arg.num_epoch,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.arg.scheduler_args.get('pct_start', 0.1),
                div_factor=self.arg.scheduler_args.get('div_factor', 25),
                final_div_factor=float(self.arg.scheduler_args.get('final_div_factor', 1e4)),
                anneal_strategy=self.arg.scheduler_args.get('anneal_strategy', 'cos')
            )

    def load_data(self):
        """Initialize data loaders"""
        Feeder = self._import_class(self.arg.feeder)
        
        if self.arg.phase == 'train':
            # Prepare dataset splits
            builder = prepare_smartfallmm(self.arg)
            
            # Split subjects for training and validation
            total_subjects = len(self.arg.subjects)
            val_size = max(1, int(total_subjects * 0.2))  # 20% for validation
            self.train_subjects = self.arg.subjects[:-val_size]
            self.val_subjects = self.arg.subjects[-val_size:]
            
            # Prepare train data
            train_data = filter_subjects(builder, self.train_subjects)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args, dataset=train_data),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                pin_memory=True
            )
            
            # Prepare validation data if needed
            if getattr(self.arg, 'include_val', True):
                val_data = filter_subjects(builder, self.val_subjects)
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.val_feeder_args, dataset=val_data),
                    batch_size=self.arg.val_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker,
                    pin_memory=True
                )

    def load_loss(self):
        """Initialize loss function based on config"""
        if hasattr(self.arg, 'distillation'):
            # Use distillation loss if specified
            self.criterion = DistillationLoss(
                temperature=self.arg.distillation['temperature'],
                alpha=self.arg.distillation['alpha']
            )
        else:
            # Use standard cross entropy loss with label smoothing if specified
            label_smoothing = 0.0
            if hasattr(self.arg, 'loss_args') and 'label_smoothing' in self.arg.loss_args:
                label_smoothing = self.arg.loss_args['label_smoothing']
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def record_time(self):
        """Record current time."""
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        """Get split of time since last record."""
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch):
        '''
        Training function matching main.py implementation
        '''
        use_cuda = torch.cuda.is_available()
        self.model.train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader=0.001, model=0.001, stats=0.001)
        acc_value = []
        accuracy = 0
        cnt = 0
        train_loss = 0

        process = tqdm(loader, ncols=80)

        for batch_idx, (inputs, targets, idx) in enumerate(process):
            with torch.no_grad():
                acc_data = inputs[self.intertial_modality].to(self.device)
                skl_data = inputs['skeleton'].to(self.device)
                targets = targets.to(self.device)

            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()
            logits = self.model(acc_data.float(), skl_data.float())
            loss = self.criterion(logits, targets)
            loss.mean().backward()
            self.optimizer.step()

            timer['model'] += self.split_time()
            
            with torch.no_grad():
                train_loss += loss.mean().item()
                accuracy += (torch.argmax(F.log_softmax(logits, dim=1), 1) == targets).sum().item()

            cnt += len(targets)
            timer['stats'] += self.split_time()

            if batch_idx % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                        batch_idx + 1, len(loader), loss.item(),
                        self.optimizer.param_groups[0]['lr']))

        train_loss /= cnt
        accuracy *= 100. / cnt

        self.train_loss_summary.append(train_loss)
        acc_value.append(accuracy)
        
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        
        self.print_log(
            f'\tTraining Loss: {train_loss:.4f}. Training Acc: {accuracy:.2f}%'
        )
        self.print_log(
            f'\tTime consumption: [Data]{proportion["dataloader"]}, '
            f'[Network]{proportion["model"]}'
        )
        
        return train_loss, accuracy

    @torch.no_grad()
    def eval(self, epoch, loader_name='val', result_file=None):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        process = tqdm(self.data_loader[loader_name], desc='Validation')
        for inputs, targets, idx in process:
            acc_data = inputs[self.intertial_modality].to(self.device)
            skl_data = inputs['skeleton'].to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            with torch.amp.autocast('cuda'):
                logits = self.model(acc_data.float(), skl_data.float())
            loss = self.criterion(logits, targets)
            
            # Calculate metrics
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100. * correct / total
        f1 = 100. * f1_score(all_targets, all_preds, average='macro')
        loss = total_loss / len(self.data_loader[loader_name])
        
        self.logger.info(
            f'{loader_name.capitalize()} Loss: {loss:.4f}. '
            f'{loader_name.capitalize()} Acc: {accuracy:.2f}% f1: {f1:.2f}'
        )
        
        # Save best model
        if self.arg.phase == 'train' and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_f1 = f1
            self.best_loss = loss
            torch.save(
                self.model,
                f'{self.arg.work_dir}/{self.arg.model_saved_name}.pth'
            )
            self.logger.info('Weights Saved')
        
        return loss

    def start(self):
        """Start training/testing."""
        if self.arg.phase == 'train':
            self.train_loss_summary = []
            self.val_loss_summary = []
            self.best_accuracy = float('-inf')
            self.best_f1 = float('-inf')
            
            # Print parameters
            self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
            
            # Setup data splits
            test_subject = self.arg.subjects[-6:]  # Last 6 subjects for testing
            train_subjects = [x for x in self.arg.subjects if x not in test_subject]
            self.test_subject = test_subject
            self.train_subjects = train_subjects
            
            # Initialize model and data
            self.model = self.load_model(self.arg.model, self.arg.model_args)
            self.print_log(f'Model Parameters: {self.count_parameters(self.model)}')
            
            # Load data and optimizer
            self.load_data()
            self.load_optimizer()
            
            # Initialize global step
            self.global_step = 0  # Start from 0 for new training
            
            # Training loop
            for epoch in range(self.arg.num_epoch):
                train_loss, train_acc = self.train(epoch)
                
                if self.arg.eval_interval > 0 and (epoch + 1) % self.arg.eval_interval == 0:
                    val_loss = self.eval(epoch, loader_name='val')
                    self.val_loss_summary.append(val_loss)
                
                # Save model if specified
                if (epoch + 1) % self.arg.save_interval == 0:
                    state_dict = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_accuracy': self.best_accuracy,
                    }
                    torch.save(
                        state_dict,
                        f'{self.arg.work_dir}/checkpoint_{epoch+1}.pth'
                    )
            
            # Print final results
            self.print_log(f'Best accuracy: {self.best_accuracy:.2f}%')
            self.print_log(f'Best F1-score: {self.best_f1:.4f}')
            self.print_log(f'Model saved: {self.arg.work_dir}')
            self.print_log(f'Training completed')
        
        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name='test')
            self.print_log('Done.\n')

    @staticmethod
    def _import_class(name):
        mod_str, _sep, class_str = name.rpartition('.')
        __import__(mod_str)
        try:
            return getattr(sys.modules[mod_str], class_str)
        except AttributeError:
            raise ImportError('Class %s cannot be found (%s)' % (
                class_str, traceback.format_exception(*sys.exc_info())
            ))

    def print_log(self, msg, print_time=True):
        """Print logs with optional timestamp."""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            msg = f"{localtime} - {msg}"
        print(msg)
        if hasattr(self, 'log_file'):
            with open(self.log_file, 'a') as f:
                print(msg, file=f)

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config['device'] = [args.device]
    config['phase'] = args.phase
    
    # Initialize seed
    init_seed(config['seed'])
    
    # Create trainer and start training
    trainer = Trainer(argparse.Namespace(**config))
    trainer.start()

if __name__ == '__main__':
    main()