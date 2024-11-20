import traceback
from typing import List
import random
import sys
import os
import time
import shutil
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
from Models import lightweight_student
from Feeder.Make_Dataset import UTD_mm
def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Accelerometer-Only Fall Detection Training')
    
    # Basic parameters
    parser.add_argument('--config', default='./config/smartfallmm/accel_only.yaml', help='path to config file')
    parser.add_argument('--phase', type=str, default='train', help='phase: train or test')
    parser.add_argument('--dataset', type=str, default='smartfallmm', help='dataset to use')
    
    # Model parameters
    parser.add_argument('--model', default=None, help='model architecture')
    parser.add_argument('--model-args', default=dict(), help='model arguments')
    parser.add_argument('--weights', type=str, help='path to pretrained weights')
    parser.add_argument('--model-saved-name', type=str, default='accel_model', help='name for saved model')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=16, help='testing batch size')
    parser.add_argument('--num-epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--optimizer', type=str, default='adamw', help='type of optimizer')
    parser.add_argument('--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='GPU device IDs')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--work-dir', type=str, default='work_dir/temp', help='working directory')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--num-worker', type=int, default=0, help='number of workers for data loading')
    
    return parser

class AccelTrainer:
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_accuracy = float('-inf')
        self.best_f1 = float('-inf')
        
        # Split subjects for training and validation
        all_subjects = self.arg.subjects
        num_train = int(len(all_subjects) * 0.8)
        self.train_subjects = all_subjects[:num_train]
        self.test_subjects = all_subjects[num_train:]
        
        # Create working directory
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            if hasattr(self.arg, 'config'):
                shutil.copy(self.arg.config, f'{self.arg.work_dir}/{os.path.basename(self.arg.config)}')
        
        # Setup device
        self.device = f'cuda:{self.arg.device[0]}' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model and training components
        self.initialize_model()
        
        # Print model info
        self.print_model_info()

    def load_model(self):
        """Load and initialize the model."""
        from Models.lightweight_student import LightweightStudent
        import torch.serialization
        
        # Add LightweightStudent to safe globals
        torch.serialization.add_safe_globals([LightweightStudent])
        
        Model = self.import_class(self.arg.model)
        model = Model(**self.arg.model_args).to(self.device)
        
        if self.arg.weights:
            try:
                # Load the state dict without weights_only first
                checkpoint = torch.load(self.arg.weights, map_location=self.device)
                
                if isinstance(checkpoint, type(model)):
                    # If the checkpoint is the model itself
                    state_dict = checkpoint.state_dict()
                elif isinstance(checkpoint, dict):
                    # If the checkpoint is a state dict
                    state_dict = checkpoint
                else:
                    # If the checkpoint is something else
                    raise ValueError("Unexpected checkpoint format")
                
                # Filter and load the state dict
                model_dict = model.state_dict()
                filtered_dict = {
                    k: v for k, v in state_dict.items()
                    if k in model_dict and v.size() == model_dict[k].size()
                }
                
                if filtered_dict:
                    model.load_state_dict(filtered_dict, strict=False)
                    print(f"Successfully loaded {len(filtered_dict)} layers from teacher model")
                    print("Loaded layers:", list(filtered_dict.keys()))
                else:
                    print("Warning: No compatible layers found in teacher model")
                
            except Exception as e:
                print(f"Error loading pretrained weights: {str(e)}")
                print("Training will continue with randomly initialized weights")
        
        return model

    def initialize_model(self):
        """Initialize model, criterion, and optimizer."""
        if self.arg.phase == 'train':
            self.model = self.load_model()
            self.criterion = torch.nn.CrossEntropyLoss()
            self.setup_optimizer()
        else:
            self.model = self.load_model()

    @staticmethod
    def import_class(name):
        """Import a class from a string."""
        try:
            components = name.split('.')
            mod = __import__(components[0])
            for comp in components[1:]:
                mod = getattr(mod, comp)
            return mod
        except Exception as e:
            print(f"Error importing class {name}: {str(e)}")
            raise
    def set_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.arg.seed)
        np.random.seed(self.arg.seed)
        torch.manual_seed(self.arg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.arg.seed)
    

    def setup_optimizer(self):
        """Setup optimizer based on arguments."""
        if self.arg.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")
    
    def print_model_info(self):
        """Print model information."""
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'# Parameters: {num_params}')
        print(f'Model size: {num_params * 4 / (1024 * 1024):.2f} MB')
    
    def load_data(self):
        """Load and prepare datasets."""
        from utils.dataset import prepare_smartfallmm, filter_subjects
        
        builder = prepare_smartfallmm(self.arg)
        train_data = filter_subjects(builder, self.train_subjects)
        val_data = filter_subjects(builder, self.test_subjects)
        
        # Import the dataset class dynamically
        Feeder = self.import_class(self.arg.feeder)
        
        # Create datasets
        train_dataset = Feeder(**self.arg.train_feeder_args, dataset=train_data)
        val_dataset = Feeder(**self.arg.val_feeder_args, dataset=val_data)
        
        # Create dataloaders
        self.data_loader = {
            'train': torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker
            ),
            'val': torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker
            )
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.data_loader['train'], desc='Training')
        for batch_idx, (inputs, labels, idx) in enumerate(pbar):
            # Get accelerometer data - try both possible keys
            if 'accelerometer' in inputs:
                data = inputs['accelerometer'].to(self.device)
            elif 'acc_data' in inputs:
                data = inputs['acc_data'].to(self.device)
            else:
                raise KeyError("No accelerometer data found in inputs")
                
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data.float())
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(labels)
            pred = outputs.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += len(labels)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': total_loss/total_samples,
                'acc': total_correct/total_samples*100,
                'f1': f1_score(all_labels, all_preds, average='macro')*100
            })
        
        return total_loss/total_samples, total_correct/total_samples*100
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (inputs, labels, idx) in enumerate(self.data_loader['val']):
            # Get accelerometer data - try both possible keys
            if 'accelerometer' in inputs:
                data = inputs['accelerometer'].to(self.device)
            elif 'acc_data' in inputs:
                data = inputs['acc_data'].to(self.device)
            else:
                raise KeyError("No accelerometer data found in inputs")
                
            labels = labels.to(self.device)
            
            outputs = self.model(data.float())
            loss = self.criterion(outputs, labels)
            
            pred = outputs.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            total_correct += (pred == labels).sum().item()
            total_samples += len(labels)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, average='macro') * 100
        return (
            total_loss/total_samples,
            total_correct/total_samples*100,
            f1
        )
    
    def train(self):
        """Main training loop."""
        best_acc = 0
        best_f1 = 0
        
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            print(f'\nEpoch {epoch+1}/{self.arg.num_epoch}')
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_f1 = self.validate()
            
            # Print metrics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%')
            
            # Save best model
            if val_acc > best_acc or (val_acc == best_acc and val_f1 > best_f1):
                best_acc = val_acc
                best_f1 = val_f1
                model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}_best.pth'
                torch.save(self.model.state_dict(), model_path)
                print(f'Model saved to {model_path}!')
    
    @staticmethod
    def import_class(name):
        """Dynamically import a class."""
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

def main():
    """Main function."""
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Load config file
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update args with config file values
        for k, v in config.items():
            setattr(args, k, v)
    
    # Create trainer
    trainer = AccelTrainer(args)
    
    # Start training or testing
    if args.phase == 'train':
        trainer.load_data()
        trainer.train()
    else:
        print("Testing phase not implemented yet.")

if __name__ == '__main__':
    main()
