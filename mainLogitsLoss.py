import os
import sys
import time
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from main import import_class

import traceback
from typing import List
import argparse
import yaml

# Environmental imports
import pandas as pd
from tqdm import tqdm
import argparse
import yaml

# Local imports 
from utils.dataset import prepare_smartfallmm, filter_subjects

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_args():


    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--config' , default = './config/smartfallmm/teacher.yaml')
    parser.add_argument('--dataset', type = str, default= 'utd' )
    # Training
    parser.add_argument('--batch-size', type = int, default = 16, metavar = 'N',
                        help = 'input batch size for training (default: 8)')

    parser.add_argument('--test-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size for testing(default: 1000)')
    parser.add_argument('--val-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size for testing(default: 1000)')

    parser.add_argument('--num-epoch', type = int , default = 150, metavar = 'N', 
                        help = 'number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type = int, default = 0)

    # Optimizer
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--base-lr', type = float, default = 0.001, metavar = 'LR',
                        help = 'learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type = float , default=0.0004)

    # Model
    parser.add_argument('--model' ,default= None, help = 'Name of Model to load')

    # Model args
    parser.add_argument('--device', nargs='+', default=[0], type = int)

    parser.add_argument('--model-args', default= str, help = 'A dictionary for model args')
    parser.add_argument('--weights', type = str, help = 'Location of weight file')
    parser.add_argument('--model-saved-name', type = str, help = 'Weight name', default='test')

    # Loss args
    parser.add_argument('--loss', default='loss.BCE' , help = 'Name of loss function to use' )
    parser.add_argument('--loss-args', default ="{}", type = str,  help = 'A dictionary for loss')
    parser.add_argument('--loss-type', type=str, default='bce', help='Type of loss function to use (bce or focal)')
    
    # Dataset args 
    parser.add_argument('--dataset-args', default=str, help = 'Arguments for dataset')

    # Dataloader 
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--feeder', default= None , help = 'Dataloader location')
    parser.add_argument('--train-feeder-args',default=str, help = 'A dict for dataloader args' )
    parser.add_argument('--val-feeder-args', default=str , help = 'A dict for validation data loader')
    parser.add_argument('--test_feeder_args',default=str, help= 'A dict for test data loader')
    parser.add_argument('--include-val', type = str2bool, default= True , help = 'If we will have the validation set or not')

    # Initialization
    parser.add_argument('--seed', type =  int , default = 2 , help = 'random seed (default: 1)') 

    parser.add_argument('--log-interval', type = int , default = 10, metavar = 'N',
                        help = 'how many batches to wait before logging training status')

   
    parser.add_argument('--work-dir', type = str, default = 'simple', metavar = 'F', help = "Working Directory")
    parser.add_argument('--print-log',type=str2bool,default=True,help='print logging or not')
    
    parser.add_argument('--phase', type = str, default = 'train')
    
    parser.add_argument('--num-worker', type = int, default= 0)
    parser.add_argument('--result-file', type = str, help = 'Name of result file')


    return parser

def str2bool(v):
    '''
    Function to parse boolean from text
    '''
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
def init_seed(seed):
    '''
    Initial seed for reproducibility of the results
    '''
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.enabled = True
    # training speed is too slow if set to True
    torch.backends.cudnn.benchmark = True

    # on cuda 11 cudnn8, the default algorithm is very slow
    # unlike on cuda 10, the default works well

def import_class(import_str):
    if import_str is None:
        raise ValueError("import_str is None")
        
    print(f"Importing: {import_str}")  # Debug print
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

# Add this near the start of main.py
print("Current directory:", os.getcwd())
print("PYTHONPATH:", os.environ.get('PYTHONPATH'))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class Trainer():
    
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.init_seed(arg.seed)

        # Initialize training metrics and summaries
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.test_loss_summary = []
        self.train_acc_summary = []
        self.val_acc_summary = []
        self.test_acc_summary = []
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.global_step = 0

        # Initialize sensors and modalities
        self.inertial_sensors = []
        for modality in arg.dataset_args['modalities']:
            if modality != 'skeleton':
                self.inertial_sensors.extend(
                    [f"{modality}_{sensor}" for sensor in arg.dataset_args['sensors'][modality]]
                )

        # Initialize model components
        self.load_data()  # Load data regardless of phase
        self.load_model()  # Load model before optimizer
        self.load_optimizer()  # Now load optimizer
        self.load_loss()  # Move loss loading after data is loaded

        # Initialize early stopping parameters
        self.best_metrics = {
            'accuracy': 0,
            'f1': 0,
            'recall': 0,
            'precision': 0,
            'loss': float('inf')
        }
        self.early_stop = False
        self.early_stop_counter = 0
        self.patience = 25  # Number of epochs to wait for improvement
        self.min_delta = 0.0001  # Minimum change to qualify as an improvement

        # Start appropriate phase
        if arg.phase == 'train':
            self.start()  # Start training
        elif arg.phase == 'test':
            self.load_weights(self.arg.weights)
            self.eval()  # Start evaluation
        else:
            raise ValueError(f"Unknown phase: {arg.phase}")

    def init_seed(self, seed):
        """
        Initialize random seeds for reproducibility
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.print_log(f'Random seed set to {seed}')

    def save_arg(self):
        '''
        Function to save configuration file
        ''' 
        print(f'{self.arg.work_dir}/{self.arg.config.rpartition("/")[-1]}') 
        shutil.copy(self.arg.config, f'{self.arg.work_dir}/{self.arg.config.rpartition("/")[-1]}')
    
    def load_loss(self):
        """
        Initialize the loss function with class weights to handle imbalance
        """
        if not hasattr(self, 'data_loader'):
            raise RuntimeError("Data loader must be initialized before loading loss function")

        # Calculate class weights based on training data distribution
        labels = self.data_loader['train'].dataset.labels
        fall_samples = sum(labels == 1)
        non_fall_samples = sum(labels == 0)
        total_samples = len(labels)

        # Print length of each sample
        print(f"Total samples: {total_samples}")
        print(f"Fall samples: {fall_samples}")
        print(f"Non-fall samples: {non_fall_samples}")

        # Calculate weights inversely proportional to class frequencies
        weight_for_0 = total_samples / (2.0 * non_fall_samples)
        weight_for_1 = total_samples / (2.0 * fall_samples)
        
        class_weights = torch.tensor([weight_for_0, weight_for_1], device=self.arg.device[0])
        
        # Initialize the loss function based on configuration
        loss_type = getattr(self.arg, 'loss_type', 'bce')  # Default to BCE if not specified
        
        if loss_type.lower() == 'focal':
            self.loss = FocalLoss(alpha=class_weights, gamma=2.0)
            self.print_log(f"Using Focal Loss with class weights: {class_weights}")
        else:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_for_1/weight_for_0], device=self.arg.device[0]))
            self.print_log(f"Using BCE Loss with positive weight: {weight_for_1/weight_for_0}")

        return True

    def count_parameters(self, model):
        '''
        Function to count the trainable parameters
        '''
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.nelement() * buffer.element_size()
        return total_size

    def load_model(self, model=None, model_args=None):
        '''
        Function to load model 
        '''
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        if model is None:
            Model = import_class(self.arg.model)
            self.model = Model(**self.arg.model_args).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
        else:
            self.model = model.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
        return self.model 
    
    def load_optimizer(self) -> None:
        '''
        Loads Optimizers
        '''
        
        
        if self.arg.optimizer.lower() == "adam" :
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr = self.arg.base_lr,
                # weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr = self.arg.base_lr, 
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr = self.arg.base_lr,
                weight_decay = self.arg.weight_decay
            )
        
        else :
           raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")
    
    def load_weights(self):
        '''
        Load weights to the model
        '''
        self.model.load_state_dict(torch.load(self.arg.weights))
    
    def load_data(self):
        """
        Loads training, validation and test datasets with matched trials and subject filtering.
        Uses the same approach as main.py with prepare_smartfallmm and filter_subjects.
        """
        Feeder = import_class(self.arg.feeder)
        
        # Get all matched trials using prepare_smartfallmm
        builder = prepare_smartfallmm(self.arg)

        # Get and sort all available subjects
        all_subjects = sorted(list({trial.subject_id for trial in builder.dataset.matched_trials}))
        
        # Split subjects according to specification
        self.test_subjects = all_subjects[-2:]  # Last 2 subjects for testing
        self.val_subjects = all_subjects[-4:-2]  # Next to last 2 subjects for validation
        self.train_subjects = all_subjects[:-4]  # All remaining subjects for training
        
        self.print_log(f"\nSubject Split:")
        self.print_log(f"Training subjects: {self.train_subjects}")
        self.print_log(f"Validation subjects: {self.val_subjects}")
        self.print_log(f"Test subjects: {self.test_subjects}")

        self.data_loader = {}

        # Create dataloaders for each split using filter_subjects
        for split, subjects in [('train', self.train_subjects), 
                            ('val', self.val_subjects),
                            ('test', self.test_subjects)]:
            
            # Filter data for current subjects
            filtered_data = filter_subjects(builder, subjects)
            if not filtered_data:
                self.print_log(f"No {split} data was loaded!")
                return False
                
            # Create dataset with filtered data
            dataset = Feeder(dataset=filtered_data, batch_size=self.arg.batch_size)
            
            # For training set, create balanced sampler
            if split == 'train':
                labels = dataset.labels
                class_counts = np.bincount(labels)
                total_samples = len(labels)
                
                # Calculate weights inversely proportional to class frequencies
                class_weights = total_samples / (len(class_counts) * class_counts)
                sample_weights = class_weights[labels]
                
                sampler = WeightedRandomSampler(
                    weights=torch.DoubleTensor(sample_weights),
                    num_samples=len(dataset),
                    replacement=True
                )
                
                self.data_loader[split] = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.arg.batch_size,
                    sampler=sampler,  # Use balanced sampler for training
                    num_workers=self.arg.num_worker,
                    drop_last=True
                )
            else:
                # For validation and test sets, no sampling needed
                self.data_loader[split] = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=getattr(self.arg, f'{split}_batch_size'),
                    shuffle=False,
                    num_workers=self.arg.num_worker
                )
            
            self.print_log(f"Created {split} dataloader with {len(dataset)} samples")
            
            # Store class distribution information
            if split == 'train':
                self.train_class_counts = class_counts
                self.print_log(f"Training set class distribution: {class_counts}")

        return True

    def record_time(self):
        '''
        Function to record time
        '''
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        '''
        Split time 
        '''
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string : str, print_time = True) -> None:
        '''
        Prints log to a file
        '''
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            string = f"[ {localtime} ] {string}"
        print(string)
        if hasattr(self, 'log_path'):
            with open(self.log_path, 'a') as f:
                print(string, file=f)

    def start(self):
        """
        Main training loop
        """
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            self.train(epoch)
            self.eval(epoch)

    def train(self, epoch):
        use_cuda = torch.cuda.is_available()
        self.model.train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader=0.001, model=0.001, stats=0.001)
        cnt = 0
        train_loss = 0
        all_targets = []
        all_probs = []

        process = tqdm(loader, ncols=80)

        for batch_idx, (inputs, targets, idx) in enumerate(process):
            sensor_data = {}
            for sensor in self.inertial_sensors:
                sensor_data[sensor] = inputs[sensor].to(
                    f'cuda:{self.output_device}' if use_cuda else 'cpu'
                )

            targets = targets.float().unsqueeze(1).to(
                f'cuda:{self.output_device}' if use_cuda else 'cpu'
            )

            self.optimizer.zero_grad()
            logits = self.model(sensor_data)

            if logits is None:
                print("Model returned None. Skipping batch.")
                continue

            loss = self.loss(logits, targets)
            loss.mean().backward()
            self.optimizer.step()

            timer['model'] += self.split_time()

            with torch.no_grad():
                train_loss += loss.mean().item() * targets.size(0)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            cnt += targets.size(0)

        # Compute metrics
        train_loss /= cnt
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        preds = (all_probs > 0.5).astype(int)

        accuracy = accuracy_score(all_targets, preds) * 100
        recall = recall_score(all_targets, preds) * 100
        f1 = f1_score(all_targets, preds) * 100
        roc_auc = roc_auc_score(all_targets, all_probs) * 100

        self.train_loss_summary.append(train_loss)

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }

        self.print_log(
            '\tTraining Loss: {:.4f}. Accuracy: {:.2f}%, Recall: {:.2f}%, F1 Score: {:.2f}%, ROC AUC: {:.2f}%'.format(
                train_loss, accuracy, recall, f1, roc_auc
            )
        )
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))


    def eval(self, epoch, loader_name='test', result_file=None):
        self.model.eval()
        loss_values = []
        all_targets = []
        all_probs = []
        process = tqdm(self.data_loader[loader_name], ncols=80)
        use_cuda = torch.cuda.is_available()

        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                sensor_data = {}
                # Process sensor data
                for sensor in self.inertial_sensors:
                    sensor_data[sensor] = inputs[sensor].to(
                        f'cuda:{self.output_device}' if use_cuda else 'cpu'
                    )

                targets = targets.float().unsqueeze(1).to(
                    f'cuda:{self.output_device}' if use_cuda else 'cpu'
                )

                logits = self.model(sensor_data)
                if logits is None:
                    continue

                loss = self.loss(logits, targets)
                loss_values.append(loss.mean().item() * targets.size(0))
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Convert to numpy arrays for metric calculation
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        preds = (all_probs > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(all_targets, preds) * 100
        recall = recall_score(all_targets, preds) * 100
        precision = precision_score(all_targets, preds) * 100
        f1 = f1_score(all_targets, preds) * 100
        roc_auc = roc_auc_score(all_targets, all_probs) * 100
        loss = np.sum(loss_values) / len(all_targets)

        # Store metrics in a dictionary
        metrics = {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'roc_auc': roc_auc,
            'loss': loss
        }

        # Print evaluation results
        self.print_log(
            '\t{} Loss: {:.4f}. Accuracy: {:.2f}%, Recall: {:.2f}%, F1 Score: {:.2f}%, ROC AUC: {:.2f}%'.format(
                loader_name.capitalize(), loss, accuracy, recall, f1, roc_auc
            )
        )

        return metrics

    def is_best_model(self, current_metrics):
        """
        Determines if the current model is the best so far based on F1 score and other metrics
        """
        is_best = False
        
        # Check if current F1 score is better than the best so far
        if current_metrics['f1'] > self.best_metrics['f1']:
            is_best = True
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            self.print_log(f'No improvement for {self.early_stop_counter} epochs')
            self.print_log(f"Best F1: {self.best_metrics['f1']:.4f}, Current F1: {current_metrics['f1']:.4f}")

        # Update all metrics if this is the best model
        if is_best:
            for metric in ['accuracy', 'recall', 'precision', 'f1', 'loss']:
                self.best_metrics[metric] = current_metrics[metric]

        # Check for early stopping
        if self.early_stop_counter >= self.patience:
            self.early_stop = True

        return is_best

    def load_best_checkpoint(self):
        """Load the best model checkpoint before final testing"""
        try:
            checkpoint = torch.load(f'{self.arg.work_dir}/best_model.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            best_val_metrics = checkpoint['validation_metrics']
            self.print_log("\nLoaded best model from checkpoint:")
            self.print_log(f"Best validation accuracy: {best_val_metrics['accuracy']:.2f}%")
            self.print_log(f"Best validation F1 score: {best_val_metrics['f1']:.2f}%")
            return True
        except:
            self.print_log("No best model checkpoint found!")
            return False

    def print_final_results(self, metrics, split='test'):
        """Print final evaluation results in a clear format"""
        self.print_log(f"\nFinal {split.capitalize()} Results:")
        self.print_log("=" * 50)
        self.print_log(f"Accuracy:     {metrics['accuracy']:.2f}%")
        self.print_log(f"F1 Score:     {metrics['f1']:.2f}%")
        self.print_log(f"Recall:       {metrics['recall']:.2f}%")
        self.print_log(f"Precision:    {metrics['precision']:.2f}%")
        self.print_log(f"ROC AUC:      {metrics['roc_auc']:.2f}%")
        self.print_log(f"Loss:         {metrics['loss']:.4f}")
        self.print_log("=" * 50)

    def save_final_results(self, train_metrics, val_metrics, test_metrics):
        """
        Save test results to a CSV file, appending new results with metadata.
        Includes model name, checkpoint file, and timestamp.
        """
        from datetime import datetime
        import pandas as pd
        
        # Get model name from the full path
        model_name = self.arg.model.split('.')[-1]
        
        # Get the best model checkpoint filename
        checkpoint_path = f'{self.arg.work_dir}/best_model.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            checkpoint_name = f'{model_name}_f1val_{checkpoint["validation_metrics"]["f1"]:.2f}_acc_{checkpoint["validation_metrics"]["accuracy"]:.2f}_loss_{checkpoint["validation_metrics"]["loss"]:.4f}.pth'
        else:
            checkpoint_name = "no_checkpoint_found"
        
        # Create new results row with metadata and only test metrics
        results = pd.DataFrame({
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'model_name': [model_name],
            'checkpoint_file': [checkpoint_name],
            'test_subjects': [str(self.test_subjects)],
            'accuracy': [test_metrics['accuracy']],
            'f1': [test_metrics['f1']],
            'recall': [test_metrics['recall']],
            'precision': [test_metrics['precision']],
            'roc_auc': [test_metrics['roc_auc']],
            'loss': [test_metrics['loss']]
        })
        
        # Append to existing CSV or create new one
        results_path = f'{self.arg.work_dir}/test_results_history.csv'
        if os.path.exists(results_path):
            existing_results = pd.read_csv(results_path)
            updated_results = pd.concat([existing_results, results], ignore_index=True)
        else:
            updated_results = results
            
        # Save with nice formatting
        updated_results.to_csv(results_path, index=False, float_format='%.4f')
        self.print_log(f"\nAppended test results to: {results_path}")
        
        # Also save current results separately for easy access
        current_results_path = f'{self.arg.work_dir}/current_test_results.csv'
        results.to_csv(current_results_path, index=False, float_format='%.4f')
        self.print_log(f"Current test results saved to: {current_results_path}")
        
    def load_checkpoint_if_specified(self):
        """Load model checkpoint if specified"""
        if self.arg.weights is not None:
            try:
                checkpoint = torch.load(self.arg.weights)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.global_step = checkpoint['global_step']
                self.print_log(f"Loaded checkpoint from {self.arg.weights}")
            except:
                self.print_log(f"Failed to load checkpoint from {self.arg.weights}")

    def start(self):
        """Main training and evaluation loop with comprehensive testing"""
        if not self.load_data():
            return

        if self.arg.phase == 'train':
            self.print_log("\nStarting Training...")
            self.print_log("=" * 50)
            self.print_log(f"Early stopping patience: {self.patience} epochs")
            self.print_log(f"Minimum improvement threshold: {self.min_delta}")
            
            # Training loop
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.print_log(f"\nEpoch {epoch}/{self.arg.num_epoch-1}")
                self.print_log("-" * 30)
                
                # Training phase
                self.train(epoch)
                
                # Validation phase
                val_metrics = self.eval(epoch, loader_name='val')
                
                # Save if best model
                if self.is_best_model(val_metrics):
                    # Extract model name from the full path
                    model_name = self.arg.model.split('.')[-1]
                    
                    model_save_path = (f'{self.arg.work_dir}/{model_name}_'
                                    f'f1val_{val_metrics["f1"]:.2f}_'
                                    f'acc_{val_metrics["accuracy"]:.2f}_'
                                    f'loss_{val_metrics["loss"]:.4f}.pth')
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'validation_metrics': val_metrics,
                        'model_name': model_name,
                        'training_info': {
                            'train_subjects': self.train_subjects,
                            'val_subjects': self.val_subjects,
                            'test_subjects': self.test_subjects,
                            'batch_size': self.arg.batch_size,
                            'learning_rate': self.arg.base_lr,
                            'optimizer': self.arg.optimizer
                        }
                    }
                    
                    torch.save(checkpoint, model_save_path)
                    # Also save as best_model.pth for easy reference
                    torch.save(checkpoint, f'{self.arg.work_dir}/best_model.pth')
                    
                    self.print_log(
                        f'\nEpoch {epoch}: New best model saved! ({model_name})'
                        f'\nValidation Metrics:'
                        f'\n - Accuracy: {val_metrics["accuracy"]:.2f}%'
                        f'\n - F1 Score: {val_metrics["f1"]:.2f}%'
                        f'\n - Recall: {val_metrics["recall"]:.2f}%'
                        f'\n - Precision: {val_metrics["precision"]:.2f}%'
                        f'\n - ROC AUC: {val_metrics["roc_auc"]:.2f}%'
                        f'\n - Loss: {val_metrics["loss"]:.4f}'
                        f'\nSaved to: {model_save_path}'
                    )
                
                # Check for early stopping
                if self.early_stop:
                    self.print_log(f"\nStopping training early at epoch {epoch}")
                    break

            # After training, evaluate on all splits using best model
            self.print_log("\nTraining Complete! Evaluating final performance...")
            
            # Load best model
            if self.load_best_checkpoint():
                # Get final metrics for all splits
                train_metrics = self.eval(0, loader_name='train')
                val_metrics = self.eval(0, loader_name='val')
                test_metrics = self.eval(0, loader_name='test')
                
                # Print comprehensive results
                self.print_log("\nFinal Evaluation Results")
                self.print_log("=" * 50)
                self.print_final_results(train_metrics, 'train')
                self.print_final_results(val_metrics, 'validation')
                self.print_final_results(test_metrics, 'test')
                
                # Save all results
                self.save_final_results(train_metrics, val_metrics, test_metrics)
                
            else:
                self.print_log("Could not perform final evaluation - no best model found!")
        
        else:  # Testing phase
            self.print_log("\nRunning inference with best model...")
            
            # Load the best model from training
            best_model_path = f'{self.arg.work_dir}/best_model.pth'
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                model_name = checkpoint.get('model_name', 'Unknown')
                best_val_metrics = checkpoint.get('validation_metrics', {})
                
                self.print_log(f"Loaded best model: {model_name}")
                self.print_log("Best validation metrics:")
                self.print_log(f" - Accuracy: {best_val_metrics.get('accuracy', 'N/A'):.2f}%")
                self.print_log(f" - F1 Score: {best_val_metrics.get('f1', 'N/A'):.2f}%")
                self.print_log(f" - Loss: {best_val_metrics.get('loss', 'N/A'):.4f}")
            else:
                self.print_log("Warning: No best model found, using current model state")
                
            test_metrics = self.eval(0, loader_name='test')
            self.print_final_results(test_metrics, 'test')


if __name__ == "__main__":
    parser = get_args()

    # Load arg from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding= 'utf-8') as f:
            default_arg = yaml.safe_load(f)
        
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    trainer = Trainer(arg)
    trainer.start()
