
import traceback
from typing import List
import random 
import sys
import os
import time
import shutil
import argparse
import yaml

# Environmental imports
import numpy as np 
import pandas as pd
import torch

import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score, accuracy_score


# Local imports 
from utils.dataset import prepare_smartfallmm, filter_subjects

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_args():


    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--config' , default = './config/smartfallmm/teacher.yaml')
    parser.add_argument('--dataset', type = str, default= 'utd' )
    # Training
    #parser.add_argument('--patience', type = int, default = 25, metavar = 'N', help = 'early stopping patience')
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
    # torch.backends.cudnn.enabled = True
    # training speed is too slow if set to True
    torch.backends.cudnn.deterministic = False

    # on cuda 11 cudnn8, the default algorithm is very slow
    # unlike on cuda 10, the default works well
    torch.backends.cudnn.benchmark = True

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

class Trainer():
    
    def __init__(self, arg):
        self.best_recall = 0
        self.best_roc_auc = 0
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_f1 = 0
        self.best_loss = 0 
        self.best_accuracy = 0 
        self.train_subjects = []
        self.test_subject = []
        self.optimizer = None
        self.data_loader = dict()
        self.inertial_sensors = []
        self.patience = 25
        # Initialize best metrics dictionary
        self.best_metrics = {
            'accuracy': float('-inf'),
            'f1': float('-inf'),
            'recall': float('-inf'),
            'roc_auc': float('-inf'),
            'loss': float('inf')
        }
        
        # Process modalities
        for modality in arg.dataset_args['modalities']:
            if modality != 'skeleton':
                self.inertial_sensors.extend(
                    [f"{modality}_{sensor}" for sensor in arg.dataset_args['sensors'][modality]]
                )
            
                
    
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)                     
            self.save_config(arg.config, arg.work_dir)
        
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else: 
            use_cuda = torch.cuda.is_available()
            self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
            self.model = torch.load(self.arg.weights)
        
        self.load_loss()
        self.load_optimizer()  # Add this line to initialize the optimizer
        
        self.include_val = arg.include_val
        
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size : {num_params/ (1024 ** 2):.2f} MB')

    
    def save_config(self,src_path : str, desc_path : str) -> None: 
        '''
        Function to save configuration file
        ''' 
        print(f'{desc_path}/{src_path.rpartition("/")[-1]}') 
        shutil.copy(src_path, f'{desc_path}/{src_path.rpartition("/")[-1]}')
    def movement_aware_loss(predictions, labels, watch_smoothness, phone_smoothness):
        """
        Custom loss function that considers both classification accuracy and
        movement characteristics to prevent misclassification of normal activities.
        """
        # Base binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, labels)
        
        # Additional smoothness-based penalty
        avg_smoothness = (watch_smoothness + phone_smoothness) / 2
        smooth_penalty = torch.where(
            (avg_smoothness > 0.8) & (predictions > 0) & (labels == 0),
            torch.ones_like(predictions) * 0.5,
            torch.zeros_like(predictions)
        ).mean()
        
        return bce_loss + smooth_penalty
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
    def is_best_model(self, current_metrics):
        """
        Determines if the current model is the best so far and handles early stopping logic.
        Returns True if model improved, False otherwise.
        """
        improved = False
        
        # First check if accuracy has improved
        if current_metrics['accuracy'] > self.best_metrics['accuracy'] + self.min_delta:
            improved = True
        # If accuracy is equal, check if F1 score has improved
        elif (abs(current_metrics['accuracy'] - self.best_metrics['accuracy']) < self.min_delta and 
            current_metrics['f1'] > self.best_metrics['f1'] + self.min_delta):
            improved = True
        # If accuracy and F1 are equal, use loss as tiebreaker (lower is better)
        elif (abs(current_metrics['accuracy'] - self.best_metrics['accuracy']) < self.min_delta and 
            abs(current_metrics['f1'] - self.best_metrics['f1']) < self.min_delta and 
            current_metrics['loss'] < self.best_metrics['loss'] - self.min_delta):
            improved = True

        # Update early stopping counter and flag
        if improved:
            self.early_stop_counter = 0
            # Update all best metrics when we find a better model
            for metric in self.best_metrics:
                self.best_metrics[metric] = current_metrics[metric]
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience: #patience
                self.early_stop = True
                self.print_log(f"\nEarly stopping triggered after {self.patience} epochs without improvement")

        return improved
    
        
    def load_model(self, model, model_args):
        '''
        Function to load model 
        '''
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        Model = import_class(model)
        model = Model(**model_args).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
        return model 
    
    def load_loss(self):
        '''
        Loading loss function for the model's training. Using BCEWithLogitsLoss for binary 
        fall detection as it combines sigmoid activation and binary cross entropy in a 
        numerically stable way.
        '''
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def load_weights(self):
        '''
        Load weights to the model
        '''
        self.model.load_state_dict(torch.load(self.arg.weights))
    
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
    
    def load_data(self):
        '''
        Loads training, validation and test datasets with specific subject splitting:
        - Test: Last 2 subjects
        - Validation: Next to last 2 subjects 
        - Train: All remaining subjects
        '''
        Feeder = import_class(self.arg.feeder)
        
        # Get all matched trials
        builder = prepare_smartfallmm(self.arg)

        # Get and sort all available subjects
        all_subjects = sorted(list({trial.subject_id for trial in builder.dataset.matched_trials}))
        
        # Split subjects according to specification
        self.val_subjects = all_subjects[-2:]  # Last 2 subjects
        print("val_subjects", self.val_subjects)
        
        self.test_subjects = all_subjects[-4:-2]  # Next to last 2 subjects
        print("test_subjects", self.test_subjects)

        self.train_subjects = all_subjects[:-4]  # All remaining subjects
        print("train_subjects", self.train_subjects)
        
        print(f"\nSubject Split:")
        print(f"Training subjects: {self.train_subjects}")
        print(f"Validation subjects: {self.val_subjects}")
        print(f"Test subjects: {self.test_subjects}")

        # Create dataloaders for each split
        for split, subjects in [('train', self.train_subjects), 
                            ('val', self.val_subjects),
                            ('test', self.test_subjects)]:
            filtered_data = filter_subjects(builder, subjects)
            if not filtered_data:
                print(f"No {split} data was loaded!")
                return False
                
            self.data_loader[split] = torch.utils.data.DataLoader(
                dataset=Feeder(dataset=filtered_data, 
                            batch_size=getattr(self.arg, f'{split}_batch_size' if split != 'train' else 'batch_size')),
                batch_size=getattr(self.arg, f'{split}_batch_size' if split != 'train' else 'batch_size'),
                shuffle=(split == 'train'),
                num_workers=self.arg.num_worker
            )
            print(f"Created {split} dataloader with {len(self.data_loader[split].dataset)} samples")
        
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
        print(string)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(string, file = f)

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
            # Handle IMU data
            for sensor in self.inertial_sensors:
                sensor_data[sensor] = inputs[sensor].to(
                    f'cuda:{self.output_device}' if use_cuda else 'cpu'
                )

            # Handle skeleton data
            if 'skeleton' in self.arg.dataset_args['modalities']:
                sensor_data['skeleton'] = inputs['skeleton'].to(
                    f'cuda:{self.output_device}' if use_cuda else 'cpu'
                )

            targets = targets.float().unsqueeze(1).to(
                f'cuda:{self.output_device}' if use_cuda else 'cpu'
            )

            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()
            logits = self.model(sensor_data)

            if logits is None:
                print("Model returned None. Skipping batch.")
                continue

            loss = self.criterion(logits, targets)
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
        use_cuda = torch.cuda.is_available()
        if result_file is not None:
            f_r = open(result_file, 'w', encoding='utf-8')
        self.model.eval()

        loss = 0
        cnt = 0
        all_targets = []
        all_probs = []

        process = tqdm(self.data_loader[loader_name], ncols=80)
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                sensor_data = {}
                # Process sensor data
                for sensor in self.inertial_sensors:
                    sensor_data[sensor] = inputs[sensor].to(
                        f'cuda:{self.output_device}' if use_cuda else 'cpu'
                    )

                # Handle skeleton data
                if 'skeleton' in self.arg.dataset_args['modalities']:
                    sensor_data['skeleton'] = inputs['skeleton'].to(
                        f'cuda:{self.output_device}' if use_cuda else 'cpu'
                    )

                targets = targets.float().unsqueeze(1).to(
                    f'cuda:{self.output_device}' if use_cuda else 'cpu'
                )

                logits = self.model(sensor_data)

                batch_loss = self.criterion(logits, targets)
                loss += batch_loss.sum().item()

                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                cnt += targets.size(0)

        # Compute metrics
        loss /= cnt
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        preds = (all_probs > 0.5).astype(int)

        accuracy = accuracy_score(all_targets, preds) * 100
        recall = recall_score(all_targets, preds) * 100
        f1 = f1_score(all_targets, preds) * 100
        roc_auc = roc_auc_score(all_targets, all_probs) * 100

        # Create metrics dictionary
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

        self.print_log('{} Loss: {:.4f}. Accuracy: {:.2f}%, Recall: {:.2f}%, F1 Score: {:.2f}%, ROC AUC: {:.2f}%'.format(
            loader_name.capitalize(), loss, accuracy, recall, f1, roc_auc
        ))

        # Save best model during training
        if self.arg.phase == 'train':
            if accuracy > self.best_accuracy:
                self.best_loss = loss
                self.best_accuracy = accuracy
                self.best_f1 = f1
                self.best_recall = recall
                self.best_roc_auc = roc_auc
                torch.save(self.model, f'{self.arg.work_dir}/{self.arg.model_saved_name}.pth')
                self.print_log('Weights Saved')

        return metrics  # Return the metrics dictionary instead of just loss

        



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
        self.print_log(f"ROC AUC:      {metrics['roc_auc']:.2f}%")
        self.print_log(f"Loss:         {metrics['loss']:.4f}")
        self.print_log("=" * 50)

    def save_final_results(self, train_metrics, val_metrics, test_metrics):
        """Save all results to a CSV file"""
        results = pd.DataFrame({
            'train_subjects': [str(self.train_subjects)],
            'val_subjects': [str(self.val_subjects)],
            'test_subjects': [str(self.test_subjects)],
            'train_accuracy': [train_metrics['accuracy']],
            'train_f1': [train_metrics['f1']],
            'train_recall': [train_metrics['recall']],
            'val_accuracy': [val_metrics['accuracy']],
            'val_f1': [val_metrics['f1']],
            'val_recall': [val_metrics['recall']],
            'test_accuracy': [test_metrics['accuracy']],
            'test_f1': [test_metrics['f1']],
            'test_recall': [test_metrics['recall']]
        })
        results.to_csv(f'{self.arg.work_dir}/final_results.csv', index=False)
        self.print_log("\nSaved final results to: final_results.csv")

    def start(self):
        """Main training and evaluation loop with comprehensive testing"""
        if not self.load_data():
            return

        if self.arg.phase == 'train':
            self.print_log("\nStarting Training...")
            self.print_log("=" * 50)
            self.print_log(f"Early stopping patience: {self.patience} epochs")
            
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
                    model_save_path = (f'{self.arg.work_dir}/best_model_'
                                    f'acc_{val_metrics["accuracy"]:.2f}_'
                                    f'f1_{val_metrics["f1"]:.2f}_'
                                    f'loss_{val_metrics["loss"]:.4f}.pth')
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'validation_metrics': val_metrics,
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
                    torch.save(checkpoint, f'{self.arg.work_dir}/best_model.pth')
                    
                    self.print_log(
                        f'\nEpoch {epoch}: New best model saved!'
                        f'\nValidation Metrics:'
                        f'\n - Accuracy: {val_metrics["accuracy"]:.2f}%'
                        f'\n - F1 Score: {val_metrics["f1"]:.2f}%'
                        f'\n - Recall: {val_metrics["recall"]:.2f}%'
                        f'\n - ROC AUC: {val_metrics["roc_auc"]:.2f}%'
                        f'\n - Loss: {val_metrics["loss"]:.4f}'
                    )
                
                # Check for early stopping
                if self.early_stop:
                    self.print_log(f"\nStopping early at epoch {epoch}")
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
            self.print_log("\nRunning inference only...")
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

    init_seed(arg.seed)
    trainer = Trainer(arg)
    trainer.start()
