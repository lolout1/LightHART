'''
Script to train the models
'''
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
    '''
    Function to build Argument Parser
    '''

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

    parser.add_argument('--num-epoch', type = int , default = 70, metavar = 'N', 
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
        Loads training and validation datasets
        '''
        Feeder = import_class(self.arg.feeder)
    
        # First get all matched trials
        builder = prepare_smartfallmm(self.arg)

        # Get available subjects from matched trials
        all_trial_subjects = set(trial.subject_id for trial in builder.dataset.matched_trials)
        print(f"All available subjects in matched trials: {sorted(list(all_trial_subjects))}")
        
        # Filter the subjects that are in our arg.subjects list and in the matched trials
        available_subjects = sorted(list(all_trial_subjects & set(self.arg.subjects)))
        print(f"Subjects available for training/validation: {available_subjects}")
        
        if not available_subjects:
            print("No subjects available that match both matched trials and requested subjects!")
            return False
            
        # Split for training and validation
        val_subjects = available_subjects[-3:]  # Take last 3 subjects for validation
        train_subjects = available_subjects[:-3]  # Rest for training
        
        print(f"\nSplit:")
        print(f"Training subjects: {train_subjects}")
        print(f"Validation subjects: {val_subjects}")

        self.train_subjects = train_subjects
        self.test_subject = val_subjects  # Store for later use

        # Prepare training data
        print("\nPreparing training data...")
        norm_train = filter_subjects(builder, train_subjects)
        if not norm_train:
            print("No training data was loaded. Exiting dataset preparation.")
            return False

        print("\nCreating training dataloader...")
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(dataset=norm_train, batch_size=self.arg.batch_size),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker
        )
        
        print(f"Created training dataloader with {len(self.data_loader['train'].dataset)} samples")

        # Prepare validation data
        if self.include_val:
            print("\nPreparing validation data...")
            norm_val = filter_subjects(builder, val_subjects)
            if not norm_val:
                print("No validation data was loaded. Continuing without validation.")
            else:
                self.data_loader['val'] = torch.utils.data.DataLoader(
                    dataset=Feeder(dataset=norm_val, batch_size=self.arg.batch_size),  # Set batch_size=1 for real-time inference
                    batch_size=self.arg.batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker
                )
                print(f"Created validation dataloader with {len(self.data_loader['val'].dataset)} samples")
    
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
        else:
            return preds, all_targets, None

        return loss

        

    def start(self):
        '''Function to start the training'''
        if not self.load_data():
            return
        
        # Initialize the results DataFrame
        results = pd.DataFrame()
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_accuracy = float('-inf')
        self.best_f1 = float('-inf')
        self.best_recall = float('-inf')
        self.best_roc_auc = float('-inf')

        self.print_log('Parameters: \n{}\n'.format(str(vars(self.arg))))
        
        
        self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
        
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            self.train(epoch)
            # Evaluate on validation set
            self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
        
        self.print_log(f'Best accuracy: {self.best_accuracy}')
        self.print_log(f'Best recall: {self.best_recall}')
        self.print_log(f'Best F-Score: {self.best_f1}')
        self.print_log(f'Best ROC AUC: {self.best_roc_auc}')
        self.print_log(f'Model name: {self.arg.work_dir}')
        self.print_log(f'Weight decay: {self.arg.weight_decay}')
        self.print_log(f'Base LR: {self.arg.base_lr}')
        self.print_log(f'Batch Size: {self.arg.batch_size}')
        self.print_log(f'Seed: {self.arg.seed}')
        
        # Save results
        subject_result = pd.Series({
            'train_subjects': str(self.train_subjects),
            'val_subjects': str(self.test_subject),
            'accuracy': round(self.best_accuracy, 2),
            'f1_score': round(self.best_f1, 2),
            'recall': round(self.best_recall, 2),
            'roc_auc': round(self.best_roc_auc, 2)
        })
        results = results.append(subject_result, ignore_index=True)
        results.to_csv(f'{self.arg.work_dir}/scores.csv')



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
