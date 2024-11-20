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

# environmental imports
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

# local imports
from utils.dataset import prepare_smartfallmm, filter_subjects


def get_args():
    '''
    Function to build Argument Parser
    '''
    parser = argparse.ArgumentParser(description='Distillation')
    parser.add_argument('--config', default='./config/smartfallmm/teacher.yaml')
    parser.add_argument('--dataset', type=str, default='utd')
    # training
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for testing(default: 1000)')
    parser.add_argument('--val-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for testing(default: 1000)')
    parser.add_argument('--num-epoch', type=int, default=70, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0)
    # optimization
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0004)
    # model
    parser.add_argument('--model', default=None, help='Name of Model to load')
    # model args
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--model-args', default=str, help='A dictionary for model args')
    parser.add_argument('--weights', type=str, help='Location of weight file')
    parser.add_argument('--model-saved-name', type=str, help='Weight name', default='test')
    # loss args
    parser.add_argument('--loss', default='loss.BCE', help='Name of loss function to use')
    parser.add_argument('--loss-args', default="{}", type=str, help='A dictionary for loss')
    # dataset args
    parser.add_argument('--dataset-args', default=str, help='Arguments for dataset')
    # dataloader
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--feeder', default=None, help='Dataloader location')
    parser.add_argument('--train-feeder-args', default=str, help='A dict for dataloader args')
    parser.add_argument('--val-feeder-args', default=str, help='A dict for validation data loader')
    parser.add_argument('--test_feeder_args', default=str, help='A dict for test data loader')
    parser.add_argument('--include-val', type=lambda v: v.lower() in ('yes', 'true', 't', 'y', '1'), default=True,
                        help='If we will have the validation set or not')
    # initialization
    parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--work-dir', type=str, default='simple', metavar='F', help="Working Directory")
    parser.add_argument('--print-log', type=lambda v: v.lower() in ('yes', 'true', 't', 'y', '1'), default=True,
                        help='print logging or not')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str, help='Name of result file')
    return parser


def init_seed(seed):
    '''
    Initialize seed for reproducibility of the results
    '''
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def import_class(import_str):
    '''
    Imports a class dynamically
    '''
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Trainer:
    def __init__(self, arg):
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
        # Update modalities based on dataset args
        self.inertial_modalities = arg.dataset_args.get('sensors', ['watch', 'phone'])
        
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            self.save_config(arg.config, arg.work_dir)
            
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            use_cuda = torch.cuda.is_available()
            self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
            self.model = torch.load(self.arg.weights)
            
        self.load_loss()
        self.include_val = arg.include_val

        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size : {num_params / (1024 ** 2):.2f} MB')

    def process_inputs(self, inputs, use_cuda):
        """Process inputs to handle accelerometer data for both watch and phone"""
        device = f'cuda:{self.output_device}' if use_cuda else 'cpu'
        
        if 'accelerometer' in inputs:
            # Convert inputs to float32 and move to device
            acc_data = inputs['accelerometer'].float().to(device)
            skl_data = inputs['skeleton'].float().to(device)
            
            # Debug prints
            self.print_log(f"Processing batch:")
            self.print_log(f"Accelerometer shape: {acc_data.shape}")
            self.print_log(f"Skeleton shape: {skl_data.shape}")
            
            # Return processed inputs with consistent dtype
            return {
                'acc_data1': acc_data,  # For phone processor
                'acc_data2': acc_data,  # For watch processor
                'skl_data': skl_data
            }
        else:
            self.print_log("Warning: No accelerometer data found in inputs")
            return None

    def create_df(self, columns=['test_subject', 'train_subjects', 'accuracy', 'f1_score']) -> pd.DataFrame:
        """
        Initializes a new DataFrame with specified columns for storing evaluation metrics.
        """
        return pd.DataFrame(columns=columns)

    def save_config(self, src_path: str, dest_path: str) -> None:
        '''
        Function to save configuration file
        '''
        shutil.copy(src_path, f'{dest_path}/{src_path.rpartition("/")[-1]}')

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
        Loading loss function for the models training
        '''
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_optimizer(self) -> None:
        '''
        Loads Optimizers
        '''
        if self.arg.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                # weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError()

    def load_data(self):
        '''
        Loads different datasets
        '''
        Feeder = import_class(self.arg.feeder)

        if self.arg.phase == 'train':
            builder = prepare_smartfallmm(self.arg)

            norm_train = filter_subjects(builder, self.train_subjects)
            norm_val = filter_subjects(builder, self.test_subject)

            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args, dataset=norm_train),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)

            # Removed visualization call
            # self.distribution_viz(norm_train['labels'], self.arg.work_dir, 'train')

            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args, dataset=norm_val),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            # Removed visualization call
            # self.distribution_viz(norm_val['labels'], self.arg.work_dir, 'val')
        else:
            builder = prepare_smartfallmm(self.arg)
            norm_test = filter_subjects(builder, self.test_subject)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset=norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)

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

    def print_log(self, string: str, print_time=True) -> None:
        '''
        Prints log to a file
        '''
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)

    def train(self, epoch):
        '''
        Trains the model for multiple epochs
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
        total_batches = 0
        skipped_batches = 0

        process = tqdm(loader, ncols=80)

        for batch_idx, (inputs, targets, idx) in enumerate(process):
            total_batches += 1
            
            # Process inputs
            processed_inputs = self.process_inputs(inputs, use_cuda)
            if processed_inputs is None:
                skipped_batches += 1
                continue
                
            targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()
            
            # Forward pass with processed inputs
            logits = self.model(
                processed_inputs['acc_data1'],
                processed_inputs['acc_data2'],
                processed_inputs['skl_data']
            )
            
            loss = self.criterion(logits, targets)
            loss.mean().backward()
            self.optimizer.step()

            timer['model'] += self.split_time()
            
            with torch.no_grad():
                train_loss += loss.mean().item()
                accuracy += (torch.argmax(F.log_softmax(logits, dim=1), 1) == targets).sum().item()
                cnt += len(targets)
                
            timer['stats'] += self.split_time()

        # Check if any batches were processed
        if cnt == 0:
            self.print_log(f"ERROR: No valid batches processed in epoch {epoch}!")
            self.print_log(f"Total batches: {total_batches}, Skipped batches: {skipped_batches}")
            return

        train_loss /= cnt
        accuracy *= 100. / cnt

        self.train_loss_summary.append(train_loss)
        acc_value.append(accuracy)
        proportion = {k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values())))) for k, v in timer.items()}
        
        self.print_log(
            f'\tTraining Loss: {train_loss:.4f}. Training Acc: {accuracy:.2f}%'
        )
        self.print_log(f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')
        self.print_log(f'\tProcessed batches: {total_batches - skipped_batches}/{total_batches}')
        
        val_loss = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
        self.val_loss_summary.append(val_loss)

    def eval(self, epoch, loader_name='test', result_file=None):
        '''
        Evaluates the model
        '''
        use_cuda = torch.cuda.is_available()
        if result_file is not None:
            f_r = open(result_file, 'w', encoding='utf-8')
        self.model.eval()

        loss = 0
        cnt = 0
        accuracy = 0
        label_list = []
        pred_list = []
        wrong_idx = []

        process = tqdm(self.data_loader[loader_name], ncols=80)
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                processed_inputs = self.process_inputs(inputs, use_cuda)
                if processed_inputs is None:
                    continue
                    
                targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                
                logits = self.model(
                    processed_inputs['acc_data1'],
                    processed_inputs['acc_data2'],
                    processed_inputs['skl_data']
                )
                
                batch_loss = self.criterion(logits, targets)
                loss += batch_loss.sum().item()
                accuracy += (torch.argmax(F.log_softmax(logits, dim=1), 1) == targets).sum().item()
                label_list.extend(targets.tolist())
                pred_list.extend(torch.argmax(F.log_softmax(logits, dim=1), 1).tolist())
                cnt += len(targets)

        loss /= cnt
        target = np.array(label_list)
        y_pred = np.array(pred_list)
        f1 = f1_score(target, y_pred, average='macro') * 100
        accuracy *= 100. / cnt

        if result_file is not None:
            predict = pred_list
            true = label_list

            for i, x in enumerate(predict):
                f_r.write(f'{x} ==> {true[i]}\n')
            f_r.close()

        self.print_log(f'{loader_name.capitalize()} Loss: {loss:.4f}. {loader_name.capitalize()} Acc: {accuracy:.2f}% f1: {f1:.2f}')
        if self.arg.phase == 'train':
            if accuracy > self.best_accuracy:
                self.best_loss = loss
                self.best_accuracy = accuracy
                self.best_f1 = f1
                torch.save(self.model, f'{self.arg.work_dir}/{self.arg.model_saved_name}.pth')
                self.print_log('Weights Saved')
        else:
            return pred_list, label_list, wrong_idx
        return loss

    def start(self):
        '''
        Function to start the training
        '''
        if self.arg.phase == 'train':
            self.train_loss_summary = []
            self.val_loss_summary = []
            self.best_accuracy = float('-inf')
            self.best_f1 = float('inf')
            self.print_log(f'Parameters: \n{str(vars(self.arg))}\n')

            results = self.create_df()
            test_subject = self.arg.subjects[-6:]
            train_subjects = [x for x in self.arg.subjects if x not in test_subject]
            self.test_subject = test_subject
            self.train_subjects = train_subjects
            self.model = self.load_model(self.arg.model, self.arg.model_args)
            self.print_log(f'Model Parameters: {self.count_parameters(self.model)}')
            self.load_data()
            self.load_optimizer()

            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch)
            self.print_log(f'Train Subjects : {self.train_subjects}')
            self.print_log(f' ------------ Test Subject {self.test_subject} -------')
            self.print_log(f'Best accuracy for : {self.best_accuracy}')
            self.print_log(f'Best F-Score: {self.best_f1}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'seed: {self.arg.seed}')
            # Removed visualization call
            # self.loss_viz(self.train_loss_summary, self.val_loss_summary)
            subject_result = pd.Series({'test_subject': str(self.test_subject), 'train_subjects': str(self.train_subjects),
                                        'accuracy': round(self.best_accuracy, 2), 'f1_score': round(self.best_f1, 2)})
            results.loc[len(results)] = subject_result
            self.best_accuracy = 0
            self.best_f1 = 0
            results.to_csv(f'{self.arg.work_dir}/scores.csv')


if __name__ == "__main__":
    parser = get_args()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r', encoding='utf-8') as f:
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

