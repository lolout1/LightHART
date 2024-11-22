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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

# Local imports
from utils.dataset import prepare_smartfallmm, filter_subjects


def get_args():
    '''
    Function to build Argument Parser
    '''

    parser = argparse.ArgumentParser(description='Distillation')
    parser.add_argument('--config', default='./config/smartfallmm/teacher.yaml')
    parser.add_argument('--dataset', type=str, default='utd')
    # Training
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')

    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for testing(default: 1000)')
    parser.add_argument('--val-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for testing(default: 1000)')

    parser.add_argument('--num-epoch', type=int, default=70, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--base-lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0004)

    # Model
    parser.add_argument('--model', default=None, help='Name of Model to load')

    # Model args
    parser.add_argument('--device', nargs='+', default=[0], type=int)

    parser.add_argument('--model-args', default=str, help='A dictionary for model args')
    parser.add_argument('--weights', type=str, help='Location of weight file')
    parser.add_argument('--model-saved-name', type=str, help='Weight name', default='test')

    # Loss args
    parser.add_argument('--loss', default='loss.BCE', help='Name of loss function to use')
    parser.add_argument('--loss-args', default="{}", type=str, help='A dictionary for loss')

    # Dataset args
    parser.add_argument('--dataset-args', default=str, help='Arguments for dataset')

    # Dataloader
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--feeder', default=None, help='Dataloader location')
    parser.add_argument('--train-feeder-args', default=str, help='A dict for dataloader args')
    parser.add_argument('--val-feeder-args', default=str, help='A dict for validation data loader')
    parser.add_argument('--test-feeder-args', default=str, help='A dict for test data loader')
    parser.add_argument('--include-val', type=str2bool, default=True, help='If we will have the validation set or not')

    # Initialization
    parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--work-dir', type=str, default='simple', metavar='F', help="Working Directory")
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')

    parser.add_argument('--phase', type=str, default='train')

    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str, help='Name of result file')
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='Gradient clipping value (default: None)')
    parser.add_argument('--scheduler', type=str, default=None,
                        help='Learning rate scheduler type')
    parser.add_argument('--scheduler-args', type=str, default="{}",
                        help='Arguments for learning rate scheduler')

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
    Initialize seed for reproducibility of the results
    '''
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = True
    # Training speed is too slow if set to True
    torch.backends.cudnn.deterministic = False

    # On CUDA 11 cuDNN8, the default algorithm is very slow
    # Unlike on CUDA 10, the default works well
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


class Trainer():

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
        self.intertial_modality = next((modality for modality in arg.dataset_args['modalities'] if modality != 'skeleton'), None)
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

        self.include_val = arg.include_val

        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size : {num_params / (1024 ** 2):.2f} MB')

    def save_config(self, src_path: str, desc_path: str) -> None:
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

    def load_scheduler(self):
        """
        Loads learning rate scheduler if specified
        """
        if self.arg.scheduler is None:
            return None

        if self.arg.scheduler.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.arg.num_epoch,
                eta_min=float(self.arg.scheduler_args.get('min_lr', 0.00001))
            )
        elif self.arg.scheduler.lower() == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(self.arg.scheduler_args.get('step_size', 30)),
                gamma=float(self.arg.scheduler_args.get('gamma', 0.1))
            )
        elif self.arg.scheduler.lower() == 'warmup_cosine':
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.arg.base_lr,
                epochs=self.arg.num_epoch,
                steps_per_epoch=len(self.data_loader['train']),
                pct_start=float(self.arg.scheduler_args.get('warmup_ratio', 0.1))
            )

    def load_model(self, model, model_args):
        '''
        Function to load model
        '''
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
        Model = import_class(model)
        model = Model(**model_args).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
        return model

    def load_loss(self):
        '''
        Loading loss function for the model training
        '''
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_weights(self):
        '''
        Load weights to the model
        '''
        self.model.load_state_dict(torch.load(self.arg.weights))

    def load_optimizer(self) -> None:
        '''
        Loads Optimizer
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
            raise ValueError(f"Unsupported optimizer type: {self.arg.optimizer}")

    def distribution_viz(self, labels: np.array, work_dir: str, mode: str) -> None:
        '''
        Visualizes the training, validation/test dataset distribution
        '''
        values, count = np.unique(labels, return_counts=True)
        plt.bar(x=values, height=count)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title(f'{mode.capitalize()} Label Distribution')
        plt.savefig(os.path.join(work_dir, f'{mode}_Label_Distribution.png'))
        plt.close()

    def load_data(self):
        '''
        Loads different datasets
        '''
        Feeder = import_class(self.arg.feeder)

        if self.arg.phase == 'train':
            # Dataset class for further processing
            builder = prepare_smartfallmm(self.arg)

            norm_train = filter_subjects(builder, self.train_subjects)
            norm_val = filter_subjects(builder, self.test_subject)

            # Training dataset
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args,
                               dataset=norm_train),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)

            # Validation dataset
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args,
                               dataset=norm_val),
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker)
        else:
            builder = prepare_smartfallmm(self.arg)
            norm_test = filter_subjects(builder, self.test_subject)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset=norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
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
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(string, file=f)

    def loss_viz(self, train_loss: List[float], val_loss: List[float]) -> None:
        '''
        Visualizes the val and train loss curves together
        '''
        epochs = range(len(train_loss))
        plt.plot(epochs, train_loss, 'b', label="Training Loss")
        plt.plot(epochs, val_loss, 'r', label="Validation Loss")
        plt.title('Train vs Val Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.arg.work_dir, 'train_vs_val_loss.png'))
        plt.close()

    def cm_viz(self, y_pred: List[int], y_true: List[int]) -> None:
        '''
        Visualizes the confusion matrix
        '''
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks(np.unique(y_true))
        plt.yticks(np.unique(y_true))
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.arg.work_dir, 'confusion_matrix.png'))
        plt.close()

    def create_df(self, columns=['test_subject', 'train_subjects', 'accuracy', 'f1_score']) -> pd.DataFrame:
        '''
        Initiates a new dataframe
        '''
        df = pd.DataFrame(columns=columns)
        return df

    def train(self, epoch):
        '''
        Trains the model for one epoch
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
            self.record_time()
            phone_acc_data = inputs['phone_accelerometer'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
            watch_acc_data = inputs['watch_accelerometer'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
            skl_data = inputs['skeleton'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
            targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')

            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()
            outputs = self.model(watch_acc_data.float(), skl_data.float())
            if isinstance(outputs, tuple):
                logits, intermediate_features = outputs
            else:
                logits = outputs
                intermediate_features = None
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            timer['model'] += self.split_time()

            with torch.no_grad():
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                accuracy += (preds == targets).sum().item()

            cnt += len(targets)
            timer['stats'] += self.split_time()

        train_loss /= cnt
        accuracy *= 100.0 / cnt

        self.train_loss_summary.append(train_loss)
        acc_value.append(accuracy)
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tTraining Loss: {:.4f}. Training Acc: {:.2f}%'.format(train_loss, accuracy)
        )
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        val_loss = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
        self.val_loss_summary.append(val_loss)
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.print_log(f'\tCurrent Learning Rate: {current_lr:.6f}')

    def eval(self, epoch, loader_name='test', result_file=None):
        '''
        Evaluates the model's performance
        '''
        use_cuda = torch.cuda.is_available()
        if result_file is not None:
            f_r = open(result_file, 'w', encoding='utf-8')
        self.model.eval()

        self.print_log('Eval epoch: {}'.format(epoch + 1))

        loss = 0
        cnt = 0
        accuracy = 0
        label_list = []
        pred_list = []

        process = tqdm(self.data_loader[loader_name], ncols=80)
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(process):
                
                #phone_acc_data = inputs['phone_accelerometer'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                watch_acc_data = inputs['watch_accelerometer'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                skl_data = inputs['skeleton'].to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
                targets = targets.to(f'cuda:{self.output_device}' if use_cuda else 'cpu')

                outputs = self.model(watch_acc_data.float(), skl_data.float())
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs

                batch_loss = self.criterion(logits, targets)
                loss += batch_loss.item()
                preds = torch.argmax(logits, dim=1)
                accuracy += (preds == targets).sum().item()
                label_list.extend(targets.cpu().tolist())
                pred_list.extend(preds.cpu().tolist())
                cnt += len(targets)

            loss /= cnt
            target = np.array(label_list)
            y_pred = np.array(pred_list)
            f1 = f1_score(target, y_pred, average='macro') * 100
            accuracy *= 100.0 / cnt

        if result_file is not None:
            for i, x in enumerate(pred_list):
                f_r.write(str(x) + ' ==>' + str(label_list[i]) + '\n')
            f_r.close()

        self.print_log('{} Loss: {:.4f}. {} Acc: {:.2f}% F1 Score: {:.2f}'.format(
            loader_name.capitalize(), loss, loader_name.capitalize(), accuracy, f1))
        if self.arg.phase == 'train':
            if accuracy > self.best_accuracy:
                self.best_loss = loss
                self.best_accuracy = accuracy
                self.best_f1 = f1
                torch.save(self.model.state_dict(), f'{self.arg.work_dir}/{self.arg.model_saved_name}.pth')
                self.print_log('Weights Saved')
        else:
            return pred_list, label_list
        return loss

    def start(self):
        '''
        Function to start the training
        '''
        if self.arg.phase == 'train':
            self.train_loss_summary = []
            self.val_loss_summary = []
            self.best_accuracy = float('-inf')

            self.best_f1 = float('-inf')
            self.print_log('Parameters: \n{}\n'.format(str(vars(self.arg))))

            results = self.create_df()
            test_subject = self.arg.subjects[-3:]
            train_subjects = [x for x in self.arg.subjects if x not in test_subject]
            self.test_subject = test_subject
            self.train_subjects = train_subjects
            self.model = self.load_model(self.arg.model, self.arg.model_args)
            self.print_log(f'Model Parameters: {self.count_parameters(self.model)}')
            self.load_data()
            self.load_optimizer()
            self.scheduler = self.load_scheduler()
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch)
            self.print_log(f'Train Subjects : {self.train_subjects}')
            self.print_log(f' ------------ Test Subject {self.test_subject} -------')
            self.print_log(f'Best accuracy: {self.best_accuracy}')
            self.print_log(f'Best F1-Score: {self.best_f1}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Seed: {self.arg.seed}')
            self.loss_viz(self.train_loss_summary, self.val_loss_summary)
            subject_result = pd.Series({'test_subject': str(self.test_subject), 'train_subjects': str(self.train_subjects),
                                        'accuracy': round(self.best_accuracy, 2), 'f1_score': round(self.best_f1, 2)})
            results = results.append(subject_result, ignore_index=True)
            self.best_accuracy = 0
            self.best_f1 = 0
            results.to_csv(f'{self.arg.work_dir}/scores.csv', index=False)


if __name__ == "__main__":
    parser = get_args()

    # Load arg from config file
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

