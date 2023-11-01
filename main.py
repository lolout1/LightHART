import argparse
import yaml
import traceback
import random 
import sys
import os
import time

#environmental import
import numpy as np 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import warnings
import json
import torch.nn.functional as F
from torchsummary import summary

#local import 
from Feeder.augmentation import TSFilpper
from utils.dataprocessing import utd_processing , bmhad_processing,normalization

def get_args():

    parser = argparse.ArgumentParser(description = 'Distillation')
    parser.add_argument('--config' , default = './config/utd/student.yaml')
    parser.add_argument('--dataset', type = str, default= 'utd' )
    #training
    parser.add_argument('--batch-size', type = int, default = 16, metavar = 'N',
                        help = 'input batch size for training (default: 8)')

    parser.add_argument('--test-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size for testing(default: 1000)')
    parser.add_argument('--val-batch-size', type = int, default = 8, 
                        metavar = 'N', help = 'input batch size for testing(default: 1000)')

    parser.add_argument('--num-epoch', type = int , default = 70, metavar = 'N', 
                        help = 'number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type = int, default = 0)

    #optim
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--base-lr', type = float, default = 0.001, metavar = 'LR', 
                        help = 'learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type = float , default=0.0004)

    #model
    parser.add_argument('--model' ,default= None, help = 'Name of Model to load')

    #model args
    parser.add_argument('--device', nargs='+', default=[0], type = int)
    parser.add_argument('--model-args', default= str, help = 'A dictionary for model args')
    parser.add_argument('--weights', type = str, help = 'Location of weight file')
    parser.add_argument('--model-saved-name', type = str, help = 'Weigt name', default='test')

    #loss args
    parser.add_argument('--loss', default='loss.BCE' , help = 'Name of loss function to use' )
    parser.add_argument('--loss-args', default ="{}", type = str,  help = 'A dictionary for loss')
    # parser.add_argument('--loss-args', default=str, help = 'A dictonary for loss args' )

    #dataloader 
    parser.add_argument('--feeder', default= None , help = 'Dataloader location')
    parser.add_argument('--train-feeder-args',default=str, help = 'A dict for dataloader args' )
    parser.add_argument('--val-feeder-args', default=str , help = 'A dict for validation data loader')
    parser.add_argument('--test_feeder_args',default=str, help= 'A dict for test data loader')
    parser.add_argument('--include-val', type = str2bool, default= True , help = 'If we will have the validation set or not')

    #initializaiton
    parser.add_argument('--seed', type =  int , default = 2 , help = 'random seed (default: 1)') 

    parser.add_argument('--log-interval', type = int , default = 10, metavar = 'N',
                        help = 'how many bathces to wait before logging training status')


   
    parser.add_argument('--work-dir', type = str, default = 'simple', metavar = 'F', help = "Working Directory")
    parser.add_argument('--print-log',type=str2bool,default=True,help='print logging or not')
    
    parser.add_argument('--phase', type = str, default = 'train')
    
    parser.add_argument('--num-worker', type = int, default= 0)
    parser.add_argument('--result-file', type = str, help = 'Name of resutl file')
    
    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
def init_seed(seed):
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
    mod_str , _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

class Trainer():
    
    def __init__(self, arg):
        self.arg = arg
        # self.save_arg()
        self.load_model(arg.model, arg.model_args)
        self.load_loss()
        self.load_optimizer()
        self.load_data()
        self.include_val = arg.include_val
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

        if self.arg.phase == 'test':
            self.load_weights(self.arg.weights)
        
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')

    def count_parameters(self, model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            

    
    def load_model(self, model, model_args):
        use_cuda = torch.cuda.is_available()
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        Model = import_class(model)
        self.model = Model(**model_args).to(f'cuda:{self.output_device}' if use_cuda else 'cpu')
    
    def load_loss(self):
        criterion= import_class(self.arg.loss)
        #loss_args = yaml.safe_load(arg.loss_args)
        self.criterion = criterion()
    
    def load_weights(self, weights):
        self.model.load_state_dict(torch.load(self.arg.weights))
    
    def load_optimizer(self):
        
        if self.arg.optimizer == "Adam" :
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr = self.arg.base_lr,
                # weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr = self.arg.base_lr, 
                weight_decay=self.arg.weight_decay
            )
        
        else :
           raise ValueError()
        
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        ## need to change it to dynamic import 
        self.data_loader = dict()
        if self.arg.phase == 'train':
            if self.arg.dataset == 'utd':
                train_data =  utd_processing(mode = self.arg.phase)
                norm_train, acc_scaler, skl_scaler =  normalization(data = train_data, mode = 'fit')
                val_data =  utd_processing(mode = 'val')
                norm_val, _, _ =  normalization(data = val_data,acc_scaler=acc_scaler,
                                               skl_scaler=skl_scaler, mode = 'transform')
            else:
                #train_data = bmhad_processing(mode = arg.phase)
                train_data = np.load('data/berkley_mhad/bhmad_sliding_stride10_train.npz')
                norm_train, acc_scaler, skl_scaler = normalization(data= train_data, mode = 'fit' )
                #val_data  = bmhad_processing(mode = 'val')
                val_data = np.load('data/berkley_mhad/bhmad_sliding_stride10_val.npz')
                norm_val, _, _ = normalization(data = val_data , mode = 'transform')
            self.acc_scaler = acc_scaler
            self.skl_scaler = skl_scaler
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args,dataset = norm_train, transform =None),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
            
            
            self.data_loader['val'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.val_feeder_args, dataset = norm_val),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker)
        else:
            if self.arg.dataset == 'utd':
                test_data =  utd_processing(mode = 'test')
            else:
                test_data = np.load('data/berkley_mhad/bhmad_sliding_stride10_test.npz')

            norm_test, _, _ =  normalization(data = test_data, mode = 'fit')
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args, dataset = norm_test),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker)
    


    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, string, print_time = True):
        print(string)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(string, file = f)

    def train(self, epoch):
        self.model.train()
        self.record_time()
        loader = self.data_loader['train']
        timer = dict(dataloader = 0.001, model = 0.001, stats = 0.001)
        loss_value = []
        acc_value = []
        accuracy = 0
        cnt = 0
        train_loss = 0
        process = tqdm(loader, ncols = 80)

        for batch_idx, (inputs, targets) in enumerate(process):
            with torch.no_grad():
                acc_data = inputs['acc_data'].cuda(self.output_device) #print("Input batch: ",inputs)
                skl_data = inputs['skl_data'].cuda(self.output_device)
                targets = targets.cuda(self.output_device)
            
            timer['dataloader'] += self.split_time()

            self.optimizer.zero_grad()

            # Ascent Step
            #print("labels: ",targets)
            masks, logits,predictions = self.model(acc_data.float(), skl_data.float())
            #logits = self.model(acc_data.float(), skl_data.float())
            #print("predictions: ",torch.argmax(predictions, 1) )
            # bce_loss = self.criterion(logits, targets)
            # slim_loss = 0
            # for mask in masks: 
            #     slim_loss += sum([self.slim_penalty(m) for m in mask])
            # loss = bce_loss + (0.3*slim_loss)
            loss = self.criterion(masks, logits, targets)
            loss.mean().backward()
            self.optimizer.step()

            timer['model'] += self.split_time()
            with torch.no_grad():
                train_loss += loss.sum().item()
                #accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                accuracy += (torch.argmax(F.log_softmax(logits,dim =1), 1) == targets).sum().item()
            cnt += len(targets) 
            timer['stats'] += self.split_time()
        train_loss /= cnt
        accuracy *= 100. / cnt
        loss_value.append(train_loss)
        acc_value.append(accuracy) 
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tTraining Loss: {:4f}. Training Acc: {:2f}%'.format(train_loss, accuracy)
        )
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        if not self.include_val:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    state_dict = self.model.state_dict()
                    #weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                    torch.save(state_dict, self.arg.work_dir + '/' + self.arg.model_saved_name+ str(epoch)+ '.pt')
                    self.print_log('Weights Saved') 
        
        else: 
            self.eval(epoch, loader_name='val', result_file=self.arg.result_file)

        #Still need to work with this one
        # if save_model:
        #     state_dict = self.model.state_dict()
        #     #weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        #     torch.save(state_dict, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
    
    def eval(self, epoch, loader_name = 'test', result_file = None):

        if result_file is not None : 
            f_r = open (result_file, 'w')
        
        self.model.eval()

        self.print_log('Eval epoch: {}'.format(epoch+1))

        loss = 0
        cnt = 0
        accuracy = 0
        label_list = []
        pred_list = []
        
        process = tqdm(self.data_loader[loader_name], ncols=80)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(process):
                label_list.extend(targets.tolist())
                #inputs = inputs.cuda(self.output_device)
                acc_data = inputs['acc_data'].cuda(self.output_device) #print("Input batch: ",inputs)
                skl_data = inputs['skl_data'].cuda(self.output_device)
                targets = targets.cuda(self.output_device)

                #_,logits,predictions = self.model(inputs.float())
                masks,logits,predictions = self.model(acc_data.float(), skl_data.float())
                #logits = self.model(acc_data.float(), skl_data.float())
                # bce_loss = self.criterion(logits, targets)
                # slim_loss = 0
                # for mask in masks: 
                #     slim_loss += sum([self.slim_penalty(m) for m in mask])
                # batch_loss = bce_loss + (0.3*slim_loss)
                batch_loss = self.criterion(masks, logits, targets)
                loss += batch_loss.sum().item()
                # accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                # pred_list.extend(torch.argmax(predictions ,1).tolist())
                accuracy += (torch.argmax(F.log_softmax(logits,dim =1), 1) == targets).sum().item()
                pred_list.extend(torch.argmax(F.log_softmax(logits,dim =1) ,1).tolist())
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100./cnt
        
        if result_file is not None:
            predict = pred_list
            true = label_list

            for i, x in enumerate(predict):
                f_r.write(str(x) +  '==>' + str(true[i]) + '\n')
        
        self.print_log('{} Loss: {:4f}. {} Acc: {:2f}%'.format(loader_name.capitalize(),loss,loader_name.capitalize(), accuracy))
        if self.arg.phase == 'train':
            if accuracy > self.best_accuracy :
                    self.best_accuracy = accuracy
                    state_dict = self.model.state_dict()
                    #weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                    torch.save(state_dict, self.arg.work_dir + '/' + self.arg.model_saved_name+ '.pt')
                    self.print_log('Weights Saved')        

    def start(self):
        #summary(self.model,[(model_args['acc_frames'],3), (model_args['mocap_frames'], model_args['num_joints'],3)] , dtypes=[torch.float, torch.float] )
        if self.arg.phase == 'train':
            self.best_accuracy  = 0
            self.print_log('Parameters: \n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch)
                
            self.print_log(f'Best accuracy: {self.best_accuracy}')
            #self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            #self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'seed: {self.arg.seed}')
        
        elif self.arg.phase == 'test' :
            if self.arg.weights is None: 
                raise ValueError('Please appoint --weights')
            self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)

    # def save_arg(self):
    #     #save arg
    #     arg_dict = vars(self.arg)


if __name__ == "__main__":
    parser = get_args()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
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
