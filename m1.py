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
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# local imports
from utils.dataset import prepare_smartfallmm
from utils.loader import DatasetBuilder
from Feeder.time2vec_varlen import (
    Time2VecVarLenFeeder,
    time2vec_varlen_collate_fn
)

################################################################
# 1) A small Transformer for variable-length data
################################################################
class FallTime2VecTransformer(torch.nn.Module):
    def __init__(self, feat_dim=11, d_model=32, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.input_proj = torch.nn.Linear(feat_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x => shape (B, N, featDim)
        mask => shape (B, N), True => ignore/pad
        """
        B, N, C = x.shape
        x_proj = self.input_proj(x)      # => (B, N, d_model)
        x_proj = x_proj.transpose(0, 1)  # => (N, B, d_model)

        out = self.transformer(x_proj, src_key_padding_mask=mask)
        out = out.transpose(0,1)         # => (B, N, d_model)

        # average the unpadded frames
        if mask is not None:
            lengths = (~mask).sum(dim=-1, keepdim=True)  # shape (B,1)
            out = out * (~mask).unsqueeze(-1)
            out = out.sum(dim=1) / torch.clamp(lengths, min=1e-9)
        else:
            out = out.mean(dim=1)

        logits = self.classifier(out)  # => (B, num_classes)
        return logits

################################################################
# 2) Arg Parsing
################################################################
def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported boolean string')

def get_args():
    parser = argparse.ArgumentParser(description='Fall detection w/ variable_time + Time2Vec')
    parser.add_argument('--config', default='./config/smartfallmm/time2vec_fall.yaml')

    # Common training args
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=16)
    parser.add_argument('--num-epoch', type=int, default=30)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--base-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0004)

    parser.add_argument('--model', default=None, help='Name of the model class')
    parser.add_argument('--model-saved-name', type=str, default='time2vec_transformer')
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--loss', default='CE', help='Loss function type')
    parser.add_argument('--loss-args', default="{}", type=str)
    parser.add_argument('--dataset_args', default={}, type=dict)
    parser.add_argument('--feeder', default=None, help='Dataloader class path')
    parser.add_argument('--train_feeder_args', default={}, type=dict)
    parser.add_argument('--val_feeder_args', default={}, type=dict)
    parser.add_argument('--test_feeder_args', default={}, type=dict)
    parser.add_argument('--include-val', type=str2bool, default=True)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--work-dir', type=str, default='exps', help='Working directory')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--seed', type=int, default=2)

    # Additional arguments often in config:
    parser.add_argument('--dataset', type=str, default='smartfallmm')
    parser.add_argument('--subjects', nargs='+', type=int, default=None)
    parser.add_argument('--model_args', default={}, type=dict)

    return parser

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

################################################################
# 3) Trainer
################################################################
class Trainer():
    def __init__(self, arg):
        self.arg = arg

        # Must define self.work_dir before print_log is called
        self.work_dir = arg.work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        self.print_log(f"Args: {vars(arg)}")

        # Optionally copy config to work_dir
        if arg.config and os.path.exists(arg.config):
            shutil.copy(arg.config, os.path.join(self.work_dir, os.path.basename(arg.config)))

        self.best_accuracy = 0
        self.best_f1 = 0
        self.best_loss = 9999

    def print_log(self, msg):
        print(msg)
        if self.arg.print_log:
            log_path = os.path.join(self.work_dir, 'log.txt')
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

    def load_data(self):
        """
        1) Build a DatasetBuilder from the user config
        2) create train/test sets by subject
        3) do global normalization (if desired)
        """
        # subjects from config
        subjects = self.arg.subjects
        if not subjects:
            subjects = []  # fallback

        # We'll do a naive train/test split => last subject as test
        if len(subjects) < 2:
            train_subj = subjects
            test_subj = []
        else:
            test_subj = [subjects[-1]]
            train_subj = subjects[:-1]

        self.print_log(f"Train Subjects={train_subj}, Test Subject={test_subj}")

        # 1) Prepare the dataset
        builder_train = prepare_smartfallmm(self.arg)
        builder_train.make_dataset(train_subj)
        train_data = builder_train.processed_data

        # 2) Global normalization across all train windows
        # Flatten
        all_arrays = train_data.get('accelerometer', [])
        if len(all_arrays)>0 and isinstance(all_arrays, list) and isinstance(all_arrays[0], np.ndarray):
            stacked = np.concatenate(all_arrays, axis=0)  # shape (sumLen, featDim)
            scaler = StandardScaler()
            scaler.fit(stacked)
            def transform_fn(x):
                return scaler.transform(x)
        else:
            def transform_fn(x): return x

        # 3) Build test
        builder_test = prepare_smartfallmm(self.arg)
        builder_test.make_dataset(test_subj)
        test_data = builder_test.processed_data

        return train_data, test_data, transform_fn

    def load_model(self):
        """
        Creates the transformer model with the user-specified model_args
        """
        # read your model_args from config
        feat_dim   = self.arg.model_args.get('feat_dim', 11)
        d_model    = self.arg.model_args.get('d_model', 32)
        nhead      = self.arg.model_args.get('nhead', 4)
        num_layers = self.arg.model_args.get('num_layers', 2)
        num_classes= self.arg.model_args.get('num_classes', 2)

        model = FallTime2VecTransformer(
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes
        )
        device = f"cuda:{self.arg.device[0]}" if (torch.cuda.is_available() and len(self.arg.device)>0) else "cpu"
        model.to(device)
        return model

    def run(self):
        if self.arg.phase == 'train':
            self.train()
        else:
            self.test()

    def train(self):
        # 1) load data
        train_data, test_data, transform_fn = self.load_data()

        # 2) Build Feeder => Time2VecVarLenFeeder
        train_ds = Time2VecVarLenFeeder(train_data, transform=transform_fn)
        test_ds  = Time2VecVarLenFeeder(test_data, transform=transform_fn)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.arg.batch_size, shuffle=True,
            collate_fn=time2vec_varlen_collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.arg.batch_size, shuffle=False,
            collate_fn=time2vec_varlen_collate_fn
        )

        # 3) load model
        model = self.load_model()
        device = f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu"

        # 4) define optimizer
        if self.arg.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        criterion = torch.nn.CrossEntropyLoss()

        # 5) train loop
        for epoch in range(self.arg.num_epoch):
            model.train()
            total_loss=0
            correct=0
            total=0

            for data_batch, labels_batch, mask_batch in tqdm(train_loader, ncols=80):
                data_batch = data_batch.to(device)
                labels_batch = labels_batch.to(device)
                mask_batch = mask_batch.to(device)

                optimizer.zero_grad()
                logits = model(data_batch, mask_batch)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()*len(labels_batch)
                preds = logits.argmax(dim=-1)
                correct += (preds==labels_batch).sum().item()
                total += len(labels_batch)

            avg_loss = total_loss/total if total>0 else 0
            accuracy = correct/total if total>0 else 0
            self.print_log(f"Epoch {epoch}: Train loss={avg_loss:.4f}, acc={accuracy:.4f}")

            # Evaluate on test after each epoch
            model.eval()
            test_loss=0
            test_correct=0
            test_total=0
            with torch.no_grad():
                for data_batch, labels_batch, mask_batch in test_loader:
                    data_batch = data_batch.to(device)
                    labels_batch= labels_batch.to(device)
                    mask_batch = mask_batch.to(device)

                    logits = model(data_batch, mask_batch)
                    loss = criterion(logits, labels_batch)
                    test_loss += loss.item()*len(labels_batch)
                    preds = logits.argmax(dim=-1)
                    test_correct += (preds==labels_batch).sum().item()
                    test_total += len(labels_batch)

            test_avg_loss = test_loss/test_total if test_total>0 else 0
            test_acc = test_correct/test_total if test_total>0 else 0
            self.print_log(f"    >> Test loss={test_avg_loss:.4f}, acc={test_acc:.4f}")

            # keep best
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                save_path = os.path.join(self.work_dir, f"{self.arg.model_saved_name}.pth")
                torch.save(model.state_dict(), save_path)
                self.print_log(f"Saved best model (acc={test_acc:.4f}) => {save_path}")

    def test(self):
        # 1) load data
        _, test_data, transform_fn = self.load_data()
        test_ds = Time2VecVarLenFeeder(test_data, transform=transform_fn)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.arg.batch_size, shuffle=False,
            collate_fn=time2vec_varlen_collate_fn
        )

        # 2) load model
        model = self.load_model()
        device = f"cuda:{self.arg.device[0]}" if torch.cuda.is_available() else "cpu"

        # attempt to load weights
        weights_path = os.path.join(self.work_dir, f"{self.arg.model_saved_name}.pth")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            self.print_log(f"Loaded weights from {weights_path}")
        else:
            self.print_log("No saved weights found; evaluating with random init.")

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        test_loss=0
        correct=0
        total=0
        all_preds = []
        all_labels= []
        with torch.no_grad():
            for data_batch, labels_batch, mask_batch in tqdm(test_loader, ncols=80):
                data_batch = data_batch.to(device)
                labels_batch= labels_batch.to(device)
                mask_batch = mask_batch.to(device)

                logits = model(data_batch, mask_batch)
                loss = criterion(logits, labels_batch)
                test_loss += loss.item()*len(labels_batch)

                preds = logits.argmax(dim=-1)
                correct += (preds==labels_batch).sum().item()
                total += len(labels_batch)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        avg_loss = test_loss/total if total>0 else 0
        accuracy = correct/total if total>0 else 0
        macro_f1 = f1_score(all_labels, all_preds, average='macro') if total>0 else 0
        self.print_log(f"[Test] loss={avg_loss:.4f}, acc={accuracy:.4f}, f1={macro_f1:.4f}")

################################################################
# 4) Main entry
################################################################
def main():
    parser = get_args()
    p = parser.parse_args()

    # If config is provided, load & override defaults
    if p.config is not None and os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        # Set parser defaults from the YAML
        parser.set_defaults(**default_arg)
        p = parser.parse_args()

    # Set random seeds
    init_seed(p.seed)

    # Build trainer and run
    trainer = Trainer(p)
    trainer.run()

if __name__ == "__main__":
    main()

