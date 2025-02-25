# File: offline_train_single.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from Feeder.single_input_feeder import SingleInputFeeder, single_input_collate_fn
from Models.fall_time2vec_transformer_single import FallTime2VecTransformerSingle

# We'll assume we only need to do offline with the new 'single_input' mode from base.py
# so we simulate your "DatasetBuilder" approach or a simpler approach to gather final (128,4) windows.

def parse_args():
    parser = argparse.ArgumentParser(description="Offline single-input training with [x,y,z,time].")
    parser.add_argument("--csv-list", type=str, required=True,
                        help="Text file listing all CSV paths + labels, e.g. lines: /path/to/file.csv <label>")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0004)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="exps/single_input")
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # 1) Read CSV paths + labels
    lines = open(args.csv_list).read().strip().split("\n")
    all_paths = []
    all_labels = []
    for line in lines:
        p, lbl_str = line.split()
        lbl = int(lbl_str)
        all_paths.append(p)
        all_labels.append(lbl)

    # 2) Build final windows => for each path we do:
    from utils.processor.base import Processor
    samples_4 = []
    labels_4 = []

    for path, lbl in zip(all_paths, all_labels):
        # Make a Processor with single_input mode => it will do parse_watch_csv => sliding_windows => reorder
        proc = Processor(file_path=path, mode="single_input", max_length=128,
                         window_size_sec=4.0, stride_sec=1.0)
        raw_data = proc.load_file(is_skeleton=False)  # => shape(N,4), [time,x,y,z]
        if raw_data.shape[0] < 1:
            continue
        windows = proc.process(raw_data)  # => a list of (128,4) => [x,y,z,time]

        for w in windows:
            if w.shape == (128,4):
                samples_4.append(w)
                labels_4.append(lbl)

    # Shuffle
    perm = np.random.permutation(len(samples_4))
    samples_4 = [samples_4[i] for i in perm]
    labels_4  = [labels_4[i] for i in perm]

    # Simple train/val split
    split_idx = int(len(samples_4)*0.8)
    train_samp = samples_4[:split_idx]
    train_lbls = labels_4[:split_idx]
    val_samp   = samples_4[split_idx:]
    val_lbls   = labels_4[split_idx:]

    train_ds = SingleInputFeeder(train_samp, train_lbls)
    val_ds   = SingleInputFeeder(val_samp,   val_lbls)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=single_input_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=single_input_collate_fn)

    # 3) Build model
    model = FallTime2VecTransformerSingle(
        time2vec_dim=8,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        dropout=0.1,
        dim_feedforward=128
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.num_epochs):
        # train
        model.train()
        total_loss = 0.0
        correct = 0
        total_count = 0
        for data_4, lbl in train_loader:
            data_4 = data_4.to(device)
            lbl    = lbl.to(device)
            optimizer.zero_grad()
            logits = model(data_4)
            loss   = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()* len(lbl)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds==lbl).sum().item()
            total_count += len(lbl)
        avg_loss = total_loss / total_count if total_count>0 else 0
        train_acc = 100.0* correct / total_count if total_count>0 else 0
        print(f"Epoch {epoch+1}/{args.num_epochs} [Train] => Loss={avg_loss:.4f}, Acc={train_acc:.2f}%")

        # val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_count   = 0
        with torch.no_grad():
            for data_4, lbl in val_loader:
                data_4 = data_4.to(device)
                lbl    = lbl.to(device)
                logits = model(data_4)
                loss = criterion(logits, lbl)
                val_loss += loss.item()* len(lbl)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds==lbl).sum().item()
                val_count   += len(lbl)

        avg_val_loss = val_loss/ val_count if val_count>0 else 0
        val_acc = 100.0* val_correct / val_count if val_count>0 else 0
        print(f"Epoch {epoch+1}/{args.num_epochs} [Val] => Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%")

        if val_acc> best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, "single_input_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Best model updated => {save_path}, ValAcc={val_acc:.2f}%")

    print(f"Training done. Best ValAcc={best_val_acc:.2f}%")

if __name__ == "__main__":
    main()

