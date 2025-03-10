# distill_student.py

import argparse
import yaml
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from importlib import import_module

from Utils.dataset import SmartFallMMBuilder
from Feeder.multi_sensor_feeder import MultiSensorFeeder, multi_sensor_collate_fn
from Utils.loss import ExtendedDistillationLoss

def str2bool(v):
    return v.lower() in ("true","1")

def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser= argparse.ArgumentParser("Distill student from teacher")
    parser.add_argument("--config", type=str, default="config/smartfallmm/distill_student_fixed.yaml")
    parser.add_argument("--teacher-weights", type=str, default="exps/teacher_tt4/teacher_tt4_best.pth")
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--print-log", type=str2bool, default=True)
    return parser.parse_args()

class DistillTrainer:
    def __init__(self, arg):
        self.arg= arg
        with open(arg.config,"r") as f:
            self.cfg= yaml.safe_load(f)
        os.environ["CUDA_VISIBLE_DEVICES"]= arg.device
        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        init_seed(arg.seed)
        self._build_data()
        self._build_models()
        self._load_teacher_weights(arg.teacher_weights)
        for p in self.teacher.parameters():
            p.requires_grad=False
        self._build_loss_optim()

    def _build_data(self):
        print("DEBUG: Distill => building dataset for student (fixed-len).")
        builder= SmartFallMMBuilder(
            root_dir="data/smartfallmm",
            subjects=self.cfg["subjects"],
            dataset_args=self.cfg["dataset_args"]
        )
        data_dict= builder.build_data()
        ds_all= MultiSensorFeeder(data_dict)
        n= len(ds_all)
        idx= np.arange(n)
        np.random.shuffle(idx)
        train_sz= int(n*0.8)
        train_idx, val_idx= idx[:train_sz], idx[train_sz:]
        from torch.utils.data import Subset
        ds_train= Subset(ds_all, train_idx)
        ds_val= Subset(ds_all, val_idx)
        self.train_loader= DataLoader(
            ds_train,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            collate_fn= multi_sensor_collate_fn
        )
        self.val_loader= DataLoader(
            ds_val,
            batch_size=self.cfg["val_batch_size"],
            shuffle=False,
            collate_fn= multi_sensor_collate_fn
        )

    def _build_models(self):
        # teacher
        t_str= self.cfg["teacher_model"]
        t_args= self.cfg["teacher_args"]
        t_parts= t_str.split(".")
        t_mod_str= ".".join(t_parts[:-1])
        t_cls_str= t_parts[-1]
        t_mod= import_module(t_mod_str)
        Tcls= getattr(t_mod, t_cls_str)
        self.teacher= Tcls(**t_args).to(self.device)

        # student
        s_str= self.cfg["student_model"]
        s_args= self.cfg["student_args"]
        s_parts= s_str.split(".")
        s_mod_str= ".".join(s_parts[:-1])
        s_cls_str= s_parts[-1]
        s_mod= import_module(s_mod_str)
        Scls= getattr(s_mod, s_cls_str)
        self.student= Scls(**s_args).to(self.device)

    def _load_teacher_weights(self, teacher_ckpt):
        print(f"DEBUG: loading teacher weights => {teacher_ckpt}")
        if not os.path.exists(teacher_ckpt):
            raise ValueError(f"Teacher checkpoint not found: {teacher_ckpt}")
        state= torch.load(teacher_ckpt, map_location=self.device)
        self.teacher.load_state_dict(state, strict=True)
        print("DEBUG: Teacher model loaded successfully.")

    def _build_loss_optim(self):
        dist_args= self.cfg.get("distill_args",{})
        self.kd_loss= ExtendedDistillationLoss(**dist_args).to(self.device)
        opt_name= self.cfg["optimizer"].lower()
        if opt_name=="adam":
            self.optimizer= torch.optim.Adam(
                self.student.parameters(),
                lr=self.cfg["base_lr"],
                weight_decay=self.cfg["weight_decay"]
            )
        else:
            self.optimizer= torch.optim.AdamW(
                self.student.parameters(),
                lr=self.cfg["base_lr"],
                weight_decay=self.cfg["weight_decay"]
            )
        self.num_epoch= self.cfg["num_epoch"]

    def start(self):
        best_val_acc= 0.0
        for ep in range(self.num_epoch):
            self._train_epoch(ep)
            val_loss, val_acc= self._eval_epoch(ep)
            if val_acc> best_val_acc:
                best_val_acc= val_acc
                os.makedirs(self.cfg["work_dir"], exist_ok=True)
                sp= os.path.join(self.cfg["work_dir"], f"{self.cfg['model_saved_name']}_best.pth")
                torch.save(self.student.state_dict(), sp)
            if self.arg.print_log:
                print(f"[Ep {ep+1}/{self.num_epoch}] valLoss={val_loss:.4f}, valAcc={val_acc:.2f}, best={best_val_acc:.2f}")

        print(f"Distillation complete. Best ValAcc={best_val_acc:.2f}")

    def _train_epoch(self, ep):
        self.student.train()
        self.teacher.eval()
        total_loss=0.0
        correct=0
        cnt=0
        for (t_in, t_mask, s_in, sk_in, sk_mask, labels) in self.train_loader:
            t_in, t_mask= t_in.to(self.device), t_mask.to(self.device)
            s_in= s_in.to(self.device)
            sk_in, sk_mask= sk_in.to(self.device), sk_mask.to(self.device)
            labels= labels.to(self.device)

            with torch.no_grad():
                t_out= self.teacher(
                    teacher_inert=t_in, teacher_inert_mask=t_mask,
                    skeleton= sk_in, skeleton_mask= sk_mask
                )
            s_out= self.student(s_in)
            loss= self.kd_loss(
                student_logits= s_out["logits"],
                teacher_logits= t_out["logits"],
                labels= labels,
                student_feat= s_out["feat"],
                teacher_feat= t_out["inert_feat"]
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss+= loss.item()* len(labels)
            preds= s_out["logits"].argmax(dim=-1)
            correct+= (preds==labels).sum().item()
            cnt+= len(labels)
        avg_loss= total_loss/cnt
        acc= 100.*correct/cnt
        if self.arg.print_log:
            print(f"[DistillEp {ep+1}] trainLoss={avg_loss:.4f}, trainAcc={acc:.2f}")

    def _eval_epoch(self, ep):
        self.student.eval()
        self.teacher.eval()
        total_loss=0.0
        correct=0
        cnt=0
        with torch.no_grad():
            for (t_in, t_mask, s_in, sk_in, sk_mask, labels) in self.train_loader:
                t_in, t_mask= t_in.to(self.device), t_mask.to(self.device)
                s_in= s_in.to(self.device)
                sk_in, sk_mask= sk_in.to(self.device), sk_mask.to(self.device)
                labels= labels.to(self.device)

                t_out= self.teacher(
                    teacher_inert= t_in, teacher_inert_mask= t_mask,
                    skeleton= sk_in, skeleton_mask= sk_mask
                )
                s_out= self.student(s_in)
                loss= self.kd_loss(
                    student_logits= s_out["logits"],
                    teacher_logits= t_out["logits"],
                    labels= labels,
                    student_feat= s_out["feat"],
                    teacher_feat= t_out["inert_feat"]
                )
                total_loss+= loss.item()* len(labels)
                preds= s_out["logits"].argmax(dim=-1)
                correct+= (preds==labels).sum().item()
                cnt+= len(labels)
        if cnt==0:
            return 0.0, 0.0
        avg_loss= total_loss/cnt
        acc= 100.* correct/cnt
        return avg_loss, acc

def main():
    arg= parse_args()
    trainer= DistillTrainer(arg)
    trainer.start()

if __name__=="__main__":
    main()

