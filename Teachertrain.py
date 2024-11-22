import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import yaml
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import os

def setup_logger(save_dir):
    log_file = save_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, device):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup directories
        self.save_dir = Path(config['work_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.save_dir)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['base_lr'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        total_steps = len(train_loader) * config['num_epoch']
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['base_lr'],
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Best metric tracking
        self.best_acc = 0
        self.best_f1 = 0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        for step, (acc_data, skl_data, targets) in enumerate(pbar):
            # Move data to device
            acc_data = acc_data.to(self.device)
            skl_data = skl_data.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                logits = self.model(acc_data, skl_data)
                loss = self.criterion(logits, targets)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate metrics
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        return total_loss / len(self.train_loader), correct / total
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        for acc_data, skl_data, targets in pbar:
            # Move data to device
            acc_data = acc_data.to(self.device)
            skl_data = skl_data.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits = self.model(acc_data, skl_data)
            loss = self.criterion(logits, targets)
            
            # Calculate metrics
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        val_acc = correct / total
        return total_loss / len(self.val_loader), val_acc
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(state, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(state, best_path)
            self.logger.info(f'New best model saved at epoch {epoch}')
    
    def train(self):
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['num_epoch']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Log metrics
            self.logger.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%'
            )
            
            # Save checkpoint
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
            
            self.save_checkpoint(epoch, metrics, is_best)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = MMTransformer(**config['model_args'])
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create data loaders
    from Feeder.Make_Dataset import UTD_mm
    train_dataset = UTD_mm(config, phase='train')
    val_dataset = UTD_mm(config, phase='val')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_worker'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_worker'],
        pin_memory=True
    )
    
    # Create trainer and start training
    trainer = Trainer(model, config, train_loader, val_loader, device)
    trainer.train()

if __name__ == '__main__':
    main()