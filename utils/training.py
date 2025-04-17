# utils/training.py
import torch
import numpy as np
import os
import logging
from tqdm import tqdm
from utils.metrics import calculate_metrics

logger = logging.getLogger("training")

def train_epoch(model, loader, criterion, optimizer, device, clip_grad=1.0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    valid_batches = 0
    
    for batch_idx, (acc_data, labels, _) in enumerate(tqdm(loader, desc="Training")):
        try:
            acc_data = acc_data.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            outputs, _ = model(acc_data)
            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels)
            
            if not torch.isfinite(loss):
                logger.warning(f"Skipping batch {batch_idx} due to non-finite loss")
                continue
                
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            running_loss += loss.item()
            valid_batches += 1
            
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    epoch_loss = running_loss / max(valid_batches, 1)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return epoch_loss, all_preds, all_labels, metrics

def validate(model, loader, criterion, device, prefix="val"):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    valid_batches = 0
    
    with torch.no_grad():
        for acc_data, labels, _ in tqdm(loader, desc=f"{prefix.capitalize()}"):
            try:
                acc_data = acc_data.to(device)
                labels = labels.to(device).float()
                
                outputs, _ = model(acc_data)
                outputs = outputs.squeeze(1)
                
                loss = criterion(outputs, labels)
                
                if not torch.isfinite(loss):
                    continue
                    
                running_loss += loss.item()
                valid_batches += 1
                
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error in validation: {e}")
                continue
    
    epoch_loss = running_loss / max(valid_batches, 1)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return epoch_loss, all_preds, all_labels, metrics

def save_model(model, fold_work_dir, epoch, metrics, device, is_best=False):
    save_path = os.path.join(fold_work_dir, 'best_model.pth' if is_best else f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    
    try:
        dummy_input = torch.randn(1, 128, 4).to(device)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        script_path = os.path.join(fold_work_dir, 'best_model_scripted.pt' if is_best else f'model_epoch_{epoch}_scripted.pt')
        torch.jit.save(traced_model, script_path)
        logger.info(f"Saved TorchScript model to {script_path}")
    except Exception as e:
        logger.warning(f"Could not save TorchScript model: {e}")
