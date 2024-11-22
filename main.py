import argparse
import os
import sys
import yaml
import torch
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from Feeder.Make_Dataset import UTD_mm, prepare_dataset
from Models.mms4 import MultiModalStudentModel
from sklearn.metrics import f1_score

def init_seed(seed):
    """
    Initialize random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--config', default='./config/smartfallmm/mms4.yaml', help='Path to the config file')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    return parser

def main():
    parser = get_args()
    arg = parser.parse_args()

    # Load configuration from YAML file
    if arg.config is not None:
        with open(arg.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(arg, key, value)

    # Initialize seed for reproducibility
    init_seed(arg.seed)

    # Set device
    device = torch.device(f'cuda:{arg.device[0]}' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    print("Preparing dataset...")
    try:
        dataset_dict = prepare_dataset(arg)
        if dataset_dict is None:
            print("Dataset preparation failed. Exiting.")
            sys.exit(1)
        train_dataset = UTD_mm(dataset_dict['train'])
        val_dataset = UTD_mm(dataset_dict['val'])
        test_dataset = UTD_mm(dataset_dict['test'])
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        sys.exit(1)
    print("Dataset prepared successfully!")

    # Create data loaders
    data_loader = dict()
    data_loader['train'] = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True, num_workers=arg.num_worker)
    data_loader['val'] = DataLoader(val_dataset, batch_size=arg.val_batch_size, shuffle=False, num_workers=arg.num_worker)
    data_loader['test'] = DataLoader(test_dataset, batch_size=arg.test_batch_size, shuffle=False, num_workers=arg.num_worker)

    # Initialize model
    model = MultiModalStudentModel(**arg.model_args)
    model = model.to(device)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=arg.base_lr, weight_decay=arg.weight_decay)

    # Training loop
    num_epochs = arg.num_epoch
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch_idx, (inputs, targets, idx) in enumerate(data_loader['train']):
            # Move data to device
            for modality in inputs:
                inputs[modality] = inputs[modality].to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and accuracy
            total_loss += loss.item() * targets.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            # Print training information
            if batch_idx % arg.log_interval == 0 and batch_idx > 0:
                avg_loss = total_loss / total_samples
                avg_acc = 100.0 * total_correct / total_samples
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(data_loader["train"])}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%')

        # Validation step
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        val_targets = []
        val_preds = []
        with torch.no_grad():
            for inputs, targets, idx in data_loader['val']:
                for modality in inputs:
                    inputs[modality] = inputs[modality].to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * targets.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == targets).sum().item()
                val_samples += targets.size(0)
                val_targets.extend(targets.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
        if val_samples == 0:
            print("No validation samples available.")
            avg_val_loss = 0
            avg_val_acc = 0
            val_f1 = 0
        else:
            avg_val_loss = val_loss / val_samples
            avg_val_acc = 100.0 * val_correct / val_samples
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.2f}%, F1 Score: {val_f1:.4f}')

            # Save the best model
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                os.makedirs(arg.work_dir, exist_ok=True)
                model_save_path = os.path.join(arg.work_dir, f'{arg.model_saved_name}.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f'New best model saved to {model_save_path}')

    print(f'Training complete. Best Validation Accuracy: {best_val_acc:.2f}%')

if __name__ == "__main__":
    main()
