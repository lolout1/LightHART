# evaluate_model.py

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Import your model and dataset classes
from Models.student_model import FallDetectionStudentModel  # Adjust the import path as necessary
from Models.teacher_model import FallDetectionTeacherModel  # If you want to evaluate the teacher model
from utils.dataset import YourDataset  # Replace with your actual dataset class

def parse_args():
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--config', default='./config/eval.yaml', help='Path to the configuration file')
    parser.add_argument('--weights', required=True, help='Path to the model weights file')
    parser.add_argument('--device', default='cuda', help='Device to run the evaluation on (e.g., "cpu" or "cuda")')
    parser.add_argument('--model-type', choices=['student', 'teacher'], default='student', help='Type of model to evaluate')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, device, model_type='student'):
    # Initialize the model
    if model_type == 'student':
        model_args = config['student_model_args']
        model = FallDetectionStudentModel(**model_args)
    elif model_type == 'teacher':
        model_args = config['teacher_model_args']
        model = FallDetectionTeacherModel(**model_args)
    else:
        raise ValueError("Invalid model type. Choose 'student' or 'teacher'.")
    model = model.to(device)
    # Load the weights
    model.load_state_dict(torch.load(config['weights'], map_location=device))
    model.eval()
    return model

def load_data(config):
    # Initialize the dataset and dataloader
    dataset_args = config['dataset_args']
    val_dataset = YourDataset(train=False, **dataset_args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    return val_loader

def evaluate(model, data_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            acc_data = batch['accelerometer'].to(device)
            skl_data = batch.get('skeleton', None)
            if skl_data is not None:
                skl_data = skl_data.to(device)
            labels = batch['label'].to(device)

            outputs = model(acc_data.float(), skl_data.float() if skl_data is not None else None)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    return avg_loss, accuracy, f1

def main():
    args = parse_args()
    config = load_config(args.config)
    config['weights'] = args.weights
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(config, device, model_type=args.model_type)
    val_loader = load_data(config)
    loss, acc, f1 = evaluate(model, val_loader, device)
    print(f'Validation Loss: {loss:.4f}')
    print(f'Validation Accuracy: {acc*100:.2f}%')
    print(f'Validation F1 Score: {f1*100:.2f}%')

if __name__ == '__main__':
    main()

