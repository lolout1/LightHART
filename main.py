import os, torch, numpy as np, argparse, logging, json, time, random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn, optim
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib, sys, inspect, pandas as pd
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FallDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.accelerometer = data.get('accelerometer', None)
        self.gyroscope = data.get('gyroscope', None)
        self.quaternion = data.get('quaternion', None)
        self.linear_acceleration = data.get('linear_acceleration', None)
        self.raw_acceleration = data.get('raw_acceleration', None)
        self.fusion_features = data.get('fusion_features', None)
        self.labels = data.get('labels', None)
        self.subjects = data.get('subjects', None)
        if self.labels is not None and (self.subjects is None or len(self.subjects) == 0):
            self.subjects = np.zeros(len(self.labels), dtype=np.int32)
    def __len__(self): 
        return 0 if self.labels is None else len(self.labels)
    def __getitem__(self, idx):
        data_dict = {}
        if hasattr(self, 'accelerometer') and self.accelerometer is not None:
            data_dict['accelerometer'] = torch.from_numpy(self.accelerometer[idx]).float()
            data_dict['acc'] = torch.from_numpy(self.accelerometer[idx]).float()
        if hasattr(self, 'gyroscope') and self.gyroscope is not None:
            data_dict['gyroscope'] = torch.from_numpy(self.gyroscope[idx]).float()
            data_dict['gyro'] = torch.from_numpy(self.gyroscope[idx]).float()
        if hasattr(self, 'quaternion') and self.quaternion is not None:
            data_dict['quaternion'] = torch.from_numpy(self.quaternion[idx]).float()
            data_dict['quat'] = torch.from_numpy(self.quaternion[idx]).float()
        if hasattr(self, 'linear_acceleration') and self.linear_acceleration is not None:
            data_dict['linear_acceleration'] = torch.from_numpy(self.linear_acceleration[idx]).float()
        if hasattr(self, 'raw_acceleration') and self.raw_acceleration is not None:
            data_dict['raw_acceleration'] = torch.from_numpy(self.raw_acceleration[idx]).float()
        if hasattr(self, 'fusion_features') and self.fusion_features is not None:
            data_dict['fusion_features'] = torch.from_numpy(self.fusion_features[idx]).float()
            data_dict['features'] = torch.from_numpy(self.fusion_features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        subject = torch.tensor(self.subjects[idx], dtype=torch.long)
        return data_dict, label, subject

class DataAdapter(nn.Module):
    def __init__(self, model):
        super(DataAdapter, self).__init__()
        self.model = model
        self.field_mappings = {
            'accelerometer': 'acc',
            'gyroscope': 'gyro',
            'quaternion': 'quat',
            'fusion_features': 'features'
        }
    def forward(self, x):
        if isinstance(x, dict):
            adapted_dict = {}
            for k, v in x.items():
                adapted_dict[k] = v
                if k in self.field_mappings and self.field_mappings[k] not in x:
                    adapted_dict[self.field_mappings[k]] = v
            try:
                return self.model(adapted_dict)
            except KeyError as e:
                missing_key = str(e).strip("'")
                if missing_key == 'acc' and 'accelerometer' in adapted_dict:
                    adapted_dict['acc'] = adapted_dict['accelerometer']
                elif missing_key == 'gyro' and 'gyroscope' in adapted_dict:
                    adapted_dict['gyro'] = adapted_dict['gyroscope']
                elif missing_key == 'quat' and 'quaternion' in adapted_dict:
                    adapted_dict['quat'] = adapted_dict['quaternion']
                elif missing_key == 'features' and 'fusion_features' in adapted_dict:
                    adapted_dict['features'] = adapted_dict['fusion_features']
                else:
                    available = list(adapted_dict.keys())
                    raise KeyError(f"Model requires field '{missing_key}' which is not available. Available fields: {available}")
                return self.model(adapted_dict)
        else:
            return self.model(x)

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    unique_preds, unique_true = np.unique(y_pred), np.unique(y_true)
    if len(unique_preds) == 1 and len(unique_true) == 1:
        if unique_preds[0] == unique_true[0]:
            return {'accuracy': 100.0, 'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
        else:
            return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    try:
        metrics = {
            'accuracy': 100 * accuracy_score(y_true, y_pred),
            'f1': 100 * f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision': 100 * precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': 100 * recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        class_metrics = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        for cls in class_metrics:
            if cls in ['0', '1']:
                metrics[f'f1_class{cls}'] = 100 * class_metrics[cls]['f1-score']
                metrics[f'precision_class{cls}'] = 100 * class_metrics[cls]['precision']
                metrics[f'recall_class{cls}'] = 100 * class_metrics[cls]['recall']
        return metrics
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    if len(loader) == 0:
        logger.warning("Training loader is empty - skipping epoch")
        return 0.0, [], [], {}
    for batch_idx, (data, labels, _) in enumerate(tqdm(loader, desc="Training")):
        for k, v in data.items():
            data[k] = v.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            if len(outputs.shape) == 2 and outputs.shape[1] == 2:
                n_labels = outputs.shape[0]
                target = torch.zeros(n_labels, 2, device=device)
                target[torch.arange(n_labels), labels] = 1
                loss = criterion(outputs, target)
            elif len(outputs.shape) == 2 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels.float())
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if len(outputs.shape) == 2 and outputs.shape[1] > 1:
            _, preds = torch.max(outputs, 1)
        else:
            preds = (torch.sigmoid(outputs) > 0.5).int()
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        if batch_idx % 50 == 0 and batch_idx > 0:
            logger.info(f"Train Batch {batch_idx}/{len(loader)}: Loss={loss.item():.4f}")
    epoch_loss = running_loss / len(loader) if total > 0 else 0.0
    metrics = calculate_metrics(all_labels, all_preds)
    return epoch_loss, all_preds, all_labels, metrics

def validate(model, loader, criterion, device, prefix="val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    if len(loader) == 0:
        logger.warning(f"{prefix.capitalize()} loader is empty - skipping evaluation")
        return 0.0, [], [], {}
    with torch.no_grad():
        for data, labels, _ in tqdm(loader, desc=f"{prefix.capitalize()}"):
            for k, v in data.items():
                data[k] = v.to(device)
            labels = labels.to(device)
            outputs = model(data)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                if len(outputs.shape) == 2 and outputs.shape[1] == 2:
                    n_labels = outputs.shape[0]
                    target = torch.zeros(n_labels, 2, device=device)
                    target[torch.arange(n_labels), labels] = 1
                    loss = criterion(outputs, target)
                elif len(outputs.shape) == 2 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels.float())
                else:
                    loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            if len(outputs.shape) == 2 and outputs.shape[1] > 1:
                _, preds = torch.max(outputs, 1)
            else:
                preds = (torch.sigmoid(outputs) > 0.5).int()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader) if total > 0 else 0.0
    metrics = calculate_metrics(all_labels, all_preds)
    return epoch_loss, all_preds, all_labels, metrics

def create_subject_folds():
    val_subjects = [38, 46]
    always_train_subjects = [45, 36, 29]
    eligible_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
    folds = []
    for i, test_subject in enumerate(eligible_subjects):
        test_subjects = [test_subject]
        train_subjects = always_train_subjects + [s for s in eligible_subjects if s != test_subject]
        folds.append({'test': test_subjects, 'val': val_subjects, 'train': train_subjects})
    return folds

def log_dataset_statistics(dataset, split_name):
    if dataset is None or len(dataset) == 0:
        logger.warning(f"{split_name} dataset is empty!")
        return
    logger.info(f"{split_name} dataset size: {len(dataset)} samples")
    all_labels, all_subjects = [], []
    for i in range(len(dataset)):
        _, label, subject = dataset[i]
        all_labels.append(label.item())
        all_subjects.append(subject.item())
    all_labels, all_subjects = np.array(all_labels), np.array(all_subjects)
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    class_dist = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
    logger.info(f"{split_name} class distribution: {class_dist}")
    unique_subjects, subject_counts = np.unique(all_subjects, return_counts=True)
    subject_dist = {int(subject): int(count) for subject, count in zip(unique_subjects, subject_counts)}
    logger.info(f"{split_name} subject distribution: {subject_dist}")

def prepare_datasets(args, fold_idx=0):
    from utils.dataset import split_by_subjects, prepare_smartfallmm
    folds = create_subject_folds()
    if fold_idx >= len(folds):
        logger.error(f"Fold index {fold_idx} out of range. Max fold: {len(folds)-1}")
        return None, None, None
    
    # Load data first to check available subjects
    current_fold = folds[fold_idx]
    logger.info(f"Fold {fold_idx+1}/{len(folds)}: Train subjects={current_fold['train']}, "
               f"Val subjects={current_fold['val']}, Test subjects={current_fold['test']}")
    
    # Use all subjects for initial data loading
    all_eligible_subjects = sum([fold['train'] + fold['val'] + fold['test'] for fold in folds], [])
    args.subjects = list(set(all_eligible_subjects))  # Remove duplicates
    
    try:
        if hasattr(args, 'dataset_args') and 'fusion_options' in args.dataset_args:
            args.dataset_args['fusion_options']['filter_type'] = args.filter_type
            args.dataset_args['fusion_options']['visualize'] = args.visualize
            logger.info(f"Using filter type: {args.filter_type} for data alignment and fusion with visualize={args.visualize}")
        
        logger.info(f"Loading and preprocessing data with filter: {args.filter_type}")
        start_time = time.time()
        is_raw_acc = True
        if 'fusion_options' in args.dataset_args:
            args.dataset_args['fusion_options']['is_raw_acc'] = is_raw_acc
        
        data = split_by_subjects(prepare_smartfallmm(args), args.subjects, args.fuse)
        logger.info(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        
        if 'subjects' not in data or len(data.get('subjects', [])) == 0:
            if 'labels' in data:
                data['subjects'] = np.zeros(len(data['labels']), dtype=np.int32)
                logger.warning(f"Created {len(data['labels'])} dummy subject IDs")
        
        # Find which subjects actually exist in the data
        available_subjects = set(np.unique(data['subjects']).astype(int))
        logger.info(f"Available subjects in dataset: {sorted(available_subjects)}")
        
        # Adjust fold assignment based on available subjects
        adjusted_fold = {
            'train': [s for s in current_fold['train'] if s in available_subjects],
            'val': [s for s in current_fold['val'] if s in available_subjects],
            'test': [s for s in current_fold['test'] if s in available_subjects]
        }
        
        # If test is empty, take one subject from train
        if not adjusted_fold['test'] and adjusted_fold['train']:
            # Move one subject from train to test (preferably from eligible_subjects)
            eligible_test = [s for s in adjusted_fold['train'] 
                            if s in [32, 39, 30, 31, 33, 34, 35, 37, 43, 44] and s not in [45, 36, 29]]
            if eligible_test:
                test_subject = eligible_test[0]
            else:
                test_subject = adjusted_fold['train'][0]
            
            adjusted_fold['test'] = [test_subject]
            adjusted_fold['train'].remove(test_subject)
            logger.info(f"Reassigned subject {test_subject} from train to test set")
        
        logger.info(f"Adjusted fold: Train={adjusted_fold['train']}, "
                   f"Val={adjusted_fold['val']}, Test={adjusted_fold['test']}")
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                logger.info(f"Loaded modality '{key}' with shape {value.shape}")
        
        full_dataset = FallDataset(data)
        logger.info(f"Dataset loaded with {len(full_dataset)} samples")
        
        train_indices, val_indices, test_indices = [], [], []
        for i in range(len(full_dataset)):
            _, _, subject = full_dataset[i]
            subject = subject.item()
            if subject in adjusted_fold['train']:
                train_indices.append(i)
            elif subject in adjusted_fold['val']:
                val_indices.append(i)
            elif subject in adjusted_fold['test']:
                test_indices.append(i)
        
        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)
        test_set = Subset(full_dataset, test_indices)
        
        log_dataset_statistics(train_set, "Training")
        log_dataset_statistics(val_set, "Validation")
        log_dataset_statistics(test_set, "Test")
        
        logger.info(f"Split dataset into {len(train_set)} train, {len(val_set)} validation, "
                   f"and {len(test_set)} test samples")
        
        if len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0:
            logger.warning(f"Skipping fold {fold_idx} due to empty dataset split")
            return None, None, None
        
        return train_set, val_set, test_set
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

def get_model(args):
    if args.model.startswith('Models.fusion_transformer.FusionTransModel'):
        try:
            sys.path.append(os.getcwd())
            from Models.fusion_transformer import FusionTransModel
            sig = inspect.signature(FusionTransModel.__init__)
            params = {}
            param_mapping = {
                'input_channels': ['acc_input_dim', 'acc_coords', 'input_dim'],
                'hidden_channels': ['hidden_dim', 'embed_dim', 'feature_dim'],
                'num_layers': ['num_layers'],
                'dropout': ['dropout'], 
                'num_classes': ['num_classes'],
            }
            for arg_name, model_params in param_mapping.items():
                arg_value = getattr(args, arg_name)
                for param in model_params:
                    if param in sig.parameters:
                        params[param] = arg_value
            fixed_params = {
                'acc_frames': 128,
                'mocap_frames': 128, 
                'num_heads': 4,
                'use_batch_norm': True,
                'use_features': args.use_features,
                'fusion_type': 'concat',
                'quat_coords': 4
            }
            for param, value in fixed_params.items():
                if param in sig.parameters:
                    params[param] = value
            logger.info(f"Initializing FusionTransModel with parameters: {params}")
            model = FusionTransModel(**params)
            return DataAdapter(model)
        except Exception as e:
            logger.error(f"Failed to initialize FusionTransModel: {e}")
            try:
                from Models.lstm import LSTMModel
                logger.info("Falling back to LSTMModel")
                return LSTMModel(
                    input_channels=args.input_channels,
                    hidden_channels=args.hidden_channels,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    num_classes=args.num_classes,
                    use_fusion=args.fuse
                )
            except ImportError:
                logger.warning("LSTMModel module not found, defining inline")
                class LSTMModel(nn.Module):
                    def __init__(self, input_channels=3, hidden_channels=64, num_layers=2,
                                dropout=0.5, num_classes=2, use_fusion=True):
                        super(LSTMModel, self).__init__()
                        self.lstm = nn.LSTM(
                            input_size=input_channels,
                            hidden_size=hidden_channels,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True
                        )
                        self.fusion_layer = None
                        if use_fusion:
                            self.fusion_layer = nn.Linear(43, hidden_channels*2)
                        self.fc = nn.Linear(hidden_channels * 2, num_classes)
                    def forward(self, x):
                        if isinstance(x, dict):
                            if 'accelerometer' in x:
                                x_acc = x['accelerometer']
                                lstm_out, _ = self.lstm(x_acc)
                                lstm_out = lstm_out[:, -1, :]
                                if self.fusion_layer is not None and 'fusion_features' in x:
                                    fusion_out = self.fusion_layer(x['fusion_features'])
                                    if lstm_out.size() == fusion_out.size():
                                        lstm_out = lstm_out + fusion_out
                                return self.fc(lstm_out)
                            else:
                                raise ValueError("Accelerometer data is required")
                        else:
                            lstm_out, _ = self.lstm(x)
                            lstm_out = lstm_out[:, -1, :]
                            return self.fc(lstm_out)
                return LSTMModel(
                    input_channels=args.input_channels,
                    hidden_channels=args.hidden_channels,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    num_classes=args.num_classes,
                    use_fusion=args.fuse
                )
    else:
        try:
            from Models.lstm import LSTMModel
            logger.info("Using LSTMModel")
            return LSTMModel(
                input_channels=args.input_channels,
                hidden_channels=args.hidden_channels,
                num_layers=args.num_layers,
                dropout=args.dropout,
                num_classes=args.num_classes,
                use_fusion=args.fuse
            )
        except ImportError:
            logger.warning("LSTMModel module not found, defining inline")
            class LSTMModel(nn.Module):
                def __init__(self, input_channels=3, hidden_channels=64, num_layers=2,
                            dropout=0.5, num_classes=2, use_fusion=True):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_channels,
                        hidden_size=hidden_channels,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0,
                        bidirectional=True
                    )
                    self.fusion_layer = None
                    if use_fusion:
                        self.fusion_layer = nn.Linear(43, hidden_channels*2)
                    self.fc = nn.Linear(hidden_channels * 2, num_classes)
                def forward(self, x):
                    if isinstance(x, dict):
                        if 'accelerometer' in x:
                            x_acc = x['accelerometer']
                            lstm_out, _ = self.lstm(x_acc)
                            lstm_out = lstm_out[:, -1, :]
                            if self.fusion_layer is not None and 'fusion_features' in x:
                                fusion_out = self.fusion_layer(x['fusion_features'])
                                if lstm_out.size() == fusion_out.size():
                                    lstm_out = lstm_out + fusion_out
                            return self.fc(lstm_out)
                        else:
                            raise ValueError("Accelerometer data is required")
                    else:
                        lstm_out, _ = self.lstm(x)
                        lstm_out = lstm_out[:, -1, :]
                        return self.fc(lstm_out)
            return LSTMModel(
                input_channels=args.input_channels,
                hidden_channels=args.hidden_channels,
                num_layers=args.num_layers,
                dropout=args.dropout,
                num_classes=args.num_classes,
                use_fusion=args.fuse
            )

def get_loss(args):
    if args.loss.lower() == 'bce':
        logger.info("Using BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()
    elif args.loss.lower() == 'ce':
        logger.info("Using CrossEntropyLoss")
        return nn.CrossEntropyLoss()
    else:
        logger.warning(f"Unknown loss {args.loss}, using CrossEntropyLoss")
        return nn.CrossEntropyLoss()

def get_optimizer(args, model):
    if args.optimizer.lower() == 'adam':
        logger.info(f"Using Adam optimizer with lr={args.base_lr}, weight_decay={args.weight_decay}")
        return optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        logger.info(f"Using AdamW optimizer with lr={args.base_lr}, weight_decay={args.weight_decay}")
        return optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        logger.info(f"Using SGD optimizer with lr={args.base_lr}, momentum=0.9, weight_decay={args.weight_decay}")
        return optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        logger.warning(f"Unknown optimizer {args.optimizer}, using Adam")
        return optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

def plot_metrics(fold_work_dir, metrics_history, title_prefix=""):
    try:
        if not metrics_history or len(metrics_history.get('train_loss', [])) == 0:
            logger.warning("No training metrics to plot")
            return
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in metrics_history and metrics_history.get('val_loss', []) and len(metrics_history['val_loss']) > 0:
            plt.plot(epochs[:len(metrics_history['val_loss'])], metrics_history['val_loss'], 'r-', label='Validation Loss')
        if 'test_loss' in metrics_history and len(metrics_history.get('test_loss', [])) > 0:
            plt.plot(epochs[:len(metrics_history['test_loss'])], metrics_history['test_loss'], 'g-', label='Test Loss')
        plt.title(f'{title_prefix}Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 2)
        if 'train_accuracy' in metrics_history and len(metrics_history.get('train_accuracy', [])) > 0:
            plt.plot(epochs[:len(metrics_history['train_accuracy'])], metrics_history['train_accuracy'], 'b-', label='Training Accuracy')
        if 'val_accuracy' in metrics_history and len(metrics_history.get('val_accuracy', [])) > 0:
            plt.plot(epochs[:len(metrics_history['val_accuracy'])], metrics_history['val_accuracy'], 'r-', label='Validation Accuracy')
        if 'test_accuracy' in metrics_history and len(metrics_history.get('test_accuracy', [])) > 0:
            plt.plot(epochs[:len(metrics_history['test_accuracy'])], metrics_history['test_accuracy'], 'g-', label='Test Accuracy')
        plt.title(f'{title_prefix}Accuracy Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 3)
        if 'train_f1' in metrics_history and len(metrics_history.get('train_f1', [])) > 0:
            plt.plot(epochs[:len(metrics_history['train_f1'])], metrics_history['train_f1'], 'b-', label='Training F1')
        if 'val_f1' in metrics_history and len(metrics_history.get('val_f1', [])) > 0:
            plt.plot(epochs[:len(metrics_history['val_f1'])], metrics_history['val_f1'], 'r-', label='Validation F1')
        if 'test_f1' in metrics_history and len(metrics_history.get('test_f1', [])) > 0:
            plt.plot(epochs[:len(metrics_history['test_f1'])], metrics_history['test_f1'], 'g-', label='Test F1')
        plt.title(f'{title_prefix}F1 Score Curves')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 4)
        if 'val_precision' in metrics_history and len(metrics_history.get('val_precision', [])) > 0:
            plt.plot(epochs[:len(metrics_history['val_precision'])], metrics_history['val_precision'], 'b-', label='Val Precision')
        if 'val_recall' in metrics_history and len(metrics_history.get('val_recall', [])) > 0:
            plt.plot(epochs[:len(metrics_history['val_recall'])], metrics_history['val_recall'], 'r-', label='Val Recall')
        if 'test_precision' in metrics_history and metrics_history.get('test_precision', []) and len(metrics_history['test_precision']) > 0:
            plt.plot(epochs[:len(metrics_history['test_precision'])], metrics_history['test_precision'], 'g-', label='Test Precision')
        if 'test_recall' in metrics_history and metrics_history.get('test_recall', []) and len(metrics_history['test_recall']) > 0:
            plt.plot(epochs[:len(metrics_history['test_recall'])], metrics_history['test_recall'], 'm-', label='Test Recall')
        plt.title(f'{title_prefix}Precision/Recall Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_work_dir, f'{title_prefix.lower().replace(" ", "_")}metrics.png'), dpi=300)
        plt.close()
        if 'val_f1' in metrics_history and len(metrics_history.get('val_f1', [])) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, metrics_history['val_f1'], 'r-', linewidth=2, label='Validation F1')
            if 'val_f1_class0' in metrics_history and len(metrics_history.get('val_f1_class0', [])) > 0:
                plt.plot(epochs[:len(metrics_history['val_f1_class0'])], metrics_history['val_f1_class0'], 'b--', label='Val F1 - Class 0 (No Fall)')
            if 'val_f1_class1' in metrics_history and len(metrics_history.get('val_f1_class1', [])) > 0:
                plt.plot(epochs[:len(metrics_history['val_f1_class1'])], metrics_history['val_f1_class1'], 'g--', label='Val F1 - Class 1 (Fall)')
            best_epoch = np.argmax(metrics_history['val_f1']) + 1
            best_f1 = max(metrics_history['val_f1'])
            plt.axvline(x=best_epoch, color='k', linestyle='--', alpha=0.7)
            plt.text(best_epoch + 0.5, best_f1 - 5, f'Best F1: {best_f1:.2f}% (Epoch {best_epoch})', 
                    bbox=dict(facecolor='white', alpha=0.8))
            plt.title(f'{title_prefix}Validation F1 Score Analysis')
            plt.xlabel('Epochs')
            plt.ylabel('F1 Score (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fold_work_dir, f'{title_prefix.lower().replace(" ", "_")}f1_analysis.png'), dpi=300)
            plt.close()
        logger.info(f"Created metric plots at {fold_work_dir}")
    except Exception as e:
        logger.error(f"Error creating metric plots: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=-1, help='Fold index (-1 means all folds)')
    parser.add_argument('--subjects', type=str, default='30,31,32,33,34,35,37,39,43,44,45,36,29,38,46')
    parser.add_argument('--fuse', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--filters', type=str, default='madgwick,kalman,ekf')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--num-worker', type=int, default=24)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--base-lr', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--input-channels', type=int, default=3)
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--use-gpu', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--work-dir', type=str, default='./results/')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--model', type=str, default='Models.fusion_transformer.FusionTransModel')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_features', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--filter-type', type=str, default='ekf', help='Filter type to use (madgwick, kalman, ekf)')
    parser.add_argument('--visualize', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose logging')
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(args, 'filters') and isinstance(args.filters, str):
        args.filters = args.filters.split(',')
    else:
        args.filters = ['madgwick', 'kalman', 'ekf']
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    args.dataset_args = {
        'age_group': ['young', 'old'],
        'modalities': ['accelerometer', 'gyroscope'],
        'sensors': ['watch'],
        'mode': 'sliding_window',
        'max_length': 128,
        'task': 'fd',
        'fusion_options': {
            'enabled': args.fuse,
            'filter_type': args.filter_type,
            'visualize': args.visualize,
            'save_aligned': True,
            'is_raw_acc': True
        }
    }
    os.makedirs(args.work_dir, exist_ok=True)
    filter_dir = args.work_dir
    os.makedirs(filter_dir, exist_ok=True)
    logger.info(f"Working directory: {filter_dir}")
    run_config = vars(args).copy()
    for k, v in run_config.items():
        if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
            run_config[k] = str(v)
    with open(os.path.join(filter_dir, 'args.json'), 'w') as f:
        json.dump(run_config, f, indent=4)
    folds = create_subject_folds()
    with open(os.path.join(filter_dir, 'folds.json'), 'w') as f:
        json.dump(folds, f, indent=4)
    if args.fold == -1:
        fold_indices = list(range(len(folds)))
        logger.info(f"Processing all {len(folds)} folds")
    else:
        fold_indices = [args.fold]
        logger.info(f"Processing single fold: {args.fold}")
    results = defaultdict(list)
    fold_results_df = pd.DataFrame()
    for fold_idx in fold_indices:
        fold_start_time = time.time()
        logger.info("="*80)
        logger.info(f"Processing fold {fold_idx+1}/{len(folds)}")
        logger.info("="*80)
        fold_work_dir = os.path.join(filter_dir, f'fold_{fold_idx}')
        os.makedirs(fold_work_dir, exist_ok=True)
        if args.phase.lower() == 'train':
            train_set, val_set, test_set = prepare_datasets(args, fold_idx)
            if train_set is None or val_set is None or test_set is None:
                logger.warning(f"Skipping fold {fold_idx} due to missing datasets")
                continue
            train_loader = DataLoader(
                train_set, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_worker, 
                pin_memory=True,
                drop_last=False
            )
            val_loader = DataLoader(
                val_set, 
                batch_size=args.test_batch_size, 
                shuffle=False, 
                num_workers=args.num_worker, 
                pin_memory=True,
                drop_last=False
            )
            test_loader = DataLoader(
                test_set, 
                batch_size=args.test_batch_size, 
                shuffle=False, 
                num_workers=args.num_worker, 
                pin_memory=True,
                drop_last=False
            )
            logger.info(f"Loaders created - Train: {len(train_loader)} batches, "
                       f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
            model = get_model(args).to(device)
            criterion = get_loss(args)
            optimizer = get_optimizer(args, model)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=args.patience//2, factor=0.5)
            best_val_f1, best_epoch = 0, 0
            best_val_metrics, best_test_metrics = {}, {}
            metrics_history = defaultdict(list)
            early_stop_counter = 0
            for epoch in range(args.max_epoch):
                epoch_start = time.time()
                logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}/{args.max_epoch}")
                try:
                    train_loss, train_preds, train_labels, train_metrics = train_epoch(
                        model, train_loader, criterion, optimizer, device)
                    val_loss, val_preds, val_labels, val_metrics = validate(
                        model, val_loader, criterion, device, prefix="val")
                    scheduler.step(val_loss)
                    epoch_time = time.time() - epoch_start
                    for key, value in train_metrics.items():
                        metrics_history[key].append(value)
                    for key, value in val_metrics.items():
                        metrics_history[key].append(value)
                    metrics_history['train_loss'].append(train_loss)
                    metrics_history['val_loss'].append(val_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    metrics_history['learning_rate'].append(current_lr)
                    logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                              f"Train F1: {train_metrics.get('f1', 0):.2f}%, "
                              f"Val F1: {val_metrics.get('f1', 0):.2f}%, "
                              f"LR: {current_lr:.6f}")
                    val_f1 = val_metrics.get('f1', 0)
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_epoch = epoch
                        best_val_metrics = val_metrics
                        early_stop_counter = 0
                        torch.save({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'val_metrics': val_metrics,
                            'filter_type': args.filter_type
                        }, os.path.join(fold_work_dir, 'best_model.pth'))
                        logger.info(f"New best model saved with validation F1: {val_f1:.2f}%")
                        if len(val_preds) > 0:
                            plt.figure(figsize=(10, 8))
                            cm_val = confusion_matrix(val_labels, val_preds)
                            plt.imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
                            plt.title('Validation Confusion Matrix')
                            plt.colorbar()
                            plt.tight_layout()
                            plt.ylabel('True label')
                            plt.xlabel('Predicted label')
                            plt.savefig(os.path.join(fold_work_dir, 'val_confusion_matrix.png'))
                            plt.close()
                            val_report = classification_report(val_labels, val_preds, output_dict=True)
                            with open(os.path.join(fold_work_dir, 'val_classification_report.json'), 'w') as f:
                                json.dump(val_report, f, indent=4)
                    else:
                        early_stop_counter += 1
                    if (epoch + 1) % args.save_interval == 0:
                        torch.save({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'val_metrics': val_metrics,
                            'filter_type': args.filter_type
                        }, os.path.join(fold_work_dir, f'checkpoint_epoch{epoch+1}.pth'))
                    if early_stop_counter >= args.patience:
                        logger.info(f"Early stopping triggered (no improvement for {args.patience} epochs)")
                        break
                except Exception as e:
                    logger.error(f"Error during fold {fold_idx}, epoch {epoch+1}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            logger.info("Training complete - Loading best model for test evaluation")
            best_model_path = os.path.join(fold_work_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                try:
                    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['state_dict'])
                    test_loss, test_preds, test_labels, test_metrics = validate(
                        model, test_loader, criterion, device, prefix="test")
                    logger.info(f"Test evaluation with best model (epoch {best_epoch+1}): "
                              f"F1: {test_metrics.get('f1', 0):.2f}%, "
                              f"Accuracy: {test_metrics.get('accuracy', 0):.2f}%, " 
                              f"Precision: {test_metrics.get('precision', 0):.2f}%, "
                              f"Recall: {test_metrics.get('recall', 0):.2f}%")
                    best_test_metrics = test_metrics
                    if len(test_preds) > 0:
                        plt.figure(figsize=(10, 8))
                        cm_test = confusion_matrix(test_labels, test_preds)
                        plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
                        plt.title('Test Confusion Matrix')
                        plt.colorbar()
                        plt.tight_layout()
                        plt.ylabel('True label')
                        plt.xlabel('Predicted label')
                        plt.savefig(os.path.join(fold_work_dir, 'test_confusion_matrix.png'))
                        plt.close()
                        test_report = classification_report(test_labels, test_preds, output_dict=True)
                        with open(os.path.join(fold_work_dir, 'test_classification_report.json'), 'w') as f:
                            json.dump(test_report, f, indent=4)
                except Exception as e:
                    logger.error(f"Error loading best model: {e}")
                    best_test_metrics = {}
                    try:
                        checkpoint = torch.load(best_model_path, map_location=device)
                        model.load_state_dict(checkpoint['state_dict'])
                        test_loss, test_preds, test_labels, test_metrics = validate(
                            model, test_loader, criterion, device, prefix="test")
                        best_test_metrics = test_metrics
                    except Exception as e2:
                        logger.error(f"Fallback loading also failed: {e2}")
            else:
                logger.error(f"Best model file not found at {best_model_path}")
                best_test_metrics = {}
            fold_training_time = time.time() - fold_start_time
            logger.info(f"Fold {fold_idx+1} processing completed in {fold_training_time:.2f} seconds")
            plot_metrics(fold_work_dir, metrics_history, f"Fold {fold_idx+1} ")
            fold_summary = {
                'fold': fold_idx,
                'best_epoch': best_epoch + 1,
                'best_val_f1': best_val_f1,
                'training_time': fold_training_time,
                **best_val_metrics,
                **best_test_metrics
            }
            with open(os.path.join(fold_work_dir, 'fold_summary.json'), 'w') as f:
                json.dump(fold_summary, f, indent=4)
            results['fold'].append(fold_idx)
            results['val_f1'].append(best_val_f1)
            results['test_f1'].append(best_test_metrics.get('f1', 0))
            results['val_accuracy'].append(best_val_metrics.get('accuracy', 0))
            results['test_accuracy'].append(best_test_metrics.get('accuracy', 0))
            results['val_precision'].append(best_val_metrics.get('precision', 0))
            results['test_precision'].append(best_test_metrics.get('precision', 0))
            results['val_recall'].append(best_val_metrics.get('recall', 0))
            results['test_recall'].append(best_test_metrics.get('recall', 0))
            fold_row = {
                'fold': fold_idx,
                'filter': args.filter_type,
                'best_epoch': best_epoch + 1,
                'training_time': fold_training_time,
                **{f'val_{k}': v for k, v in best_val_metrics.items() if not k.endswith('_history')},
                **{f'test_{k}': v for k, v in best_test_metrics.items() if not k.endswith('_history')}
            }
            fold_results_df = pd.concat([fold_results_df, pd.DataFrame([fold_row])], ignore_index=True)
        elif args.phase.lower() == 'test':
            logger.info(f"Testing fold {fold_idx}")
            _, _, test_set = prepare_datasets(args, fold_idx)
            if test_set is None or len(test_set) == 0:
                logger.warning(f"Skipping fold {fold_idx} due to empty test set")
                continue
            test_loader = DataLoader(
                test_set, 
                batch_size=args.test_batch_size, 
                shuffle=False,
                num_workers=args.num_worker, 
                pin_memory=True,
                drop_last=False
            )
            model_path = os.path.join(fold_work_dir, 'best_model.pth')
            if not os.path.exists(model_path):
                logger.error(f"Best model not found at {model_path}. Skipping fold {fold_idx}.")
                continue
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model = get_model(args).to(device)
                model.load_state_dict(checkpoint['state_dict'])
                logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
                test_loss, test_preds, test_labels, test_metrics = validate(
                    model, test_loader, get_loss(args), device, prefix="test")
                logger.info(f"Fold {fold_idx+1} Test Results: F1={test_metrics.get('f1', 0):.2f}%, "
                          f"Accuracy={test_metrics.get('accuracy', 0):.2f}%, "
                          f"Precision={test_metrics.get('precision', 0):.2f}%, "
                          f"Recall={test_metrics.get('recall', 0):.2f}%")
                results['fold'].append(fold_idx)
                results['test_f1'].append(test_metrics.get('f1', 0))
                results['test_accuracy'].append(test_metrics.get('accuracy', 0))
                results['test_precision'].append(test_metrics.get('precision', 0))
                results['test_recall'].append(test_metrics.get('recall', 0))
                fold_row = {
                    'fold': fold_idx,
                    'filter': args.filter_type,
                    **{f'test_{k}': v for k, v in test_metrics.items() if not k.endswith('_history')}
                }
                fold_results_df = pd.concat([fold_results_df, pd.DataFrame([fold_row])], ignore_index=True)
                if len(test_preds) > 0:
                    cm_test = confusion_matrix(test_labels, test_preds)
                    test_report = classification_report(test_labels, test_preds, output_dict=True)
                    try:
                        plt.figure(figsize=(10, 8))
                        plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
                        plt.title(f'Fold {fold_idx+1} Test Confusion Matrix')
                        plt.colorbar()
                        plt.tight_layout()
                        plt.ylabel('True label')
                        plt.xlabel('Predicted label')
                        plt.savefig(os.path.join(fold_work_dir, 'test_confusion_matrix.png'))
                        plt.close()
                    except Exception as e:
                        logger.error(f"Error saving test confusion matrix for fold {fold_idx}: {e}")
                    try:
                        with open(os.path.join(fold_work_dir, 'test_classification_report.json'), 'w') as f:
                            json.dump(test_report, f, indent=4)
                    except Exception as e:
                        logger.error(f"Error saving test classification report for fold {fold_idx}: {e}")
            except Exception as e:
                logger.error(f"Error during testing fold {fold_idx}: {e}")
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    model = get_model(args).to(device)
                    model.load_state_dict(checkpoint['state_dict'])
                    test_loss, test_preds, test_labels, test_metrics = validate(
                        model, test_loader, get_loss(args), device, prefix="test")
                    logger.info(f"Fallback loaded - Fold {fold_idx+1} Test Results: F1={test_metrics.get('f1', 0):.2f}%")
                except Exception as e2:
                    logger.error(f"Fallback testing also failed: {e2}")
    if not fold_results_df.empty:
        csv_path = os.path.join(filter_dir, f'all_folds_results_{args.filter_type}.csv')
        fold_results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved all fold results to {csv_path}")
    if len(results.get('fold', [])) > 0:
        logger.info("Overall results across all folds:")
        avg_val_f1 = np.mean(results['val_f1']) if 'val_f1' in results and len(results['val_f1']) > 0 else 0
        avg_test_f1 = np.mean(results['test_f1']) if len(results['test_f1']) > 0 else 0
        avg_val_accuracy = np.mean(results['val_accuracy']) if 'val_accuracy' in results and len(results['val_accuracy']) > 0 else 0
        avg_test_accuracy = np.mean(results['test_accuracy']) if len(results['test_accuracy']) > 0 else 0
        avg_val_precision = np.mean(results['val_precision']) if 'val_precision' in results and len(results['val_precision']) > 0 else 0
        avg_test_precision = np.mean(results['test_precision']) if len(results['test_precision']) > 0 else 0
        avg_val_recall = np.mean(results['val_recall']) if 'val_recall' in results and len(results['val_recall']) > 0 else 0
        avg_test_recall = np.mean(results['test_recall']) if len(results['test_recall']) > 0 else 0
        if 'val_f1' in results and len(results['val_f1']) > 0:
            logger.info(f"Average validation F1: {avg_val_f1:.2f}%")
            logger.info(f"Average validation accuracy: {avg_val_accuracy:.2f}%")
            logger.info(f"Average validation precision: {avg_val_precision:.2f}%")
            logger.info(f"Average validation recall: {avg_val_recall:.2f}%")
        logger.info(f"Average test F1: {avg_test_f1:.2f}%")
        logger.info(f"Average test accuracy: {avg_test_accuracy:.2f}%")
        logger.info(f"Average test precision: {avg_test_precision:.2f}%")
        logger.info(f"Average test recall: {avg_test_recall:.2f}%")
        overall_metrics = {
            'num_folds': len(results['fold']),
            'filter_type': args.filter_type,
            'avg_val_f1': float(avg_val_f1) if 'val_f1' in results and len(results['val_f1']) > 0 else 0,
            'avg_val_accuracy': float(avg_val_accuracy) if 'val_accuracy' in results and len(results['val_accuracy']) > 0 else 0,
            'avg_val_precision': float(avg_val_precision) if 'val_precision' in results and len(results['val_precision']) > 0 else 0,
            'avg_val_recall': float(avg_val_recall) if 'val_recall' in results and len(results['val_recall']) > 0 else 0,
            'avg_test_f1': float(avg_test_f1),
            'avg_test_accuracy': float(avg_test_accuracy),
            'avg_test_precision': float(avg_test_precision),
            'avg_test_recall': float(avg_test_recall),
            'fold_results': {
                'fold': results['fold'],
                'val_f1': results.get('val_f1', []),
                'val_accuracy': results.get('val_accuracy', []),
                'val_precision': results.get('val_precision', []),
                'val_recall': results.get('val_recall', []),
                'test_f1': results['test_f1'],
                'test_accuracy': results['test_accuracy'],
                'test_precision': results['test_precision'],
                'test_recall': results['test_recall']
            }
        }
        with open(os.path.join(filter_dir, f'overall_results_{args.filter_type}.json'), 'w') as f:
            json.dump(overall_metrics, f, indent=4)
        try:
            if len(results['fold']) > 1:
                plt.figure(figsize=(12, 8))
                x = list(range(len(results['fold'])))
                width = 0.15
                metrics_to_plot = []
                if 'val_accuracy' in results and len(results['val_accuracy']) > 0:
                    metrics_to_plot.extend(['val_accuracy', 'val_f1', 'val_precision', 'val_recall'])
                metrics_to_plot.extend(['test_accuracy', 'test_f1', 'test_precision', 'test_recall'])
                positions = []
                start_pos = -(len(metrics_to_plot) - 1) * width / 2
                for i in range(len(metrics_to_plot)):
                    positions.append(start_pos + i * width)
                for i, metric in enumerate(metrics_to_plot):
                    if metric in results and len(results[metric]) > 0:
                        plt.bar([p + positions[i] for p in x], results[metric], width, label=metric.replace('_', ' ').title())
                plt.axhline(y=avg_test_f1, color='r', linestyle='--', alpha=0.5)
                plt.text(len(results['fold'])-1, avg_test_f1, f'Avg Test F1: {avg_test_f1:.2f}%')
                plt.title(f'Performance Metrics Across Folds (Filter: {args.filter_type})')
                plt.xlabel('Fold')
                plt.ylabel('Metric Value (%)')
                plt.xticks(x, [f'{i}' for i in results['fold']])
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(filter_dir, f'fold_comparison_{args.filter_type}.png'))
                plt.close()
        except Exception as e:
            logger.error(f"Error creating fold comparison plot: {e}")
    else:
        logger.warning("No results to report across folds.")
    logger.info("="*80)
    logger.info(f"Training completed for filter type: {args.filter_type}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
