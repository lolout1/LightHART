import os, torch, numpy as np, argparse, logging, json, time, random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn, optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib
import sys
import inspect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FallDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.accelerometer = data.get('accelerometer', None)
        self.gyroscope = data.get('gyroscope', None)
        self.quaternion = data.get('quaternion', None)
        self.linear_acceleration = data.get('linear_acceleration', None)
        self.fusion_features = data.get('fusion_features', None)
        self.labels = data.get('labels', None)
        self.subjects = data.get('subjects', None)
        if self.labels is not None and (self.subjects is None or len(self.subjects) == 0):
            self.subjects = np.zeros(len(self.labels), dtype=np.int32)
            logger.warning(f"Created dummy subject IDs for {len(self.labels)} samples")
        
    def __len__(self): 
        return 0 if self.labels is None else len(self.labels)
        
    def __getitem__(self, idx):
        data_dict = {}
        # Map to the expected keys for FusionTransModel
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
                logger.error(f"Model expects field '{missing_key}' which is not available. Available fields: {list(adapted_dict.keys())}")
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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for data, labels, _ in tqdm(loader, desc="Training"):
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
    return running_loss/len(loader), 100*correct/total, all_preds, all_labels

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels, _ in tqdm(loader, desc="Validation"):
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
    return running_loss/len(loader), 100*correct/total, all_preds, all_labels

def create_subject_folds():
    # Define constant validation subjects
    val_subjects = [38, 46]
    
    # Define subjects that are always in training (no fall data)
    always_train_subjects = [45, 36, 29]
    
    # Define eligible subjects for test (have fall data)
    eligible_subjects = [32, 39, 30, 31, 33, 34, 35, 37, 43, 44]
    
    # Generate folds with 2 test subjects per fold
    folds = []
    for i in range(0, len(eligible_subjects), 2):
        if i + 1 < len(eligible_subjects):
            test_subjects = [eligible_subjects[i], eligible_subjects[i+1]]
            train_subjects = always_train_subjects + [s for s in eligible_subjects if s not in test_subjects]
            folds.append({
                'test': test_subjects,
                'val': val_subjects,
                'train': train_subjects
            })
    
    return folds

def prepare_datasets(args, fold_idx=0):
    from utils.dataset import split_by_subjects, prepare_smartfallmm
    
    # Get subject split for current fold
    folds = create_subject_folds()
    if fold_idx >= len(folds):
        logger.error(f"Fold index {fold_idx} out of range. Max fold: {len(folds)-1}")
        return None, None, None
    
    current_fold = folds[fold_idx]
    logger.info(f"Fold {fold_idx+1}: Train subjects={current_fold['train']}, "
               f"Val subjects={current_fold['val']}, Test subjects={current_fold['test']}")
    
    # Set subjects argument to include all subjects for data loading
    all_subjects = current_fold['train'] + current_fold['val'] + current_fold['test']
    args.subjects = all_subjects
    
    try:
        # Load all data
        data = split_by_subjects(prepare_smartfallmm(args), all_subjects, args.fuse)
        
        if 'subjects' not in data or len(data.get('subjects', [])) == 0:
            if 'labels' in data:
                data['subjects'] = np.zeros(len(data['labels']), dtype=np.int32)
                logger.warning(f"Created {len(data['labels'])} dummy subject IDs")
        
        # Create full dataset
        full_dataset = FallDataset(data)
        logger.info(f"Dataset loaded with {len(full_dataset)} samples")
        
        # Create indices for train, val, test based on subject IDs
        train_indices = []
        val_indices = []
        test_indices = []
        
        for i in range(len(full_dataset)):
            _, _, subject = full_dataset[i]
            subject = subject.item()
            if subject in current_fold['train']:
                train_indices.append(i)
            elif subject in current_fold['val']:
                val_indices.append(i)
            elif subject in current_fold['test']:
                test_indices.append(i)
        
        # Create subset datasets
        train_set = Subset(full_dataset, train_indices)
        val_set = Subset(full_dataset, val_indices)
        test_set = Subset(full_dataset, test_indices)
        
        logger.info(f"Split dataset into {len(train_set)} train, {len(val_set)} validation, "
                   f"and {len(test_set)} test samples")
        
        return train_set, val_set, test_set
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

class LSTMModel(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, num_layers=2, dropout=0.5, num_classes=2, use_fusion=True):
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
                    else:
                        logger.warning(f"Shape mismatch: lstm_out {lstm_out.size()}, fusion_out {fusion_out.size()}")
                
                return self.fc(lstm_out)
            else:
                raise ValueError("Accelerometer data is required")
        else:
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]
            return self.fc(lstm_out)

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
                'acc_frames': 64,
                'mocap_frames': 64, 
                'num_heads': 4,
                'use_batch_norm': True,
                'use_features': args.use_features,
                'fusion_type': 'concat',
                'quat_coords': 4
            }
            
            for param, value in fixed_params.items():
                if param in sig.parameters:
                    params[param] = value
            
            model = FusionTransModel(**params)
            return DataAdapter(model)
            
        except Exception as e:
            logger.error(f"Failed to initialize FusionTransModel: {e}")
            return LSTMModel(
                input_channels=args.input_channels,
                hidden_channels=args.hidden_channels,
                num_layers=args.num_layers,
                dropout=args.dropout,
                num_classes=args.num_classes,
                use_fusion=args.fuse
            )
    else:
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
        return nn.BCEWithLogitsLoss()
    elif args.loss.lower() == 'ce':
        return nn.CrossEntropyLoss()
    else:
        logger.warning(f"Unknown loss {args.loss}, using CrossEntropyLoss")
        return nn.CrossEntropyLoss()

def get_optimizer(args, model):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        logger.warning(f"Unknown optimizer {args.optimizer}, using Adam")
        return optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--subjects', type=str, default='30,31,32,33,34,35,37,39,43,44,45,36,29')
    parser.add_argument('--fuse', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--filters', type=str, default='madgwick')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--num-worker', type=int, default=4)
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--input-channels', type=int, default=3)
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--use-gpu', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--work-dir', type=str, default='./work_dir/')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_features', type=lambda x: x.lower() == 'true', default=False)
    args = parser.parse_args()
    
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Parse filters
    if hasattr(args, 'filters') and isinstance(args.filters, str):
        args.filters = args.filters.split(',')
    else:
        args.filters = ['madgwick']
    
    # Configure device (GPU or CPU)
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Setup dataset args
    args.dataset_args = {
        'age_group': ['young', 'old'],
        'modalities': ['accelerometer', 'gyroscope'],
        'sensors': ['watch'],
        'mode': 'sliding_window',
        'max_length': 64,
        'task': 'fd',
        'fusion_options': {
            'enabled': args.fuse,
            'filter_type': args.filters[0] if args.filters else 'madgwick',
            'visualize': False,
            'save_aligned': True,
        }
    }
    
    # Create working directory for the fold
    fold_work_dir = os.path.join(args.work_dir, f'fold_{args.fold}')
    os.makedirs(fold_work_dir, exist_ok=True)
    with open(os.path.join(fold_work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Generate folds information and save it
    folds = create_subject_folds()
    with open(os.path.join(args.work_dir, 'folds.json'), 'w') as f:
        json.dump(folds, f, indent=4)
    
    if args.phase.lower() == 'train':
        # Load datasets with subject-based split
        train_set, val_set, test_set = prepare_datasets(args, args.fold)
        
        if train_set is None or val_set is None or test_set is None:
            logger.error("Error preparing datasets. Exiting.")
            return
            
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.num_worker, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, 
                              num_workers=args.num_worker, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=args.num_worker, pin_memory=True)
        
        # Initialize model, loss, optimizer, and scheduler
        model = get_model(args).to(device)
        criterion = get_loss(args)
        optimizer = get_optimizer(args, model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.patience//2, factor=0.5, verbose=True)
        
        best_val_acc, best_epoch, best_test_acc = 0, 0, 0
        results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_acc': []}
        early_stop_counter = 0
        
        for epoch in range(args.max_epoch):
            logger.info(f"Epoch {epoch+1}/{args.max_epoch}")
            start_time = time.time()
            
            try:
                # Training phase
                train_loss, train_acc, train_preds, train_labels = train_epoch(
                    model, train_loader, criterion, optimizer, device)
                
                # Validation phase
                val_loss, val_acc, val_preds, val_labels = validate(
                    model, val_loader, criterion, device)
                
                # Test phase during training (for monitoring only)
                test_loss, test_acc, test_preds, test_labels = validate(
                    model, test_loader, criterion, device)
                
                scheduler.step(val_loss)
                epoch_time = time.time() - start_time
                
                results['train_loss'].append(train_loss)
                results['train_acc'].append(train_acc)
                results['val_loss'].append(val_loss)
                results['val_acc'].append(val_acc)
                results['test_acc'].append(test_acc)
                
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                          f"Test Acc={test_acc:.2f}% [Time: {epoch_time:.2f}s]")
                
                # Save best model based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
                    early_stop_counter = 0
                    
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'val_acc': val_acc,
                        'test_acc': test_acc,
                    }, os.path.join(fold_work_dir, 'best_model.pth'))
                    
                    logger.info(f"Best model saved with validation accuracy: {val_acc:.2f}%, "
                              f"test accuracy: {test_acc:.2f}%")
                    
                    # Save confusion matrices for validation and test sets
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
                    
                    # Save classification reports for validation and test sets
                    val_report = classification_report(val_labels, val_preds, output_dict=True)
                    with open(os.path.join(fold_work_dir, 'val_classification_report.json'), 'w') as f:
                        json.dump(val_report, f, indent=4)
                    
                    test_report = classification_report(test_labels, test_preds, output_dict=True)
                    with open(os.path.join(fold_work_dir, 'test_classification_report.json'), 'w') as f:
                        json.dump(test_report, f, indent=4)
                else:
                    early_stop_counter += 1
                
                # Save regular checkpoints
                if (epoch + 1) % args.save_interval == 0:
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'val_acc': val_acc,
                        'test_acc': test_acc,
                    }, os.path.join(fold_work_dir, f'checkpoint_epoch{epoch+1}.pth'))
                
                # Early stopping check
                if early_stop_counter >= args.patience:
                    logger.info(f"Early stopping triggered (no improvement for {args.patience} epochs)")
                    break
                    
            except Exception as e:
                logger.error(f"Error during epoch {epoch+1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # Save training history plots and summary
        try:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(results['train_loss'], label='Train')
            plt.plot(results['val_loss'], label='Validation')
            plt.title('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(results['train_acc'], label='Train')
            plt.plot(results['val_acc'], label='Validation')
            plt.plot(results['test_acc'], label='Test')
            plt.title('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(args.work_dir, f'training_history_fold{args.fold}.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error saving training plots: {e}")

        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
    
    # Test phase: Load best model and evaluate on the test set only
    elif args.phase.lower() == 'test':
        logger.info(f"Testing fold {args.fold}")
        
        # Load datasets with subject-based split (test set only)
        _, _, test_set = prepare_datasets(args, args.fold)
        if test_set is None:
            logger.error("Error preparing test dataset. Exiting.")
            return
            
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_worker, pin_memory=True)
        
        # Load best model from the fold directory
        fold_work_dir = os.path.join(args.work_dir, f'fold_{args.fold}')
        model_path = os.path.join(fold_work_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            logger.error(f"Best model not found at {model_path}. Exiting.")
            return
        
        checkpoint = torch.load(model_path, map_location=device)
        model = get_model(args).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded best model from {model_path} with validation accuracy {checkpoint.get('val_acc', 'N/A')}% and test accuracy {checkpoint.get('test_acc', 'N/A')}%")
        
        # Evaluate the test set
        test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, get_loss(args), device)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Compute confusion matrix and classification report
        cm_test = confusion_matrix(test_labels, test_preds)
        test_report = classification_report(test_labels, test_preds, output_dict=True)
        
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Test Confusion Matrix')
            plt.colorbar()
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(fold_work_dir, 'test_confusion_matrix.png'))
            plt.close()
        except Exception as e:
            logger.error(f"Error saving test confusion matrix: {e}")
        
        try:
            with open(os.path.join(fold_work_dir, 'test_classification_report.json'), 'w') as f:
                json.dump(test_report, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving test classification report: {e}")
    
    else:
        logger.error(f"Unknown phase: {args.phase}")

if __name__ == "__main__":
    main()

