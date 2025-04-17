import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    try:
        if np.sum(y_pred) == 0 and np.sum(y_true) == 0:
            return {'accuracy': 100.0, 'f1': 100.0, 'precision': 100.0, 'recall': 100.0}
        elif np.sum(y_pred) == 0:
            return {'accuracy': 100.0 * (1 - np.mean(y_true)), 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        elif np.sum(y_true) == 0:
            return {'accuracy': 100.0 * (1 - np.mean(y_pred)), 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        metrics = {
            'accuracy': 100 * accuracy_score(y_true, y_pred),
            'f1': 100 * f1_score(y_true, y_pred, average='binary', zero_division=0),
            'precision': 100 * precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': 100 * recall_score(y_true, y_pred, average='binary', zero_division=0)
        }
        return metrics
    except Exception as e:
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if not np.isfinite(val_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
