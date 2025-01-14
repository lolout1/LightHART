import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from scipy.optimize import minimize

class CalibrationUtils:
    """Handles probability calibration and threshold optimization"""
    def __init__(self):
        self.temperature = 1.0
        self.optimal_threshold = 0.5
        
    def calibrate_temperature(self, logits, labels, max_iter=100):
        """
        Calibrates confidence using temperature scaling
        Args:
            logits: Raw model outputs before sigmoid
            labels: True binary labels
            max_iter: Maximum optimization iterations
        """
        def nll_loss(t):
            # Scaled sigmoid
            scaled_probs = torch.sigmoid(logits / t)
            # Negative log likelihood
            loss = F.binary_cross_entropy(scaled_probs, labels)
            return loss.item()
        
        # Optimize temperature parameter
        opt_result = minimize(nll_loss, x0=1.0, method='nelder-mead', 
                            options={'maxiter': max_iter})
        self.temperature = opt_result.x[0]
        
    def find_optimal_threshold(self, probabilities, labels, target_fpr=0.05):
        """
        Finds threshold that achieves target false positive rate
        Args:
            probabilities: Calibrated probabilities
            labels: True binary labels
            target_fpr: Target false positive rate
        """
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        # Find threshold closest to target FPR
        optimal_idx = np.argmin(np.abs(fpr - target_fpr))
        self.optimal_threshold = thresholds[optimal_idx]
        
    def apply_calibration(self, logits):
        """Applies temperature scaling and thresholding"""
        calibrated_probs = torch.sigmoid(logits / self.temperature)
        predictions = (calibrated_probs > self.optimal_threshold).float()
        return calibrated_probs, predictions

class EnhancedMetrics:
    """Comprehensive metrics for fall detection evaluation"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.probabilities = []
        self.labels = []
        
    def update(self, probs, preds, labels):
        self.probabilities.extend(probs.cpu().numpy())
        self.predictions.extend(preds.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        
    def compute_all(self):
        """Computes comprehensive metrics"""
        probs = np.array(self.probabilities)
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Basic metrics
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Advanced metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        
        # Fall detection specific metrics
        fall_detection_rate = recall  # Same as sensitivity
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'npv': npv,
            'fall_detection_rate': fall_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'miss_rate': miss_rate,
            'num_false_positives': int(fp),
            'num_false_negatives': int(fn)
        }
