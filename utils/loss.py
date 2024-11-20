import torch
import torch.nn.functional as F
import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self, T=2.0, alpha=0.7, feature_weight=0.1):
        """
        Args:
            T: Temperature for softmax scaling (default: 2.0)
            alpha: Weight for soft label loss (default: 0.7)
            feature_weight: Weight for feature distillation loss (default: 0.1)
        """
        super(DistillationLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_output, teacher_output, labels):
        """
        Compute the distillation loss combining soft targets, hard targets, and feature alignment.
        
        Args:
            student_output: Tuple of (logits, features) from student model
            teacher_output: Tuple of (logits, features) from teacher model
            labels: Ground truth labels
        """
        # Unpack outputs
        if isinstance(student_output, tuple):
            student_logits, student_features = student_output
        else:
            student_logits = student_output
            student_features = None

        if isinstance(teacher_output, tuple):
            teacher_logits, teacher_features = teacher_output
        else:
            teacher_logits = teacher_output
            teacher_features = None

        # Knowledge Distillation Loss (soft targets)
        soft_targets = F.softmax(teacher_logits / self.T, dim=1)
        soft_prob = F.log_softmax(student_logits / self.T, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.T ** 2)

        # Standard Cross-Entropy Loss (hard targets)
        hard_loss = self.criterion(student_logits, labels)

        # Feature Distillation Loss
        feature_loss = 0.0
        if student_features is not None and teacher_features is not None:
            if 'transformer_output' in teacher_features and 'transformer_output' in student_features:
                feature_loss = self.mse_loss(
                    student_features['transformer_output'],
                    teacher_features['transformer_output']
                )

        # Combine all losses
        total_loss = (self.alpha * soft_loss) + \
                    ((1 - self.alpha) * hard_loss) + \
                    (self.feature_weight * feature_loss)
        
        return total_loss
