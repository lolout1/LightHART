import torch 
import torch.nn.functional as F
import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.3):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)

        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.temperature**2)

        label_loss = self.criterion(student_logits, labels)

        loss = self.alpha * soft_targets_loss + (1-self.alpha) * label_loss
        return loss

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        if self.class_weights is None:
            device = logits.device
            class_counts = torch.bincount(targets)
            total = class_counts.sum()
            weights = total / (class_counts * len(class_counts))
            weights = weights.to(device)
        else:
            weights = self.class_weights
            
        return F.cross_entropy(logits, targets, weight=weights)

class EnhancedDistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5, class_weights=None):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = self.ce_loss(student_logits, labels)
        
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.temperature**2)
        
        loss = self.alpha * soft_loss + (1-self.alpha) * hard_loss
        return loss
