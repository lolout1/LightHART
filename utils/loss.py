import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with imbalanced data
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the positive class
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
        
    def forward(self, inputs, targets):
        # Get probabilities
        inputs = torch.clamp(torch.sigmoid(inputs), min=self.eps, max=1-self.eps)
        
        # For binary classification
        targets = targets.float()
        
        # Calculate focal weight
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Apply focal weight
        loss = -alpha * (1 - pt) ** self.gamma * torch.log(pt)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FallDetectionLoss(nn.Module):
    """
    Combined loss function for fall detection
    
    Combines focal loss with class-balanced BCE loss
    """
    def __init__(self, focal_weight=0.5, pos_weight=2.0, gamma=2.0):
        super(FallDetectionLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=gamma)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.focal_weight = focal_weight
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * bce
