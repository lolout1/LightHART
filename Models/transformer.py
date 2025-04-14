import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np

class SimpleFFN(nn.Module):
    """Simple but effective feed-forward network with strong regularization"""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TemporalCNN(nn.Module):
    """Efficient temporal feature extractor with multi-scale receptive fields"""
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.conv_small = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//3, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm1d(out_channels//3),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.conv_medium = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//3, kernel_size=7, padding=3, groups=1),
            nn.BatchNorm1d(out_channels//3),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.conv_large = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//3, kernel_size=15, padding=7, groups=1),
            nn.BatchNorm1d(out_channels//3),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.combine = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        x_small = self.conv_small(x)
        x_medium = self.conv_medium(x)
        x_large = self.conv_large(x)
        
        # Concatenate features from different scales
        x_cat = torch.cat([x_small, x_medium, x_large], dim=1)
        
        # Mix features
        x = self.combine(x_cat)
        return x

class DomainInvariantModel(nn.Module):
    """A model designed for robust cross-subject generalization"""
    def __init__(self, mocap_frames=128, num_joints=32, acc_frames=128, num_classes=1, 
                 acc_coords=4, embed_dim=32, dropout=0.5, **kwargs):
        super().__init__()
        
        # Feature extraction with reduced parameters
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Multi-scale temporal features
        self.temporal_conv = TemporalCNN(embed_dim, embed_dim)
        
        # Global statistics pooling (captures more than just mean)
        self.pool = StatisticsPooling(embed_dim)
        
        # Gradual dimensionality reduction with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Special augmentation for domain generalization
        self.spec_aug = SpecAugment(time_drop_width=8, time_stripes_num=2)
        
        # Data normalization layer
        self.instance_norm = nn.InstanceNorm1d(acc_coords)
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, acc_data, skl_data=None, training=True, **kwargs):
        # Process accelerometer data [batch, time, channels]
        x = rearrange(acc_data, 'b t c -> b c t')
        
        # Normalize input features to make them more subject-invariant
        x = self.instance_norm(x)
        
        # Data augmentation during training for better generalization
        if self.training and training:
            x = self.spec_aug(x)
        
        # Extract features
        x = self.feature_extractor(x)
        x = self.temporal_conv(x)
        
        # Apply statistics pooling (mean + std)
        x = self.pool(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits, x

class StatisticsPooling(nn.Module):
    """Pooling that captures both mean and standard deviation"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 2
        
    def forward(self, x):
        # x shape: [batch, channels, time]
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat([mean, std], dim=1)

class SpecAugment(nn.Module):
    """Time-domain augmentation for improved generalization"""
    def __init__(self, time_drop_width=8, time_stripes_num=2):
        super().__init__()
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
    
    def forward(self, x):
        if not self.training or self.time_stripes_num <= 0:
            return x
        
        # Create a copy to modify
        x_aug = x.clone()
        batch_size, channels, time_len = x.shape
        
        # Apply time masking
        for i in range(self.time_stripes_num):
            start = torch.randint(0, max(1, time_len - self.time_drop_width), (1,))[0]
            end = min(time_len, start + self.time_drop_width)
            if (end - start) > 1:  # Ensure we're masking something
                x_aug[:, :, start:end] = 0.0
                
        return x_aug

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Aliases for backward compatibility
class OptimizedFallDetector(DomainInvariantModel):
    pass

class TransModel(DomainInvariantModel):
    pass
