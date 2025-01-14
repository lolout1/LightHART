import torch
import torch.nn as nn
import torch.nn.functional as F

class FallDetectionBlock(nn.Module):
    """Simplified and regularized detection block"""
    def __init__(self, channels, kernel_size, drop_rate=0.3):
        super().__init__()
        
        # Depthwise separable convolutions for efficiency and regularization
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        
        # Simple residual connection
        self.shortcut = nn.Identity()
        
        # Learnable scale for residual
        self.scale = nn.Parameter(torch.ones(1) * 0.1)  # Initialize small

    def forward(self, x, training=True):
        identity = self.shortcut(x)
        out = self.conv1(x)
        
        # Scaled residual connection
        return identity + out * self.scale

class StudentModel(nn.Module):
    """Regularized student model focused on accelerometer patterns"""
    def __init__(self,
                 input_channels=4,  # x, y, z, magnitude
                 hidden_dim=64,     # Reduced from original
                 num_blocks=3,      # Reduced from 4
                 dropout_rate=0.3): # Increased dropout
        super().__init__()
        
        # Input projection with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Temporal processing blocks
        self.temporal_blocks = nn.ModuleList([
            FallDetectionBlock(
                hidden_dim, 
                kernel_size=2*i + 3,  # Different kernel sizes for multi-scale
                drop_rate=dropout_rate
            ) for i in range(num_blocks)
        ])
        
        # Progressive feature reduction
        self.feature_reducer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )
        
        # Fall detection head with confidence modeling
        self.fall_detector = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(dropout_rate * 1.2),  # Extra dropout in final layer
            nn.Linear(hidden_dim // 8, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights carefully
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def compute_magnitude(self, acc_data):
        """Enhanced SMV computation with time-domain features"""
        # Basic magnitude
        magnitude = torch.sqrt(torch.sum(acc_data ** 2, dim=-1, keepdim=True))
        
        # Add rate of change (first derivative)
        acc_diff = torch.diff(acc_data, dim=1, prepend=acc_data[:, :1, :])
        jerk = torch.sqrt(torch.sum(acc_diff ** 2, dim=-1, keepdim=True))
        
        # Combine features
        return torch.cat([magnitude, jerk], dim=-1)

    def forward(self, x, training=True):
        """Forward pass with regularization during training"""
        # Compute enhanced SMV features
        magnitude_features = self.compute_magnitude(x)
        x = torch.cat([x, magnitude_features], dim=-1)
        
        # Project inputs
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # [B, C, T]
        
        # Apply temporal blocks with residual connections
        features = []
        for block in self.temporal_blocks:
            x = block(x, training)
            features.append(x)
        
        # Multi-scale feature fusion
        multi_scale = torch.stack([
            F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
            for feat in features
        ]).mean(0)
        
        # Feature reduction with dropout
        fall_feat = self.feature_reducer(multi_scale)
        if training:
            fall_feat = F.dropout(fall_feat, p=0.2, training=True)
        
        # Final prediction with confidence
        logit = self.fall_detector(fall_feat).squeeze(-1)
        fall_prob = self.sigmoid(logit)
        
        # Add prediction noise during training to prevent overconfidence
        if training:
            fall_prob = fall_prob * 0.9 + torch.rand_like(fall_prob) * 0.1
        
        return fall_prob, fall_feat