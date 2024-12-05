import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionFeatureBlock(nn.Module):
    """
    Analyzes motion characteristics to differentiate between sudden falls and controlled movements.
    Uses multi-scale convolutions to capture both rapid and gradual changes in acceleration.
    """
    def __init__(self, in_channels, out_channels):
        super(MotionFeatureBlock, self).__init__()
        
        # Fast motion pathway for detecting sudden movements (falls)
        self.fast_pathway = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels//2),
            nn.GELU(),
            nn.Conv1d(out_channels//2, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels//2),
            nn.GELU()
        )
        
        # Slow motion pathway for analyzing controlled movements (sitting)
        self.slow_pathway = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//2, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels//2),
            nn.GELU(),
            nn.Conv1d(out_channels//2, out_channels//2, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels//2),
            nn.GELU()
        )
        
        # Motion smoothness analyzer
        self.smoothness_conv = nn.Conv1d(out_channels, 1, kernel_size=5, padding=2)

    def forward(self, x):
        # Analyze fast and slow motion components
        fast_features = self.fast_pathway(x)
        slow_features = self.slow_pathway(x)
        
        # Combine features
        combined = torch.cat([fast_features, slow_features], dim=1)
        
        # Calculate motion smoothness
        smoothness = torch.sigmoid(self.smoothness_conv(combined))
        
        return combined, smoothness

class TemporalBlock(nn.Module):
    """
    Enhanced temporal block that maintains temporal relationships while being
    sensitive to both sudden and gradual movements.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                     padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                     padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class AttentionModule(nn.Module):
    """
    Self-attention module to focus on important motion patterns while
    maintaining temporal context.
    """
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        
        self.scale = channels ** -0.5
        
        self.feature_mixing = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention = torch.bmm(q.transpose(1, 2), k) * self.scale
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(attention, v.transpose(1, 2)).transpose(1, 2)
        out = self.feature_mixing(out)
        return out + x

class EnhancedFallDetectionModel(nn.Module):
    """
    Complete fall detection model that effectively distinguishes between
    falls and normal activities by analyzing motion patterns, temporal
    relationships, and movement characteristics.
    """
    def __init__(self, seq_length=128, num_channels=3, num_filters=64):
        super(EnhancedFallDetectionModel, self).__init__()
        
        # Motion analysis for watch and phone
        self.watch_motion = MotionFeatureBlock(num_channels, num_filters)
        self.phone_motion = MotionFeatureBlock(num_channels, num_filters)
        
        # Temporal feature extraction
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(num_filters, num_filters, kernel_size=3, dilation=d)
            for d in [1, 2, 4, 8]
        ])
        
        # Attention modules for focusing on relevant patterns
        self.watch_attention = AttentionModule(num_filters)
        self.phone_attention = AttentionModule(num_filters)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2 + 2, num_filters * 2),  # +2 for smoothness values
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(num_filters * 2, num_filters),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(num_filters, 1)
        )

    def forward(self, x):
        # Process watch data
        watch = x['accelerometer_watch'].permute(0, 2, 1)
        watch_features, watch_smoothness = self.watch_motion(watch)
        
        # Process phone data
        phone = x['accelerometer_phone'].permute(0, 2, 1)
        phone_features, phone_smoothness = self.phone_motion(phone)
        
        # Apply temporal blocks
        for block in self.temporal_blocks:
            watch_features = block(watch_features)
            phone_features = block(phone_features)
        
        # Apply attention
        watch_features = self.watch_attention(watch_features)
        phone_features = self.phone_attention(phone_features)
        
        # Global average pooling
        watch_features = torch.mean(watch_features, dim=2)
        phone_features = torch.mean(phone_features, dim=2)
        
        # Combine features with smoothness measures
        combined_features = torch.cat([
            watch_features,
            phone_features,
            watch_smoothness.squeeze(1).mean(dim=1, keepdim=True),
            phone_smoothness.squeeze(1).mean(dim=1, keepdim=True)
        ], dim=1)
        
        # Final classification
        logits = self.classifier(combined_features)
        return logits