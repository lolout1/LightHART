import torch
import torch.nn as nn
from einops import rearrange


class ResidualBlock1D(nn.Module):
    """A residual block for 1D convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(self.gelu(self.bn2(self.conv2(x))))
        return x + residual


class IMUProcessor(nn.Module):
    """Processes accelerometer data with residual convolutional blocks."""
    def __init__(self, acc_coords, embed_dim, num_blocks=3, dropout=0.2):
        super().__init__()
        self.xyz_processor = nn.Sequential(
            nn.Conv1d(acc_coords - 1, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            *[ResidualBlock1D(embed_dim // 2, embed_dim // 2, dropout=dropout) for _ in range(num_blocks)],
        )
        self.smv_processor = nn.Sequential(
            nn.Conv1d(1, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            *[ResidualBlock1D(embed_dim // 2, embed_dim // 2, dropout=dropout) for _ in range(num_blocks)],
        )

    def forward(self, x):
        xyz = x[:, :-1]  # Extract XYZ coordinates
        smv = x[:, -1:]  # Extract SMV
        xyz_features = self.xyz_processor(rearrange(xyz, 'b t c -> b c t'))
        smv_features = self.smv_processor(rearrange(smv, 'b t c -> b c t'))
        return torch.cat([xyz_features, smv_features], dim=-1)


class SkeletonProcessor(nn.Module):
    """Processes skeleton data with 2D convolutions and residual blocks."""
    def __init__(self, num_joints, in_chans, embed_dim, num_blocks=3, dropout=0.2):
        super().__init__()
        self.num_joints = num_joints  # Store num_joints as an instance variable
        self.embed = nn.Conv2d(in_chans, embed_dim, kernel_size=1)
        layers = [ResidualBlock1D(embed_dim, embed_dim, dropout=dropout)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock1D(embed_dim, embed_dim, dropout=dropout))
        self.conv_blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        # Use self.num_joints instead of num_joints
        x = rearrange(x, 'b t (j c) -> b c t j', j=self.num_joints)  # [batch, channels, seq_len, num_joints]
        x = self.embed(x)  # [batch, embed_dim, seq_len, num_joints]
        x = torch.mean(x, dim=-1)  # Average over joints [batch, embed_dim, seq_len]
        x = self.conv_blocks(x)  # Apply 1D convolutions
        x = rearrange(x, 'b c t -> b t c')  # Back to [batch, seq_len, embed_dim]
        return x

class MultiModalFusion(nn.Module):
    """Fuses data from multiple modalities with attention and residual blocks."""
    def __init__(self, embed_dim, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=dropout)
        self.fusion_with_skeleton = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fusion_without_skeleton = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, features):
        phone, _ = self.attention(features['phone'], features['phone'], features['phone'])
        watch, _ = self.attention(features['watch'], features['watch'], features['watch'])
        
        if 'skeleton' in features:
            skeleton, _ = self.attention(features['skeleton'], features['skeleton'], features['skeleton'])
            combined = torch.cat([phone.mean(1), watch.mean(1), skeleton.mean(1)], dim=-1)
            fused = self.fusion_with_skeleton(combined)
        else:
            combined = torch.cat([phone.mean(1), watch.mean(1)], dim=-1)
            fused = self.fusion_without_skeleton(combined)
        
        return fused


class EnhancedFallDetectionModel(nn.Module):
    """Final model combining IMU and skeleton processing for fall detection."""
    def __init__(self, num_joints, in_chans, acc_coords, embed_dim, num_classes, dropout=0.2):
        super().__init__()
        self.phone_processor = IMUProcessor(acc_coords, embed_dim, dropout=dropout)
        self.watch_processor = IMUProcessor(acc_coords, embed_dim, dropout=dropout)
        self.skeleton_processor = SkeletonProcessor(num_joints, in_chans, embed_dim, dropout=dropout)
        self.fusion = MultiModalFusion(embed_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, data):
        features = {
            'phone': self.phone_processor(data['accelerometer_phone']),
            'watch': self.watch_processor(data['accelerometer_watch']),
        }
        if 'skeleton' in data:
            features['skeleton'] = self.skeleton_processor(data['skeleton'])
        
        fused = self.fusion(features)
        logits = self.classifier(fused)
        return logits
