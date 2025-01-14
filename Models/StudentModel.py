import torch
import torch.nn as nn
import torch.nn.functional as F

class PrecisionBlock(nn.Module):
    """
    Convolution-based residual block for capturing detailed temporal dynamics.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Weighted residual connection
        self.res_weight = nn.Parameter(torch.ones(1))
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + self.res_weight * identity
        return F.relu(out)

class TemporalAttention(nn.Module):
    """
    Multi-scale temporal attention for highlighting critical fall segments.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        avg_pooled = self.avg_pool(x).view(b, c)
        max_pooled = self.max_pool(x).view(b, c)

        # Combine avg + max
        combined = torch.cat([avg_pooled, max_pooled], dim=1)
        scale = self.fc(combined).view(b, c, 1)
        return x * scale.expand_as(x)

class StudentModel(nn.Module):
    """
    Student model: Single-modality (watch accelerometer).
    Incorporates magnitude computation similar to TeacherModel.
    Uses a stack of PrecisionBlocks + TemporalAttention to capture fall patterns.
    """
    def __init__(self,
                 input_channels=4,  # x, y, z, magnitude
                 hidden_dim=128,
                 num_blocks=4,
                 dropout_rate=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(    
            nn.Linear(input_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)  # mild dropout in projection
        )

        # Stacked temporal blocks
        self.temporal_blocks = nn.ModuleList([
            PrecisionBlock(hidden_dim, hidden_dim, kernel_size=(2*i + 3))
            for i in range(num_blocks)
        ])

        self.attention = TemporalAttention(channels=hidden_dim)

        # Classification head
        self.fall_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, T, 3] => raw watch accelerometer data (x, y, z)
        Returns:
            student_prob: [B], predicted fall probability
            student_feat: [B, hidden_dim], final feature
        """
        # Compute magnitude and append to input channels
        magnitude = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))  # [B, T, 1]
        x = torch.cat([x, magnitude], dim=-1)  # Now [B, T, 4]

        # Project inputs => [B, T, hidden_dim]
        x = self.input_proj(x)
        # => [B, hidden_dim, T]
        x = x.transpose(1, 2)

        # Pass through multiple blocks + attention
        for block in self.temporal_blocks:
            x = block(x)
            x = self.attention(x)

        # Global average pool => [B, hidden_dim]
        student_feat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        student_prob = self.fall_confidence(student_feat).squeeze(-1)
        # = self.sigmoid(logit)
        return student_prob, student_feat