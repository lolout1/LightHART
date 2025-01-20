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

        self.res_weight = nn.Parameter(torch.ones(1))
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        """
        x shape: [batch_size, in_channels, length]
        """
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
        """
        x shape: [batch_size, channels, length]
        """
        b, c, t = x.size()
        avg_pooled = self.avg_pool(x).view(b, c)
        max_pooled = self.max_pool(x).view(b, c)
        combined = torch.cat([avg_pooled, max_pooled], dim=1)
        scale = self.fc(combined).view(b, c, 1)
        return x * scale.expand_as(x)

class StudentModel(nn.Module):
    """
    Student model for watch accelerometer data
    expecting input of shape [batch_size, window_size, channels].
    """
    def __init__(self,
                 input_channels=3,
                 hidden_dim=48,
                 num_blocks=4,
                 dropout_rate=0.2):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        self.temporal_blocks = nn.ModuleList([
            PrecisionBlock(hidden_dim, hidden_dim, kernel_size=(2 * i + 3))
            for i in range(num_blocks)
        ])
        self.attention = TemporalAttention(channels=hidden_dim)
        self.fall_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Expects x with shape: [batch_size, window_size, channels]
        """
        # Transpose to [batch_size, channels, window_size] for Conv1d
        x = x.permute(0, 2, 1)  # => [B, channels, length]

        # (1) Project each time step from input_channels -> hidden_dim
        # We flatten [B, channels, length] => reshape for linear => revert
        b, c, length = x.shape
        x = x.reshape(b * length, c)  # => [B*length, channels]
        x = self.input_proj(x)        # => [B*length, hidden_dim]
        x = x.reshape(b, length, self.hidden_dim)
        # Transpose to [B, hidden_dim, length] for conv
        x = x.permute(0, 2, 1)  # => [B, hidden_dim, length]

        # (2) Pass through multiple temporal blocks + attention
        for block in self.temporal_blocks:
            x = block(x)
            x = self.attention(x)

        # (3) Global average pool => [B, hidden_dim]
        # shape => [B, hidden_dim, length]
        student_feat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        # (4) Classification
        logits = self.fall_confidence(student_feat).squeeze(-1)
        student_prob = torch.sigmoid(logits)  # range [0,1]
        return student_prob, student_feat
