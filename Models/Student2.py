import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSeparableBlock(nn.Module):
    """
    A residual block with depthwise separable convolutions and
    learnable gating, focused on both accuracy and regularization.
    """
    def __init__(self, channels, kernel_size, dropout_rate=0.3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels  # Depthwise
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # Residual gating for controlled addition
        self.scale = nn.Parameter(torch.ones(1) * 0.1)  # small initial scale

    def forward(self, x):
        identity = x
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.gelu(out)
        out = self.dropout(out)

        # Weighted residual connection
        return identity + self.scale * out


class StudentModel(nn.Module):
    """
    Optimized Student Model for watch accelerometer data only (x, y, z),
    plus derived magnitude and jerk features, aimed at high accuracy & recall.
    """
    def __init__(
        self,
        input_channels=3,  # watch accelerometer input: x, y, z
        hidden_dim=64,
        num_blocks=3,      # fewer blocks for capacity control
        dropout_rate=0.3   # higher dropout for strong regularization
    ):
        super().__init__()

        # The student will compute magnitude and jerk internally => total input = 3 + 2
        self.proj_in = nn.Linear(input_channels + 2, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)
        self.proj_act = nn.GELU()
        self.proj_drop = nn.Dropout(dropout_rate * 0.5)  # mild dropout

        # Stacked temporal blocks (depthwise separable + gating)
        self.blocks = nn.ModuleList([
            ResidualSeparableBlock(channels=hidden_dim,
                                   kernel_size=(2 * i + 3),
                                   dropout_rate=dropout_rate)
            for i in range(num_blocks)
        ])

        # Feature aggregator
        # (optionally, use a multi-scale aggregator if you store intermediate features)
        self.aggregate_norm = nn.LayerNorm(hidden_dim)

        # Classifier head => single sigmoid output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def compute_features(self, acc_data):
        """
        Compute magnitude and jerk (rate of change) to match teacher's approach.
        acc_data shape: [B, T, 3]
        Returns: [B, T, 2] => [magnitude, jerk_magnitude]
        """
        # Basic magnitude
        magnitude = torch.sqrt(torch.sum(acc_data ** 2, dim=-1, keepdim=True))

        # Jerk (first derivative of accelerometer data)
        # Prepend the first frame to keep sequence length
        acc_diff = torch.diff(acc_data, dim=1, prepend=acc_data[:, :1, :])
        jerk_mag = torch.sqrt(torch.sum(acc_diff ** 2, dim=-1, keepdim=True))

        return torch.cat([magnitude, jerk_mag], dim=-1)

    def forward(self, x):
        """
        x: [B, T, 3] => raw watch accelerometer data
        Output: single sigmoid probability for fall detection
        """
        # Compute teacher-like features => [B, T, 2]
        feat = self.compute_features(x)
        # Concatenate => total input = [B, T, 5]
        x = torch.cat([x, feat], dim=-1)

        # Project => [B, T, hidden_dim]
        x = self.proj_in(x)
        x = self.proj_norm(x)
        x = self.proj_act(x)
        x = self.proj_drop(x)

        # Switch to [B, hidden_dim, T] for conv blocks
        x = x.transpose(1, 2)

        # Pass through residual depthwise separable blocks
        for block in self.blocks:
            x = block(x)

        # Normalize final feature map
        x = x.transpose(1, 2)   # back to [B, T, hidden_dim]
        x = self.aggregate_norm(x)

        # Global average pooling => [B, hidden_dim]
        features = torch.mean(x, dim=1)

        # Classification => single logit => final probability
        logit = self.classifier(features)  # [B, 1]
        prob = self.sigmoid(logit)  # [B, 1]
        return prob.squeeze(-1), features  # return probability [B] and features [B, hidden_dim]
