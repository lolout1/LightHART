

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResTCNBlock(nn.Module):
    """
    Residual Temporal Convolutional Network Block.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResTCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class AttentionModule(nn.Module):
    """
    Multi-Head Self-Attention Module.
    """
    def __init__(self, embed_dim, num_heads):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                               batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        out = self.layer_norm(x + attn_output)  # Residual connection
        return out

class FallDetectionModel(nn.Module):
    """
    Optimized Fall Detection Model using ResTCN and Attention Mechanisms.
    """
    def __init__(self, seq_len=128, num_channels=3, num_filters=64,
                 num_classes=1):
        super(FallDetectionModel, self).__init__()
        self.seq_len = seq_len

        # Process accelerometer_watch data
        self.watch_tcn = nn.Sequential(
            ResTCNBlock(num_channels, num_filters, kernel_size=5, dilation=1),
            ResTCNBlock(num_filters, num_filters, kernel_size=5, dilation=2),
            ResTCNBlock(num_filters, num_filters, kernel_size=5, dilation=4),
        )

        # Process accelerometer_phone data
        self.phone_tcn = nn.Sequential(
            ResTCNBlock(num_channels, num_filters, kernel_size=5, dilation=1),
            ResTCNBlock(num_filters, num_filters, kernel_size=5, dilation=2),
            ResTCNBlock(num_filters, num_filters, kernel_size=5, dilation=4),
        )

        # Attention Modules
        self.watch_attention = AttentionModule(embed_dim=num_filters,
                                               num_heads=4)
        self.phone_attention = AttentionModule(embed_dim=num_filters,
                                               num_heads=4)

        # Fusion Layer
        self.fusion_attention = AttentionModule(embed_dim=num_filters * 2,
                                                num_heads=4)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
            # No sigmoid activation; use BCEWithLogitsLoss
        )

    def forward(self, x):
        """
        Args:
            x (dict): Contains 'accelerometer_watch' and 'accelerometer_phone' tensors.
                      Each tensor has shape (batch_size, seq_len, num_channels).
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Process watch data
        watch = x['accelerometer_watch']  # (batch_size, seq_len, 3)
        watch = watch.permute(0, 2, 1)    # (batch_size, 3, seq_len)
        watch_feat = self.watch_tcn(watch)  # (batch_size, num_filters, seq_len)
        watch_feat = watch_feat.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        watch_feat = self.watch_attention(watch_feat)
        watch_feat = torch.mean(watch_feat, dim=1)  # (batch_size, num_filters)

        # Process phone data
        phone = x['accelerometer_phone']  # (batch_size, seq_len, 3)
        phone = phone.permute(0, 2, 1)    # (batch_size, 3, seq_len)
        phone_feat = self.phone_tcn(phone)  # (batch_size, num_filters, seq_len)
        phone_feat = phone_feat.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        phone_feat = self.phone_attention(phone_feat)
        phone_feat = torch.mean(phone_feat, dim=1)  # (batch_size, num_filters)

        # Fuse features
        combined_feat = torch.cat([watch_feat, phone_feat], dim=1)  # (batch_size, num_filters * 2)
        combined_feat = combined_feat.unsqueeze(1)  # (batch_size, 1, num_filters * 2)
        combined_feat = self.fusion_attention(combined_feat)
        fused_feat = combined_feat.squeeze(1)  # (batch_size, num_filters * 2)

        # Classification
        logits = self.classifier(fused_feat)
        return logits
