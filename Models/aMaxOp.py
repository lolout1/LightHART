import torch
import torch.nn as nn
import torch.nn.functional as F

class ResTCNBlock(nn.Module):
    """
    Enhanced Residual Temporal Convolutional Network Block optimized for fall detection.
    Uses deeper feature extraction and skip connections for better temporal pattern recognition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResTCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        # Increased intermediate channels for richer feature extraction
        intermediate_channels = out_channels * 2
        
        # Two-stage convolution for deeper feature extraction
        self.conv1 = nn.Conv1d(in_channels, intermediate_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        
        # Added dropout for better generalization
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.conv2 = nn.Conv1d(intermediate_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Added second dropout layer
        self.dropout2 = nn.Dropout(p=0.2)
        
        # Using GELU activation for better gradient flow
        self.activation = nn.GELU()
        
        # Enhanced residual connection with 1x1 convolution
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        # Enhanced forward path with dropout and GELU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        # Residual connection with scaled addition
        out = out * 0.5 + residual
        out = self.activation(out)
        return out

class AttentionModule(nn.Module):
    """
    Enhanced Multi-Head Self-Attention Module with relative positional encoding
    for better temporal relationship modeling.
    """
    def __init__(self, embed_dim, num_heads):
        super(AttentionModule, self).__init__()
        
        # Multi-head attention with increased heads for better feature capture
        self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                             batch_first=True, dropout=0.1)
        
        # Added feed-forward network for enhanced feature transformation
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Dual layer normalization for stable training
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # First attention block with residual
        residual = x
        x = self.layer_norm1(x)
        attn_output, _ = self.attention(x, x, x)
        x = residual + self.dropout(attn_output)
        
        # Feed-forward block with residual
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x

class FallDetectionModel(nn.Module):
    def __init__(self, seq_len=128, num_channels=3, num_filters=128, num_classes=1):
        super(FallDetectionModel, self).__init__()
        
        # Add frequency-aware convolutions for impact detection
        self.frequency_conv = nn.Sequential(
            nn.Conv1d(num_channels, num_filters//2, kernel_size=15, padding=7),
            nn.BatchNorm1d(num_filters//2),
            nn.GELU(),
            nn.Conv1d(num_filters//2, num_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.GELU()
        )
        
        # Modify TCN blocks for multi-scale feature extraction
        self.watch_tcn = nn.ModuleList([
            ResTCNBlock(num_filters, num_filters, kernel_size=k, dilation=d)
            for k, d in [(3,1), (5,2), (7,4), (9,8)]  # Multiple kernel sizes
        ])
        
        # Add explicit fall pattern detection module
        self.fall_pattern_detector = nn.Sequential(
            nn.Linear(num_filters * 2, num_filters),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(num_filters, 3),  # Detect pre-fall, impact, and post-fall
            nn.Softmax(dim=-1)
        )
        
        # Modified classifier with fall pattern awareness
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2 + 3, num_filters * 2),  # Added fall pattern features
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(num_filters * 2, num_filters),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(num_filters, 1)
        )

    def forward(self, x):
        # Process watch data with enhanced feature extraction
        watch = x['accelerometer_watch']
        watch = watch.permute(0, 2, 1)
        watch_feat = self.watch_tcn(watch)
        watch_feat = watch_feat.permute(0, 2, 1)
        watch_feat = self.watch_attention(watch_feat)
        
        # Weighted temporal pooling instead of simple mean
        watch_weights = F.softmax(torch.sum(watch_feat, dim=-1), dim=-1)
        watch_feat = torch.sum(watch_feat * watch_weights.unsqueeze(-1), dim=1)
        
        # Process phone data with similar enhancements
        phone = x['accelerometer_phone']
        phone = phone.permute(0, 2, 1)
        phone_feat = self.phone_tcn(phone)
        phone_feat = phone_feat.permute(0, 2, 1)
        phone_feat = self.phone_attention(phone_feat)
        
        phone_weights = F.softmax(torch.sum(phone_feat, dim=-1), dim=-1)
        phone_feat = torch.sum(phone_feat * phone_weights.unsqueeze(-1), dim=1)
        
        # Enhanced feature fusion
        combined_feat = torch.cat([watch_feat, phone_feat], dim=1)
        combined_feat = combined_feat.unsqueeze(1)
        fused_feat = self.fusion_attention(combined_feat)
        fused_feat = fused_feat.squeeze(1)
        
        # Classification with dropout for robustness
        logits = self.classifier(fused_feat)
        return logits