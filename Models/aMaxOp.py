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
    """
    Enhanced Fall Detection Model with optimized architecture for 32ms sampling rate
    and improved feature fusion for better accuracy.
    """
    def __init__(self, seq_len=128, num_channels=3, num_filters=128, num_classes=1):
        super(FallDetectionModel, self).__init__()
        
        # Increased number of filters for richer feature extraction
        self.seq_len = seq_len
        
        # Enhanced TCN for watch data with increased depth
        self.watch_tcn = nn.Sequential(
            ResTCNBlock(num_channels, num_filters, kernel_size=7, dilation=1),
            ResTCNBlock(num_filters, num_filters, kernel_size=7, dilation=2),
            ResTCNBlock(num_filters, num_filters, kernel_size=7, dilation=4),
            ResTCNBlock(num_filters, num_filters, kernel_size=7, dilation=8)
        )
        
        # Enhanced TCN for phone data
        self.phone_tcn = nn.Sequential(
            ResTCNBlock(num_channels, num_filters, kernel_size=7, dilation=1),
            ResTCNBlock(num_filters, num_filters, kernel_size=7, dilation=2),
            ResTCNBlock(num_filters, num_filters, kernel_size=7, dilation=4),
            ResTCNBlock(num_filters, num_filters, kernel_size=7, dilation=8)
        )
        
        # Enhanced attention modules with more heads
        self.watch_attention = AttentionModule(embed_dim=num_filters, num_heads=8)
        self.phone_attention = AttentionModule(embed_dim=num_filters, num_heads=8)
        
        # Advanced feature fusion with cross-attention
        self.fusion_attention = AttentionModule(embed_dim=num_filters * 2, num_heads=8)
        
        # Enhanced classifier with deeper architecture
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2, num_filters * 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(num_filters * 4, num_filters * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(num_filters * 2, 1)
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