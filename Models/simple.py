import torch
from torch import nn
import torch.nn.functional as F
import math

class FallDetectionTransformer(nn.Module):
    def __init__(self, acc_frames=64, num_classes=2, num_heads=4, acc_coords=3, 
                 num_layers=2, embed_dim=32, dropout=0.2, use_batch_norm=True, **kwargs):
        super().__init__()
        self.acc_frames = acc_frames
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        self.num_classes = num_classes
        
        # Encoder for accelerometer data
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(), nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )
        
        # Encoder for gyroscope data
        self.gyro_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(), nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )
        
        # Combined feature dimension
        self.feature_dim = embed_dim * 2
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=num_heads, dim_feedforward=self.feature_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=num_layers, 
            norm=nn.LayerNorm(self.feature_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.classifier[-1].bias)
        fan_in = self.classifier[-1].weight.size(1)
        nn.init.normal_(self.classifier[-1].weight, 0, 1/math.sqrt(fan_in))
    
    def forward(self, acc_data, gyro_data):
        """
        Forward pass with explicit tensor arguments for AI Edge Torch compatibility.
        
        Args:
            acc_data: Accelerometer data [batch, seq_len, channels] or [batch, channels, seq_len]
            gyro_data: Gyroscope data [batch, seq_len, channels] or [batch, channels, seq_len]
            
        Returns:
            Classification logits [batch, num_classes]
        """
        # Handle different input formats
        if acc_data.shape[1] == self.acc_coords and len(acc_data.shape) == 3:
            # Already in [batch, channels, seq_len] format
            acc_data_conv = acc_data
        else:
            # Convert to [batch, channels, seq_len] format
            acc_data_conv = acc_data.transpose(1, 2)
            
        if gyro_data.shape[1] == self.acc_coords and len(gyro_data.shape) == 3:
            # Already in [batch, channels, seq_len] format
            gyro_data_conv = gyro_data
        else:
            # Convert to [batch, channels, seq_len] format
            gyro_data_conv = gyro_data.transpose(1, 2)
        
        # Process accelerometer data
        acc_features = self.acc_encoder(acc_data_conv)
        # Process gyroscope data
        gyro_features = self.gyro_encoder(gyro_data_conv)
        
        # Convert to sequence-first format for transformer
        acc_features = acc_features.transpose(1, 2)
        gyro_features = gyro_features.transpose(1, 2)
        
        # Concatenate features
        fused_features = torch.cat([acc_features, gyro_features], dim=2)
        
        # Process through transformer
        transformer_output = self.transformer(fused_features)
        
        # Global average pooling
        pooled = torch.mean(transformer_output, dim=1)
        
        # Classification
        return self.classifier(pooled)
