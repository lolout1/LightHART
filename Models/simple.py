import torch
from torch import nn
import torch.nn.functional as F
import math

class FallDetectionTransformer(nn.Module):
    def __init__(self, seq_length=64, num_classes=2, num_heads=2, input_channels=3, num_layers=2, embed_dim=32, dropout=0.2, use_batch_norm=True, **kwargs):
        super().__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Accelerometer encoder - uses fixed-size operations for TFLite
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        # Gyroscope encoder - uses fixed-size operations for TFLite
        self.gyro_encoder = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        # Fusion dimension
        self.fusion_dim = embed_dim * 2
        
        # Pre-computed positional embeddings (no Embedding layer for TFLite)
        pos_enc = torch.zeros(1, seq_length, self.fusion_dim)
        position = torch.arange(seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.fusion_dim, 2).float() * (-math.log(10000.0) / self.fusion_dim))
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("position_embeddings", pos_enc)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=num_heads,
            dim_feedforward=self.fusion_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.fusion_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Weight initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, acc_data, gyro_data):
        """
        Forward pass with explicit tensor inputs for TFLite compatibility
        
        Args:
            acc_data: Tensor of shape [batch_size, seq_length, channels] (NHWC format)
            gyro_data: Tensor of shape [batch_size, seq_length, channels] (NHWC format)
            
        Returns:
            Tensor of shape [batch_size, num_classes] with class logits
        """
        # Get batch_size for reshaping operations
        batch_size = acc_data.shape[0]
        
        # Convert from channel-last (NHWC) to channel-first (NCHW) for Conv1D
        acc_data_conv = acc_data.transpose(1, 2)  # [batch, seq, ch] -> [batch, ch, seq]
        gyro_data_conv = gyro_data.transpose(1, 2)  # [batch, seq, ch] -> [batch, ch, seq]
        
        # Process through encoders
        acc_features = self.acc_encoder(acc_data_conv)  # [batch, embed_dim, seq]
        gyro_features = self.gyro_encoder(gyro_data_conv)  # [batch, embed_dim, seq]
        
        # Convert back to channel-last for transformer
        acc_features = acc_features.transpose(1, 2)  # [batch, seq, embed_dim]
        gyro_features = gyro_features.transpose(1, 2)  # [batch, seq, embed_dim]
        
        # Fusion by concatenation
        fused_features = torch.cat([acc_features, gyro_features], dim=2)  # [batch, seq, fusion_dim]
        
        # Get sequence length for positional embedding
        seq_len = fused_features.shape[1]
        
        # Slice positional embeddings to match sequence length (fixed operation for TFLite)
        if seq_len <= self.position_embeddings.shape[1]:
            pos_emb = self.position_embeddings[:, :seq_len, :]
        else:
            # Handle case where input is longer than positional embeddings
            pos_emb = torch.nn.functional.pad(
                self.position_embeddings,
                (0, 0, 0, seq_len - self.position_embeddings.shape[1], 0, 0)
            )
        
        # Add positional embeddings - broadcasting handles batch dimension
        fused_features = fused_features + pos_emb
        
        # Apply transformer
        transformer_output = self.transformer(fused_features)
        
        # Global average pooling (mean across sequence dimension)
        pooled_output = torch.mean(transformer_output, dim=1)
        
        # Classification
        return self.classifier(pooled_output)
