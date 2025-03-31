# Models/lightweight_transformer.py
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

class LightFallTransformer(nn.Module):
    def __init__(self, 
                 acc_frames=64,
                 num_classes=2, 
                 num_heads=2, 
                 acc_coords=3,
                 num_layers=2, 
                 embed_dim=32, 
                 dropout=0.2,
                 use_batch_norm=True):
        super().__init__()
        
        self.acc_frames = acc_frames
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        self.num_classes = num_classes
        
        # Encoder for accelerometer data - simplified
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        # Encoder for gyroscope data - simplified
        self.gyro_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        # Fusion layer - simple concatenation
        self.fusion_dim = embed_dim * 2
        
        # Position encoding - simpler version
        self.pos_embedding = nn.Parameter(torch.zeros(1, acc_frames, self.fusion_dim))
        
        # Transformer with fewer parameters
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim, 
            nhead=num_heads,
            dim_feedforward=self.fusion_dim * 2,  # Smaller feedforward layer
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers, 
            num_layers=num_layers
        )
        
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Basic weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Initialize position embedding
        nn.init.normal_(self.pos_embedding, std=0.02)
                
    def forward(self, inputs):
        if isinstance(inputs, dict):
            acc_data = inputs['accelerometer']
            gyro_data = inputs['gyroscope']
        else:
            # For inference or direct passing
            acc_data, gyro_data = inputs
        
        # Handle batch dimension properly
        if len(acc_data.shape) == 2:  # [batch, features]
            acc_data = acc_data.unsqueeze(1)  # [batch, 1, features]
        if len(gyro_data.shape) == 2:
            gyro_data = gyro_data.unsqueeze(1)
            
        batch_size = acc_data.shape[0]
        
        # Process accelerometer data
        acc_data = rearrange(acc_data, 'b l c -> b c l')
        acc_features = self.acc_encoder(acc_data)
        acc_features = rearrange(acc_features, 'b c l -> b l c')
        
        # Process gyroscope data
        gyro_data = rearrange(gyro_data, 'b l c -> b c l')
        gyro_features = self.gyro_encoder(gyro_data)
        gyro_features = rearrange(gyro_features, 'b c l -> b l c')
        
        # Simple concatenation fusion
        fused_features = torch.cat([acc_features, gyro_features], dim=2)
        
        # Add positional encoding
        fused_features = fused_features + self.pos_embedding
        
        # Transformer processing
        transformer_output = self.transformer(fused_features)
        
        # Global average pooling
        pooled = torch.mean(transformer_output, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    # Compatibility methods for your existing system
    def forward_accelerometer_only(self, acc_data):
        # Create dummy gyro data (zeros) in case only accelerometer data is available
        if len(acc_data.shape) == 2:
            acc_data = acc_data.unsqueeze(1)
            
        batch_size, seq_len, _ = acc_data.shape
        dummy_gyro = torch.zeros_like(acc_data)
        
        return self.forward((acc_data, dummy_gyro))
    
    def forward_multi_sensor(self, acc_data, gyro_data):
        return self.forward((acc_data, gyro_data))
        
    # Simplified inference method for TFLite conversion
    def forward_inference(self, acc_data, gyro_data):
        with torch.no_grad():
            return self.forward((acc_data, gyro_data))
