import torch
from torch import nn
from typing import Dict, Optional, Union, Tuple
import torch.nn.functional as F
from einops import rearrange
import math
import logging

# Configure logging
logger = logging.getLogger("model")

class FusionTransModel(nn.Module):
    def __init__(self,
                acc_frames=128,
                num_classes=2,
                num_heads=4,
                acc_coords=3,
                quat_coords=4,
                num_layers=2,
                embed_dim=24,
                fusion_type='concat',
                dropout=0.3,
                use_batch_norm=True,
                **kwargs):
        """
        Optimized transformer model for IMU fusion with linear acceleration and quaternion.
        Args:
            acc_frames: Number of frames in acceleration data
            num_classes: Number of output classes
            num_heads: Number of attention heads
            acc_coords: Number of linear acceleration coordinates (3)
            quat_coords: Number of quaternion coordinates (4)
            num_layers: Number of transformer layers
            embed_dim: Embedding dimension for features
            fusion_type: How to combine different sensor data ('concat', 'attention')
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        print(f"Initializing FusionTransModel with fusion_type={fusion_type}")
        self.fusion_type = fusion_type
        self.seq_len = acc_frames
        self.embed_dim = embed_dim

        # Linear acceleration encoder (the data is already linear acceleration)
        self.linear_acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        # Quaternion orientation encoder
        self.quat_encoder = nn.Sequential(
            nn.Conv1d(quat_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        # Determine feature dimension based on fusion type
        if fusion_type == 'concat':
            # We concatenate linear acceleration and quaternion embeddings
            feature_dim = embed_dim * 2
        elif fusion_type == 'attention':
            # We use attention to combine the embeddings
            feature_dim = embed_dim
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            # Default to concatenation
            feature_dim = embed_dim * 2
            print(f"Unknown fusion type '{fusion_type}', defaulting to 'concat'")
            self.fusion_type = 'concat'

        self.feature_dim = feature_dim

        # Positional embedding for transformer
        self.position_embedding = nn.Parameter(torch.randn(1, self.seq_len, feature_dim) * 0.02)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Use pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers,
            norm=nn.LayerNorm(feature_dim)
        )

        # Classification head with regularization
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, data_dict):
        """
        Forward pass using linear acceleration and quaternion data.
        
        Args:
            data_dict: Dictionary containing 'linear_acceleration' and 'quaternion'
            
        Returns:
            Class logits
        """
        # Extract inputs
        if isinstance(data_dict, dict):
            # Get linear acceleration (this is already linear, not raw accelerometer)
            linear_acc = data_dict.get('linear_acceleration')
            
            # Get quaternion data for orientation
            quaternion = data_dict.get('quaternion')
            
            if linear_acc is None:
                raise ValueError("Missing required input: linear_acceleration")
                
            if quaternion is None:
                raise ValueError("Missing required input: quaternion")
                
        else:
            raise ValueError("Input must be a dictionary with 'linear_acceleration' and 'quaternion' keys")
        
        batch_size = linear_acc.shape[0]
        
        # Process linear acceleration
        linear_acc = rearrange(linear_acc, 'b l c -> b c l')
        linear_acc_features = self.linear_acc_encoder(linear_acc)
        linear_acc_features = rearrange(linear_acc_features, 'b c l -> b l c')
        
        # Process quaternion
        quaternion = rearrange(quaternion, 'b l c -> b c l')
        quat_features = self.quat_encoder(quaternion)
        quat_features = rearrange(quat_features, 'b c l -> b l c')
        
        # Combine features based on fusion type
        if self.fusion_type == 'concat':
            features = torch.cat([linear_acc_features, quat_features], dim=2)
        elif self.fusion_type == 'attention':
            # Use attention to combine acc and quaternion features
            linear_acc_q = rearrange(linear_acc_features, 'b l c -> l b c')
            quat_k = rearrange(quat_features, 'b l c -> l b c')
            quat_v = quat_k
            attn_output, _ = self.fusion_attention(linear_acc_q, quat_k, quat_v)
            features = rearrange(attn_output, 'l b c -> b l c')
        
        # Add positional encoding
        pos_embed = self.position_embedding[:, :features.size(1), :features.size(2)]
        features = features + pos_embed
        
        # Apply transformer encoder
        transformer_output = self.transformer(features)
        
        # Global average pooling with attention weights
        pooled = torch.mean(transformer_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
