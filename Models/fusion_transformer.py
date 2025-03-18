# Models/fusion_transformer.py
import torch
from torch import nn
from typing import Dict, Optional, Union, Tuple, Any
import torch.nn.functional as F
from einops import rearrange
import math
import logging
import traceback

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
                feature_dim=None,
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
            fusion_type: How to combine different sensor data ('concat', 'attention', 'acc_only')
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            feature_dim: Optional explicit feature dimension (computed automatically if None)
        """
        super().__init__()
        logger.info(f"Initializing FusionTransModel with fusion_type={fusion_type}")
        self.fusion_type = fusion_type
        self.seq_len = acc_frames
        self.embed_dim = embed_dim
        self.acc_only = fusion_type == 'acc_only'
        
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

        # Quaternion orientation encoder if not using accelerometer only
        if not self.acc_only:
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
            if feature_dim is None:
                feature_dim = embed_dim * 2
            else:
                logger.info(f"Using explicit feature dimension: {feature_dim}")
        elif fusion_type == 'attention':
            # We use attention to combine the embeddings
            feature_dim = embed_dim
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        elif fusion_type == 'acc_only':
            # Use only accelerometer data
            feature_dim = embed_dim
        else:
            # Default to concatenation
            feature_dim = embed_dim * 2
            logger.warning(f"Unknown fusion type '{fusion_type}', defaulting to 'concat'")
            self.fusion_type = 'concat'

        self.feature_dim = feature_dim
        
        # Feature projector for handling fusion features
        self.fusion_feature_projector = nn.Linear(43, embed_dim)  # 43 is typical fusion feature dimension
        self.feature_projector = nn.Linear(embed_dim * 2, feature_dim)  # For mismatched dimensions

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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward_fusion(self, acc_data, fusion_features):
        """
        Forward pass using accelerometer data and pre-extracted fusion features.
        
        Args:
            acc_data: Linear acceleration data [batch, seq_len, 3]
            fusion_features: Pre-computed fusion features [batch, feature_dim]
            
        Returns:
            Class logits
        """
        # Process linear acceleration
        acc_data = rearrange(acc_data, 'b l c -> b c l')
        acc_features = self.linear_acc_encoder(acc_data)
        acc_features = rearrange(acc_features, 'b c l -> b l c')
        
        # Process fusion features
        batch_size, seq_len, _ = acc_features.shape
        
        # Project fusion features to match embedding dimension
        fusion_embed = self.fusion_feature_projector(fusion_features)
        
        # Expand fusion features to match sequence length
        expanded_features = fusion_embed.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine features
        combined = torch.cat([acc_features, expanded_features], dim=2)
        
        # Project to desired feature dimension if needed
        if combined.size(2) != self.feature_dim:
            features = self.feature_projector(combined)
        else:
            features = combined
        
        # Add positional encoding (only use what we need)
        pos_embed = self.position_embedding[:, :seq_len, :]
        features = features + pos_embed
        
        # Apply transformer encoder
        transformer_output = self.transformer(features)
        
        # Global average pooling
        pooled = torch.mean(transformer_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

    def forward_quaternion(self, linear_acc, quaternion):
        """
        Forward pass using linear acceleration and quaternion data.
        
        Args:
            linear_acc: Linear acceleration data [batch, seq_len, 3]
            quaternion: Quaternion orientation data [batch, seq_len, 4]
            
        Returns:
            Class logits
        """
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
            # Check if we need to project to match dimensions
            combined = torch.cat([linear_acc_features, quat_features], dim=2)
            if combined.size(2) != self.feature_dim:
                features = self.feature_projector(combined)
            else:
                features = combined
        elif self.fusion_type == 'attention':
            # Use attention to combine acc and quaternion features
            linear_acc_q = rearrange(linear_acc_features, 'b l c -> l b c')
            quat_k = rearrange(quat_features, 'b l c -> l b c')
            quat_v = quat_k
            attn_output, _ = self.fusion_attention(linear_acc_q, quat_k, quat_v)
            features = rearrange(attn_output, 'l b c -> b l c')
        else:
            # Default to concatenation
            features = torch.cat([linear_acc_features, quat_features], dim=2)
            if features.size(2) != self.feature_dim:
                features = self.feature_projector(features)
        
        # Add positional encoding - handle variable sequence lengths
        seq_len = features.size(1)
        pos_embed = self.position_embedding[:, :seq_len, :]
        features = features + pos_embed
        
        # Apply transformer encoder
        transformer_output = self.transformer(features)
        
        # Global average pooling
        pooled = torch.mean(transformer_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

    def forward_accelerometer(self, acc_data):
        """
        Forward pass using only accelerometer data.
        
        Args:
            acc_data: Accelerometer data [batch, seq_len, 3]
            
        Returns:
            Class logits
        """
        # Process accelerometer data
        acc_data = rearrange(acc_data, 'b l c -> b c l')
        acc_features = self.linear_acc_encoder(acc_data)
        acc_features = rearrange(acc_features, 'b c l -> b l c')
        
        # Add positional encoding
        seq_len = acc_features.size(1)
        pos_embed = self.position_embedding[:, :seq_len, :self.embed_dim]
        features = acc_features + pos_embed[:, :, :acc_features.size(2)]
        
        # Apply transformer
        transformer_output = self.transformer(features)
        
        # Global average pooling
        pooled = torch.mean(transformer_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

    def forward(self, data):
        """
        General forward pass with robust handling of different input formats.
        
        Args:
            data: Either a dictionary with sensor data or direct tensor input
            
        Returns:
            Class logits
        """
        try:
            # Handle dictionary input
            if isinstance(data, dict):
                # Check for pre-computed fusion features
                if 'fusion_features' in data and data['fusion_features'] is not None:
                    if 'accelerometer' not in data or data['accelerometer'] is None:
                        raise ValueError("Missing required input: accelerometer data for fusion")
                    return self.forward_fusion(data['accelerometer'], data['fusion_features'])
                
                # Check for quaternion data
                elif 'quaternion' in data and data['quaternion'] is not None:
                    # Use linear_acceleration if available, otherwise use accelerometer
                    acc_data = data.get('linear_acceleration', data.get('accelerometer'))
                    if acc_data is None:
                        raise ValueError("Missing required input: accelerometer or linear_acceleration")
                    return self.forward_quaternion(acc_data, data['quaternion'])
                
                # Check for gyroscope data
                elif 'gyroscope' in data and data['gyroscope'] is not None and not self.acc_only:
                    # Use linear_acceleration if available, otherwise use accelerometer
                    acc_data = data.get('linear_acceleration', data.get('accelerometer'))
                    if acc_data is None:
                        raise ValueError("Missing required input: accelerometer or linear_acceleration")
                    # Use same processing as quaternion but with gyro data
                    return self.forward_quaternion(acc_data, data['gyroscope'])
                
                # Fall back to accelerometer-only processing
                else:
                    # Get accelerometer data
                    acc_data = data.get('accelerometer')
                    if acc_data is None:
                        raise ValueError("Missing required input: accelerometer")
                    
                    return self.forward_accelerometer(acc_data)
            
            # Handle direct tensor input (assumed to be accelerometer data)
            else:
                return self.forward_accelerometer(data)
                
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise
