# Models/acc_only_model.py

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import logging
import traceback

logger = logging.getLogger("acc_only_model")

class AccOnlyModel(nn.Module):
    """
    Specialized model that uses only accelerometer data for fall detection.
    This allows evaluating the importance of accelerometer features alone.
    """
    def __init__(self,
                acc_frames=64,
                num_classes=2,
                num_heads=4,
                acc_coords=3,
                num_layers=2,
                embed_dim=32,
                dropout=0.3,
                use_batch_norm=True,
                feature_dim=None,
                **kwargs):
        super().__init__()
        self.acc_frames = acc_frames
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        self.num_classes = num_classes
        
        # Set the sequence length to match accelerometer frames
        self.seq_len = self.acc_frames
        
        # Linear accelerometer encoder - enhanced for single-modality use
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding='same'),  # Larger kernel to capture more temporal patterns
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )
        
        # Add positional encoding for better temporal modeling
        self.pos_encoding = nn.Parameter(torch.zeros(1, acc_frames, embed_dim))
        
        # Feature dimension will be just the embedding dimension
        self.feature_dim = embed_dim
            
        # Ensure number of heads divides feature dimension evenly
        if self.feature_dim % num_heads != 0:
            adjusted_heads = max(1, self.feature_dim // (self.feature_dim // num_heads))
            if adjusted_heads != num_heads:
                logger.info(f"Adjusting number of heads from {num_heads} to {adjusted_heads} to match feature dimension")
            num_heads = adjusted_heads
            
        # Create transformer encoder with more layers for single-modality
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=self.feature_dim * 4,  # Larger feedforward layer for more expressivity
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers + 1,  # Add an extra layer for better processing
            norm=nn.LayerNorm(self.feature_dim)
        )

        # Classification head optimized for accelerometer-only
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.LayerNorm(128) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, num_classes)
        )
        
        # Add attention pooling to focus on important parts of the sequence
        self.attn_pool = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights properly
        self._init_weights()
        
        logger.info(f"Initialized AccOnlyModel with embed_dim={self.embed_dim}, feature_dim={self.feature_dim}")
    
    def _init_weights(self):
        """Initialize weights with better values for convergence"""
        # Initialize transformer layers
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize positional encoding
        nn.init.normal_(self.pos_encoding, 0, 0.02)
        
        # Initialize final classification layer
        nn.init.zeros_(self.classifier[-1].bias)
        fan_in = self.classifier[-1].weight.size(1)
        nn.init.normal_(self.classifier[-1].weight, 0, 1/math.sqrt(fan_in))

    def forward(self, data):
        """
        Process accelerometer data for fall detection.
        
        Args:
            data: Dictionary with 'accelerometer' tensor or directly the accelerometer tensor
            
        Returns:
            Classification logits
        """
        try:
            # Extract accelerometer data from input
            if isinstance(data, dict):
                acc_data = data['accelerometer']
            else:
                acc_data = data
                
            # Ensure data is float
            acc_data = acc_data.float()
            
            # Process accelerometer data
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            # Add positional encoding
            acc_features = acc_features + self.pos_encoding[:, :acc_features.shape[1], :]
            
            # Apply transformer encoder
            transformer_output = self.transformer(acc_features)
            
            # Apply attention pooling
            attn_weights = self.attn_pool(transformer_output)
            pooled = torch.sum(transformer_output * attn_weights, dim=1)
            
            # Apply classifier
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Provide a graceful fallback
            batch_size = acc_data.shape[0] if isinstance(acc_data, torch.Tensor) else 1
            return torch.zeros((batch_size, self.num_classes), device=acc_data.device if isinstance(acc_data, torch.Tensor) else 'cpu')
