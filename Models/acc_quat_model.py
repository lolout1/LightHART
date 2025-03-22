# Models/acc_quat_model.py

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import logging
import traceback

logger = logging.getLogger("acc_quat_model")

class AccQuatModel(nn.Module):
    """
    Model that uses accelerometer and quaternion data (from IMU fusion) for fall detection.
    This allows evaluating the benefits of orientation information without raw gyroscope.
    """
    def __init__(self,
                acc_frames=64,
                num_classes=2,
                num_heads=4,
                acc_coords=3,
                quat_coords=4,
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
        self.quat_coords = quat_coords
        self.num_classes = num_classes
        
        # Set the sequence length to match accelerometer frames
        self.seq_len = self.acc_frames
        
        # Linear accelerometer encoder
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        # Quaternion encoder with special handling for orientation data
        self.quat_encoder = nn.Sequential(
            nn.Conv1d(quat_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )
        
        # Add positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, acc_frames, embed_dim * 2))

        # Determine feature dimension for concatenated features
        if feature_dim is None:
            self.feature_dim = embed_dim * 2  # acc + quat
        else:
            self.feature_dim = feature_dim
            
        # Feature adapter to ensure correct dimensions
        self.feature_adapter = nn.Linear(embed_dim * 2, self.feature_dim)
        
        # Ensure number of heads divides feature dimension evenly
        if self.feature_dim % num_heads != 0:
            adjusted_heads = max(1, self.feature_dim // (self.feature_dim // num_heads))
            if adjusted_heads != num_heads:
                logger.info(f"Adjusting number of heads from {num_heads} to {adjusted_heads} to match feature dimension")
            num_heads = adjusted_heads
            
        # Create transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=self.feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.feature_dim)
        )

        # Classification head
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
        
        # Add attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized AccQuatModel with embed_dim={self.embed_dim}, feature_dim={self.feature_dim}")
    
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
        Process accelerometer and quaternion data for fall detection.
        
        Args:
            data: Dictionary with 'accelerometer' and 'quaternion' tensors
            
        Returns:
            Classification logits
        """
        try:
            # Extract data from input dictionary
            if isinstance(data, dict):
                acc_data = data['accelerometer']
                quat_data = data.get('quaternion')
                
                # Ensure we have quaternion data
                if quat_data is None or torch.all(quat_data == 0):
                    raise ValueError("This model requires quaternion data")
            else:
                raise ValueError("Input must be a dictionary with 'accelerometer' and 'quaternion' data")
            
            # Process accelerometer data
            acc_data = rearrange(acc_data.float(), 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            # Process quaternion data
            quat_data = rearrange(quat_data.float(), 'b l c -> b c l')
            quat_features = self.quat_encoder(quat_data)
            quat_features = rearrange(quat_features, 'b c l -> b l c')
            
            # Concat features
            fused_features = torch.cat([acc_features, quat_features], dim=2)
            
            # Add positional encoding
            seq_len = fused_features.shape[1]
            fused_features = fused_features + self.pos_encoding[:, :seq_len, :]
            
            # Adapt features if needed
            if fused_features.shape[2] != self.feature_dim:
                fused_features = self.feature_adapter(fused_features)
            
            # Apply transformer encoder
            transformer_output = self.transformer(fused_features)
            
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
            batch_size = acc_data.shape[0] if 'acc_data' in locals() and isinstance(acc_data, torch.Tensor) else 1
            device = acc_data.device if 'acc_data' in locals() and isinstance(acc_data, torch.Tensor) else 'cpu'
            return torch.zeros((batch_size, self.num_classes), device=device)

    def forward_accelerometer_only(self, acc_data):
        """
        Fallback method for accelerometer-only inference.
        Not recommended as the model is designed for acc+quaternion.
        """
        logger.warning("Using accelerometer-only mode for a model designed for acc+quaternion")
        
        # Process accelerometer data
        acc_data = rearrange(acc_data.float(), 'b l c -> b c l')
        acc_features = self.acc_encoder(acc_data)
        acc_features = rearrange(acc_features, 'b c l -> b l c')
        
        # Create zero tensor for quaternion features
        batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
        quat_features = torch.zeros(batch_size, seq_len, self.embed_dim, device=acc_data.device)
        
        # Concat features
        fused_features = torch.cat([acc_features, quat_features], dim=2)
        
        # Add positional encoding
        fused_features = fused_features + self.pos_encoding[:, :seq_len, :]
        
        # Adapt features if needed
        if fused_features.shape[2] != self.feature_dim:
            fused_features = self.feature_adapter(fused_features)
        
        # Apply transformer encoder
        transformer_output = self.transformer(fused_features)
        
        # Apply attention pooling
        attn_weights = self.attn_pool(transformer_output)
        pooled = torch.sum(transformer_output * attn_weights, dim=1)
        
        # Apply classifier
        logits = self.classifier(pooled)
        
        return logits
