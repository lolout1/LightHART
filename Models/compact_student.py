import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processor.base import Time2Vec

class CompactStudent(nn.Module):
    def __init__(self,
                 accel_dim=3,
                 time2vec_dim=16,    # Match teacher's time embedding
                 hidden_dim=48,      # Reduced from teacher's 128
                 num_heads=3,        # Efficient multi-head attention
                 num_layers=2,
                 dropout=0.2,
                 **kwargs):
        super().__init__()
        
        # Time embedding (matching teacher's input structure)
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        
        # Efficient initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(accel_dim + time2vec_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Lightweight temporal convolution for local patterns
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 
                     kernel_size=5, padding='same', groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Efficient transformer with fewer parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # Reduced feedforward size
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Simple projection to match teacher's feature space
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # Project to teacher's dimension for distillation
            nn.LayerNorm(128)
        )
        
        # Compact classifier
        self.classifier = nn.Linear(128, 2)
        
    def forward(self, accel_seq, accel_mask=None, accel_time=None, return_features=False):
        """Forward pass with efficient processing"""
        B, T, _ = accel_seq.shape
        
        # Time embedding
        t_emb = self.time2vec(accel_time.view(B * T, 1)).view(B, T, -1)
        
        # Initial projection
        x = torch.cat([accel_seq, t_emb], dim=-1)
        x = self.input_proj(x)
        
        # Efficient temporal processing
        x_local = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        
        # Global dependencies with transformer
        x = self.transformer(x_local, src_key_padding_mask=accel_mask)
        
        # Global pooling with mask handling
        if accel_mask is not None:
            mask_float = (~accel_mask).float().unsqueeze(-1)
            x = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-6)
        else:
            x = x.mean(dim=1)
            
        # Project to teacher's feature space
        features = self.feature_proj(x)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            return logits, {
                'accel_features': features,  # Aligned with teacher's dimension
                'fused_features': features   # Same features for fusion alignment
            }
        
        return logits
