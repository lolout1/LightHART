# EdgeFallTransformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class EdgeFallTransformer(nn.Module):
    def __init__(self,
                mocap_frames=128,
                num_joints=32,
                acc_frames=128,
                num_classes=1,
                num_heads=4,
                acc_coords=3,
                av=False,
                num_layer=2,
                norm_first=True,
                embed_dim=32,
                activation='relu',
                dropout=0.5,
                **kwargs):
        super().__init__()
        
        # Input projection - handle 4 channels (x,y,z,smv)
        self.input_proj = nn.Sequential(
            nn.Conv1d(4, embed_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(embed_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=norm_first
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layer,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Output layers
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, num_classes)
    
    @classmethod
    def from_config(cls, config):
        return cls(
            acc_frames=128,
            num_classes=config.get('num_classes', 1),
            num_heads=config.get('num_heads', 4),
            acc_coords=config.get('acc_coords', 3),
            num_layer=config.get('num_layer', 2),
            embed_dim=config.get('embed_dim', 32),
            activation=config.get('activation', 'relu'),
            dropout=config.get('dropout', 0.5)
        )
        
    def forward(self, acc_data, skl_data=None, **kwargs):
        # Process input - convert to channel-first for Conv1D
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x)
        
        # Convert to sequence-first for transformer
        x = rearrange(x, 'b c l -> l b c')
        
        # Apply transformer
        x = self.encoder(x)
        
        # Convert back to batch-first
        x = rearrange(x, 'l b c -> b l c')
        
        # Apply normalization
        x = self.temporal_norm(x)
        feature = x
        
        # Global pooling and output projection
        x = rearrange(x, 'b f c -> b c f')
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)
        x = rearrange(x, 'b c f -> b (c f)')
        x = self.output(x)
        
        return x, feature
