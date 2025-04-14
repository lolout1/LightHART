import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        context = torch.sum(x * attn_weights, dim=1)
        return context

class OptimizedFallDetector(nn.Module):
    def __init__(self, mocap_frames=64, num_joints=32, acc_frames=64, num_classes=1, num_heads=4, 
                 acc_coords=4, num_layer=2, embed_dim=64, activation='gelu', dropout=0.3, **kwargs):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*2,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layer
        )
        
        self.temporal_attn = TemporalAttention(embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, acc_data, skl_data=None, **kwargs):
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x)
        x = rearrange(x, 'b c l -> b l c')
        features = self.transformer(x)
        context = self.temporal_attn(features)
        logits = self.classifier(context)
        return logits, features

class TransModel(OptimizedFallDetector):
    pass
