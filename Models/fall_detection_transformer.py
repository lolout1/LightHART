# models/fall_detection_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.01)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        
    def forward(self, x):
        B, L, C = x.shape
        
        # Compute QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        
        # Output projection
        x = self.proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Initialize weights
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm architecture
        y = self.norm1(x)
        y = self.attn(y)
        y = self.dropout(y)
        x = x + y
        
        y = self.norm2(x)
        y = self.mlp(y)
        y = self.dropout(y)
        x = x + y
        
        return x

class FallDetectionTransformer(nn.Module):
    def __init__(self, 
                 acc_frames=128, 
                 num_classes=1, 
                 num_heads=2, 
                 acc_coords=3, 
                 num_layer=2, 
                 embed_dim=16, 
                 dropout=0.5):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(4, embed_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(embed_dim)
        )
        
        # Initialize Conv1d
        for m in self.input_proj:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=embed_dim * 2,
                dropout=dropout
            ) for _ in range(num_layer)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, num_classes)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)
    
    def forward(self, acc_data, *args):
        # Input shape: (batch_size, seq_len, channels)
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x)
        
        # Rearrange for transformer blocks
        x = rearrange(x, 'b c l -> b l c')
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Store features for return
        features = x
        
        # Global average pooling using 1D conv as in original
        x = rearrange(x, 'b f c -> b c f')
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)
        x = rearrange(x, 'b c f -> b (c f)')
        
        # Output projection
        logits = self.output(x)
        
        return logits, features
