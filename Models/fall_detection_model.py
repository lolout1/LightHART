# fall_detection_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EdgeFallTransformer(nn.Module):
    """Fall detection model using transformer architecture with improved numerical stability"""
    
    def __init__(self, acc_frames=128, num_classes=1, num_heads=4, 
                 acc_coords=4, num_layer=2, embed_dim=32, dropout=0.1):
        super().__init__()
        
        # Input projection with batch norm for stability
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU()  # More stable activation function
        )
        
        # Positional encoding
        position = torch.arange(acc_frames).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_enc = torch.zeros(1, acc_frames, embed_dim)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_enc)
        
        # Transformer blocks with standard PyTorch components for stability
        self.blocks = nn.ModuleList()
        for _ in range(num_layer):
            self.blocks.append(
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    dropout=dropout
                )
            )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)
        
        # Initialize with smaller weights for stability
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)  # Smaller initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Input projection
        x = x.transpose(1, 2)  # [B, F, L]
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # [B, L, F]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final processing
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        
        logits = self.output(x)
        
        return logits, x

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Use smaller initialization for better stability
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.01)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V with careful reshaping
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scale query for stability
        q = q * self.scale
        
        # Compute attention with careful numerical handling
        attn = (q @ k.transpose(-2, -1))
        attn = torch.clamp(attn, min=-10.0, max=10.0)  # Clamp for stability
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        # Pre-norm architecture for better stability
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = SelfAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),  # More stable than ReLU
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Initialize MLP weights
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
