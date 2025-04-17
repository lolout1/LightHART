import torch
import torch.nn as nn
from Models.attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Initialize feedforward layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Pre-norm architecture for stability
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x
