import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.01)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x):
        b, l, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, l, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Manual attention calculation with clipping to prevent overflow
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = torch.clamp(scores, -10.0, 10.0)  # Prevent extreme values
        
        # Attention weights calculation without softmax
        attention_weights = torch.exp(scores - scores.max(dim=-1, keepdim=True)[0])
        attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).reshape(b, l, c)
        return self.proj(out)
