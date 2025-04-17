import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from normalization import LayerNorm  # custom Edge‑Torch layer
from scaled_dot_product_attention import scaled_dot_product_attention  # SDPA kernel
from feed_forward import SequentialFeedForward  # MLP block

# ----------------------------------------------------------------------------
# Self‑Attention (bidirectional or causal) ------------------------------------------------

class SelfAttention(nn.Module):
    """Export‑safe multi‑head self‑attention.

    Args:
        dim (int): embedding dimension
        num_heads (int): number of heads
        causal (bool): if True, future positions are masked (for streaming)
    """

    def __init__(self, dim: int, num_heads: int = 2, causal: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def _make_causal_mask(self, T: int, device: torch.device):
        mask = torch.full((1, 1, T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)  # upper‑triangular
        return mask

    def forward(self, x: torch.Tensor):  # (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B, T, N, H)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        mask = None
        if self.causal:
            mask = self._make_causal_mask(T, x.device)

        y = scaled_dot_product_attention(q, k, v, self.head_dim, mask=mask, scale=self.scale)
        y = y.reshape(B, T, D)  # merge heads
        return self.proj(y)


# ----------------------------------------------------------------------------
# Transformer block -----------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, causal: bool = False, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, causal)
        self.ln2 = LayerNorm(dim)
        self.ff = SequentialFeedForward(
            dim=dim,
            hidden_dim=dim * mlp_ratio,
            activation=F.gelu,
        )

    def forward(self, x: torch.Tensor):  # (B, T, D)
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ----------------------------------------------------------------------------
# Edge‑friendly fall‑detection Transformer ------------------------------------

class EdgeFallTransformer(nn.Module):
    """Transformer for 1‑D sensor windows, ready for TFLite via AI‑Edge‑Torch.

    Input:  (B, T, C)  where C=3, T=seq_len.
    Output: (B, num_classes)  Sigmoid scores.
    """

    def __init__(
        self,
        seq_len: int = 128,
        in_channels: int = 3,
        embed_dim: int = 48,
        depth: int = 2,
        num_heads: int = 2,
        num_classes: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len

        # 1‑D conv stem expects (B, C, T) so we'll permute in forward.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=8, stride=1, padding="same"),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, causal)
            for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor):  # x: (B, T, C)
        # Rearrange for conv: (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.stem(x)               # (B, D, T)
        x = x.permute(0, 2, 1)         # (B, T, D)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)               # (B, T, D)
        x = x.mean(dim=1)              # global average over time -> (B, D)
        logits = self.head(x)          # (B, num_classes)
        return torch.sigmoid(logits)


# ----------------------------------------------------------------------------
# Quick export smoke‑test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    model = EdgeFallTransformer().eval()
    sample = torch.randn(1, 128, 3)

    # Forward pass
    out = model(sample)
    print("Output shape:", out.shape)  # should be [1, num_classes]

