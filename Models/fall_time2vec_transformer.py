# Models/fall_time2vec_transformer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Time2Vec(nn.Module):
    """
    Minimal Time2Vec example. If you have your own in utils.processor.base, import that instead.
    This version returns an embedding of shape (N, out_dim).
    """
    def __init__(self, out_dim=8):
        super().__init__()
        self.out_dim = out_dim
        # Phase / frequencies can be learned. Simplified approach below
        self.freq = nn.Parameter(torch.randn(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, t):
        """
        t: shape (N, 1) or (B*T, 1)
        returns: shape (N, out_dim)
        """
        # We do something like sin(freq * t + bias). This is a toy example.
        # You can adapt to the original T2V approach from your code.
        return torch.sin(t * self.freq + self.bias)


class FallTime2VecTransformer(nn.Module):
    """
    Example: a simple Transformer-based model that can take a (B, T, 3) accelerometer input
    plus an optional mask (B, T). We create an 8-dim Time2Vec, then cat => total feat=3+8=11.
    Then we do a standard TransformerEncoder and output (B, num_classes).
    """

    def __init__(self,
                 feat_dim=11,  # total dimension after we cat time2vec + accel
                 d_model=48,
                 nhead=4,
                 num_layers=2,
                 num_classes=2,
                 time2vec_dim=8,    # we'll do 8-D time2vec
                 dropout=0.2,
                 dim_feedforward=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.time2vec_dim = time2vec_dim
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward

        # 1) Time2Vec for the time axis
        self.time2vec = Time2Vec(out_dim=time2vec_dim)

        # 2) Input projection from feat_dim -> d_model
        # But note: raw accel=3 + time2vec=8 => total=11
        self.input_proj = nn.Linear(feat_dim, d_model)

        # 3) Standard PyTorch TransformerEncoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # we want (B, T, d_model)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 4) Output layer
        self.fc = nn.Linear(d_model, self.num_classes)

    def forward(self, accel_seq, accel_mask=None, accel_time=None):
        """
        accel_seq: shape (B, T, 3)
        accel_mask: shape (B, T) bool, True => padding
        accel_time: shape (B, T) or None
        """
        B, T, C = accel_seq.shape   # expect C=3 if you only have 3 accelerometer channels

        # 1) If no time provided => create dummy time
        if accel_time is None:
            # shape => (B, T) indexing from 0..T-1
            time_idx = torch.arange(T, device=accel_seq.device).unsqueeze(0).expand(B, T).float()
        else:
            time_idx = accel_time

        # 2) Flatten time => apply Time2Vec => shape => (B*T, 1)
        time_flat = time_idx.reshape(-1, 1)  # => (B*T, 1)
        t_emb_flat = self.time2vec(time_flat)  # => (B*T, time2vec_dim=8)
        t_emb = t_emb_flat.view(B, T, self.time2vec_dim)  # => (B, T, 8)

        # 3) Concatenate => shape => (B, T, 3+8=11)
        x = torch.cat([accel_seq, t_emb], dim=-1)

        # 4) Project => (B, T, d_model)
        x_proj = self.input_proj(x)

        # 5) Pass to Transformer encoder
        out = self.encoder(x_proj, src_key_padding_mask=accel_mask)  # => (B, T, d_model)

        # 6) Global average pool across time
        out = out.mean(dim=1)  # => (B, d_model)

        # 7) final linear => (B, num_classes)
        logits = self.fc(out)
        return logits

