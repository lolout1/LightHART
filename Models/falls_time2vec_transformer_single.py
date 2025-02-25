# File: Models/fall_time2vec_transformer_single.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Time2Vec(nn.Module):
    """
    Minimal learned Time2Vec
    """
    def __init__(self, out_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self.lin_weight = nn.Parameter(torch.randn(1))
        self.lin_bias   = nn.Parameter(torch.randn(1))
        if out_channels > 1:
            self.per_weight = nn.Parameter(torch.randn(out_channels - 1))
            self.per_bias   = nn.Parameter(torch.randn(out_channels - 1))
        else:
            self.per_weight = None
            self.per_bias   = None

    def forward(self, t):
        # t => shape (B*T,1)
        t_lin = self.lin_weight * t + self.lin_bias
        if self.per_weight is not None:
            alpha = self.per_weight.unsqueeze(0)
            beta  = self.per_bias.unsqueeze(0)
            t_per = torch.sin(alpha * t + beta)
            return torch.cat([t_lin, t_per], dim=-1)
        else:
            return t_lin

class FallTime2VecTransformerSingle(nn.Module):
    """
    Single-input model for [x,y,z,time].
    1) Extract time channel, pass to Time2Vec,
    2) Cat with [x,y,z],
    3) Transformer => classification
    """
    def __init__(self,
                 time2vec_dim=3,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.1,
                 dim_feedforward=128):
        super().__init__()
        self.time2vec_dim = time2vec_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward

        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        in_dim = 3 + time2vec_dim  # [x,y,z] + t_emb
        self.input_proj = nn.Linear(in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x_in):
        """
        x_in => (B,T,4) => [x,y,z,time].
        Return => (B,num_classes) logits
        """
        B, T, C = x_in.shape
        assert C == 4, f"Expect last dim=4 => [x,y,z,time], got {C}"

        # separate time => shape(B,T,1) => flatten
        time_vals = x_in[:,:,3:4]  # => (B,T,1)
        time_flat = time_vals.reshape(-1,1)  # => (B*T,1)
        t_emb_flat = self.time2vec(time_flat) # =>(B*T,time2vec_dim)
        t_emb = t_emb_flat.view(B,T,self.time2vec_dim)

        # separate xyz =>(B,T,3)
        xyz = x_in[:,:,:3]

        # combine =>(B,T, 3+time2vec_dim)
        combined = torch.cat([xyz, t_emb], dim=-1)

        # project =>(B,T,d_model)
        proj = self.input_proj(combined)

        # transform
        out_seq = self.encoder(proj)  # =>(B,T,d_model)

        # global avg
        feats = out_seq.mean(dim=1)

        logits = self.fc(feats)
        return logits

