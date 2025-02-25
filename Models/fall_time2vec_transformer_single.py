import torch
import torch.nn as nn
import torch.nn.functional as F

class Time2Vec(nn.Module):
    """
    Minimal learned Time2Vec for 1D time input
    """
    def __init__(self, out_channels=8):
        super().__init__()
        self.lin_weight = nn.Parameter(torch.randn(1))
        self.lin_bias = nn.Parameter(torch.randn(1))

        if out_channels > 1:
            self.per_weight = nn.Parameter(torch.randn(out_channels - 1))
            self.per_bias   = nn.Parameter(torch.randn(out_channels - 1))
        else:
            self.per_weight = None
            self.per_bias   = None

    def forward(self, t):
        # t => shape (N,1)
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
    Single-input model: (B, T, 4) => [x, y, z, timeElapsed].
    1) extract time => Time2Vec
    2) cat => [x,y,z, t_emb]
    3) pass to Transformer => classification
    """
    def __init__(self,
                 time2vec_dim=8,
                 d_model=48,
                 nhead=4,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.1,
                 dim_feedforward=128):
        super().__init__()
        self.time2vec_dim = time2vec_dim
        self.time2vec = Time2Vec(out_channels=time2vec_dim)

        in_dim = 3 + time2vec_dim  # x,y,z plus time embedding
        self.input_proj = nn.Linear(in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        """
        x => (B,T,4). The 4th channel is time.
        mask => (B,T) boolean, True => pad
        Returns => (B,num_classes).
        """
        B, T, C = x.shape
        assert C == 4, "Expected 4 channels => [x, y, z, timeElapsed]"

        # separate time => shape(B,T,1)
        time_vals = x[..., 3].unsqueeze(-1)  # => (B,T,1)
        # flatten => (B*T,1)
        time_flat = time_vals.reshape(-1,1)
        t_emb_flat = self.time2vec(time_flat)   # => (B*T, time2vec_dim)
        t_emb = t_emb_flat.view(B, T, self.time2vec_dim)  # => (B,T,time2vec_dim)

        # slice xyz => shape(B,T,3)
        xyz = x[..., :3]

        # cat => shape(B,T, 3 + time2vec_dim)
        combined = torch.cat([xyz, t_emb], dim=-1)

        # project => (B,T,d_model)
        proj = self.input_proj(combined)

        # pass to transformer
        out_seq = self.encoder(proj, src_key_padding_mask=mask)  # => (B,T,d_model)

        # global avg
        feats = out_seq.mean(dim=1)  # => (B,d_model)

        logits = self.fc(feats)
        return logits
