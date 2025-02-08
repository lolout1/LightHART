# File: Models/t4.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processor.base import Time2Vec  # <-- Added import

class TransformerTeacher(nn.Module):
    """
    Teacher model with cross-attention fusion for skeleton + accelerometer.
    """
    def __init__(
        self,
        num_joints=32,
        joint_dim=3,
        hidden_dim=96,
        accel_dim=3,
        time2vec_dim=16,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.3,
        num_classes=2
    ):
        super().__init__()

        # 1) Skeleton Encoder
        self.skel_embed = nn.Linear(num_joints * joint_dim, hidden_dim)
        self.pos_skel = nn.Parameter(torch.randn(1, 120, hidden_dim))  # up to 120 skeleton frames

        encoder_layer_skel = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.skel_encoder = nn.TransformerEncoder(encoder_layer_skel, num_layers=num_layers)

        # 2) Accelerometer Encoder
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        self.accel_proj = nn.Linear(accel_dim + time2vec_dim, hidden_dim)

        encoder_layer_accel = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.accel_encoder = nn.TransformerEncoder(encoder_layer_accel, num_layers=num_layers)

        # 3) Cross‐Attention for Fusion
        cross_attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn = nn.TransformerEncoder(cross_attn_layer, num_layers=1)

        # 4) Classification Head
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, skel_seq, accel_seq, accel_time, skel_mask=None, accel_mask=None):
        """
        Return: {
          'logits': ...,
          'features': {
             'skel': <B, Ts, hidden>,
             'accel': <B, Ta, hidden>,
             'cross': <fusion representation (B, hidden * 2)>,
          }
        }
        """
        # ----- Skeleton Encoding -----
        B, Ts, _ = skel_seq.shape
        skel_x = self.skel_embed(skel_seq) + self.pos_skel[:, :Ts, :]
        skel_enc = self.skel_encoder(skel_x, src_key_padding_mask=skel_mask)

        # ----- Accelerometer Encoding -----
        B, Ta, _ = accel_seq.shape
        t_emb = self.time2vec(accel_time.view(-1, 1)).view(B, Ta, -1)
        accel_in = torch.cat([accel_seq, t_emb], dim=-1)
        accel_in = self.accel_proj(accel_in)
        accel_enc = self.accel_encoder(accel_in, src_key_padding_mask=accel_mask)

        # ----- Cross‐Attention Fusion -----
        combined = torch.cat([skel_enc, accel_enc], dim=1)  # shape (B, Ts+Ta, hidden_dim)
        if skel_mask is not None and accel_mask is not None:
            combined_mask = torch.cat([skel_mask, accel_mask], dim=1)  # shape (B, Ts+Ta)
        else:
            combined_mask = None
        cross_out = self.cross_attn(combined, src_key_padding_mask=combined_mask)

        # Global average pooling to get a single vector
        cross_mean = cross_out.mean(dim=1)  # (B, hidden_dim)

        # Also pool skeleton and accelerometer separately for skip connections
        skel_mean = skel_enc.mean(dim=1)
        accel_mean = accel_enc.mean(dim=1)

        # Final fusion
        fused = torch.cat([cross_mean, skel_mean + accel_mean], dim=-1)  # (B, hidden_dim * 2)

        logits = self.fusion_fc(fused)
        return {
            'logits': logits,
            'features': {
                'skel': skel_enc,
                'accel': accel_enc,
                'cross': fused
            }
        }


