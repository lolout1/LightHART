# File: Models/master_t3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processor.base import Time2Vec

def masked_mean(features, mask):
    """
    Computes the mean over the time dimension for each sample, ignoring padded positions.
    mask: boolean tensor with True for padded positions.
    """
    if mask is not None:
        valid = ~mask  # valid positions are where mask is False
        features = features * valid.unsqueeze(-1).float()
        return features.sum(dim=1) / valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return features.mean(dim=1)

class TransformerTeacher(nn.Module):
    def __init__(self,
                 num_joints=32,
                 joint_dim=3,
                 hidden_skel=128,
                 accel_dim=3,
                 time2vec_dim=8,
                 hidden_accel=128,
                 accel_heads=4,
                 accel_layers=3,
                 skeleton_heads=4,
                 skeleton_layers=2,
                 fusion_hidden=256,
                 num_classes=2,
                 dropout=0.3,
                 dim_feedforward=256,
                 **kwargs):
        """
        Teacher model for skeleton+accelerometer data.
        Returns a dict with:
          - 'logits': final classification
          - 'accel_feat': the final accelerometer feature (for distillation)
        """
        super().__init__()

        # 1) Skeleton Transformer Branch
        self.skel_embed = nn.Linear(num_joints * joint_dim, hidden_skel)
        self.skel_pos = nn.Parameter(torch.randn(1, 120, hidden_skel))  # up to 120 frames
        skel_layer = nn.TransformerEncoderLayer(
            d_model=hidden_skel,
            nhead=skeleton_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.skel_transformer = nn.TransformerEncoder(skel_layer, num_layers=skeleton_layers)

        # 2) Accelerometer Branch
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        self.accel_proj = nn.Linear(accel_dim + time2vec_dim, hidden_accel)
        accel_layer = nn.TransformerEncoderLayer(
            d_model=hidden_accel,
            nhead=accel_heads,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.accel_transformer = nn.TransformerEncoder(accel_layer, num_layers=accel_layers)

        # 3) Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_skel + hidden_accel, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, skel_seq, accel_seq, accel_time, accel_mask=None):
        """
        Args:
         - skel_seq: (B, T_s, num_joints*joint_dim)
         - accel_seq: (B, T_a, accel_dim)
         - accel_time: (B, T_a)
         - accel_mask: (B, T_a) boolean mask
        Return dict:
         - 'logits': final (B, num_classes)
         - 'accel_feat': final pooled accel feature (B, hidden_accel)
        """
        # Skeleton branch
        B, Ts, _ = skel_seq.shape
        skel_emb = self.skel_embed(skel_seq) + self.skel_pos[:, :Ts, :]
        skel_feat_seq = self.skel_transformer(skel_emb)  # (B, T_s, hidden_skel)
        skel_feat = skel_feat_seq.mean(dim=1)            # (B, hidden_skel)

        # Accelerometer branch
        B, Ta, _ = accel_seq.shape
        t_emb = self.time2vec(accel_time.view(B * Ta, 1)).view(B, Ta, -1)
        accel_in = torch.cat([accel_seq, t_emb], dim=-1)  # (B, T_a, accel_dim + time2vec_dim)
        accel_in = F.gelu(self.accel_proj(accel_in))
        accel_feat_seq = self.accel_transformer(accel_in, src_key_padding_mask=accel_mask)
        accel_feat = masked_mean(accel_feat_seq, accel_mask)  # (B, hidden_accel)

        # Fusion
        fused = self.fusion(torch.cat([skel_feat, accel_feat], dim=-1))  # (B, fusion_hidden)
        logits = self.classifier(fused)  # (B, num_classes)

        return {
            'logits': logits,
            'accel_feat': accel_feat
        }

