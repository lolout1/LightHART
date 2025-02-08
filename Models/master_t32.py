# File: Models/master_t32.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processor.base import Time2Vec

def masked_mean(features, mask):
    """
    Computes the mean over the time dimension for each sample, ignoring padded positions.
    mask: boolean tensor with True for padded positions => invalid.
    """
    if mask is not None:
        valid = ~mask  # valid = where mask is False
        # Expand mask so we can zero out padded positions in the feature dimension:
        features = features * valid.unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)  # avoid divide-by-zero
        return features.sum(dim=1) / denom
    else:
        return features.mean(dim=1)

class TransformerTeacher(nn.Module):
    """
    Teacher model with:
      1) Convolutional preprocessing for skeleton data (skel_preconv).
      2) Convolutional + Transformer for accelerometer (accel_conv + accel_transformer).
      3) Optional linear heads for KD (accel_feature_proj, skel_feature_proj).
      4) Fusion MLP + final classifier.
    Returns:
      - logits (B, num_classes)
      - accel_feat_kd (B, 128) => for knowledge distillation if needed
    """
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
        super().__init__()

        # 1) Enhanced Skeleton Branch
        #    We apply 1D convolutions across time (dimension=Ts).
        #    input shape: (B, Ts, num_joints*joint_dim) => we transpose to (B, C, Ts)
        self.skel_preconv = nn.Sequential(
            nn.Conv1d(num_joints * joint_dim, hidden_skel // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_skel // 2, hidden_skel, kernel_size=3, padding=1)
        )
        self.skel_pos = nn.Parameter(torch.randn(1, 120, hidden_skel))  # up to 120 frames
        self.skel_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_skel,
                nhead=skeleton_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=skeleton_layers
        )

        # 2) Accelerometer Branch with Feature Pyramid
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        self.accel_proj = nn.Linear(accel_dim + time2vec_dim, hidden_accel)

        # A convolution block to expand from hidden_accel -> hidden_accel * 2
        self.accel_conv = nn.Sequential(
            nn.Conv1d(hidden_accel, hidden_accel * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Then a Transformer on top of that
        self.accel_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_accel * 2,
                nhead=accel_heads,
                dim_feedforward=dim_feedforward * 2,  # bigger since we doubled the channels
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=accel_layers
        )

        # 3) Feature Projection Heads for KD
        #    E.g. you might want a 128-dim representation for knowledge distillation
        self.accel_feature_proj = nn.Linear(hidden_accel * 2, 128)
        self.skel_feature_proj  = nn.Linear(hidden_skel, 128)

        # 4) Enhanced Fusion
        #    Combine skeleton feature + expanded accelerometer feature => feed MLP => final logits
        self.fusion = nn.Sequential(
            nn.Linear(hidden_skel + (hidden_accel * 2), fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU()
        )
        self.classifier = nn.Linear(fusion_hidden // 2, num_classes)

    def forward(self, skel_seq, accel_seq, accel_time, accel_mask=None):
        """
        skel_seq:   (B, Ts, num_joints*joint_dim)
        accel_seq:  (B, Ta, accel_dim)
        accel_time: (B, Ta)
        accel_mask: (B, Ta) boolean => True = "PAD"
        """
        B, Ts, _ = skel_seq.shape
        # (1) Skeleton Preconv => shape => (B, hidden_skel, Ts)
        #     we need to transpose to (B, Channels, Ts), then back
        skel_conv = self.skel_preconv(skel_seq.permute(0, 2, 1)).permute(0, 2, 1)
        # Add positional embedding
        max_skel_len = self.skel_pos.shape[1]  # e.g. 120
        if Ts > max_skel_len:
            skel_conv = skel_conv[:, :max_skel_len, :]
            Ts = max_skel_len
        skel_emb = skel_conv + self.skel_pos[:, :Ts, :]
        # Transformer
        skel_feat_seq = self.skel_transformer(skel_emb)
        skel_feat = skel_feat_seq.mean(dim=1)

        # (2) Accelerometer
        B, Ta, _ = accel_seq.shape
        # Apply Time2Vec on flattened time
        t_emb = self.time2vec(accel_time.view(B * Ta, 1)).view(B, Ta, -1)
        # Project to hidden_accel
        accel_in = F.gelu(self.accel_proj(torch.cat([accel_seq, t_emb], dim=-1)))
        # Now conv => shape => (B, hidden_accel*2, Ta)
        accel_conv_out = self.accel_conv(accel_in.permute(0, 2, 1)).permute(0, 2, 1)
        # Transformer
        accel_feat_seq = self.accel_transformer(accel_conv_out, src_key_padding_mask=accel_mask)
        accel_feat = masked_mean(accel_feat_seq, accel_mask)

        # (3) Feature Projections (for KD if needed)
        accel_feat_kd = self.accel_feature_proj(accel_feat)  # e.g. (B, 128)
        # If you need skeleton KD => skel_feat_kd = self.skel_feature_proj(skel_feat)

        # (4) Fusion
        fused = self.fusion(torch.cat([skel_feat, accel_feat], dim=-1))
        logits = self.classifier(fused)

        # return final logits + whichever KD feats you want
        return logits

