
# Models/t1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processor.base import Time2Vec  # Use the common Time2Vec implementation
import math

def masked_mean(features, mask):
    """
    Computes the mean over the time dimension for each sample, ignoring padded positions.
    mask: boolean tensor with True for padded positions.
    """
    if mask is not None:
        valid = ~mask  # valid positions are False in the mask
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
        TransformerTeacher model for fall detection using skeleton and accelerometer data.
        
        Parameters:
         - num_joints: number of joints in the skeleton.
         - joint_dim: typically 3 (x,y,z) for each joint.
         - hidden_skel: embedding dimension for the skeleton branch.
         - accel_dim: number of accelerometer channels (typically 3).
         - time2vec_dim: output channels for the Time2Vec embedding.
         - hidden_accel: embedding dimension for the accelerometer branch.
         - accel_heads: number of attention heads for the accelerometer Transformer.
         - accel_layers: number of layers for the accelerometer Transformer.
         - skeleton_heads: number of attention heads for the skeleton Transformer.
         - skeleton_layers: number of layers for the skeleton Transformer.
         - fusion_hidden: hidden dimension for the fusion MLP.
         - num_classes: number of output classes.
         - dropout: dropout rate.
         - dim_feedforward: feedforward dimension in Transformer layers.
        """
        super().__init__()

        # 1) Skeleton Transformer Branch
        self.skel_embed = nn.Linear(num_joints * joint_dim, hidden_skel)
        # Learnable positional encoding (assume max skeleton sequence length = 64)
        self.skel_pos = nn.Parameter(torch.randn(1, 120, hidden_skel))
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
        # The accelerometer input is raw (B, T_a, 3) and a corresponding time vector (B, T_a)
        # Time2Vec will embed each scalar time into a vector of size time2vec_dim.
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        self.accel_proj = nn.Linear(accel_dim + time2vec_dim, hidden_accel)
        accel_layer = nn.TransformerEncoderLayer(
            d_model=hidden_accel,
            nhead=accel_heads,
            dim_feedforward=dim_feedforward * 2,  # increased capacity
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.accel_transformer = nn.TransformerEncoder(accel_layer, num_layers=accel_layers)

        # 3) Fusion
        # We fuse the skeleton feature (hidden_skel) with the accelerometer feature (hidden_accel)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_skel + hidden_accel, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, skel_seq, accel_seq, accel_time, accel_mask=None):
        """
        Forward pass:
         - skel_seq: (B, T_s, num_joints*joint_dim) raw skeleton input (e.g. flattened joints)
         - accel_seq: (B, T_a, accel_dim) accelerometer signals (e.g. [x, y, z])
         - accel_time: (B, T_a) time stamps for each accelerometer sample (in seconds)
         - accel_mask: (B, T_a) Boolean mask (True for padded positions)
        """
        B, Ts, _ = skel_seq.shape
        # Skeleton branch
        # Embed skeleton and add positional encoding
        skel_emb = self.skel_embed(skel_seq) + self.skel_pos[:, :Ts, :]
        # Process with Transformer and average over time dimension
        skel_feat = self.skel_transformer(skel_emb).mean(dim=1)

        # Accelerometer branch
        B, Ta, _ = accel_seq.shape
        # Apply Time2Vec on the flattened time values (reshape to (B*Ta,1) then back)
        t_emb = self.time2vec(accel_time.view(B * Ta, 1)).view(B, Ta, -1)
        # Concatenate raw accelerometer input with time embedding and project
        accel_in = F.gelu(self.accel_proj(torch.cat([accel_seq, t_emb], dim=-1)))
        # Process with Transformer encoder
        accel_feat_seq = self.accel_transformer(accel_in, src_key_padding_mask=accel_mask)
        # Compute masked mean along the time dimension
        accel_feat = masked_mean(accel_feat_seq, accel_mask)

        # Fusion
        fused = self.fusion(torch.cat([skel_feat, accel_feat], dim=-1))
        logits = self.classifier(fused)
        return logits

def masked_mean(features, mask):
    if mask is not None:
        valid = ~mask  # valid positions are where mask is False
        features = features * valid.unsqueeze(-1).float()
        return features.sum(dim=1) / valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return features.mean(dim=1)

