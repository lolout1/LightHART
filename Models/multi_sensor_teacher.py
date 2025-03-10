# multi_sensor_teacher.py

import torch
import torch.nn as nn

class MultiSensorTeacher(nn.Module):
    """
    Teacher merges phone+watch inertial + skeleton (variable-len => mask).
    """
    def __init__(self,
                 skeleton_hidden=128,
                 watch_inertial_hidden=128,
                 skeleton_heads=4,
                 skeleton_layers=2,
                 watch_inert_heads=4,
                 watch_inert_layers=2,
                 fusion_dim=256,
                 dropout=0.3,
                 orientation_rep="quat",
                 num_classes=2):
        super().__init__()
        # skeleton => shape (B, T_sk, 1+96)
        self.skel_in = nn.Linear(1+96, skeleton_hidden)
        in_dim = 8 if orientation_rep=="quat" else 7
        self.inert_in= nn.Linear(in_dim, watch_inertial_hidden)

        sk_layer= nn.TransformerEncoderLayer(
            d_model=skeleton_hidden, nhead=skeleton_heads,
            dropout=dropout, batch_first=True
        )
        self.skel_encoder= nn.TransformerEncoder(sk_layer, skeleton_layers)

        in_layer= nn.TransformerEncoderLayer(
            d_model=watch_inertial_hidden, nhead=watch_inert_heads,
            dropout=dropout, batch_first=True
        )
        self.inert_encoder= nn.TransformerEncoder(in_layer, watch_inert_layers)

        self.fuse = nn.Linear(skeleton_hidden + watch_inertial_hidden, fusion_dim)
        self.dropout= nn.Dropout(dropout)
        self.head= nn.Linear(fusion_dim, num_classes)

    def forward(self, teacher_inert, teacher_inert_mask,
                skeleton, skeleton_mask):
        # teacher_inert => (B, T_t, in_dim), teacher_inert_mask => (B, T_t)
        # skeleton => (B, T_sk, 1+96), skeleton_mask => (B, T_sk)
        sk_proj = self.skel_in(skeleton)
        in_proj = self.inert_in(teacher_inert)

        # debug prints
        # print(f"DEBUG: Teacher forward => teacher_inert={teacher_inert.shape}, skeleton={skeleton.shape}")

        sk_enc = self.skel_encoder(sk_proj, src_key_padding_mask=skeleton_mask)
        in_enc = self.inert_encoder(in_proj, src_key_padding_mask=teacher_inert_mask)

        sk_feat= sk_enc.mean(dim=1)
        in_feat= in_enc.mean(dim=1)
        fused  = torch.cat([sk_feat, in_feat], dim=-1)
        fused  = self.fuse(fused)
        fused  = self.dropout(fused)
        logits = self.head(fused)

        return {
            "logits": logits,
            "skel_feat": sk_feat,
            "inert_feat": in_feat,
            "fused_feat": fused
        }

