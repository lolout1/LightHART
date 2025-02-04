import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processor.base import Time2Vec

class CrossJointTransformer(nn.Module):
    def __init__(self, num_joints=32, num_heads=4, hidden_dim=128):
        super().__init__()
        self.adj_matrix = nn.Parameter(torch.eye(num_joints) * 0.5)
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads//2, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads//2, batch_first=True)

    def forward(self, x):
        B, T, J, C = x.shape
        x_spatial = x.view(B*T, J, C)
        attn_mask = torch.sigmoid(self.adj_matrix).repeat(B*T,1,1)
        spatial_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial, attn_mask=attn_mask)
        spatial_out = spatial_out.view(B, T, J, C)
        temporal_out, _ = self.temporal_attn(spatial_out, spatial_out, spatial_out)
        return temporal_out

class TeacherModel(nn.Module):
    def __init__(self,
                 num_joints=32,
                 joint_dim=3,
                 hidden_skel=128,
                 accel_dim=3,
                 time2vec_dim=8,
                 hidden_accel=64,
                 accel_heads=4,
                 accel_layers=2,
                 skeleton_heads=4,
                 skeleton_layers=1,
                 fusion_hidden=128,
                 num_classes=2,
                 dropout=0.2,
                 dim_feedforward=128,
                 **kwargs):
        super().__init__()

        # Skeleton Branch
        self.skel_embed = nn.Linear(num_joints*joint_dim, hidden_skel)
        self.joint_transformer = CrossJointTransformer(
            num_joints=num_joints,
            num_heads=skeleton_heads,
            hidden_dim=hidden_skel
        )
        
        # Accelerometer Branch 
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        self.accel_proj = nn.Linear(accel_dim + time2vec_dim, hidden_accel)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_accel,
            nhead=accel_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.accel_transformer = nn.TransformerEncoder(enc_layer, num_layers=accel_layers)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_skel + hidden_accel, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, skel_seq, accel_seq, accel_time, accel_mask=None):
        # Skeleton: (B, T, J*3) -> (B, T, J, 3)
        B, T = skel_seq.shape[:2]
        skel_reshaped = skel_seq.view(B, T, 32, 3)
        skel_emb = self.skel_embed(skel_reshaped.flatten(2))
        skel_feat = self.joint_transformer(skel_emb.view(B,T,32,-1)).mean((1,2))
        
        # Accelerometer
        t_emb = self.time2vec(accel_time.unsqueeze(-1)).squeeze(2)
        accel_in = F.gelu(self.accel_proj(torch.cat([accel_seq, t_emb], -1)))
        accel_feat = self.accel_transformer(accel_in, src_key_padding_mask=accel_mask)
        accel_feat = accel_feat.mean(1) if accel_mask is None else \
                     (accel_feat * (~accel_mask).unsqueeze(-1)).sum(1) / (~accel_mask).sum(1, keepdim=True)

        # Fusion
        return self.fusion(torch.cat([skel_feat, accel_feat], -1))
