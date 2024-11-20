import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from functools import partial

class OptimizedTeacherModel(nn.Module):
    def __init__(
        self,
        device='cuda',
        mocap_frames=128,
        acc_frames=128,
        num_joints=32,
        in_chans=3,  # xyz for skeleton
        acc_coords=4,  # xyz + svm for accelerometer
        spatial_embed=128,
        num_heads=8,
        sdepth=4,
        tdepth=6,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=None,
        num_classes=2,
    ):
        super().__init__()
        print("\nInitializing Enhanced Teacher Model")
        
        # Save important parameters
        self.spatial_embed = spatial_embed
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.acc_coords = acc_coords
        self.mocap_frames = mocap_frames
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # Skeleton joint embedding
        self.joint_embed = nn.Sequential(
            nn.Linear(in_chans, spatial_embed),
            norm_layer(spatial_embed),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Skeleton spatial encoder
        self.skeleton_encoder = nn.Sequential(
            nn.Linear(num_joints * spatial_embed, spatial_embed),
            norm_layer(spatial_embed),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Enhanced accelerometer processing
        self.acc_encoder = nn.Sequential(
            # First process xyz
            nn.Linear(3, spatial_embed // 2),
            norm_layer(spatial_embed // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            
            # Process SVM and combine
            nn.Linear(spatial_embed // 2 + 1, spatial_embed),  # +1 for SVM
            norm_layer(spatial_embed),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Motion attention for better temporal dynamics
        self.motion_attention = nn.Sequential(
            nn.Linear(spatial_embed, spatial_embed // 2),
            nn.GELU(),
            nn.Linear(spatial_embed // 2, spatial_embed),
            nn.Sigmoid()
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            spatial_embed, num_heads, dropout=attn_drop_rate, batch_first=True
        )

        # Advanced fusion module
        self.fusion_gate = nn.Sequential(
            nn.Linear(spatial_embed * 2, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(spatial_embed, spatial_embed),
            nn.Sigmoid()
        )

        # Position embeddings and tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        self.pos_embed = nn.Parameter(torch.zeros(1, mocap_frames + 1, spatial_embed))

        # Transformer blocks for temporal modeling
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=spatial_embed,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(tdepth)
        ])

        # Enhanced classifier
        self.classifier = nn.Sequential(
            norm_layer(spatial_embed),
            nn.Linear(spatial_embed, spatial_embed // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(spatial_embed // 2, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, acc_data, skl_data):
        """
        Args:
            acc_data: tensor of shape [batch_size, frames, 4] (xyz + svm)
            skl_data: tensor of shape [batch_size, frames, num_joints, 3] (xyz)
        """
        B = acc_data.shape[0]
        
        # Process skeleton data
        skl_reshaped = skl_data.view(B, -1, self.num_joints, 3)
        joint_features = []
        for i in range(self.num_joints):
            joint_data = skl_reshaped[:, :, i, :]
            joint_embed = self.joint_embed(joint_data)
            joint_features.append(joint_embed)
        
        joint_features = torch.cat(joint_features, dim=-1)
        skl_features = self.skeleton_encoder(joint_features)

        # Process accelerometer data
        acc_xyz = acc_data[:, :, :3]  # First 3 channels are xyz
        svm = acc_data[:, :, 3:4]     # Last channel is SVM
        
        # Process XYZ coordinates
        acc_features = self.acc_encoder[0:4](acc_xyz)
        # Concatenate SVM and process
        acc_features = torch.cat([acc_features, svm], dim=-1)
        acc_features = self.acc_encoder[4:](acc_features)

        # Apply motion attention
        motion_weights = self.motion_attention(acc_features)
        acc_features = acc_features * motion_weights

        # Cross-modal attention
        cross_features = self.cross_attention(
            skl_features, acc_features, acc_features
        )[0]

        # Fusion with gating
        fusion_weights = self.fusion_gate(torch.cat([cross_features, skl_features], dim=-1))
        fused_features = fusion_weights * cross_features + (1 - fusion_weights) * skl_features

        # Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, fused_features], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]

        # Transformer blocks with residual connections
        for block in self.transformer_blocks:
            x = block(x)

        # Classification using CLS token
        x = x[:, 0]
        logits = self.classifier(x)

        return logits

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
