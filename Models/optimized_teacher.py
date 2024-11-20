import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .model_utils import Block

class OptimizedTeacher(nn.Module):
    def __init__(
        self,
        device='cuda',
        mocap_frames=128,
        acc_frames=128,
        num_joints=32,
        in_chans=3,
        num_patch=4,
        acc_coords=4,
        spatial_embed=64,
        sdepth=2,
        adepth=2,
        tdepth=2,
        num_heads=2,
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        op_type='all',
        embed_type='lin',
        drop_rate=0.2,
        attn_drop_rate=0.2,
        drop_path_rate=0.2,
        norm_layer=None,
        num_classes=2
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        temp_embed = spatial_embed
        acc_embed = temp_embed
        self.acc_embed = acc_embed
        self.num_patch = num_patch
        self.mocap_frames = mocap_frames
        self.skl_patch_size = mocap_frames // num_patch
        self.acc_patch_size = acc_frames // num_patch
        self.temp_frames = mocap_frames
        self.op_type = op_type
        self.embed_type = embed_type
        self.sdepth = sdepth
        self.adepth = adepth
        self.tdepth = tdepth
        self.num_joints = num_joints
        self.joint_coords = in_chans
        self.acc_frames = acc_frames
        self.acc_coords = acc_coords
        self.skl_encoder_size = temp_embed
        
        # Initialize with original effective architecture
        if self.embed_type == 'lin':
            self.Skeleton_embedding = nn.Sequential(
                nn.Linear(num_joints * in_chans, spatial_embed),
                norm_layer(spatial_embed),
                nn.GELU(),
                nn.Dropout(drop_rate)
            )
            self.Accelerometer_embedding = nn.Sequential(
                nn.Linear(acc_coords, acc_embed),
                norm_layer(acc_embed),
                nn.GELU(),
                nn.Dropout(drop_rate)
            )
        else:
            self.Skeleton_embedding = nn.Sequential(
                nn.Conv1d(in_chans, spatial_embed, kernel_size=1),
                nn.BatchNorm1d(spatial_embed),
                nn.GELU()
            )
            self.Accelerometer_embedding = nn.Sequential(
                nn.Conv1d(acc_coords, acc_embed, kernel_size=1),
                nn.BatchNorm1d(acc_embed),
                nn.GELU()
            )

        # Fall attention (lightweight addition)
        self.fall_attention = nn.Sequential(
            nn.Linear(acc_embed, acc_embed),
            nn.Sigmoid()
        )
        
        # Original position embeddings
        self.temp_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        
        # Original transformer blocks
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.tdepth)]
        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=temp_embed,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=tdpr[i],
                norm_layer=norm_layer
            )
            for i in range(self.tdepth)
        ])
        
        # Original classification head
        self.class_head = nn.Sequential(
            norm_layer(temp_embed),
            nn.Linear(temp_embed, num_classes)
        )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.temp_token, std=0.02)
        nn.init.trunc_normal_(self.Temporal_pos_embed, std=0.02)
        
    def forward(self, acc_data, skl_data):
        batch_size = skl_data.size(0)
        seq_length = skl_data.size(1)
        
        # Process skeleton data (keeping original logic)
        skl_data = skl_data.view(batch_size, seq_length, -1)
        skl_embedded = self.Skeleton_embedding(skl_data)
        
        # Process accelerometer data (keeping original logic)
        if acc_data.dim() == 2:
            acc_data = acc_data.unsqueeze(-1)
        elif acc_data.dim() == 1:
            acc_data = acc_data.unsqueeze(-1).unsqueeze(-1)
        elif acc_data.dim() > 3:
            acc_data = acc_data.view(batch_size, seq_length, -1)
            
        acc_embedded = self.Accelerometer_embedding(acc_data)
        
        # Lightweight fall attention
        if self.training:
            acc_embedded = acc_embedded * self.fall_attention(acc_embedded)
            self.features = []
        
        # Original fusion
        combined = skl_embedded + acc_embedded
        
        # Original positional encoding
        class_token = self.temp_token.expand(batch_size, -1, -1)
        combined = torch.cat((class_token, combined), dim=1)
        combined = combined + self.Temporal_pos_embed
        combined = self.pos_drop(combined)
        
        # Original transformer processing
        for blk in self.Temporal_blocks:
            combined = blk(combined)
            if self.training:
                self.features.append(combined)
        
        # Original classification
        cls_token_final = combined[:, 0]
        logits = self.class_head(cls_token_final)
        
        return logits
    
    def get_features(self):
        return self.features
