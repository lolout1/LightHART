import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiModalFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Cross-modal attention
        self.q_acc = nn.Linear(dim, dim)
        self.k_skl = nn.Linear(dim, dim)
        self.v_skl = nn.Linear(dim, dim)
        
        # Layernorm for stability
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Fusion with residual gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, acc_feat, skl_feat):
        # Apply normalization
        acc_feat = self.norm1(acc_feat)
        skl_feat = self.norm2(skl_feat)
        
        # Multi-head attention
        q = rearrange(self.q_acc(acc_feat), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.k_skl(skl_feat), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.v_skl(skl_feat), 'b t (h d) -> b h t d', h=self.num_heads)
        
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        attn = F.dropout(F.softmax(attn, dim=-1), p=0.1)
        
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        # Gated fusion
        gate = self.fusion_gate(torch.cat([acc_feat, out], dim=-1))
        return acc_feat + gate * out

class EnhancedTeacherModel(nn.Module):
    def __init__(self, 
                 num_joints=32,
                 in_chans=3,
                 acc_coords=3,
                 hidden_dim=128,
                 num_heads=4,
                 fusion_layers=2,
                 drop_rate=0.3):
        super().__init__()
        
        # Input embeddings
        self.joint_embed = nn.Sequential(
            nn.Linear(in_chans * num_joints, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        
        self.acc_embed = nn.Sequential(
            nn.Linear(acc_coords + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            MultiModalFusion(hidden_dim, num_heads)
            for _ in range(fusion_layers)
        ])
        
        # Temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=8),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        
        # Global Average Pooling (channels_first)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Final Dense layer with zero initialization
        self.classifier = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, acc_data, skeleton_data):
        # Compute features
        extra_features = self.compute_features(acc_data)
        acc_data = torch.cat([acc_data, extra_features], dim=-1)
        
        # Process inputs
        acc_feat = self.acc_embed(acc_data)
        skl_feat = self.joint_embed(skeleton_data.reshape(skeleton_data.shape[0], 
                                                       skeleton_data.shape[1], -1))
        
        # Fusion
        fused_feat = acc_feat
        for fusion_layer in self.fusion_layers:
            fused_feat = fusion_layer(fused_feat, skl_feat)
        
        # Temporal (channels_first format)
        temp_feat = self.temporal_conv(fused_feat.transpose(1, 2))
        
        # Global Average Pooling
        x = self.pool(temp_feat).squeeze(-1)  # Equivalent to GlobalAveragePooling1D
        
        # Final Dense layer with sigmoid
        output = self.sigmoid(self.classifier(x))
        
        return output.squeeze(-1)

    def compute_features(self, acc_data):
        # Signal magnitude vector
        magnitude = torch.sqrt(torch.sum(acc_data ** 2, dim=-1, keepdim=True))
        
        # Jerk (acceleration derivative)
        jerk = torch.diff(acc_data, dim=1, prepend=acc_data[:, :1, :])
        jerk_magnitude = torch.sqrt(torch.sum(jerk ** 2, dim=-1, keepdim=True))
        
        return torch.cat([magnitude, jerk_magnitude], dim=-1)