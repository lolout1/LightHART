import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiModalFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Cross-modal attention layers
        self.q_acc = nn.Linear(dim, dim)
        self.k_skl = nn.Linear(dim, dim)
        self.v_skl = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, acc_feat, skl_feat):
        # Layer normalization
        acc_feat = self.norm1(acc_feat)
        skl_feat = self.norm2(skl_feat)

        B, T, _ = acc_feat.shape

        # Multi-head attention
        q = rearrange(self.q_acc(acc_feat), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.k_skl(skl_feat), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.v_skl(skl_feat), 'b t (h d) -> b h t d', h=self.num_heads)

        # Scaled dot-product attention
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Attend to skeleton features
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')

        # Concatenate and fuse features
        fused = self.fusion(torch.cat([acc_feat, out], dim=-1))
        return fused

class TeacherModel(nn.Module):
    """Enhanced teacher model for fall detection, with feature return."""
    def __init__(self, 
                 num_joints=32,
                 in_chans=3,
                 acc_coords=4,
                 hidden_dim=256,
                 num_heads=8,
                 fusion_layers=3,
                 drop_rate=0.1):
        super().__init__()

        # Skeleton processing
        self.joint_embed = nn.Linear(in_chans * num_joints, hidden_dim)
        self.skeleton_pos_embed = nn.Parameter(torch.randn(1, 128, hidden_dim))

        # Accelerometer processing
        self.acc_embed = nn.Linear(acc_coords, hidden_dim)
        self.acc_pos_embed = nn.Parameter(torch.randn(1, 128, hidden_dim))

        # Cross-modal fusion layers
        self.fusion_layers = nn.ModuleList([
            MultiModalFusion(hidden_dim, num_heads) 
            for _ in range(fusion_layers)
        ])

        # Temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Fall detection head
        self.fall_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim // 2, 1)  # final output => single logit
        )

        self.sigmoid = nn.Sigmoid()

    def _compute_magnitude(self, acc_data):
        """Compute acceleration magnitude"""
        return torch.sqrt(torch.sum(acc_data ** 2, dim=-1, keepdim=True))

    def forward(self, acc_data, skeleton_data):
        """
        Return both final probability (teacher_prob) and final feature
        (teacher_feat) for alignment.
        """
        B = acc_data.shape[0]

        # Process accelerometer data (add magnitude as 4th channel)
        magnitude = self._compute_magnitude(acc_data)
        acc_data = torch.cat([acc_data, magnitude], dim=-1)  # [B, T, 4]

        # Embed + position
        acc_feat = self.acc_embed(acc_data)  # [B, T, hidden_dim]
        acc_feat = acc_feat + self.acc_pos_embed[:, :acc_feat.size(1), :]

        # Process skeleton data
        skeleton_data = skeleton_data.view(B, skeleton_data.shape[1], -1)
        skl_feat = self.joint_embed(skeleton_data)
        skl_feat = skl_feat + self.skeleton_pos_embed[:, :skl_feat.size(1), :]

        # Cross-modal fusion
        fused_feat = acc_feat
        for fusion_layer in self.fusion_layers:
            fused_feat = fusion_layer(fused_feat, skl_feat)  # [B, T, hidden_dim]

        # Temporal modeling => [B, hidden_dim, T]
        temp_feat = fused_feat.transpose(1, 2)
        temp_feat = self.temporal_conv(temp_feat)  # [B, hidden_dim, T]

        # Global pooling => final feature
        teacher_feat = F.adaptive_avg_pool1d(temp_feat, 1).squeeze(-1)  # [B, hidden_dim]

        # Classifier => final logit => teacher_prob
        logit = self.fall_detector(teacher_feat).squeeze(-1)  # [B]
        teacher_prob = self.sigmoid(logit)

        # Return final probability + last feature
        return teacher_prob, teacher_feat
