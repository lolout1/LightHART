import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusionTeacherModel(nn.Module):
    def __init__(
        self,
        device='cuda',
        mocap_frames=128,
        acc_frames=128,
        num_joints=32,
        in_chans=3,  # xyz for skeleton
        acc_coords=4,  # xyz + svm for accelerometer
        spatial_embed=64,
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        num_classes=2,
        distill=False,
        temperature=4.0  # Temperature scaling for soft targets
    ):
        super().__init__()

        self.device = device
        self.temperature = temperature
        self.spatial_embed = spatial_embed
        self.num_joints = num_joints
        self.distill = distill
        self.mocap_frames = mocap_frames

        # Skeleton embedding with convolutional encoding
        self.skeleton_conv = nn.Sequential(
            nn.Conv1d(in_chans, spatial_embed, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(spatial_embed, spatial_embed, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Skeleton normalization
        self.skeleton_norm = nn.LayerNorm(spatial_embed)

        # Accelerometer embedding with 4 features (xyz + svm)
        self.acc_embed = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Cross-modal attention layer
        self.cross_attention = nn.MultiheadAttention(
            spatial_embed, num_heads, dropout=attn_drop_rate, batch_first=True
        )

        # Temporal transformer for fused features
        self.temporal_transformer = nn.ModuleList([
            TransformerBlock(
                dim=spatial_embed,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
            ) for _ in range(depth)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed),
            nn.Linear(spatial_embed, num_classes)
        )

        # Distillation head (if applicable)
        if self.distill:
            self.distill_head = nn.Sequential(
                nn.LayerNorm(spatial_embed),
                nn.Linear(spatial_embed, num_classes)
            )

    def forward(self, acc_data, skl_data=None):
        B = acc_data.shape[0]
        
        # Accelerometer embedding
        acc_features = self.acc_embed(acc_data)

        if skl_data is not None:
            # Process each joint independently
            skl_embeds = []
            for i in range(self.num_joints):
                joint_data = skl_data[:, :, i, :]  # Extract data for the i-th joint (B, T, C)
                joint_embeds = self.skeleton_conv(joint_data.permute(0, 2, 1))  # Conv1D expects (B, C, T)
                joint_embeds = joint_embeds.permute(0, 2, 1)  # Restore shape to (B, T, spatial_embed)
                skl_embeds.append(joint_embeds)
            
            # Concatenate all joint embeddings along the frame dimension and reshape to [B, mocap_frames, spatial_embed]
                skl_embeds = torch.cat(skl_embeds, dim=-1)  # Shape should now be [B, T, num_joints * spatial_embed]

                # Update the reshape dimensions based on concatenated size
                skl_embeds = skl_embeds.reshape(B, self.mocap_frames, -1)  # Use -1 to infer the correct size for spatial embedding

                # Apply layer normalization to the reshaped embeddings
                skl_embeds = self.skeleton_norm(skl_embeds)
            # Cross-modal attention between skeleton and accelerometer features
            cross_features = self.cross_attention(skl_embeds, acc_features_exp, acc_features_exp)[0]

            fused_features = cross_features
        else:
            fused_features = acc_features

        # Temporal processing with Transformer
        for block in self.temporal_transformer:
            fused_features = block(fused_features)

        # Classification head output with temperature scaling
        logits = self.classifier(fused_features.mean(dim=1)) / self.temperature

        if self.distill and skl_data is not None:
            distill_logits = self.distill_head(fused_features.mean(dim=1)) / self.temperature
            return logits, distill_logits

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
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
