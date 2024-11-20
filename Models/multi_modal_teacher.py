import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .model_utils import Block

class MultiModalTeacher(nn.Module):
    def __init__(
        self,
        device='cuda',
        mocap_frames=128,
        acc_frames=128,
        num_joints=32,
        in_chans=3,
        num_patch=4,
        acc_coords=4,  # x, y, z, svm for each sensor
        spatial_embed=128,
        sdepth=4,
        adepth=4,
        tdepth=6,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        op_type='all',
        embed_type='lin',
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=None,
        num_classes=2
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Save parameters
        self.spatial_embed = spatial_embed
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.acc_coords = acc_coords  # 4 coordinates: x, y, z, svm
        self.seq_length = acc_frames

        # Skeleton embedding
        if embed_type == 'lin':
            self.Skeleton_embedding = nn.Sequential(
                nn.Linear(num_joints * in_chans, spatial_embed),
                nn.LayerNorm(spatial_embed),
                nn.GELU(),
                nn.Dropout(drop_rate)
            )
        else:
            self.Skeleton_embedding = nn.Sequential(
                nn.Conv1d(in_chans, spatial_embed, kernel_size=1),
                nn.BatchNorm1d(spatial_embed),
                nn.GELU()
            )

        # Phone accelerometer processing
        self.phone_acc_processor = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),  # Process x, y, z, svm
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Watch accelerometer processing
        self.watch_acc_processor = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),  # Process x, y, z, svm
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Fusion module for accelerometers
        self.acc_fusion = nn.Sequential(
            nn.Linear(spatial_embed * 2, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Final fusion module
        self.modality_fusion = nn.Sequential(
            nn.Linear(spatial_embed * 2, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Temporal token and position embedding
        self.temp_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        self.pos_embed = nn.Parameter(torch.zeros(1, acc_frames + 1, spatial_embed))

        # Initialize embeddings
        nn.init.trunc_normal_(self.temp_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=spatial_embed,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer
            )
            for _ in range(tdepth)
        ])

        # Classification head
        self.norm = norm_layer(spatial_embed)
        self.head = nn.Linear(spatial_embed, num_classes)

    def _validate_input(self, tensor, name):
        """Validate input tensor dimensions"""
        if tensor is None:
            raise ValueError(f"{name} is None")
        if tensor.dim() not in [3, 4]:
            raise ValueError(f"Expected {name} to have 3 or 4 dimensions, got {tensor.dim()}")
        return tensor

    def print_shapes(self, prefix, tensor):
        """Debug helper to print tensor shapes"""
        print(f"{prefix}: {tensor.shape}")

    def forward(self, acc_data1, acc_data2, skl_data):
        """
        Forward pass of the model.
        acc_data1: First accelerometer data with shape [B, T, 4] for x, y, z, svm
        acc_data2: Second accelerometer data with shape [B, T, 4] for x, y, z, svm
        skl_data: Skeleton data with shape [B, T, J, C] or [B, T, J*C]
        """
        # Input validation and type conversion
        skl_data = self._validate_input(skl_data, "skl_data").float()
        acc_data1 = self._validate_input(acc_data1, "acc_data1").float()
        acc_data2 = self._validate_input(acc_data2, "acc_data2").float()

        B = skl_data.shape[0]  # batch size
        T = skl_data.shape[1]  # sequence length

        self.print_shapes("skl_data", skl_data)
        self.print_shapes("acc_data1", acc_data1)
        self.print_shapes("acc_data2", acc_data2)

        # Process skeleton data
        if skl_data.dim() == 4:
            skl_data = skl_data.reshape(B, T, self.num_joints * self.in_chans)
        else:
            skl_data = skl_data.view(B, T, self.num_joints * self.in_chans)

        # Get features from each modality
        skl_features = self.Skeleton_embedding(skl_data)  # [B, T, C]
        acc1_features = self.phone_acc_processor(acc_data1)  # [B, T, C]
        acc2_features = self.watch_acc_processor(acc_data2)  # [B, T, C]

        self.print_shapes("skl_features", skl_features)
        self.print_shapes("acc1_features", acc1_features)
        self.print_shapes("acc2_features", acc2_features)

        # Fuse accelerometer features
        acc_features = self.acc_fusion(torch.cat([acc1_features, acc2_features], dim=-1))
        self.print_shapes("fused_acc_features", acc_features)

        # Combine all modalities
        combined = self.modality_fusion(torch.cat([skl_features, acc_features], dim=-1))
        self.print_shapes("combined_features", combined)

        # Add class token
        cls_token = self.temp_token.expand(B, -1, -1)
        x = torch.cat((cls_token, combined), dim=1)
        self.print_shapes("after_cls_token", x)

        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            self.print_shapes(f"after_block_{idx}", x)

        # Classification
        x = self.norm(x[:, 0])  # Use CLS token
        x = self.head(x)
        self.print_shapes("final_output", x)

        return x
