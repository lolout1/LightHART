import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .model_utils import Block

class MMTransformer(nn.Module):
    def __init__(
        self,
        device='cuda',
        mocap_frames=128,
        acc_frames=128,
        num_joints=32,
        in_chans=3,
        acc_coords=4,
        spatial_embed=128,  # Ensure spatial_embed is 128
        sdepth=3,
        adepth=3,
        tdepth=4,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        embed_type='conv',
        drop_rate=0.4,
        attn_drop_rate=0.3,
        drop_path_rate=0.4,
        norm_layer=None,
        num_classes=2,
        distill=False  # Enable distillation feature
    ):
        super().__init__()
        print("\nInitializing Enhanced Teacher Model with Knowledge Distillation Capability")
        
        self.device = device
        self.spatial_embed = spatial_embed
        self.acc_frames = acc_frames
        self.mocap_frames = mocap_frames
        self.distill = distill

        # Accelerometer encoders for xyz and svm (each producing spatial_embed = 128)
        self.acc_xyz_encoder = self._create_acc_encoder(in_chans=3)
        self.svm_encoder = self._create_acc_encoder(in_chans=1)

        # Skeleton encoder
        self.skeleton_embedding = nn.Sequential(
            nn.Conv1d(num_joints * in_chans, spatial_embed, kernel_size=5, padding=2),
            nn.BatchNorm1d(spatial_embed),
            nn.ReLU(),
            nn.Conv1d(spatial_embed, spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_embed),
            nn.GELU()
        )

        # Feature refinement layers
        self.acc_refine = nn.ModuleList([self._create_block(norm_layer) for _ in range(adepth)])
        self.skl_refine = nn.ModuleList([self._create_block(norm_layer) for _ in range(sdepth)])

        # Cross-attention for modality fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim=spatial_embed, num_heads=num_heads, dropout=attn_drop_rate, batch_first=True)
        
        # Position embeddings
        self.register_buffer('acc_pos_embed', self._build_pos_embed(acc_frames, spatial_embed))
        self.register_buffer('skl_pos_embed', self._build_pos_embed(mocap_frames, spatial_embed))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))

        # Transformer blocks and classifier
        self.transformer_blocks = nn.ModuleList([self._create_block(norm_layer) for _ in range(tdepth)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed),
            nn.Linear(spatial_embed, spatial_embed // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(spatial_embed // 2, num_classes)
        )

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _create_acc_encoder(self, in_chans):
        """Creates an encoder for the accelerometer input."""
        return nn.Sequential(
            nn.Conv1d(in_chans, self.spatial_embed, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.spatial_embed),
            nn.ReLU(),
            nn.Conv1d(self.spatial_embed, self.spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.spatial_embed),
            nn.GELU()
        )

    def _create_block(self, norm_layer):
        """Creates a transformer block for feature refinement and fusion."""
        return Block(
            dim=self.spatial_embed,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            drop=0.4,
            attn_drop=0.3,
            drop_path=0.4,
            norm_layer=norm_layer or partial(nn.LayerNorm, eps=1e-6)
        )

    def _build_pos_embed(self, length, dim):
        """Creates positional embeddings."""
        pos = torch.arange(length, dtype=torch.float32).reshape(-1, 1)
        omega = 1. / (10000 ** (2 * torch.arange(dim // 2, dtype=torch.float32) / dim))
        out = torch.cat([torch.sin(pos * omega), torch.cos(pos * omega)], dim=1)
        if dim % 2:
            out = torch.cat([out, torch.zeros_like(pos)], dim=1)
        return out.reshape(1, length, dim)

    def extract_features(self, x, target_len):
        """Ensure sequence length matches target length."""
        B, T, C = x.shape
        if T != target_len:
            x = x.permute(0, 2, 1)  # [B, C, T]
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # [B, T, C]
        return x

    def forward(self, acc_data, skl_data=None):
        B = acc_data.shape[0]
        
        # Process accelerometer data
        acc_xyz = acc_data[:, :, :3].permute(0, 2, 1)
        svm = acc_data[:, :, 3:].permute(0, 2, 1)
        
        # Process accelerometer features independently
        acc_features = self.acc_xyz_encoder(acc_xyz) + self.svm_encoder(svm)
        acc_features = acc_features.permute(0, 2, 1)  # Adjust to match [B, T, C] format
        acc_features = self.extract_features(acc_features, self.acc_frames) + self.acc_pos_embed

        # Process skeleton data if available
        if skl_data is not None:
            skl_data = skl_data.reshape(B, -1, skl_data.shape[1])
            skl_features = self.skeleton_embedding(skl_data).permute(0, 2, 1)
            skl_features = self.extract_features(skl_features, self.mocap_frames) + self.skl_pos_embed

            # Refine features
            for block in self.acc_refine:
                acc_features = block(acc_features)
            for block in self.skl_refine:
                skl_features = block(skl_features)

            # Cross-attention for modality fusion
            combined_features, _ = self.cross_attention(acc_features, skl_features, skl_features)
        else:
            for block in self.acc_refine:
                acc_features = block(acc_features)
            combined_features = acc_features

        # Concatenate CLS token and apply transformer blocks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, combined_features], dim=1)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        final = x[:, 0]  # CLS token
        logits = self.classifier(final)
        
        if self.distill:
            return logits, acc_features, skl_features if skl_data is not None else None
        return logits

    def _init_weights(self, m):
        """Initializes weights of the model layers."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
