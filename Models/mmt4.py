# models.py

import torch
import torch.nn as nn
from einops import rearrange
import random

class RandomApplyCustom(nn.Module):
    def __init__(self, module, p=0.5):
        super(RandomApplyCustom, self).__init__()
        self.module = module
        self.p = p

    def forward(self, x):
        if self.training and random.random() < self.p:
            return self.module(x)
        return x

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers.
    """
    def __init__(self, in_channels, out_channels, dropout):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        return out

class EnhancedTeacherModel(nn.Module):
    def __init__(
        self,
        num_joints=32,
        in_chans=3,            # Skeleton data channels (x, y, z)
        acc_coords=4,          # Accelerometer data channels (x, y, z, SMV)
        spatial_embed=256,
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.1,
        use_skeleton=True      # Ensuring skeleton data is used
    ):
        super(EnhancedTeacherModel, self).__init__()
        self.use_skeleton = use_skeleton
        self.num_joints = num_joints
        self.in_chans = in_chans

        # =====================
        # Feature Extractor for Phone with Residual Blocks
        # =====================
        self.phone_conv = nn.Sequential(
            ResidualBlock(acc_coords, 64, dropout),
            ResidualBlock(64, 128, dropout),
            ResidualBlock(128, 256, dropout),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.phone_fc = nn.Sequential(
            nn.Linear(256 * 64, spatial_embed),  # Assuming T=128 -> T/2=64
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =====================
        # Feature Extractor for Watch with Residual Blocks
        # =====================
        self.watch_conv = nn.Sequential(
            ResidualBlock(acc_coords, 64, dropout),
            ResidualBlock(64, 128, dropout),
            ResidualBlock(128, 256, dropout),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.watch_fc = nn.Sequential(
            nn.Linear(256 * 64, spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =====================
        # Skeleton Embedding with Residual Blocks and Attention
        # =====================
        if self.use_skeleton:
            self.skeleton_conv = nn.Sequential(
                nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                ResidualBlock(64, 128, dropout),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(dropout)
            )
            self.skeleton_fc = nn.Sequential(
                nn.Linear(128 * 16 * 16, spatial_embed),  # Adjust based on skeleton data dimensions
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Inter-joint Attention to capture skeletal relationships
            self.joint_attention = nn.MultiheadAttention(
                embed_dim=spatial_embed,
                num_heads=4,  # Fewer heads for inter-joint attention
                dropout=dropout,
                batch_first=True
            )

        # =====================
        # Fusion Layers with Cross-Modality Attention
        # =====================
        fusion_input_size = 2 * spatial_embed
        if self.use_skeleton:
            fusion_input_size += spatial_embed

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =====================
        # Temporal Transformer Encoder
        # =====================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed,
            nhead=num_heads,
            dim_feedforward=spatial_embed * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(spatial_embed)
        )

        # =====================
        # Classification Head
        # =====================
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed),
            nn.Linear(spatial_embed, spatial_embed // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_embed // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for convolutional and linear layers
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                # Residual blocks already handle their own initialization
                continue
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, acc_phone, acc_watch, skeleton_data):
        """
        Args:
            acc_phone (torch.Tensor): [B, T, C=4] tensor (x, y, z, SMV)
            acc_watch (torch.Tensor): [B, T, C=4] tensor (x, y, z, SMV)
            skeleton_data (torch.Tensor, optional): [B, T, J=32, C=3] tensor (x, y, z)
        Returns:
            logits (torch.Tensor): [B, num_classes]
        """
        # =====================
        # Feature Extraction
        # =====================
        # Phone
        phone_feat = rearrange(acc_phone, 'b t c -> b c t')  # [B, C=4, T=128]
        phone_feat = self.phone_conv(phone_feat)             # [B, 256, T/2=64]
        phone_feat = phone_feat.view(phone_feat.size(0), -1) # Flatten [B, 256*64=16384]
        phone_feat = self.phone_fc(phone_feat)               # [B, spatial_embed=256]

        # Watch
        watch_feat = rearrange(acc_watch, 'b t c -> b c t')  # [B, C=4, T=128]
        watch_feat = self.watch_conv(watch_feat)              # [B, 256, T/2=64]
        watch_feat = watch_feat.view(watch_feat.size(0), -1) # Flatten [B, 256*64=16384]
        watch_feat = self.watch_fc(watch_feat)               # [B, spatial_embed=256]

        # Skeleton (if used)
        if self.use_skeleton and skeleton_data is not None:
            skeleton_feat = rearrange(skeleton_data, 'b t j c -> b c j t')  # [B, C=3, J=32, T=128]
            skeleton_feat = self.skeleton_conv(skeleton_feat)             # [B, 128, J/2=16, T/2=64]
            skeleton_feat = skeleton_feat.view(skeleton_feat.size(0), -1)  # Flatten [B, 128*16*64=131072]
            skeleton_feat = self.skeleton_fc(skeleton_feat)               # [B, spatial_embed=256]

            # Inter-joint Attention
            # Reshape to [B, J, C]
            skl_embedded = skeleton_feat.view(skeleton_feat.size(0), self.num_joints, self.spatial_embed)  # [B, J=32, 256]
            skl_embedded, _ = self.joint_attention(skl_embedded, skl_embedded, skl_embedded)  # [B, J=32, 256]
            skl_embedded = skl_embedded.view(skl_embedded.size(0), -1)  # [B, 32*256=8192]

            # Optionally, you can apply further processing to skl_embedded
            skl_embedded = self.skeleton_fc(skl_embedded)  # [B, spatial_embed=256]
        else:
            skl_embedded = None

        # =====================
        # Fusion
        # =====================
        if self.use_skeleton and skl_embedded is not None:
            fused = torch.cat([phone_feat, watch_feat, skl_embedded], dim=1)  # [B, 768]
        else:
            fused = torch.cat([phone_feat, watch_feat], dim=1)                # [B, 512]

        fused = self.fusion_layer(fused)                                     # [B, 256]

        # =====================
        # Cross-Modality Attention
        # =====================
        # Adding a dummy temporal dimension since Transformer expects [B, T, C]
        fused = fused.unsqueeze(1)                                           # [B, 1, 256]

        fused, _ = self.cross_attention(fused, fused, fused)                # [B, 1, 256]

        # =====================
        # Temporal Modeling
        # =====================
        temporal_features = self.transformer_encoder(fused)                  # [B, 1, 256]

        # =====================
        # Classification
        # =====================
        pooled_features = temporal_features.mean(dim=1)                     # [B, 256]
        logits = self.classifier(pooled_features)                           # [B, num_classes=2]

        return logits