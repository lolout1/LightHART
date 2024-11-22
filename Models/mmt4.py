import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalTeacherModel(nn.Module):
    def __init__(
        self,
        num_joints=32,
        in_chans=3,
        acc_coords=4,  # x, y, z, smv
        spatial_embed=256,
        num_heads=8,
        depth=8,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.2
    ):
        super().__init__()
        self.spatial_embed = spatial_embed
        self.num_joints = num_joints
        self.in_chans = in_chans

        # Skeleton Embedding
        self.skeleton_embed = nn.Sequential(
            nn.Conv1d(in_chans, spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_embed),
            nn.GELU(),
            nn.Conv1d(spatial_embed, spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_embed),
            nn.GELU()
        )

        # Joint Attention
        self.joint_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Accelerometer Embedding for Watch
        self.acc_embed_watch = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_embed, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU()
        )

        # Accelerometer Embedding for Phone
        self.acc_embed_phone = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_embed, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU()
        )

        # Accelerometer Fusion Layer
        self.acc_fusion_layer = nn.Sequential(
            nn.Linear(2 * spatial_embed, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Cross-Modality Fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed,
            nhead=num_heads,
            dim_feedforward=spatial_embed * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(spatial_embed)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed),
            nn.Linear(spatial_embed, spatial_embed // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_embed // 2, num_classes)
        )

    def forward(self, data):
        acc_data_watch = data['watch']  # [B, T, 4]
        acc_data_phone = data['phone']  # [B, T, 4]
        skl_data = data['skeleton']     # [B, T, num_joints, in_chans]

        batch_size, seq_length, num_joints, in_chans = skl_data.size()

        # Process Skeleton Data
        skl_data = skl_data.view(batch_size * seq_length, num_joints, in_chans)
        joint_embeddings = []

        for i in range(self.num_joints):
            joint_data = skl_data[:, i, :]
            joint_data = joint_data.unsqueeze(-1)  # [B*T, in_chans, 1]
            joint_embed = self.skeleton_embed(joint_data)  # [B*T, spatial_embed, 1]
            joint_embed = joint_embed.squeeze(-1)  # [B*T, spatial_embed]
            joint_embeddings.append(joint_embed)

        # Stack and process joint relationships
        skl_embedded = torch.stack(joint_embeddings, dim=1)  # [B*T, num_joints, spatial_embed]

        # Joint Attention
        skl_embedded, _ = self.joint_attention(
            skl_embedded, skl_embedded, skl_embedded
        )

        # Mean pool joints and reshape
        skl_embedded = skl_embedded.mean(dim=1)  # [B*T, spatial_embed]
        skl_embedded = skl_embedded.view(batch_size, seq_length, -1)  # [B, T, spatial_embed]

        # Process Accelerometer Data
        acc_embedded_watch = self.acc_embed_watch(acc_data_watch)  # [B, T, spatial_embed]
        acc_embedded_phone = self.acc_embed_phone(acc_data_phone)  # [B, T, spatial_embed]

        # Combine accelerometer embeddings
        acc_embedded = torch.cat([acc_embedded_watch, acc_embedded_phone], dim=-1)  # [B, T, 2*spatial_embed]
        acc_embedded = self.acc_fusion_layer(acc_embedded)  # [B, T, spatial_embed]

        # Cross-modal Fusion
        # Concatenate modalities along the time dimension
        combined_features = torch.cat([skl_embedded, acc_embedded], dim=1)  # [B, 2*T, spatial_embed]
        fused_features, _ = self.cross_attention(
            combined_features, combined_features, combined_features
        )

        # Temporal Modeling
        temporal_features = self.transformer_encoder(fused_features)

        # Global Pooling and Classification
        pooled_features = temporal_features.mean(dim=1)
        logits = self.classifier(pooled_features)

        return logits
