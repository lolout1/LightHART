import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalStudentModel(nn.Module):
    def __init__(
        self,
        acc_coords=3,  # x, y, z
        spatial_embed=256,
        num_heads=8,
        depth=8,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.2
    ):
        super().__init__()

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
        acc_data_phone = data['phone']

        # Process Accelerometer Data
        acc_embedded_watch = self.acc_embed_watch(acc_data_watch)  # [B, T, C]
        acc_embedded_phone = self.acc_embed_phone(acc_data_phone)  # [B, T, C]

        # Combine accelerometer embeddings
        acc_embedded = torch.cat([acc_embedded_watch, acc_embedded_phone], dim=-1)  # [B, T, 2*C]
        acc_embedded = self.acc_fusion_layer(acc_embedded)  # [B, T, C]

        # Temporal Modeling
        temporal_features = self.transformer_encoder(acc_embedded)

        # Global Pooling and Classification
        pooled_features = temporal_features.mean(dim=1)
        logits = self.classifier(pooled_features)

        return logits
