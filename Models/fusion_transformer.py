import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionTransModel(nn.Module):
    """
    Fusion Transformer model for fall detection that fuses inertial sensor data.
    Supports configurable use of extra fusion features for teacher models.
    """
    def __init__(self, num_layers=3, embed_dim=32, acc_coords=3, quat_coords=4,
                 num_classes=2, acc_frames=64, mocap_frames=64, num_heads=4,
                 fusion_type='concat', dropout=0.3, use_batch_norm=True,
                 feature_dim=64, use_features=True):
        super(FusionTransModel, self).__init__()

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        self.quat_coords = quat_coords
        self.acc_frames = acc_frames
        self.mocap_frames = mocap_frames
        self.fusion_type = fusion_type
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.use_features = use_features
        self.use_batch_norm = use_batch_norm

        # Embedding layers for inertial modalities
        self.acc_embed = nn.Linear(acc_coords, embed_dim)
        self.quat_embed = nn.Linear(quat_coords, embed_dim)

        # Optional feature embedding (for teacher model)
        if self.use_features:
            self.feature_embedding = nn.Sequential(
                nn.Linear(43, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # Position encoding for each modality
        self.acc_pos_encoder = nn.Parameter(torch.zeros(1, acc_frames, embed_dim))
        self.quat_pos_encoder = nn.Parameter(torch.zeros(1, mocap_frames, embed_dim))
        nn.init.trunc_normal_(self.acc_pos_encoder, std=0.02)
        nn.init.trunc_normal_(self.quat_pos_encoder, std=0.02)

        # Transformer encoders for accelerometer and quaternion streams
        encoder_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.acc_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.quat_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        # Attention pooling layers
        self.acc_attention = nn.Linear(embed_dim, 1)
        self.quat_attention = nn.Linear(embed_dim, 1)

        # Determine fusion output dimension
        if self.fusion_type == 'concat':
            fusion_dim = embed_dim * 2 + (feature_dim if self.use_features else 0)
        elif self.fusion_type == 'sum':
            fusion_dim = embed_dim + (feature_dim if self.use_features else 0)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )

        # Batch normalization layers (applied along the channel dimension)
        if self.use_batch_norm:
            # For accelerometer, BN is applied over channels (3) not over frame length
            self.acc_bn = nn.BatchNorm1d(acc_coords)
            # For quaternion, use 4 channels
            self.quat_bn = nn.BatchNorm1d(quat_coords)
            self.fusion_bn = nn.BatchNorm1d(fusion_dim)

    def forward(self, data_dict):
        """
        Expects a dictionary with keys:
          - 'acc': [batch, frames, 3]
          - 'quat': [batch, frames, 4]
          - 'features': [batch, 43] (if use_features is True; otherwise may be omitted)
        """
        acc = data_dict['acc']      # shape: [B, T_acc, 3]
        quat = data_dict['quat']    # shape: [B, T_quat, 4]
        features = data_dict.get('features', None)

        # Apply BN over channel dimension if enabled and if batch size > 1
        if self.use_batch_norm:
            if acc.size(0) > 1:
                acc = self.acc_bn(acc.transpose(1, 2)).transpose(1, 2)
            if quat.size(0) > 1:
                quat = self.quat_bn(quat.transpose(1, 2)).transpose(1, 2)

        # Embedding + positional encoding
        acc_embedded = self.acc_embed(acc) + self.acc_pos_encoder  # [B, T_acc, embed_dim]
        quat_embedded = self.quat_embed(quat) + self.quat_pos_encoder  # [B, T_quat, embed_dim]

        # Transformer encoding
        acc_encoded = self.acc_encoder(acc_embedded)
        quat_encoded = self.quat_encoder(quat_embedded)

        # Attention pooling across time
        acc_attn = F.softmax(self.acc_attention(acc_encoded), dim=1)
        quat_attn = F.softmax(self.quat_attention(quat_encoded), dim=1)
        acc_pooled = torch.sum(acc_encoded * acc_attn, dim=1)  # [B, embed_dim]
        quat_pooled = torch.sum(quat_encoded * quat_attn, dim=1)  # [B, embed_dim]

        # Fusion: optionally include extra features if provided
        if self.use_features and (features is not None):
            features_embedded = self.feature_embedding(features)
            if self.fusion_type == 'concat':
                fused = torch.cat([acc_pooled, quat_pooled, features_embedded], dim=1)
            elif self.fusion_type == 'sum':
                fused = acc_pooled + quat_pooled + features_embedded
        else:
            if self.fusion_type == 'concat':
                fused = torch.cat([acc_pooled, quat_pooled], dim=1)
            elif self.fusion_type == 'sum':
                fused = acc_pooled + quat_pooled

        # Apply BN on fused features if enabled and if batch size > 1
        if self.use_batch_norm:
            if fused.size(0) > 1:
                fused = self.fusion_bn(fused)

        logits = self.classifier(fused)
        return logits

