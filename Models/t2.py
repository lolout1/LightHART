import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedTeacherModel(nn.Module):
    def __init__(
        self,
        num_joints=32,
        in_chans=3,
        acc_coords=4,
        spatial_embed=128,
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()
        self.spatial_embed = spatial_embed
        self.num_joints = num_joints
        self.in_chans = in_chans

        # Skeleton embedding
        self.skeleton_embed = nn.Sequential(
            nn.Conv1d(in_chans, spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_embed),
            nn.GELU(),
            nn.Conv1d(spatial_embed, spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_embed),
            nn.GELU()
        )

        # Accelerometer embedding
        self.acc_embed = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Linear(spatial_embed, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU()
        )

        # Cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Transformer encoder
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

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed),
            nn.Linear(spatial_embed, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, acc_data, skl_data):
        batch_size, seq_length = acc_data.shape[:2]
        
        # Process skeleton data
        skl_data = skl_data.view(batch_size * seq_length, self.num_joints, self.in_chans)
        joint_data = skl_data.transpose(1, 2)
        skl_embedded = self.skeleton_embed(joint_data)
        skl_embedded = skl_embedded.mean(dim=2)
        skl_embedded = skl_embedded.view(batch_size, seq_length, -1)

        # Process accelerometer data
        acc_embedded = self.acc_embed(acc_data)

        # Cross-modal fusion
        fused_features, _ = self.cross_attention(skl_embedded, acc_embedded, acc_embedded)
        
        # Temporal modeling
        transformer_output = self.transformer_encoder(fused_features)
        
        # Global pooling and classification
        pooled_output = transformer_output.mean(dim=1)
        logits = self.classifier(pooled_output)
        
        return logits
