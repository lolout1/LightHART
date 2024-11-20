import torch
import torch.nn as nn

class FallDetectionTeacherModel(nn.Module):
    def __init__(
        self,
        num_joints=32,
        in_chans=3,
        acc_coords=4,  # x, y, z, SMV
        spatial_embed=256,
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.5,
    ):
        super().__init__()

        self.spatial_embed = spatial_embed

        # Skeleton embedding
        self.skeleton_embed = nn.Sequential(
            nn.Conv1d(in_channels=in_chans, out_channels=spatial_embed, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=spatial_embed, out_channels=spatial_embed, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Accelerometer embedding for watch and phone
        self.watch_acc_embed = nn.Sequential(
            nn.Conv1d(in_channels=acc_coords, out_channels=spatial_embed, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=spatial_embed, out_channels=spatial_embed, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.phone_acc_embed = nn.Sequential(
            nn.Conv1d(in_channels=acc_coords, out_channels=spatial_embed, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=spatial_embed, out_channels=spatial_embed, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed,
            nhead=num_heads,
            dim_feedforward=spatial_embed * mlp_ratio,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(spatial_embed, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, watch_acc_data, phone_acc_data, skl_data):
        """
        watch_acc_data & phone_acc_data shape: [batch_size, seq_length, acc_coords]
        skl_data shape: [batch_size, seq_length, num_joints, in_chans]
        """
        batch_size, seq_length, num_joints, in_chans = skl_data.size()

        # Process skeleton data
        skl_data = skl_data.view(-1, in_chans, num_joints)  # [batch_size * seq_length, in_chans, num_joints]
        skl_embedded = self.skeleton_embed(skl_data)  # [batch_size * seq_length, spatial_embed, num_joints]
        skl_embedded = skl_embedded.mean(dim=2)  # Aggregate over joints
        skl_embedded = skl_embedded.view(batch_size, seq_length, -1)  # [batch_size, seq_length, spatial_embed]

        # Process accelerometer data
        # Reshape to [batch_size, acc_coords, seq_length]
        watch_acc_data = watch_acc_data.permute(0, 2, 1)  # [batch_size, acc_coords, seq_length]
        phone_acc_data = phone_acc_data.permute(0, 2, 1)  # [batch_size, acc_coords, seq_length]

        watch_acc_embedded = self.watch_acc_embed(watch_acc_data)  # [batch_size, spatial_embed, seq_length]
        phone_acc_embedded = self.phone_acc_embed(phone_acc_data)  # [batch_size, spatial_embed, seq_length]

        # Fuse accelerometer data by averaging embeddings
        acc_embedded = (watch_acc_embedded + phone_acc_embedded) / 2  # [batch_size, spatial_embed, seq_length]
        acc_embedded = acc_embedded.permute(0, 2, 1)  # [batch_size, seq_length, spatial_embed]

        # Cross-modal attention
        fused_features, _ = self.cross_attention(skl_embedded, acc_embedded, acc_embedded)

        # Transformer encoder
        transformer_output = self.transformer_encoder(fused_features)

        # Pooling over sequence length
        pooled_output = transformer_output.mean(dim=1)  # [batch_size, spatial_embed]

        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]

        return logits