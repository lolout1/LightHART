import torch
import torch.nn as nn
import torch.nn.functional as F

class FallDetectionStudentModel(nn.Module):
    def __init__(
        self,
        acc_coords=4,       # x, y, z, SVM for accelerometer data
        num_joints=32,      # Number of joints in the skeleton data
        in_chans=3,         # x, y, z for skeleton data
        spatial_embed=128,
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        num_classes=2
    ):
        super().__init__()

        self.spatial_embed = spatial_embed
        self.num_joints = num_joints
        self.in_chans = in_chans
        self.acc_coords = acc_coords

        # Accelerometer embedding
        self.acc_embed = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),
            nn.ReLU(),
            nn.Linear(spatial_embed, spatial_embed),
            nn.ReLU()
        )

        # Modality hallucination: Generate skeleton-like features from accelerometer data
        self.modality_hallucination = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),
            nn.ReLU(),
            nn.Linear(spatial_embed, num_joints * in_chans),
            nn.ReLU()
        )

        # Skeleton embedding per joint (shared across joints)
        self.skeleton_embed = nn.Sequential(
            nn.Conv1d(in_chans, spatial_embed, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(spatial_embed, spatial_embed, kernel_size=1),
            nn.ReLU()
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed, num_heads=num_heads, batch_first=True
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed,
            nhead=num_heads,
            dim_feedforward=spatial_embed * mlp_ratio,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.classifier = nn.Linear(spatial_embed, num_classes)

    def forward(self, acc_data, skl_data=None):
        """
        acc_data: [batch_size, seq_length, acc_coords]
        skl_data: Optional; [batch_size, seq_length, num_joints, in_chans]
        """

        batch_size, seq_length, _ = acc_data.size()

        # Apply accelerometer embedding
        acc_embedded = self.acc_embed(acc_data)  # [B, T, spatial_embed]

        # Generate or use skeleton data
        if skl_data is not None:
            # Use provided skeleton data
            batch_size_skl, seq_length_skl, num_joints, in_chans = skl_data.size()
            assert batch_size == batch_size_skl and seq_length == seq_length_skl, \
                "Accelerometer data and skeleton data must have the same batch size and sequence length"

            # Reshape skl_data to [B*T, num_joints, in_chans]
            skl_data = skl_data.view(batch_size * seq_length, num_joints, in_chans)

            # Process each joint individually
            joint_embeddings = []
            for i in range(self.num_joints):
                joint_data = skl_data[:, i, :]  # [B*T, in_chans]
                joint_data = joint_data.unsqueeze(-1)  # [B*T, in_chans, 1]
                joint_embed = self.skeleton_embed(joint_data)  # [B*T, spatial_embed, 1]
                joint_embed = joint_embed.squeeze(-1)  # [B*T, spatial_embed]
                joint_embeddings.append(joint_embed)

            # Stack joint embeddings to get [B*T, num_joints, spatial_embed]
            skl_embedded = torch.stack(joint_embeddings, dim=1)  # [B*T, num_joints, spatial_embed]

            # Mean pooling over joints
            skl_embedded = skl_embedded.mean(dim=1)  # [B*T, spatial_embed]

            # Reshape back to [B, T, spatial_embed]
            skl_embedded = skl_embedded.view(batch_size, seq_length, -1)  # [B, T, spatial_embed]
        else:
            # Generate skeleton-like features from accelerometer data
            # Flatten acc_data to [B*T, acc_coords]
            acc_data_flat = acc_data.view(batch_size * seq_length, -1)  # [B*T, acc_coords]
            # Generate skeleton data: [B*T, num_joints * in_chans]
            generated_skl = self.modality_hallucination(acc_data_flat)
            # Reshape to [B*T, num_joints, in_chans]
            generated_skl = generated_skl.view(batch_size * seq_length, self.num_joints, self.in_chans)

            # Process each joint individually
            joint_embeddings = []
            for i in range(self.num_joints):
                joint_data = generated_skl[:, i, :]  # [B*T, in_chans]
                joint_data = joint_data.unsqueeze(-1)  # [B*T, in_chans, 1]
                joint_embed = self.skeleton_embed(joint_data)  # [B*T, spatial_embed, 1]
                joint_embed = joint_embed.squeeze(-1)  # [B*T, spatial_embed]
                joint_embeddings.append(joint_embed)

            # Stack joint embeddings to get [B*T, num_joints, spatial_embed]
            skl_embedded = torch.stack(joint_embeddings, dim=1)  # [B*T, num_joints, spatial_embed]

            # Mean pooling over joints
            skl_embedded = skl_embedded.mean(dim=1)  # [B*T, spatial_embed]

            # Reshape back to [B, T, spatial_embed]
            skl_embedded = skl_embedded.view(batch_size, seq_length, -1)  # [B, T, spatial_embed]

        # Fuse modalities using cross-attention
        # Query: skl_embedded, Key and Value: acc_embedded
        fused_features, _ = self.cross_attention(skl_embedded, acc_embedded, acc_embedded)  # [B, T, spatial_embed]

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(fused_features)  # [B, T, spatial_embed]

        # Pooling
        pooled_output = transformer_output.mean(dim=1)  # [B, spatial_embed]

        # Classification
        logits = self.classifier(pooled_output)  # [B, num_classes]

        return logits
