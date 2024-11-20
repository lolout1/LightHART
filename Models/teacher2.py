import torch
import torch.nn as nn
import torch.nn.functional as F

class FallDetectionTeacherModel(nn.Module):
    def __init__(
        self,
        num_joints=32,
        in_chans=3,
        acc_coords=4,
        seq_length=128,
        spatial_embed=128,
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.5
    ):
        super().__init__()

        self.spatial_embed = spatial_embed
        self.num_joints = num_joints
        self.in_chans = in_chans

        # Skeleton embedding per joint
        self.skeleton_embed = nn.Sequential(
            nn.Linear(in_chans, spatial_embed),
            nn.ReLU(),
            nn.Linear(spatial_embed, spatial_embed),
            nn.ReLU()
        )

        # Accelerometer embedding
        self.acc_embed = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),
            nn.ReLU(),
            nn.Linear(spatial_embed, spatial_embed),
            nn.ReLU()
        )

        # Transformer encoder for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed, 
            nhead=num_heads, 
            dim_feedforward=spatial_embed * mlp_ratio, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(spatial_embed, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, acc_data, skl_data):
        # acc_data shape: [batch_size, seq_length, acc_coords]
        # skl_data shape: [batch_size, seq_length, num_joints, in_chans]

        batch_size, seq_length, num_joints, in_chans = skl_data.size()

        # Reshape skeleton data to [batch_size * seq_length * num_joints, in_chans]
        skl_data = skl_data.view(-1, in_chans)
        # Apply skeleton embedding
        skl_embedded = self.skeleton_embed(skl_data)  # [total_samples, spatial_embed]
        # Reshape back to [batch_size, seq_length, num_joints, spatial_embed]
        skl_embedded = skl_embedded.view(batch_size, seq_length, num_joints, self.spatial_embed)
        # Aggregate over joints (mean pooling)
        skl_embedded = skl_embedded.mean(dim=2)  # [batch_size, seq_length, spatial_embed]

        # Accelerometer embedding
        acc_embedded = self.acc_embed(acc_data)  # [batch_size, seq_length, spatial_embed]

        # Fuse modalities by concatenation
        fused_features = skl_embedded + acc_embedded  # [batch_size, seq_length, spatial_embed]

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(fused_features)  # [batch_size, seq_length, spatial_embed]

        # Pooling over sequence length
        pooled_output = transformer_output.mean(dim=1)  # [batch_size, spatial_embed]

        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]

        return logits

if __name__ == "__main__":
    acc_data = torch.randn(size=(16, 128, 4))            # [batch_size, seq_length, acc_coords]
    skl_data = torch.randn(size=(16, 128, 32, 3))        # [batch_size, seq_length, num_joints, in_chans]
    model = FallDetectionTeacherModel()
    output = model(acc_data, skl_data)
    print(output.shape)  # Should output: torch.Size([16, 2])
