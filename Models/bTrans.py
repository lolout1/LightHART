import torch
import torch.nn as nn
import torch.nn.functional as F
class FallDetectionTeacherModel(nn.Module):
    def __init__(
        self,
        num_joints=32,
        in_chans=3,
        acc_coords=4,
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
        self.skeleton_embed = nn.Sequential(
            nn.Conv1d(in_chans, spatial_embed, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(spatial_embed, spatial_embed, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.acc_embed = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),
            nn.ReLU(),
            nn.Linear(spatial_embed, spatial_embed),
            nn.ReLU()
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed, num_heads=num_heads, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed, nhead=num_heads, dim_feedforward=spatial_embed * mlp_ratio, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Linear(spatial_embed, num_classes)
    def forward(self, acc_data, skl_data):
        if skl_data.dim() != 4:
            raise ValueError(f"Expected skl_data to have 4 dimensions, but got {skl_data.dim()} dimensions")
        batch_size, seq_length, num_joints, in_chans = skl_data.size()
        if in_chans != self.in_chans or num_joints != self.num_joints:
            raise ValueError(
                f"Expected skl_data to have shape (batch_size, seq_length, {self.num_joints}, {self.in_chans}), "
                f"but got {skl_data.size()}"
            )
        skl_data = skl_data.view(batch_size * seq_length, num_joints, in_chans)
        joint_embeddings = []
        for i in range(self.num_joints):
            joint_data = skl_data[:, i, :]
            joint_data = joint_data.unsqueeze(-1)
            joint_embed = self.skeleton_embed(joint_data)
            joint_embed = joint_embed.squeeze(-1)
            joint_embeddings.append(joint_embed)
        skl_embedded = torch.stack(joint_embeddings, dim=1)
        skl_embedded = skl_embedded.mean(dim=1)
        skl_embedded = skl_embedded.view(batch_size, seq_length, -1)
        acc_embedded = self.acc_embed(acc_data)
        fused_features, _ = self.cross_attention(skl_embedded, acc_embedded, acc_embedded)
        transformer_output = self.transformer_encoder(fused_features)
        pooled_output = transformer_output.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits
