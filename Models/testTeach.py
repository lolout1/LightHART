import torch
import torch.nn as nn
import math

class FallDetectionTeacherModel(nn.Module):
    """
    Teacher Model for Fall Detection using Skeleton and Accelerometer Data.
    Processes each joint individually and uses attention mechanisms.
    """

    def __init__(
        self,
        num_joints=32,
        in_chans=3,        # x, y, z coordinates
        acc_coords=4,      # x, y, z, SMV for accelerometer data
        seq_length=128,
        spatial_embed=256,
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.1,
        use_positional_encoding=True,
        distill=False       # Enable knowledge distillation outputs
    ):
        super().__init__()

        self.num_joints = num_joints
        self.in_chans = in_chans
        self.acc_coords = acc_coords
        self.seq_length = seq_length
        self.spatial_embed = spatial_embed
        self.distill = distill

        # Skeleton Embedding for individual joints
        self.skeleton_embed = nn.Sequential(
            nn.Linear(self.in_chans, spatial_embed),
            nn.GELU(),
            nn.Linear(spatial_embed, spatial_embed),
            nn.GELU()
        )

        # Accelerometer Embedding
        self.acc_embed = nn.Sequential(
            nn.Linear(self.acc_coords, spatial_embed),
            nn.GELU(),
            nn.Linear(spatial_embed, spatial_embed),
            nn.GELU()
        )

        # Positional Encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=spatial_embed,
                max_len=seq_length
            )

        # Attention over joints
        self.joint_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-Attention Layer for Modality Fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Transformer Encoder for Temporal Modeling
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

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed),
            nn.Linear(spatial_embed, spatial_embed // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(spatial_embed // 2, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, acc_data, skl_data):
        """
        Forward pass of the teacher model.

        Args:
            acc_data (Tensor): Accelerometer data of shape [batch_size, seq_length, acc_coords].
            skl_data (Tensor): Skeleton data of shape [batch_size, seq_length, num_joints, in_chans].

        Returns:
            logits (Tensor): Output logits of shape [batch_size, num_classes].
            intermediate_features (Dict): Intermediate representations for distillation.
        """
        batch_size = acc_data.size(0)
        seq_length = acc_data.size(1)

        # Process Skeleton Data
        # skl_data shape: [batch_size, seq_length, num_joints, in_chans]
        skl_data = skl_data.view(batch_size * seq_length * self.num_joints, self.in_chans)  # [B*T*N, C]

        # Pass through skeleton embedding
        skl_embedded = self.skeleton_embed(skl_data)  # [B*T*N, spatial_embed]

        # Reshape back to [batch_size * seq_length, num_joints, spatial_embed]
        skl_embedded = skl_embedded.view(batch_size * seq_length, self.num_joints, -1)  # [B*T, N, spatial_embed]

        # Apply joint attention
        skl_embedded, joint_attention_weights = self.joint_attention(
            query=skl_embedded,
            key=skl_embedded,
            value=skl_embedded
        )  # [B*T, N, spatial_embed]

        # Aggregate over joints
        skl_embedded = skl_embedded.mean(dim=1)  # [B*T, spatial_embed]

        # Reshape back to [batch_size, seq_length, spatial_embed]
        skl_embedded = skl_embedded.view(batch_size, seq_length, -1)  # [B, T, spatial_embed]

        # Process Accelerometer Data
        acc_embedded = self.acc_embed(acc_data)  # [batch_size, seq_length, spatial_embed]

        # Add Positional Encoding if used
        if self.use_positional_encoding:
            skl_embedded = self.positional_encoding(skl_embedded)
            acc_embedded = self.positional_encoding(acc_embedded)

        # Cross-Attention Fusion
        fused_features, cross_attention_weights = self.cross_attention(
            query=skl_embedded,
            key=acc_embedded,
            value=acc_embedded
        )  # [batch_size, seq_length, spatial_embed]

        # Transformer Encoder
        transformer_output = self.transformer_encoder(fused_features)  # [batch_size, seq_length, spatial_embed]

        # Global Average Pooling over sequence length
        pooled_output = transformer_output.mean(dim=1)  # [batch_size, spatial_embed]

        # Classification Head
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]

        if self.distill:
            # Return intermediate representations for distillation
            return logits, {
                'fused_features': fused_features,                  # After cross-attention
                'transformer_output': transformer_output,          # After transformer encoder
                'skeleton_embedding': skl_embedded,                # After joint attention
                'joint_attention_weights': joint_attention_weights,  # Attention weights over joints
                'cross_attention_weights': cross_attention_weights  # Cross-attention weights
            }
        else:
            return logits

    def _init_weights(self, m):
        """Initialize weights for linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class PositionalEncoding(nn.Module):
    """Positional Encoding module for adding positional information."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant 'pe' matrix with values dependent on position and i
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

