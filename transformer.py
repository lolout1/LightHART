import torch
import torch.nn as nn
import math

class FallDetectionStudentModel(nn.Module):
    """
    Student Model for Fall Detection using Accelerometer Data Only.
    Designed to mimic the teacher model's behavior through knowledge distillation.
    """

    def __init__(
        self,
        acc_coords=4,      # x, y, z, SMV
        seq_length=128,
        embedding_dim=256,  # Match the teacher's spatial embedding size
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.1,
        use_positional_encoding=True
    ):
        super().__init__()

        self.acc_coords = acc_coords
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.use_positional_encoding = use_positional_encoding

        # Accelerometer Embedding
        self.acc_embed = nn.Sequential(
            nn.Linear(self.acc_coords, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU()
        )

        # Positional Encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=embedding_dim,
                max_len=seq_length
            )

        # Transformer Encoder for Temporal Modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embedding_dim)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, acc_data):
        """
        Forward pass of the student model.

        Args:
            acc_data (Tensor): Accelerometer data of shape [batch_size, seq_length, acc_coords].

        Returns:
            logits (Tensor): Output logits of shape [batch_size, num_classes].
            intermediate_features (Dict): Intermediate representations for distillation.
        """
        # Process Accelerometer Data
        acc_embedded = self.acc_embed(acc_data)  # [batch_size, seq_length, embedding_dim]

        # Add Positional Encoding if used
        if self.use_positional_encoding:
            acc_embedded = self.positional_encoding(acc_embedded)

        # Transformer Encoder
        transformer_output = self.transformer_encoder(acc_embedded)  # [batch_size, seq_length, embedding_dim]

        # Global Average Pooling over sequence length
        pooled_output = transformer_output.mean(dim=1)  # [batch_size, embedding_dim]

        # Classification Head
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]

        # Return intermediate features for distillation
        return logits, {
            'transformer_output': transformer_output
        }

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
