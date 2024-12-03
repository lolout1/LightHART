# Models/awatifModel.py

import torch
import torch.nn as nn
import numpy as np

def get_positional_embedding(seq_len, d_model, n=10000):
    """
    Generates sinusoidal positional embeddings.

    Args:
        seq_len (int): Length of the sequence.
        d_model (int): Dimension of the embeddings.
        n (int): Base for the positional encoding.

    Returns:
        torch.FloatTensor: Positional embeddings of shape (1, seq_len, d_model).
    """
    P = np.zeros((seq_len, d_model))
    for k in range(seq_len):
        for i in range(int(d_model / 2)):
            denominator = np.power(n, 2 * i / d_model)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    P = P[np.newaxis, :, :]  # Shape: (1, seq_len, d_model)
    return torch.FloatTensor(P)

class Encoder(nn.Module):
    """
    Encoder block consisting of Multi-Head Attention and MLP with residual connections.
    """
    def __init__(self, embed_dim, mlp_dim, num_heads, attn_drop_rate, drop_rate):
        """
        Initializes the Encoder.

        Args:
            embed_dim (int): Dimension of the embeddings.
            mlp_dim (int): Dimension of the MLP hidden layer.
            num_heads (int): Number of attention heads.
            attn_drop_rate (float): Dropout rate for attention.
            drop_rate (float): Dropout rate for other layers.
        """
        super().__init__()

        # LayerNorm and Multi-Head Attention
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop_rate,
            batch_first=True  # Ensures input shape is (batch, seq_len, embed_dim)
        )
        self.dropout1 = nn.Dropout(drop_rate)

        # LayerNorm and MLP
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(drop_rate)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using Truncated Normal distribution for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Truncated normal initialization is not directly available in PyTorch,
                # but you can approximate it using normal initialization and clipping.
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    m.weight.clamp_(-0.04, 0.04)  # Approximate truncation
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, embed_dim).
        """
        # Multi-Head Attention block
        attn_norm = self.norm1(x)
        attn_output, _ = self.attention(attn_norm, attn_norm, attn_norm)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # Residual connection

        # MLP block
        mlp_norm = self.norm2(x)
        mlp_output = self.mlp(mlp_norm)
        x = x + mlp_output  # Residual connection

        return x

class PCNNTransformer(nn.Module):
    """
    PCNN Transformer model for Fall Detection using accelerometer data.
    """
    def __init__(self, length=128, channels=3, num_layers=4, embed_dim=64,
                 mlp_dim=128, num_heads=4, dropout_rate=0.1, attention_dropout_rate=0.1):
        """
        Initializes the PCNNTransformer.

        Args:
            length (int): Sequence length.
            channels (int): Number of input channels per modality (e.g., accelerometer axes).
            num_layers (int): Number of Encoder layers.
            embed_dim (int): Dimension of the embeddings.
            mlp_dim (int): Dimension of the MLP hidden layer.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate for MLP and Dense layers.
            attention_dropout_rate (float): Dropout rate for attention.
        """
        super().__init__()

        # Adjust the input projection layer to accommodate concatenated channels
        concatenated_channels = channels * 2  # Since we have two modalities
        self.input_projection = nn.Linear(concatenated_channels, embed_dim)

        # Optional positional embedding
        # self.pos_embed = nn.Parameter(get_positional_embedding(length, embed_dim), requires_grad=False)

        # Stack of Encoder layers
        self.encoders = nn.ModuleList([
            Encoder(
                embed_dim=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                attn_drop_rate=attention_dropout_rate,
                drop_rate=dropout_rate
            ) for _ in range(num_layers)
        ])

        # Final dense layers
        self.dense1 = nn.Linear(embed_dim, 8)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(8, 16)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Output layer (no sigmoid for BCEWithLogitsLoss)
        self.output = nn.Linear(16, 1)

        # Initialize output layer weights to zero
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        """
        Forward pass of the PCNNTransformer.

        Args:
            x (dict or torch.Tensor): Input data. If dict, should contain
                                      'accelerometer_phone' and 'accelerometer_watch'.
                                      Each should be of shape (batch, seq_len, channels).

        Returns:
            torch.Tensor: Output logits of shape (batch, 1).
        """
        # Handle dictionary input for multimodal data
        if isinstance(x, dict):
            phone = x['accelerometer_phone']  # Shape: (batch, seq_len, 3)
            watch = x['accelerometer_watch']  # Shape: (batch, seq_len, 3)

            # Concatenate along the channel dimension
            x = torch.cat((phone, watch), dim=-1)  # Shape: (batch, seq_len, 6)
        else:
            # If x is a single tensor, ensure it has the correct number of channels
            x = x  # Should be of shape (batch, seq_len, concatenated_channels)

        # Initial projection to embed_dim
        x = self.input_projection(x)  # Shape: (batch, seq_len, embed_dim)

        # Optional positional embedding
        # if hasattr(self, 'pos_embed'):
        #     x = x + self.pos_embed[:, :x.size(1), :]

        # Pass through Encoder layers
        for encoder in self.encoders:
            x = encoder(x)  # Each encoder maintains shape: (batch, seq_len, embed_dim)

        # Pass through final dense layers with ReLU and Dropout
        x = self.dense1(x)       # Shape: (batch, seq_len, 8)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.dense2(x)       # Shape: (batch, seq_len, 16)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Global average pooling over the sequence length
        x = x.transpose(1, 2)    # Shape: (batch, 16, seq_len)
        x = self.pool(x)         # Shape: (batch, 16, 1)
        x = x.squeeze(-1)        # Shape: (batch, 16)

        # Output logits
        x = self.output(x)       # Shape: (batch, 1)
        # No sigmoid activation here; use BCEWithLogitsLoss during training

        return x
