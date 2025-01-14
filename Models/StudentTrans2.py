import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Compute sinusoidal positional embeddings (non-trainable).
    Similar to get_positional_embedding() in your TF code.
    """
    def __init__(self, seq_len, d_model, n=10000):
        super().__init__()
        # Create the positional embedding once in CPU
        P = torch.zeros((seq_len, d_model))
        for k in range(seq_len):
            for i in range(d_model // 2):
                denominator = (n ** (2 * i / d_model))
                angle = torch.tensor(k / denominator)
                P[k, 2*i] = torch.sin(angle)
                P[k, 2*i + 1] = torch.cos(angle)

        # Register as buffer so it's not a parameter.
        self.register_buffer('pos_embed', P.unsqueeze(0))  # shape => [1, seq_len, d_model]

    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        Returns x + positional embedding => [B, seq_len, d_model]
        """
        bsz, seq_len, d_model = x.shape
        # If desired, we can add the positional embed:
        # x = x + self.pos_embed[:, :seq_len, :]
        return x

class TransformerBlock(nn.Module):
    """
    Single Transformer Block => MultiHeadAttention + Feedforward + Residual + LayerNorm
    Based on your Keras 'encoder(...)' function structure.
    """
    def __init__(
        self, 
        embed_dim,       # e.g., 128
        num_heads,       # e.g., 4
        mlp_dim,         # e.g., 16 in your code
        dropout_rate=0.25,
        attn_drop_rate=0.25,
        channels=128     # this is 'num_channels' in your code
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=attn_drop_rate,
            batch_first=True   # so input => [B, T, E]
        )
        self.dropout_attn = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(embed_dim)

        # The feedforward part => Dense(mlp_dim) -> activation -> Dropout -> Dense(channels) -> Dropout
        # But we want dimension to remain embed_dim. 
        # In your TF code, you "Dense(units=channels)", 
        # but your 'channels' param was the input dimension to the block. 
        # In typical Transformers, we'd keep the dimension the same => embed_dim 
        # or have a feedforward dimension => mlp_dim, then come back to embed_dim.
        # We'll adapt to your code structure.

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, channels),  # or embed_dim if we want the same dimension
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # x => [B, T, E]
        # 1) Self-Attention + Residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)  # MHA
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out   # residual

        # 2) Feedforward + Residual
        normed2 = self.norm2(x)
        ff_out = self.ffn(normed2)
        x = x + ff_out
        return x

class PyTorchTransformer(nn.Module):
    """
    PyTorch version of your Keras 'transformer(...)' model.
    - Stacks multiple TransformerBlock layers.
    - Applies final dense layers [8, 16], then global average pooling, etc.
    """
    def __init__(
        self, 
        seq_len=128,       # n_timesteps
        channels=3,        # n_features
        num_layers=4, 
        embed_dim=128,
        mlp_dim=16, 
        num_heads=4,
        dropout_rate=0.25,
        attn_drop_rate=0.25
    ):
        super().__init__()

        self.seq_len = seq_len
        self.channels = channels

        # Optionally, if you want a linear projection from channels => embed_dim
        self.input_linear = nn.Linear(channels, embed_dim)

        # (Optional) pos encoding
        self.pos_encoder = PositionalEncoding(seq_len, embed_dim, n=10000)

        # Stacked transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                attn_drop_rate=attn_drop_rate,
                channels=embed_dim  # keep dimension consistent at embed_dim
            ) for _ in range(num_layers)
        ])

        # After transformer layers, your Keras code does:
        # for dim in [8, 16]:
        #   x = Dense(dim, relu), dropout
        # => Then global average pooling (channels_first)
        # => final Dense(1, 'sigmoid')
        #
        # In PyTorch, we'll do:
        self.dense_seq = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Because your code uses GlobalAveragePooling1D(data_format='channels_first'),
        # in PyTorch we do an average over dim=1 if we want to treat dimension=1 as "channels".
        # But your input shape is [B, T, E], so we'll do average over dim=1 or 2 as needed.

        self.final_linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x shape: [B, T, channels]
        Return => tuple([B, 1]) => binary classification with shape matching targets.
        """
        # 1) Optional linear embedding
        # => shape [B, T, embed_dim]
        out = self.input_linear(x)

        # 2) optional positional embedding
        out = self.pos_encoder(out)  # out shape => [B, T, embed_dim]

        # 3) pass through stacked transformer blocks
        for block in self.transformer_blocks:
            out = block(out)  # => [B, T, embed_dim]

        # 4) pass through 2 dense layers: 8, 16
        out = self.dense_seq(out)  # => [B, T, 16]

        # 5) global average pooling => average across the time dimension => dim=1
        # => [B, 16]
        out = out.mean(dim=1)

        # 6) final => Dense(1, activation='sigmoid')
        out = self.final_linear(out)  # [B, 1]
        out = self.sigmoid(out)  # Keep the [B, 1] shape for BCELoss
        
        # Return as tuple for compatibility with training loop
        return (out,)

##
## Example usage
##
if __name__ == "__main__":
    # Suppose we have random data [batch=16, time=128, channels=3]
    # to replicate your data shape in the TF code.
    sample_x = torch.randn(16, 128, 3)  # e.g. older watch accelerometer

    model = PyTorchTransformer(
        seq_len=128,
        channels=3,
        num_layers=4,
        embed_dim=128,
        mlp_dim=16,
        num_heads=4,
        dropout_rate=0.25,
        attn_drop_rate=0.25
    )

    out = model(sample_x)
    print("Output shape:", out[0].shape)  # => [16]
    print("Sample output:", out[0][:5])   # just to see
