import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        embed_dim,
        num_heads,
        mlp_dim,
        dropout_rate=0.25,
        attn_dropout_rate=0.25
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=attn_dropout_rate,
            batch_first=True
        )
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # Self-attention block
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out

        # Feed-forward network
        normed2 = self.norm2(x)
        ff_out = self.ffn(normed2)
        x = x + ff_out
        return x

class HierarchicalTransformer(nn.Module):
    def __init__(
        self, 
        window_seq_len=128,
        num_windows=3,
        channels=4,
        embed_dim=128,
        num_heads=4,
        mlp_dim=256,
        num_layers_window=2,
        num_layers_sequence=2,
        dropout_rate=0.25,
        attn_dropout_rate=0.25,
        num_classes=1
    ):
        super().__init__()
        self.window_seq_len = window_seq_len
        self.num_windows = num_windows
        self.channels = channels
        self.embed_dim = embed_dim

        # Input linear projection for windows
        self.input_linear = nn.Linear(channels, embed_dim)
        self.pos_encoder_window = PositionalEncoding(d_model=embed_dim, max_len=window_seq_len)

        # Window-level Transformer blocks
        self.transformer_blocks_window = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate
            ) for _ in range(num_layers_window)
        ])

        # Positional encoding for window sequence
        self.pos_encoder_sequence = PositionalEncoding(d_model=embed_dim, max_len=num_windows)

        # Sequence-level Transformer blocks
        self.transformer_blocks_sequence = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate
            ) for _ in range(num_layers_sequence)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, num_windows, window_seq_len, channels]
        Returns:
            Tuple containing the output predictions
        """
        batch_size, num_windows, window_seq_len, channels = x.size()
        assert num_windows == self.num_windows, f"Expected {self.num_windows} windows, got {num_windows}"
        assert window_seq_len == self.window_seq_len, f"Expected window length {self.window_seq_len}, got {window_seq_len}"
        assert channels == self.channels, f"Expected {self.channels} channels, got {channels}"

        # Process each window individually
        x = x.view(batch_size * num_windows, window_seq_len, channels)  # [B*N, W, C]
        
        # If input has 3 channels, compute SMV to expand to 4 channels
        if channels == 3:
            smv = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))  # [B*N, W, 1]
            x = torch.cat([x, smv], dim=-1)  # [B*N, W, 4]
        
        # Linear projection
        x = self.input_linear(x)  # [B*N, W, embed_dim]
        
        # Positional encoding
        x = self.pos_encoder_window(x)  # [B*N, W, embed_dim]
        
        # Window-level Transformer blocks
        for block in self.transformer_blocks_window:
            x = block(x)  # [B*N, W, embed_dim]
        
        # Aggregate window features (e.g., take the mean)
        x = x.mean(dim=1)  # [B*N, embed_dim]
        
        # Reshape to [B, N, embed_dim]
        x = x.view(batch_size, num_windows, self.embed_dim)  # [B, N, embed_dim]
        
        # Positional encoding for window sequence
        x = self.pos_encoder_sequence(x)  # [B, N, embed_dim]
        
        # Sequence-level Transformer blocks
        for block in self.transformer_blocks_sequence:
            x = block(x)  # [B, N, embed_dim]
        
        # Aggregate sequence features (e.g., take the mean)
        x = x.mean(dim=1)  # [B, embed_dim]
        
        # Classification
        out = self.classifier(x)  # [B, num_classes]
        
        return (out,)

if __name__ == "__main__":
    # Example usage
    batch_size = 16
    num_windows = 3
    window_seq_len = 128
    channels = 4  # After adding SMV

    # Simulate input: [batch_size, num_windows, window_seq_len, channels]
    sample_x = torch.randn(batch_size, num_windows, window_seq_len, channels)

    model = HierarchicalTransformer(
        window_seq_len=128,
        num_windows=3,
        channels=4,
        embed_dim=128,
        num_heads=4,
        mlp_dim=256,
        num_layers_window=2,
        num_layers_sequence=2,
        dropout_rate=0.25,
        attn_dropout_rate=0.25,
        num_classes=1
    )

    out = model(sample_x)
    print("Output shape:", out[0].shape)  # Expected: [batch_size, 1]
    print("Sample output:", out[0][:5])
