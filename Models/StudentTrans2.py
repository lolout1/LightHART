import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, n=10000):
        super().__init__()
        P = torch.zeros((seq_len, d_model))
        for k in range(seq_len):
            for i in range(d_model // 2):
                denominator = (n ** (2 * i / d_model))
                angle = torch.tensor(k / denominator)
                P[k, 2*i] = torch.sin(angle)
                P[k, 2*i + 1] = torch.cos(angle)
        self.register_buffer('pos_embed', P.unsqueeze(0))

    def forward(self, x):
        bsz, seq_len, d_model = x.shape
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        embed_dim,
        num_heads,
        mlp_dim,
        dropout_rate=0.25,
        attn_drop_rate=0.25,
        channels=128
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=attn_drop_rate,
            batch_first=True
        )
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, channels),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out
        normed2 = self.norm2(x)
        ff_out = self.ffn(normed2)
        x = x + ff_out
        return x

class PyTorchTransformer(nn.Module):
    def __init__(
        self, 
        seq_len=128,
        channels=3,
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
        self.input_linear = nn.Linear(channels, embed_dim)
        self.pos_encoder = PositionalEncoding(seq_len, embed_dim, n=10000)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                attn_drop_rate=attn_drop_rate,
                channels=embed_dim
            ) for _ in range(num_layers)
        ])
        self.dense_seq = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.final_linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.input_linear(x)
        out = self.pos_encoder(out)
        for block in self.transformer_blocks:
            out = block(out)
        out = self.dense_seq(out)
        out = out.mean(dim=1)
        out = self.final_linear(out)
        out = self.sigmoid(out)
        return (out,)

if __name__ == "__main__":
    sample_x = torch.randn(16, 128, 3)
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
    print("Output shape:", out[0].shape)
    print("Sample output:", out[0][:5])
