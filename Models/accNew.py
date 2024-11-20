import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # [1, max_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEncoderWAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_weights = []

    def forward(self, src, mask=None, src_key_padding_mask=None):
        self.attention_weights = []  # Clear previous attention weights

        def save_attention(module, input, output):
            if isinstance(module, nn.MultiheadAttention):
                # The attention weights are in the output tuple at index 1
                attn_output, attn_weights = output
                if attn_weights is not None:
                    self.attention_weights.append(attn_weights.detach().cpu().numpy())

        # Register hooks for each layer
        hooks = []
        for layer in self.encoder.layers:
            # Access the MultiheadAttention module directly from the TransformerEncoderLayer
            mha = layer.self_attn
            hooks.append(mha.register_forward_hook(save_attention))

        # Forward pass through the encoder
        output = self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return output



class TransModel(nn.Module):
    def __init__(self,
                 acc_frames=128,
                 acc_coords=4,
                 num_classes=8,
                 num_heads=4,
                 num_layers=4,
                 embed_dim=32,
                 dropout=0.4,
                 activation='relu',
                 **kwargs):
        super().__init__()
        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        size = self.data_shape[1]

        # Enhanced input projection with multi-scale convolutions
        self.input_proj = nn.Sequential(
            nn.Conv1d(size, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=dropout, max_len=self.length)

        self.encoder = TransformerEncoderWAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Improved classification head
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, acc_data):
        # acc_data shape: [batch_size, seq_length, acc_coords]
        x = acc_data.permute(0, 2, 1)  # [batch_size, acc_coords, seq_length]
        x = self.input_proj(x)         # [batch_size, embed_dim, seq_length]
        x = x.permute(0, 2, 1)         # [batch_size, seq_length, embed_dim]

        x = self.positional_encoding(x)
        x = self.encoder(x)

        x = x.mean(dim=1)  # Global average pooling
        logits = self.fc(x)
        return logits

if __name__ == "__main__":
    data = torch.randn(size=(16, 128, 4))  # [batch_size, seq_length, acc_coords]
    model = TransModel()
    output = model(data)
    print(output.shape)  # Expected output: [16, 8]
