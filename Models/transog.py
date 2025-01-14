import torch
from torch import nn
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange


class TransformerEncoderWAttention(nn.TransformerEncoder):
    """
    Custom Transformer Encoder that captures attention weights during the forward pass.
    """
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        self.attention_weights = []
        for layer in self.layers:
            output, attn = layer.self_attn(
                output, output, output,
                attn_mask=mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True
            )
            self.attention_weights.append(attn)
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TransModel(nn.Module):
    """
    Transformer-based model for fall detection using accelerometer data.
    """
    def __init__(
        self,
        acc_frames=128,
        acc_coords=4,
        num_classes: int = 2,
        num_heads=2,
        num_layers=2,
        embed_dim=32,
        activation='relu',
        **kwargs
    ):
        super().__init__()
        # Input projection using Conv1D layers
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Transformer Encoder configuration
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,  # Embed_dim consistency
            nhead=num_heads,
            dim_feedforward=128,
            activation=activation,
            dropout=0.5
        )
        self.encoder = TransformerEncoderWAttention(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # Fully connected layers for classification
        self.ln1 = nn.Linear(embed_dim, 64)
        self.ln2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(32, num_classes)

        # Initialize weights
        nn.init.normal_(self.output.weight, 0, 0.02)

    def forward(self, acc_data, skl_data=None):
        """
        Forward pass for the TransModel.

        Args:
            acc_data: Tensor of shape [batch_size, acc_frames, acc_coords].
            skl_data: Optional; Placeholder for future use.

        Returns:
            logits: Tensor of shape [batch_size, num_classes].
        """
        b, l, c = acc_data.shape

        # Input projection using Conv1D
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x)  # [batch_size, embed_dim, acc_frames]
        x = rearrange(x, 'b c l -> l b c')  # [acc_frames, batch_size, embed_dim ]

        # Transformer encoding
        x = self.encoder(x)  # [acc_frames, batch_size, embed_dim]
        x = rearrange(x, 'l b c -> b l c')  # [batch_size, acc_frames, embed_dim]

        # Global pooling (mean over frames)
        x = x.mean(dim=1)  # [batch_size, embed_dim]

        # Fully connected layers
        x = F.relu(self.ln1(x))
        x = self.dropout(x)
        x = F.relu(self.ln2(x))
        x = self.output(x)

        return x


if __name__ == "__main__":
    # Test the model with dummy data
    acc_data = torch.randn(size=(16, 128, 4))  # [batch_size, acc_frames, acc_coords]
    model = TransModel(embed_dim=32)
    output = model(acc_data)
    print(output.shape)  # Should output [16, num_classes]

