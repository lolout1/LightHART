import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LightTransformerEncoder(nn.Module):
    """
    A lightweight transformer encoder block for capturing temporal patterns
    in accelerometer data.
    """
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        # A single layer of TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # <--- ensures shape: [B, T, d_model]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, d_model] 
        """
        out = self.transformer(x)          # => [B, T, d_model]
        out = self.ln(out)                 # final layer norm
        return out

class LightTransformerStudent(nn.Module):
    """
    Lightweight Transformer-based student model for single-modality
    watch accelerometer data. Focused on accuracy (precision/recall/F1)
    rather than efficiency, but still relatively compact to reduce
    overfitting on small datasets.
    """
    def __init__(
        self,
        input_channels=4,   # watch accelerometer (x,y,z)
        d_model=64,
        nhead=4,
        num_layers=4,       # number of transformer layers
        dim_feedforward=128,
        dropout=0.3
    ):
        super().__init__()

        # Project 3D accelerometer + 1D magnitude => total 4 input channels
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels + 1, d_model),  # +1 for magnitude
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Stack multiple lightweight transformer encoders
        self.encoders = nn.ModuleList([
            LightTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # single logit
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, T, 3] -> raw watch accelerometer data (x,y,z).
        Returns:
            fall_prob: [B] => fall probability per sample
            final_feat: [B, d_model] => final hidden state for optional use
        """
        B, T, _ = x.shape

        # 1) Compute magnitude => shape [B, T, 1]


        # 2) Input projection => [B, T, d_model]
        x = self.input_proj(x)

        # 3) Pass through transformer encoders
        # The nn.Transformer modules default to shape (B, T, E) if batch_first=True
        for encoder in self.encoders:
            x = encoder(x)  # => [B, T, d_model]

        # 4) Global average pooling => [B, d_model]
        final_feat = x.mean(dim=1)  # average across time dimension

        # 5) Classification => single logit => fall probability
        logit = self.classifier(final_feat).squeeze(-1)
        fall_prob = self.sigmoid(logit)

        return fall_prob

# Example usage:
if __name__ == "__main__":
    # Suppose we have [Batch=16, Time=128, Channels=3]
    sample_acc_data = torch.randn(16, 128, 3)
    model = LightTransformerStudent(
        input_channels=3,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.3
    )
    probs, feats = model(sample_acc_data)
    print("Output shape:", probs.shape)   # [16]
    print("Feature shape:", feats.shape)  # [16, 128]
