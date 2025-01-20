import torch
import torch.nn as nn
import torch.nn.functional as F

class LightTransformerEncoder(nn.Module):
    """
    A lightweight single-layer transformer encoder for capturing temporal patterns.
    """
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        out = self.transformer(x)  # => [B, T, d_model]
        out = self.ln(out)
        return out

class TransWatchModel(nn.Module):
    """
    Transformer model for watch accelerometer data with 4 channels:
    - x, y, z
    - smv (computed from zero-mean x, y, z)
    Window size: 128, shape [B, 128, 4].
    """
    def __init__(self,
                 input_channels=4,  # x, y, z, smv
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=128,
                 dropout=0.3):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Build multiple transformer encoders
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
        x: shape [B, 128, 3] => x, y, z
        We'll compute zero-mean x, y, z => smv => cat => [B, 128, 4]
        """
        # 1) Zero-mean
        mean_vals = torch.mean(x, dim=1, keepdim=True)  # shape [B, 1, 3]
        zero_mean = x - mean_vals  # shape [B, 128, 3]

        # 2) SMV from zero-mean x, y, z
        sum_sq = torch.sum(zero_mean**2, dim=-1, keepdim=True)  # [B, 128, 1]
        smv = torch.sqrt(sum_sq)  # [B, 128, 1]

        # 3) Concat => [B, 128, 4]
        x_4 = torch.cat([x, smv], dim=-1)

        # 4) Input projection => [B, 128, d_model]
        x_4 = self.input_proj(x_4)

        # 5) Pass through multiple encoders
        for encoder in self.encoders:
            x_4 = encoder(x_4)  # => [B, 128, d_model]

        # 6) Global average pooling => [B, d_model]
        final_feat = x_4.mean(dim=1)

        # 7) Binary classification => single logit => fall probability
        logit = self.classifier(final_feat).squeeze(-1)
        fall_prob = self.sigmoid(logit)

        return fall_prob, final_feat

if __name__ == "__main__":
    # Example usage
    sample = torch.randn(8, 128, 3)  # batch_size=8, length=128, channels=3
    model = TransWatchModel(
        input_channels=4,  # x,y,z, + smv
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.3
    )
    probs, feats = model(sample)
    print("Output shape:", probs.shape)   # [8]
    print("Feature shape:", feats.shape)  # [8, 64]
