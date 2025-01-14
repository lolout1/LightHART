import torch
import torch.nn as nn
import torch.nn.functional as F

class LightStudentTransformer(nn.Module):
    """
    Student model: single-modality watch data, 
    with a smaller/lighter transformer approach.
    Outputs raw logits (no sigmoid).
    """

    def __init__(
        self,
        input_channels=3,  # watch x,y,z
        d_model=64,
        nhead=2,
        num_layers=1,      # smaller to avoid overfitting ~1000 samples
        dim_feedforward=128,
        dropout=0.3
    ):
        super().__init__()

        # watch x,y,z => plus magnitude => 4 channels
        self.in_dim = input_channels + 1

        # 1) Input linear
        self.input_proj = nn.Sequential(
            nn.Linear(self.in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # 2) Single-layer transformer or minimal layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3) Classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, x):
        """
        x: [B, 64, 3] => watch data
        return:
          student_logits: [B], raw
          student_feat:   [B, d_model], final hidden
        """
        B, T, _ = x.shape
        # compute magnitude => shape [B, T, 1]
        mag = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))
        # cat => [B, T, 4]
        x = torch.cat([x, mag], dim=-1)

        # embed => [B, T, d_model]
        x = self.input_proj(x)

        # pass through small transformer
        x = self.transformer(x)  # => [B, T, d_model]

        # global average pool
        final_feat = x.mean(dim=1)  # => [B, d_model]

        # raw logits
        student_logits = self.classifier(final_feat).squeeze(-1)
        return student_logits, final_feat
