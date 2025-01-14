import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TeacherTransformerFusion(nn.Module):
    """
    Teacher transformer that fuses skeleton + watch data for fall detection.
    Uses 2-second windows => 64 watch samples, ~60 skeleton frames.
    Output is raw logits (for knowledge distillation).
    """

    def __init__(
        self,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.3,
        skeleton_dim=96,  # 32 joints x 3 coords
        watch_dim=4       # x, y, z, plus magnitude
    ):
        super().__init__()

        # 1) Embedding layers for watch & skeleton
        self.watch_embed = nn.Linear(watch_dim, d_model)
        self.skel_embed  = nn.Linear(skeleton_dim, d_model)

        # 2) Positional embeddings (optional, length ~ 64)
        self.watch_pos = nn.Parameter(torch.randn(1, 128, d_model))
        self.skel_pos  = nn.Parameter(torch.randn(1, 128, d_model))

        # 3) Separate transformer encoders for watch & skeleton
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.watch_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.skel_encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) Fusion layer to combine watch + skeleton at each time step
        self.fusion = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 5) Optional 1D conv for final temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # 6) Final classifier => raw logits
        self.classifier = nn.Linear(d_model, 1)

    def _compute_magnitude(self, watch):
        # watch shape [B, T, 3]
        return torch.sqrt(torch.sum(watch**2, dim=-1, keepdim=True))  # => [B, T, 1]

    def forward(self, acc_data, skl_data):
        """
        acc_data: [B, 64, 3] (accelerometer x,y,z)
        skl_data: [B, 64, 32, 3] (aligned skeleton frames)
        Returns:
          teacher_probs: [B], probabilities between 0 and 1
          teacher_feat:   [B, d_model]
        """
        B, T, _ = acc_data.shape

        # 1) Add watch magnitude => shape [B, 64, 4]
        mag = self._compute_magnitude(acc_data)
        watch_in = torch.cat([acc_data, mag], dim=-1)  # => [B, T, 4]

        # 2) Flatten skeleton => shape [B, T, 96]
        skel_in = skl_data.view(B, T, -1)  # => [B, T, 96]

        # 3) Embed watch + skeleton
        w_feat = self.watch_embed(watch_in)  # => [B, T, d_model]
        s_feat = self.skel_embed(skel_in)

        # 4) Add positional embeddings (slice if T=64 is fixed)
        w_feat = w_feat + self.watch_pos[:, :T, :]
        s_feat = s_feat + self.skel_pos[:, :T, :]

        # 5) Encode separately
        w_encoded = self.watch_encoder(w_feat)  # => [B, T, d_model]
        s_encoded = self.skel_encoder(s_feat)

        # 6) Fuse at each time step via concat + MLP
        fused = torch.cat([w_encoded, s_encoded], dim=-1)  # => [B, T, 2*d_model]
        fused = self.fusion(fused)                        # => [B, T, d_model]

        # 7) temporal conv => shape => [B, d_model, T]
        fused = fused.transpose(1, 2)
        fused = self.temporal_conv(fused)  # => [B, d_model, T]

        # 8) global pooling => teacher_feat
        teacher_feat = fused.mean(dim=-1)  # => [B, d_model]

        # 9) final => logits + sigmoid for probabilities
        teacher_logits = self.classifier(teacher_feat).squeeze(-1)
        teacher_probs = torch.sigmoid(teacher_logits)
        
        

        return teacher_probs
