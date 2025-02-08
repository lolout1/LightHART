# File: Models/fall2.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processor.base import Time2Vec  # Assumed to be available in your repo

class FallTime2VecTransformer(nn.Module):
    """
    A Transformer-based student model for fall detection using watch accelerometer data
    plus Time2Vec embeddings for variable sampling times.

    The model takes as input:
      - accel_seq: accelerometer data with shape (B, T, 3)
      - accel_time: time values with shape (B, T)
      - accel_mask: a boolean mask with shape (B, T) where True indicates a padded token

    The forward pass returns a dictionary with:
      - "logits": the classification output (B, num_classes)
      - "features": a dict containing:
          "accel_seq": the sequence embeddings (B, T, d_model)
          "fusion": the pooled (mean) embedding (B, d_model)
    """
    def __init__(
        self,
        feat_dim=11,        # 3 for raw accel + 8 for Time2Vec embedding = 11 features per timestep
        d_model=48,
        nhead=4,
        num_layers=2,
        num_classes=2,
        time2vec_dim=8,
        dropout=0.4,
        dim_feedforward=128
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.time2vec_dim = time2vec_dim
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward

        # 1) Time2Vec module to embed time information.
        self.time2vec = Time2Vec(out_channels=time2vec_dim)

        # 2) Input projection: Concatenate accelerometer data with time embedding.
        #    The combined input dimension should equal feat_dim.
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        # 3) Transformer Encoder: processes the projected sequence.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) Classification head: map the pooled embedding to the number of classes.
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, accel_seq, accel_time=None, accel_mask=None):
        """
        Forward pass of the student model.

        Args:
            accel_seq (Tensor): Accelerometer data of shape (B, T, 3).
            accel_time (Tensor, optional): Time steps of shape (B, T). If not provided, a range is used.
            accel_mask (Tensor, optional): Boolean mask of shape (B, T) where True indicates padding.

        Returns:
            dict: {
                "logits": Tensor of shape (B, num_classes),
                "features": {
                    "accel_seq": Transformer encoder output (B, T, d_model),
                    "fusion": Mean-pooled representation (B, d_model)
                }
            }
        """
        B, T, _ = accel_seq.shape
        device = accel_seq.device

        # If no time information is provided, create a default time vector.
        if accel_time is None:
            accel_time = torch.arange(T, device=device).unsqueeze(0).expand(B, T).float()

        # Pass the time values through Time2Vec.
        # Flatten time to shape (B*T, 1) then reshape back to (B, T, time2vec_dim).
        t_emb_flat = self.time2vec(accel_time.reshape(-1, 1))
        t_emb = t_emb_flat.view(B, T, -1)

        # Concatenate the accelerometer data with the time embedding.
        # This forms the input of shape (B, T, feat_dim) where feat_dim = 3 + time2vec_dim.
        x = torch.cat([accel_seq, t_emb], dim=-1)

        # Project the concatenated features to the desired model dimension.
        x_proj = self.input_proj(x)
        x_proj = self.norm(x_proj)

        # Process the sequence through the Transformer encoder.
        # The src_key_padding_mask should be provided in the correct format.
        out_seq = self.encoder(x_proj, src_key_padding_mask=accel_mask)

        # Compute the mean of the sequence embeddings, taking the mask into account.
        if accel_mask is not None:
            valid_mask = ~accel_mask  # True for valid positions
            valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
            out_sum = (out_seq * valid_mask.unsqueeze(-1)).sum(dim=1)
            out_mean = out_sum / valid_counts
        else:
            out_mean = out_seq.mean(dim=1)

        # Compute the final logits.
        logits = self.fc(out_mean)

        return {
            "logits": logits,
            "features": {
                "accel_seq": out_seq,   # Sequence output from the encoder.
                "fusion": out_mean      # Pooled representation.
            }
        }

