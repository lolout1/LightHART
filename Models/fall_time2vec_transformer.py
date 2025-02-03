# Models/fall_time2vec_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FallTime2VecTransformer(nn.Module):
    """
    Optimized Transformer for variable-length sequences with Time2Vec features.
    Expects input: (B, N, feat_dim) and an optional mask of shape (B, N).
    Uses batch_first=True for better inference performance.
    """
    def __init__(self, feat_dim=11, d_model=64, nhead=4, num_layers=2, num_classes=2, 
                 dropout=0.1, dim_feedforward=128):
        super().__init__()
        # Project input features to d_model and normalize them.
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder layer with batch_first=True.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier with an extra dropout layer.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x: (B, N, feat_dim)
        mask: (B, N) boolean, where True indicates padded positions to ignore.
        """
        # Project and normalize input.
        x_proj = self.input_proj(x)         # (B, N, d_model)
        x_proj = self.input_norm(x_proj)      # (B, N, d_model)
        
        # Transformer (batch_first=True, so no transpose is needed).
        out = self.transformer(x_proj, src_key_padding_mask=mask)  # (B, N, d_model)
        
        # Global average pooling over unmasked positions.
        if mask is not None:
            lengths = (~mask).sum(dim=-1).unsqueeze(-1).float()  # (B, 1)
            out = out * (~mask).unsqueeze(-1).float()            # Zero out padded positions.
            out = out.sum(dim=1) / torch.clamp(lengths, min=1e-9)
        else:
            out = out.mean(dim=1)
        
        logits = self.classifier(out)  # (B, num_classes)
        return logits
