import torch
import torch.nn as nn
import torch.nn.functional as F

class FallTransformerFusion(nn.Module):
    """
    Student model optimized for fused IMU features.
    This model is designed for inference on smartwatch with only accelerometer/gyroscope data.
    """
    def __init__(self,
                 feat_dim=15,      # Dimension of fused features
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 num_classes=2,
                 dropout=0.1,
                 dim_feedforward=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward

        # Input projection from feature dimension to model dimension
        self.input_proj = nn.Linear(feat_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output classification layer
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, accel_seq, accel_mask=None):
        """
        Forward pass using fused IMU features

        Args:
            accel_seq: IMU data with fusion features, shape (B, T, feat_dim)
            accel_mask: Boolean mask, shape (B, T), True=padding

        Returns:
            Dictionary with logits and intermediate feature
        """
        # Project input features to model dimension
        x_proj = self.input_proj(accel_seq)

        # Apply transformer encoder
        out_seq = self.encoder(x_proj, src_key_padding_mask=accel_mask)

        # Global average pooling (accounting for padding)
        if accel_mask is not None
                        inv_mask = (~accel_mask).float().unsqueeze(-1)
            # Apply mask and compute mean
            feat = (out_seq * inv_mask).sum(dim=1) / (inv_mask.sum(dim=1) + 1e-6)
        else:
            feat = out_seq.mean(dim=1)
        
        # Classification
        logits = self.fc(feat)
        
        return {
            "logits": logits,
            "feat": feat
        }
