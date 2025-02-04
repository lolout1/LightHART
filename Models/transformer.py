# Models/fall_time2vec_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FallTime2VecTransformer(nn.Module):
    """
    A simple Transformer-based model that can take:
      def forward(self, acc_data, skl_data=None, mask=None):
        - acc_data: shape (B, T, feats)
        - skl_data: shape (B, T, something) or None (unused here)
        - mask: shape (B, T) bool => True for 'ignore'
    """
    def __init__(self, feat_dim=11, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super().__init__()
        self.feat_dim    = feat_dim
        self.d_model     = d_model
        self.nhead       = nhead
        self.num_layers  = num_layers
        self.num_classes = num_classes

        # 1) linear projection from feat_dim => d_model
        self.input_proj = nn.Linear(feat_dim, d_model)

        # 2) a basic TransformerEncoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2*d_model,
            dropout=0.1,
            batch_first=True  # so we pass (B,T,d_model)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 3) final classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, acc_data, skl_data=None, mask=None):
        """
        acc_data: (B,T,feat_dim)
        skl_data: optional (B,T,XYZ...) or None. We'll just debug-print shape if not None.
        mask: (B,T) bool => used as 'src_key_padding_mask' in the transformer
        """
        if skl_data is not None:
            print(f"[DEBUG] Skeleton data shape={skl_data.shape}, but ignoring it in forward.")
            # (We do nothing else with skeleton data.)

        # 1) Project input => (B,T,d_model)
        x_proj = self.input_proj(acc_data)

        # 2) pass to encoder. For batch_first=True, pass 'src_key_padding_mask=mask'
        #    True => ignore. So your mask_batch of shape [B,T] is correct.
        out = self.encoder(x_proj, src_key_padding_mask=mask)

        # 3) global average pool => (B,d_model)
        out = out.mean(dim=1)

        # 4) final linear => (B,num_classes)
        logits = self.fc(out)
        return logits

