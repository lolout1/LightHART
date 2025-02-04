# Models/s.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processor.base import Time2Vec

class CNNTransformerStudent(nn.Module):
    """
    A CNN + Transformer hybrid model that uses only watch accelerometer data.
    Time2Vec is integrated to provide temporal embeddings for each time step.
    This is the 'student' model for knowledge distillation.
    """
    def __init__(self,
                 accel_dim=3,
                 cnn_filters=[32, 64],
                 cnn_kernels=[5, 5],
                 cnn_strides=[1, 1],
                 time2vec_dim=4,
                 transformer_d_model=64,
                 transformer_nhead=4,
                 transformer_num_layers=2,
                 transformer_ff=128,
                 dropout=0.2,
                 num_classes=2,
                 **kwargs):
        """
        Args:
            accel_dim: number of accelerometer channels (3 for [x,y,z]).
            cnn_filters, cnn_kernels, cnn_strides: define the CNN feature extractor layers.
            time2vec_dim: number of periodic components in the Time2Vec layer (output dimension is time2vec_dim + 1).
            transformer_d_model: dimension of Transformer embedding (d_model).
            transformer_nhead: number of attention heads in Transformer.
            transformer_num_layers: number of encoder layers in Transformer.
            transformer_ff: feedforward dimension in Transformer.
            dropout: dropout probability.
            num_classes: output dimension (2 for binary fall classification).
        """
        super().__init__()

        # 1) CNN feature extractor (1D conv over time dimension)
        self.cnn_layers = nn.ModuleList()
        in_channels = accel_dim
        for out_channels, k, s in zip(cnn_filters, cnn_kernels, cnn_strides):
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s)
            self.cnn_layers.append(conv)
            in_channels = out_channels

        # 2) Time2Vec for temporal embedding
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        # We'll store final CNN channels count to combine with Time2Vec
        self.cnn_out_dim = in_channels  # after final conv

        # 3) Linear projection to transformer d_model
        self.proj = nn.Linear(self.cnn_out_dim + time2vec_dim, transformer_d_model)

        # 4) Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_num_layers)

        # 5) Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(transformer_d_model, num_classes)
        )

    def forward(self, accel_seq, accel_time=None, accel_mask=None):
        """
        Args:
          accel_seq: (B, T, C=3) accelerometer data
          accel_time: (B, T) times (seconds), or None
          accel_mask: (B, T) boolean mask (True => padded)

        Returns:
          logits => (B, num_classes)
        """
        B, T, C = accel_seq.shape
        # Debug shapes
        # print(f"[DEBUG - Student] In: accel_seq={accel_seq.shape}, mask={None if accel_mask is None else accel_mask.shape}")

        # 1) CNN feature extraction
        # We expect shape => (B, C, T) for Conv1d
        x_cnn = accel_seq.permute(0, 2, 1)  # => (B, channels=3, T)
        for conv in self.cnn_layers:
            x_cnn = conv(x_cnn)        # => (B, out_channels, T') if stride=1/k=5 => T' < T
            x_cnn = F.relu(x_cnn)
        # Now shape => (B, final_out_channels, T'?)
        # Permute back => (B, T', final_out_channels)
        x_cnn = x_cnn.permute(0, 2, 1)
        B, T_prime, F_cnn = x_cnn.shape

        # Debug
        # print(f"[DEBUG - Student] After CNN: shape={x_cnn.shape}, T_prime={T_prime}")

        # 2) Adjust mask if needed
        if accel_mask is not None:
            # If the CNN has shortened T => we must slice the mask accordingly
            # e.g. if original T=834 -> T_prime=826 => slice mask to (B, 826)
            if accel_mask.shape[1] > T_prime:
                accel_mask = accel_mask[:, :T_prime]
            # If for any reason T_prime < accel_mask.shape[1], we do similarly
            elif accel_mask.shape[1] < T_prime:
                # Rare case if CNN extended time dimension (unlikely), you could pad the mask
                pad_len = T_prime - accel_mask.shape[1]
                pad_zeros = torch.zeros((B, pad_len), dtype=torch.bool, device=accel_mask.device)
                accel_mask = torch.cat([accel_mask, pad_zeros], dim=1)
            # print(f"[DEBUG - Student] New mask shape={accel_mask.shape}")

        # 3) If no accel_time => create a default
        if accel_time is None:
            seq_len = T_prime
            t_idx = torch.arange(seq_len, device=x_cnn.device).unsqueeze(0).repeat(B, 1).float()
        else:
            # Slice time if it has more frames than T_prime
            if accel_time.shape[1] > T_prime:
                accel_time = accel_time[:, :T_prime]
            seq_len = T_prime
            t_idx = accel_time

        # Flatten => (B*T_prime, 1) for Time2Vec
        t_idx_flat = t_idx.reshape(-1, 1)  # => (B*T_prime, 1)
        t_emb_flat = self.time2vec(t_idx_flat)
        t_emb = t_emb_flat.view(B, seq_len, -1)  # => (B, T_prime, time2vec_dim)

        # 4) Concat => linear projection
        combined = torch.cat([x_cnn, t_emb], dim=-1)  # => (B, T_prime, F_cnn + time2vec_dim)
        x_proj = self.proj(combined)                  # => (B, T_prime, d_model)

        # 5) Transformer
        # pass src_key_padding_mask=accel_mask => shape must match x_proj => (B, T_prime)
        out = self.transformer(x_proj, src_key_padding_mask=accel_mask)  # => (B, T_prime, d_model)

        # 6) Pool => masked mean if mask is present
        if accel_mask is not None:
            valid = ~accel_mask  # shape => (B, T_prime)
            valid = valid.unsqueeze(-1).float()  # => (B, T_prime, 1)
            out = out * valid
            sums = out.sum(dim=1)  # => (B, d_model)
            counts = valid.sum(dim=1).clamp(min=1e-5)
            seq_repr = sums / counts
        else:
            seq_repr = out.mean(dim=1)

        logits = self.classifier(seq_repr)  # => (B, num_classes)
        return logits

