import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TeacherTransformerFusion(nn.Module):
    """
    Teacher transformer that fuses skeleton + watch data for fall detection.
    Uses adaptive average pooling to align sequences to target length.
    Supports different sampling rates between accelerometer and skeleton.
    Maintains model architecture while adding flexibility.
    """

    def __init__(
        self,
        d_model=256,
        nhead=4,
        num_layers=2,
        dropout=0.3,
        skeleton_dim=96,  # 32 joints x 3 coords
        watch_dim=4,      # x, y, z, plus magnitude
        max_len=64        # Maximum sequence length for positional embeddings
    ):
        super().__init__()

        # 1) Embedding layers for watch & skeleton
        self.watch_embed = nn.Linear(watch_dim, d_model)
        self.skel_embed  = nn.Linear(skeleton_dim, d_model)

        # 2) Positional embeddings
        self.watch_pos = nn.Parameter(torch.randn(1, max_len, d_model))
        self.skel_pos  = nn.Parameter(torch.randn(1, max_len, d_model))

        # 3) Average pooling to align sequences
        self.target_len = max_len

        # 4) Separate transformer encoders
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

        # 5) Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 6) Temporal conv
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # 7) Classifier
        self.classifier = nn.Linear(d_model, 1)

    def _compute_magnitude(self, watch):
        """Compute magnitude of acceleration"""
        if watch.dim() == 2:
            watch = watch.unsqueeze(0)  # Add batch dimension if missing
            
        # No need for mean subtraction for magnitude
        sum_squared = torch.sum(torch.square(watch), dim=-1, keepdim=True)  # [B, T, 1]
        return torch.sqrt(sum_squared)  # [B, T, 1]
        
    def _adaptive_pooling(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Average pool sequence to target length"""
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        B, T, C = x.shape
        if T == target_len:
            return x
            
        # Reshape to use F.adaptive_avg_pool1d
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.adaptive_avg_pool1d(x, target_len)  # [B, C, target_len]
        return x.transpose(1, 2)  # [B, target_len, C]
        
    def forward(self, acc_data, skl_data):
        """
        acc_data: [B, T1, 3] (accelerometer x,y,z at ~31Hz)
        skl_data: [B, T2, 32, 3] (skeleton frames at ~30Hz)
        Returns:
            teacher_probs: [B], probabilities between 0 and 1
            teacher_feat: [B, d_model]
        """
        # Add batch dimension if missing
        if acc_data.dim() == 2:
            acc_data = acc_data.unsqueeze(0)  # [1, T1, 3]
        if skl_data.dim() == 3:
            skl_data = skl_data.unsqueeze(0)  # [1, T2, 32, 3]
            
        B = acc_data.shape[0]  # Get batch size
        
        # 1) Add watch magnitude => shape [B, T1, 4]
        mag = self._compute_magnitude(acc_data)  # [B, T1, 1]
        watch_in = torch.cat([acc_data, mag], dim=-1)  # [B, T1, 4]
        
        # 2) Flatten skeleton => shape [B, T2, 96]
        skl_shape = skl_data.shape
        skel_in = skl_data.reshape(B, skl_shape[1], -1)  # [B, T2, 96]
        
        # 3) Average pool both to target length
        watch_in = self._adaptive_pooling(watch_in, self.target_len)  # [B, target_len, 4]
        skel_in = self._adaptive_pooling(skel_in, self.target_len)  # [B, target_len, 96]
        
        # 4) Embed and add positional encoding
        w_feat = self.watch_embed(watch_in)  # [B, target_len, d_model]
        s_feat = self.skel_embed(skel_in)  # [B, target_len, d_model]
        
        # Add positional embeddings with proper batch expansion
        w_feat = w_feat + self.watch_pos.expand(B, -1, -1)[:, :self.target_len, :]  # [B, target_len, d_model]
        s_feat = s_feat + self.skel_pos.expand(B, -1, -1)[:, :self.target_len, :]  # [B, target_len, d_model]
        
        # 5) Encode separately
        w_encoded = self.watch_encoder(w_feat)  # [B, target_len, d_model]
        s_encoded = self.skel_encoder(s_feat)  # [B, target_len, d_model]
        
        # 6) Fuse
        fused = torch.cat([w_encoded, s_encoded], dim=-1)  # [B, target_len, 2*d_model]
        fused = self.fusion(fused)  # [B, target_len, d_model]
        
        # 7) Temporal conv
        fused = fused.transpose(1, 2)  # [B, d_model, target_len]
        fused = self.temporal_conv(fused)  # [B, d_model, target_len]
        
        # 8) Global pooling
        teacher_feat = fused.mean(dim=-1)  # [B, d_model]
        
        # 9) Classify 
        logits = self.classifier(teacher_feat)  # [B, 1]
        probs = torch.sigmoid(logits).squeeze(-1)  # [B]
        
        return probs, teacher_feat
