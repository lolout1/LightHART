import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MobileFallDetector(nn.Module):
    def __init__(
        self,
        acc_coords=4,          # x, y, z, SMV
        sequence_length=128,    # Aligned with dataset max_length
        hidden_dim=64,         # Reduced for mobile
        num_heads=4,           # Optimized for efficiency
        depth=3,               # Reduced for mobile
        mlp_ratio=2,           # Reduced for efficiency
        num_classes=2,         # Binary classification
        dropout=0.2,           # Moderate dropout
        use_skeleton=False     # Match existing architecture
    ):
        super().__init__()
        
        # Efficient feature extraction for both phone and watch
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(acc_coords, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Depthwise separable convolution
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        """
        Forward pass handling both phone and watch accelerometer data
        
        Args:
            data (dict): Contains 'accelerometer_phone' and 'accelerometer_watch' tensors
        """
        # Process phone data
        phone_data = data['accelerometer_phone'].float()
        phone_data = rearrange(phone_data, 'b t c -> b c t')
        phone_features = self.feature_extractor(phone_data)
        phone_features = phone_features.flatten(1)
        
        # Process watch data
        watch_data = data['accelerometer_watch'].float()
        watch_data = rearrange(watch_data, 'b t c -> b c t')
        watch_features = self.feature_extractor(watch_data)
        watch_features = watch_features.flatten(1)
        
        # Fusion
        combined = torch.cat([phone_features, watch_features], dim=1)
        fused = self.fusion(combined)
        
        # Add sequence dimension
        fused = fused.unsqueeze(1)
        
        # Temporal modeling
        temporal = self.transformer(fused)
        
        # Classification
        pooled = temporal.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits