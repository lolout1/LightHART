import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class EnhancedXYZProcessor(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.xyz_encoder = nn.Sequential(
            # Initial feature extraction
            nn.Conv1d(3, hidden_dim // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Deeper feature extraction
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Multi-scale processing
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for temporal features
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # CNN feature extraction
        features = self.xyz_encoder(x)  # [B, H, T]
        
        # Apply self-attention
        features = rearrange(features, 'b c t -> t b c')
        attended_features, _ = self.attention(features, features, features)
        features = self.norm(features + attended_features)
        features = rearrange(features, 't b c -> b c t')
        
        return features

class EnhancedSMVProcessor(nn.Module):
    def __init__(self, hidden_dim, sequence_length, dropout=0.3):
        super().__init__()
        self.smv_encoder = nn.Sequential(
            # Detailed SMV processing
            nn.Conv1d(1, hidden_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Multi-scale features
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Peak detection features
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.threshold_learner = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.smv_encoder(x)
        threshold = self.threshold_learner(features)
        return features, threshold

class EnhancedDualPathFallDetector(nn.Module):
    def __init__(
        self,
        acc_coords=4,
        sequence_length=128,
        hidden_dim=128,  # Increased from 64
        num_heads=12,    # Increased from 8
        depth=6,         # Increased from 4
        mlp_ratio=4,
        num_classes=2,
        dropout=0.4,     # Increased from 0.3
        use_skeleton=False
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Enhanced processors
        self.phone_xyz_processor = EnhancedXYZProcessor(hidden_dim, dropout)
        self.phone_smv_processor = EnhancedSMVProcessor(hidden_dim, sequence_length, dropout)
        self.watch_xyz_processor = EnhancedXYZProcessor(hidden_dim, dropout)
        self.watch_smv_processor = EnhancedSMVProcessor(hidden_dim, sequence_length, dropout)
        
        # Advanced feature fusion with residual connections
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 3),
            nn.LayerNorm(hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Enhanced Transformer with more sophisticated attention
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Enhanced classifier with deeper architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Increased input features
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Additional fall-specific feature extractor
        self.fall_feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

    def forward(self, data):
        # Process phone data
        phone_xyz = data['accelerometer_phone'][:, :, :3].float()
        phone_smv = torch.norm(phone_xyz, dim=2, keepdim=True)
        
        phone_xyz = rearrange(phone_xyz, 'b t c -> b c t')
        phone_smv = rearrange(phone_smv, 'b t c -> b c t')
        
        phone_xyz_features = self.phone_xyz_processor(phone_xyz)
        phone_smv_features, phone_threshold = self.phone_smv_processor(phone_smv)
        
        # Process watch data
        watch_xyz = data['accelerometer_watch'][:, :, :3].float()
        watch_smv = torch.norm(watch_xyz, dim=2, keepdim=True)
        
        watch_xyz = rearrange(watch_xyz, 'b t c -> b c t')
        watch_smv = rearrange(watch_smv, 'b t c -> b c t')
        
        watch_xyz_features = self.watch_xyz_processor(watch_xyz)
        watch_smv_features, watch_threshold = self.watch_smv_processor(watch_smv)
        
        # Feature fusion with residual connection
        device_features = torch.cat([
            reduce(phone_xyz_features, 'b c t -> b c', 'mean'),
            reduce(phone_smv_features, 'b c t -> b c', 'mean'),
            reduce(watch_xyz_features, 'b c t -> b c', 'mean'),
            reduce(watch_smv_features, 'b c t -> b c', 'mean')
        ], dim=1)
        
        fused = self.fusion(device_features)
        
        # Add fall-specific features
        fall_features = self.fall_feature_extractor(fused)
        
        # Temporal modeling with transformer
        temporal = fused.unsqueeze(1)
        temporal = self.transformer(temporal)
        
        # Classification with concatenated features
        pooled = torch.cat([temporal.squeeze(1), fall_features], dim=1)
        logits = self.classifier(pooled)
        
        return logits, {
            'phone_smv': phone_threshold.squeeze(-1),
            'watch_smv': watch_threshold.squeeze(-1),
            'fall_features': fall_features
        }