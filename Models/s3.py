
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class KDStudentModel(nn.Module):
    def __init__(
        self,
        acc_coords=4,          # x, y, z, SMV
        seq_length=128,
        num_joints=32,         # Match teacher's joint dimension
        spatial_embed=32,
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.5
    ):
        super().__init__()
        
        # Accelerometer feature extraction
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_embed),
            nn.GELU(),
            nn.Conv1d(spatial_embed, spatial_embed * 2, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm1d(spatial_embed * 2),
            nn.GELU(),
            nn.Conv1d(spatial_embed * 2, spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_embed),
            nn.GELU()
        )
        
        # Learnable joint tokens
        self.joint_tokens = nn.Parameter(
            torch.randn(1, num_joints, spatial_embed)
        )
        
        # Motion-to-joint mapping
        self.motion_joint_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=num_heads//2,
            dropout=dropout,
            batch_first=True
        )
        
        # Joint relationship modeling (matches teacher's skeleton processing)
        self.joint_relationship = nn.Sequential(
            nn.Linear(spatial_embed, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Linear(spatial_embed, spatial_embed),
            nn.LayerNorm(spatial_embed)
        )
        
        # Cross-modal fusion (crucial for matching teacher's fusion process)
        self.modal_fusion = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed,
            nhead=num_heads,
            dim_feedforward=spatial_embed * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(spatial_embed)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed),
            nn.Linear(spatial_embed, spatial_embed // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_embed // 2, num_classes)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, acc_data):
        batch_size = acc_data.size(0)
        
        # 1. Extract accelerometer features
        acc_features = rearrange(acc_data, 'b l c -> b c l')
        acc_features = self.acc_encoder(acc_features)
        acc_features = rearrange(acc_features, 'b c l -> b l c')
        
        # 2. Generate joint-like representations
        joint_tokens = self.joint_tokens.expand(batch_size, -1, -1)
        
        # 3. Map motion features to joint space (mimics teacher's joint processing)
        joint_features, _ = self.motion_joint_attention(
            query=joint_tokens,
            key=acc_features,
            value=acc_features
        )
        
        # 4. Model joint relationships (similar to teacher's skeleton processing)
        joint_features = self.joint_relationship(joint_features)
        
        # 5. Cross-modal fusion (matches teacher's fusion mechanism)
        fused_features, _ = self.modal_fusion(
            query=joint_features,
            key=acc_features,
            value=acc_features
        )
        
        # 6. Temporal modeling
        temporal_features = self.transformer(fused_features)
        
        # 7. Global pooling and classification
        pooled_features = temporal_features.mean(dim=1)
        logits = self.classifier(pooled_features)
        
        return logits
