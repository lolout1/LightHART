import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionTransModel(nn.Module):
    """
    Fusion Transformer model for fall detection with support for IMU and orientation data.
    """
    def __init__(self, num_layers=3, embed_dim=32, acc_coords=3, quat_coords=4, 
                 num_classes=2, acc_frames=64, mocap_frames=64, num_heads=4, 
                 fusion_type='concat', dropout=0.3, use_batch_norm=True, feature_dim=64):
        super(FusionTransModel, self).__init__()
        
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        self.quat_coords = quat_coords
        self.acc_frames = acc_frames
        self.mocap_frames = mocap_frames
        self.fusion_type = fusion_type
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        # Embedding layers for different modalities
        self.acc_embed = nn.Linear(acc_coords, embed_dim)
        self.quat_embed = nn.Linear(quat_coords, embed_dim)
        
        # Feature embedding layer for fusion features
        self.feature_embedding = nn.Sequential(
            nn.Linear(43, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Position encoding
        self.acc_pos_encoder = nn.Parameter(torch.zeros(1, acc_frames, embed_dim))
        self.quat_pos_encoder = nn.Parameter(torch.zeros(1, mocap_frames, embed_dim))
        
        # Initialize position encodings
        nn.init.trunc_normal_(self.acc_pos_encoder, std=0.02)
        nn.init.trunc_normal_(self.quat_pos_encoder, std=0.02)
        
        # Transformer encoder for each modality
        encoder_norm = nn.LayerNorm(embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=False
        )
        
        self.acc_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.quat_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        
        # Attention pooling
        self.acc_attention = nn.Linear(embed_dim, 1)
        self.quat_attention = nn.Linear(embed_dim, 1)
        
        # Determine fusion output dimension
        if fusion_type == 'concat':
            fusion_dim = embed_dim * 2 + feature_dim
        elif fusion_type == 'sum':
            fusion_dim = embed_dim + feature_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.acc_bn = nn.BatchNorm1d(acc_frames)
            self.quat_bn = nn.BatchNorm1d(mocap_frames)
            self.fusion_bn = nn.BatchNorm1d(fusion_dim)
    
    def forward(self, data_dict):
        """
        Forward pass with data dictionary format
        """
        acc = data_dict['acc']  # [batch, frames, 3]
        quat = data_dict['quat']  # [batch, frames, 4]
        features = data_dict['features']  # [batch, 43]
        
        batch_size = acc.size(0)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            acc = self.acc_bn(acc.transpose(1, 2)).transpose(1, 2)
            quat = self.quat_bn(quat.transpose(1, 2)).transpose(1, 2)
        
        # Embedding
        acc_embedded = self.acc_embed(acc) + self.acc_pos_encoder
        quat_embedded = self.quat_embed(quat) + self.quat_pos_encoder
        
        # Transformer encoding
        acc_encoded = self.acc_encoder(acc_embedded)  # [batch, frames, embed_dim]
        quat_encoded = self.quat_encoder(quat_embedded)  # [batch, frames, embed_dim]
        
        # Attention pooling
        acc_attn_weights = F.softmax(self.acc_attention(acc_encoded), dim=1)  # [batch, frames, 1]
        quat_attn_weights = F.softmax(self.quat_attention(quat_encoded), dim=1)  # [batch, frames, 1]
        
        acc_pooled = torch.sum(acc_encoded * acc_attn_weights, dim=1)  # [batch, embed_dim]
        quat_pooled = torch.sum(quat_encoded * quat_attn_weights, dim=1)  # [batch, embed_dim]
        
        # Process fusion features
        features_embedded = self.feature_embedding(features)  # [batch, feature_dim]
        
        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([acc_pooled, quat_pooled, features_embedded], dim=1)
        elif self.fusion_type == 'sum':
            fused = acc_pooled + quat_pooled + features_embedded
        
        # Apply batch normalization to fused features
        if self.use_batch_norm:
            fused = self.fusion_bn(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
