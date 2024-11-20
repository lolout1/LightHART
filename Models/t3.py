
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedTeacherModel(nn.Module):
    def __init__(
        self,
        num_joints=32,
        in_chans=3,
        acc_coords=4,
        spatial_embed=128,
        num_heads=8,
        depth=6,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()
        self.spatial_embed = spatial_embed
        self.num_joints = num_joints
        self.in_chans = in_chans
        
        # Enhanced skeleton embedding with better normalization
        self.skeleton_embed = nn.Sequential(
            nn.Conv1d(in_chans, spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_embed),
            nn.GELU(),
            nn.Conv1d(spatial_embed, spatial_embed, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_embed),
            nn.GELU()
        )
        
        # Inter-joint attention for better skeletal relationships
        self.joint_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=4,  # Fewer heads for joint relationships
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced accelerometer embedding
        self.acc_embed = nn.Sequential(
            nn.Linear(acc_coords, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_embed, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.GELU()
        )
        
        # Cross-modality fusion with improved attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal modeling with pre-norm architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed,
            nhead=num_heads,
            dim_feedforward=spatial_embed * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(spatial_embed)
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed),
            nn.Linear(spatial_embed, spatial_embed // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_embed // 2, num_classes)
        )

    def forward(self, acc_data, skl_data):
        batch_size, seq_length, num_joints, in_chans = skl_data.size()
        
        # Process each joint with shared weights
        skl_data = skl_data.view(batch_size * seq_length, num_joints, in_chans)
        joint_embeddings = []
        
        for i in range(self.num_joints):
            joint_data = skl_data[:, i, :]
            joint_data = joint_data.unsqueeze(-1)
            joint_embed = self.skeleton_embed(joint_data)
            joint_embed = joint_embed.squeeze(-1)
            joint_embeddings.append(joint_embed)
            
        # Stack and process joint relationships
        skl_embedded = torch.stack(joint_embeddings, dim=1)  # [B*T, J, C]
        
        # Model inter-joint relationships
        skl_embedded, joint_attn_weights = self.joint_attention(
            skl_embedded, skl_embedded, skl_embedded
        )
        
        # Mean pool joints and reshape
        skl_embedded = skl_embedded.mean(dim=1)  # [B*T, C]
        skl_embedded = skl_embedded.view(batch_size, seq_length, -1)  # [B, T, C]
        
        # Process accelerometer data
        acc_embedded = self.acc_embed(acc_data)
        
        # Cross-modal fusion
        fused_features, cross_attn_weights = self.cross_attention(
            skl_embedded, acc_embedded, acc_embedded
        )
        
        # Temporal modeling
        temporal_features = self.transformer_encoder(fused_features)
        
        # Global pooling and classification
        pooled_features = temporal_features.mean(dim=1)
        logits = self.classifier(pooled_features)
        
        # Store intermediate features during training
        if self.training:
            # Save intermediate features and attention weights
            self.features = {
                'joint_features': skl_embedded.detach(),
                'acc_features': acc_embedded.detach(),
                'fused_features': fused_features.detach(),
                'temporal_features': temporal_features.detach(),
                'joint_attention_weights': joint_attn_weights.detach(),
                'cross_attention_weights': cross_attn_weights.detach()
            }
            
            # Print feature shapes for debugging (optional)
            print(f"Joint Features Shape: {skl_embedded}")
            print(f"Acc Features Shape: {acc_embedded}")
            print(f"Fused Features Shape: {fused_features}")
            print(f"Temporal Features Shape: {temporal_features}")
        
        return logits
    
    def get_features(self):
        """Access stored features for knowledge distillation"""
        if hasattr(self, 'features'):
            return self.features
        return None
