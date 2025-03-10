import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerTeacherFusion(nn.Module):
    """
    Enhanced transformer teacher model that can handle fused IMU data.
    """
    def __init__(self,
                 num_joints=32,
                 joint_dim=3,
                 accel_dim=15,  # Full fusion features (acc, gyro, lin_acc, orientation, derived)
                 hidden_skel=128,
                 hidden_accel=128,
                 accel_heads=4,
                 accel_layers=3,
                 skeleton_heads=4,
                 skeleton_layers=2,
                 fusion_hidden=256,
                 num_classes=2,
                 dropout=0.2,
                 dim_feedforward=256):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.accel_dim = accel_dim
        self.hidden_skel = hidden_skel
        self.hidden_accel = hidden_accel
        self.accel_heads = accel_heads
        self.accel_layers = accel_layers
        self.skeleton_heads = skeleton_heads
        self.skeleton_layers = skeleton_layers
        self.fusion_hidden = fusion_hidden
        self.num_classes = num_classes
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward

        # Feature dimensions
        self.skel_feat_dim = num_joints * joint_dim
        
        # 1. Accelerometer pathway
        # Projection: raw_accel (with fusion features) => hidden_accel
        self.accel_proj = nn.Linear(accel_dim, hidden_accel)
        
        # Transformer layers for accelerometer
        accel_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_accel,
            nhead=accel_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.accel_encoder = nn.TransformerEncoder(
            accel_encoder_layer,
            num_layers=accel_layers
        )
        
        # 2. Skeleton pathway
        # Projection: skeleton => hidden_skel
        self.skel_proj = nn.Linear(self.skel_feat_dim, hidden_skel)
        
        # Transformer layers for skeleton
        skel_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_skel,
            nhead=skeleton_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.skel_encoder = nn.TransformerEncoder(
            skel_encoder_layer,
            num_layers=skeleton_layers
        )
        
        # 3. Fusion of modalities
        # Projection: hidden_accel + hidden_skel => fusion_hidden
        self.fusion_proj = nn.Linear(hidden_accel + hidden_skel, fusion_hidden)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, skel_seq, accel_seq, accel_mask=None, skel_mask=None):
        """
        Forward pass with support for variable-length sequences and fusion features
        
        Args:
            skel_seq: List of skeleton tensors or padded tensor, each (L_i, num_joints*3) or (B, L, num_joints*3)
            accel_seq: Accelerometer tensor with fusion features, shape (B, L, accel_dim)
            accel_mask: Boolean mask for accelerometer, shape (B, L), True=padding
            skel_mask: Boolean mask for skeleton, shape (B, L), True=padding
            
        Returns:
            Dictionary with logits, accel_feat, and skel_feat
        """
        device = accel_seq.device
        B = accel_seq.shape[0]
        
        # Handle accelerometer branch first
        # Project to hidden dimension
        accel_proj = self.accel_proj(accel_seq)
        
        # Apply accelerometer transformer
        accel_encoded = self.accel_encoder(accel_proj, src_key_padding_mask=accel_mask)
        
        # Global average pooling (ignoring padding)
        if accel_mask is not None:
            # Create an inverse mask (1 for real data, 0 for padding)
            accel_inv_mask = (~accel_mask).float().unsqueeze(-1)
            # Apply mask and compute mean
            accel_feat = (accel_encoded * accel_inv_mask).sum(dim=1) / (accel_inv_mask.sum(dim=1) + 1e-6)
        else:
            accel_feat = accel_encoded.mean(dim=1)
        
        # Handle skeleton branch - can be list or padded tensor
        if isinstance(skel_seq, list):
            # Process each skeleton sequence separately
            all_skel_feats = []
            for skel in skel_seq:
                # Add batch dimension if needed
                if skel.dim() == 2:
                    skel = skel.unsqueeze(0)
                
                # Project skeleton to hidden dimension
                skel_proj = self.skel_proj(skel)
                
                # Apply skeleton transformer (no masking needed for individual sequences)
                skel_encoded = self.skel_encoder(skel_proj)
                
                # Pool across time
                skel_feat = skel_encoded.mean(dim=1)
                all_skel_feats.append(skel_feat)
            
            # Stack all skeleton features
            skel_feat = torch.cat(all_skel_feats, dim=0)
        else:
            # Padded tensor approach
            skel_proj = self.skel_proj(skel_seq)
            skel_encoded = self.skel_encoder(skel_proj, src_key_padding_mask=skel_mask)
            
            # Global average pooling (ignoring padding)
            if skel_mask is not None:
                skel_inv_mask = (~skel_mask).float().unsqueeze(-1)
                skel_feat = (skel_encoded * skel_inv_mask).sum(dim=1) / (skel_inv_mask.sum(dim=1) + 1e-6)
            else:
                skel_feat = skel_encoded.mean(dim=1)
        
        # Concatenate features from both branches
        combined_feat = torch.cat([accel_feat, skel_feat], dim=-1)
        
        # Project to fusion space
        fusion_feat = self.fusion_proj(combined_feat)
        
        # Apply classifier
        logits = self.classifier(fusion_feat)
        
        return {
            "logits": logits,
            "accel_feat": accel_feat,
            "skel_feat": skel_feat,
            "fusion_feat": fusion_feat
        }
