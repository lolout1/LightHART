"""
Enhanced transformer models with attention map extraction for cross-modal distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttentionWithMap(nn.MultiheadAttention):
    """Extended MultiheadAttention that stores attention maps."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)
        self.attention_maps = None
    
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, average_attn_weights=True):
        """
        Forward pass that also stores attention maps.
        """
        output, attn_weights = super().forward(
            query, key, value, 
            key_padding_mask=key_padding_mask,
            need_weights=True,  # Always compute weights
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights
        )
        
        # Store attention weights
        self.attention_maps = attn_weights
        
        return output, attn_weights

class TransformerEncoderLayerWithMap(nn.TransformerEncoderLayer):
    """Extended TransformerEncoderLayer that uses MultiHeadAttentionWithMap."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", batch_first=True):
        # Initialize without parent's self-attention
        nn.Module.__init__(self)
        self.self_attn = MultiHeadAttentionWithMap(d_model, nhead, dropout=dropout)
        
        # Initialize the rest of the layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = getattr(F, activation)
        self.batch_first = batch_first
    
    def get_attention_maps(self):
        """Return attention maps from self-attention layer."""
        return self.self_attn.attention_maps

class TransformerEncoderWithMap(nn.TransformerEncoder):
    """Extended TransformerEncoder that collects intermediate layer outputs and attention maps."""
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Forward pass that collects intermediate outputs and attention maps.
        
        Args:
            src: Input tensor
            mask: Attention mask
            src_key_padding_mask: Key padding mask
            
        Returns:
            Tuple of (output, intermediate_outputs, attention_maps)
        """
        output = src
        intermediate_outputs = []
        attention_maps = []
        
        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            intermediate_outputs.append(output)
            attention_maps.append(mod.get_attention_maps())
        
        if self.norm is not None:
            output = self.norm(output)
            intermediate_outputs[-1] = output
        
        return output, intermediate_outputs, attention_maps

class QuatTeacherEnhanced(nn.Module):
    """
    Enhanced teacher model with dual-branch Transformer for skeleton and IMU data.
    Collects intermediate outputs and attention maps for distillation.
    """
    
    def __init__(
        self,
        feat_dim=16,      # Fused IMU feature dimension
        d_model=64,       # Hidden dimension
        nhead=4,          # Number of attention heads
        num_layers=3,     # Number of transformer layers
        num_classes=2,    # Number of output classes
        dropout=0.2,      # Dropout rate
        dim_feedforward=128  # Feedforward network dimension
    ):
        """
        Initialize enhanced teacher model.
        
        Args:
            feat_dim: Dimension of fused IMU features
            d_model: Model hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout: Dropout rate
            dim_feedforward: Dimension of feedforward network
        """
        super().__init__()
        
        # Feature dimensions
        self.feat_dim = feat_dim
        self.d_model = d_model
        
        # === Skeleton Branch ===
        # Input projection for skeleton (96D -> d_model)
        self.skel_in = nn.Linear(96, d_model)
        
        # Positional encoding
        self.skel_pos_enc = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        skel_layer = TransformerEncoderLayerWithMap(
            d_model=d_model,
            nhead=4,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.skel_enc = TransformerEncoderWithMap(skel_layer, num_layers=2)
        
        # === IMU Branch ===
        # Input projection for fused IMU (feat_dim -> d_model)
        self.imu_in = nn.Linear(feat_dim, d_model)
        
        # Positional encoding
        self.imu_pos_enc = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        imu_layer = TransformerEncoderLayerWithMap(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.imu_enc = TransformerEncoderWithMap(imu_layer, num_layers=num_layers)
        
        # === Fusion and Classification ===
        # Fusion layer
        self.fuse = nn.Linear(d_model*2, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Classification layer
        self.out = nn.Linear(d_model, num_classes)
    
    def forward(self, skel_seq, imu_seq, skel_mask=None, imu_mask=None):
        """
        Forward pass with skeleton and IMU data.
        
        Args:
            skel_seq: Skeleton sequence of shape (batch_size, seq_len, 96)
            imu_seq: IMU sequence of shape (batch_size, seq_len, feat_dim)
            skel_mask: Mask for skeleton padding of shape (batch_size, seq_len)
            imu_mask: Mask for IMU padding of shape (batch_size, seq_len)
            
        Returns:
            Dictionary with:
                - logits: Output logits
                - skel_feat: Skeleton features
                - imu_feat: IMU features
                - fused_feat: Fused features
                - skel_intermediates: Intermediate skeleton features
                - imu_intermediates: Intermediate IMU features
                - skel_attentions: Skeleton attention maps
                - imu_attentions: IMU attention maps
        """
        # === Process Skeleton ===
        # Project to d_model
        s_proj = self.skel_in(skel_seq)
        
        # Add positional encoding
        s_proj = self.skel_pos_enc(s_proj)
        
        # Apply transformer encoder
        s_out, s_intermediates, s_attentions = self.skel_enc(
            s_proj, 
            src_key_padding_mask=skel_mask
        )
        
        # Global average pooling (ignoring padding)
        if skel_mask is not None:
            # Create float mask (1.0 for valid positions, 0.0 for padding)
            s_float_mask = (~skel_mask).float().unsqueeze(-1)
            
            # Apply mask and compute average
            s_masked = s_out * s_float_mask
            s_sum = s_masked.sum(dim=1)
            s_len = s_float_mask.sum(dim=1)
            s_feat = s_sum / (s_len + 1e-10)
        else:
            s_feat = s_out.mean(dim=1)
        
        # === Process IMU ===
        # Project to d_model
        i_proj = self.imu_in(imu_seq)
        
        # Add positional encoding
        i_proj = self.imu_pos_enc(i_proj)
        
        # Apply transformer encoder
        i_out, i_intermediates, i_attentions = self.imu_enc(
            i_proj,
            src_key_padding_mask=imu_mask
        )
        
        # Global average pooling (ignoring padding)
        if imu_mask is not None:
            # Create float mask (1.0 for valid positions, 0.0 for padding)
            i_float_mask = (~imu_mask).float().unsqueeze(-1)
            
            # Apply mask and compute average
            i_masked = i_out * i_float_mask
            i_sum = i_masked.sum(dim=1)
            i_len = i_float_mask.sum(dim=1)
            i_feat = i_sum / (i_len + 1e-10)
        else:
            i_feat = i_out.mean(dim=1)
        
        # === Fuse Features ===
        # Concatenate features
        fused = torch.cat([s_feat, i_feat], dim=-1)
        
        # Apply fusion layer
        fused = self.fuse(fused)
        fused = F.relu(fused)
        fused = self.drop(fused)
        
        # Classification
        logits = self.out(fused)
        
        # Return all features and intermediate outputs for distillation
        return {
            "logits": logits,
            "skel_feat": s_feat,
            "imu_feat": i_feat,
            "fused_feat": fused,
            "skel_intermediates": s_intermediates,
            "imu_intermediates": i_intermediates,
            "skel_attentions": s_attentions,
            "imu_attentions": i_attentions
        }

class QuatStudentEnhanced(nn.Module):
    """
    Enhanced student model using only IMU data.
    Collects intermediate outputs and attention maps for distillation.
    """
    
    def __init__(
        self,
        feat_dim=16,      # Fused IMU feature dimension
        d_model=48,       # Hidden dimension (smaller than teacher)
        nhead=4,          # Number of attention heads
        num_layers=2,     # Number of transformer layers (less than teacher)
        num_classes=2,    # Number of output classes
        dropout=0.1,      # Dropout rate (less than teacher)
        dim_feedforward=96  # Feedforward network dimension (smaller than teacher)
    ):
        """
        Initialize enhanced student model.
        
        Args:
            feat_dim: Dimension of fused IMU features
            d_model: Model hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout: Dropout rate
            dim_feedforward: Dimension of feedforward network
        """
        super().__init__()
        
        # Input projection
        self.in_proj = nn.Linear(feat_dim, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayerWithMap(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoderWithMap(encoder_layer, num_layers=num_layers)
        
        # Classification
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, num_classes)
    
    def forward(self, imu_seq, imu_mask=None):
        """
        Forward pass with IMU data only.
        
        Args:
            imu_seq: IMU sequence of shape (batch_size, seq_len, feat_dim)
            imu_mask: Mask for IMU padding of shape (batch_size, seq_len)
            
        Returns:
            Dictionary with:
                - logits: Output logits
                - feat: Final features before classification
                - intermediates: Intermediate layer outputs
                - attentions: Attention maps from each layer
        """
        # Project to d_model
        proj = self.in_proj(imu_seq)
        
        # Add positional encoding
        proj = self.pos_enc(proj)
        
        # Apply transformer encoder
        out, intermediates, attentions = self.encoder(
            proj,
            src_key_padding_mask=imu_mask
        )
        
        # Global average pooling (ignoring padding)
        if imu_mask is not None:
            # Create float mask (1.0 for valid positions, 0.0 for padding)
            float_mask = (~imu_mask).float().unsqueeze(-1)
            
            # Apply mask and compute average
            masked = out * float_mask
            sum_val = masked.sum(dim=1)
            seq_len = float_mask.sum(dim=1)
            feat = sum_val / (seq_len + 1e-10)
        else:
            feat = out.mean(dim=1)
        
        # Apply dropout
        feat = self.drop(feat)
        
        # Classification
        logits = self.out(feat)
        
        # Return features and intermediate outputs for distillation
        return {
            "logits": logits,
            "feat": feat,
            "intermediates": intermediates,
            "attentions": attentions
        }
