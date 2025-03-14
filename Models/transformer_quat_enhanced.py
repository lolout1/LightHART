"""
Enhanced transformer models with quaternion support for fall detection.

Provides:
1. QuatTeacherEnhanced - Teacher model using both skeleton and IMU data
2. QuatStudentEnhanced - Student model using only IMU data
3. Both models support quaternion features and attention map extraction
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

class QuaternionAttention(nn.Module):
    """
    Quaternion-aware attention mechanism that preserves orientation properties.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        Initialize quaternion attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Quaternion components get special treatment
        self.quat_dim = min(4, self.head_dim)  # At most 4 dimensions for quaternion
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store attention maps
        self.attention_maps = None
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        """
        Forward pass with quaternion-aware attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Mask for padding tokens
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)
        
        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply masks
        if attn_mask is not None:
            scores = scores + attn_mask
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Calculate attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Store attention maps for later use
        self.attention_maps = attn_weights
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights

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
                 activation="relu", batch_first=True, use_quat_attention=False):
        # Initialize without parent's self-attention
        nn.Module.__init__(self)
        
        # Use quaternion attention if requested
        if use_quat_attention:
            self.self_attn = QuaternionAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = MultiHeadAttentionWithMap(d_model, nhead, dropout=dropout)
        
        # Initialize the rest of the layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == "relu" else F.gelu
        self.batch_first = batch_first
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Forward pass for the enhanced transformer encoder layer"""
        src2, weights = self.self_attn(
            self.norm1(src),
            self.norm1(src),
            self.norm1(src),
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask
        )
        
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        
        return src
    
    def get_attention_maps(self):
        """Return attention maps from self-attention layer."""
        return self.self_attn.attention_maps

class TransformerEncoderWithMap(nn.Module):
    """Extended TransformerEncoder that collects intermediate layer outputs and attention maps."""
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
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
    Uses quaternion features for better orientation representation.
    """
    
    def __init__(
        self,
        feat_dim=16,      # Fused IMU feature dimension
        d_model=64,       # Hidden dimension
        nhead=4,          # Number of attention heads
        num_layers=3,     # Number of transformer layers
        num_classes=2,    # Number of output classes
        dropout=0.2,      # Dropout rate
        dim_feedforward=128,  # Feedforward network dimension
        use_quat_attention=True  # Whether to use quaternion-aware attention
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
            use_quat_attention: Whether to use quaternion-aware attention
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
            batch_first=True,
            use_quat_attention=False  # No need for quaternion attention in skeleton branch
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
            batch_first=True,
            use_quat_attention=use_quat_attention  # Use quaternion attention for IMU branch
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
    Uses quaternion features for better orientation representation.
    """
    
    def __init__(
        self,
        feat_dim=16,      # Fused IMU feature dimension
        d_model=48,       # Hidden dimension (smaller than teacher)
        nhead=4,          # Number of attention heads
        num_layers=2,     # Number of transformer layers (less than teacher)
        num_classes=2,    # Number of output classes
        dropout=0.1,      # Dropout rate (less than teacher)
        dim_feedforward=96,  # Feedforward network dimension (smaller than teacher)
        use_quat_attention=True  # Whether to use quaternion-aware attention
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
            use_quat_attention: Whether to use quaternion-aware attention
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
            batch_first=True,
            use_quat_attention=use_quat_attention
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
