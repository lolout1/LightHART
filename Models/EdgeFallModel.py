import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from einops import rearrange

logger = logging.getLogger(__name__)

class StableSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim) ** -0.5
        
        # Single projection matrix for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.01)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Project to q, k, v with careful shape handling
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scale query for numerical stability
        q = q * self.scale
        
        # Attention with gradient clamping for stability
        attn = (q @ k.transpose(-2, -1))
        attn = torch.clamp(attn, min=-10.0, max=10.0)  # Prevent extreme values
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class StableTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # Pre-norm architecture for better stability
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = StableSelfAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),  # More stable than ReLU
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Small weight initialization
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Residual connections with pre-norm
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EdgeFallTransformer(nn.Module):
    """Fall detection model optimized for AI Edge Torch conversion"""
    
    def __init__(self, 
                 acc_frames=128, 
                 num_classes=1, 
                 num_heads=4, 
                 acc_coords=4, 
                 num_layer=2, 
                 embed_dim=32, 
                 dropout=0.1,
                 debug=False):
        super().__init__()
        
        self.debug = debug
        self.acc_frames = acc_frames
        self.acc_coords = acc_coords
        
        # Input projection with SMV + 3-axis acceleration
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU()  # More stable activation
        )
        
        # Positional encoding
        self.register_buffer(
            'pos_encoding', 
            self._create_pos_encoding(acc_frames, embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            StableTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=2.0,  # Smaller ratio for stability
                dropout=dropout
            ) for _ in range(num_layer)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)
        
        # Initialize with small weights
        self._init_weights()
        
        if self.debug:
            logger.info(f"Initialized EdgeFallTransformer with {num_layer} layers, {num_heads} heads, embed_dim={embed_dim}")
    
    def _create_pos_encoding(self, length, dim):
        """Create sinusoidal positional embeddings"""
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_enc = torch.zeros(1, length, dim)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        return pos_enc
    
    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Extract shape from input
        batch_size, seq_len, features = x.shape
        
        if self.debug:
            logger.info(f"Input shape: {x.shape}")
            logger.info(f"Input stats - mean: {x.mean():.4f}, std: {x.std():.4f}, min: {x.min():.4f}, max: {x.max():.4f}")
            # Log SMV values (first channel)
            smv = x[:, :, 0]
            logger.info(f"SMV stats - mean: {smv.mean():.4f}, std: {smv.std():.4f}, min: {smv.min():.4f}, max: {smv.max():.4f}")
        
        # Input projection
        x = rearrange(x, 'b l c -> b c l')
        x = self.input_proj(x)
        x = rearrange(x, 'b c l -> b l c')
        
        if self.debug:
            logger.info(f"After input projection: {x.shape}, mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.debug and i == 0:
                logger.info(f"After block {i}: mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        # Global pooling and output projection
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        features = x
        
        if self.debug:
            logger.info(f"Features shape: {features.shape}, mean: {features.mean():.4f}, std: {features.std():.4f}")
        
        x = self.dropout(x)
        
        # Output logits (not sigmoid - will be handled in loss function)
        logits = self.output(x)
        
        if self.debug:
            logger.info(f"Output logits: {logits.shape}, values: {logits.detach().cpu().numpy()}")
        
        return logits, features
