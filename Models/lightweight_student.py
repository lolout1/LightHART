import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class EnhancedLightweightStudent(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        norm_first: bool = True,
        embed_dim: int = 32,
        activation: str = 'gelu',
        acc_coords: int = 4,
        num_classes: int = 2,
        acc_frames: int = 128,
        num_heads: int = 2,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()
        
        self.acc_frames = acc_frames
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        
        # Initial feature extraction
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Self-attention modules
        self.channel_attention = nn.Sequential(
            nn.Linear(acc_frames, acc_frames // 8),
            nn.LayerNorm(acc_frames // 8),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(acc_frames // 8, acc_frames),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 8),
            nn.LayerNorm(embed_dim // 8),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(embed_dim // 8, embed_dim),
            nn.Sigmoid()
        )
        
        # Positional encoding
        self.register_buffer(
            'pos_embedding',
            self._get_positional_embedding(acc_frames, embed_dim)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4,
                dropout=dropout,
                norm_first=norm_first
            ) for _ in range(num_layers)
        ])
        
        # Feature fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Final classifier
        final_dim = embed_dim * num_layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, embed_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.apply(self._init_weights)
        
    def _get_positional_embedding(self, seq_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, acc_data: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            B = acc_data.shape[0]
            print(f"\nInput shape: {acc_data.shape}")
            
            # Handle input dimensions and initial feature extraction
            if acc_data.dim() == 3 and acc_data.size(2) == self.acc_coords:
                x = acc_data.transpose(1, 2)
            else:
                x = acc_data.unsqueeze(0)
                
            # Feature extraction through conv layers
            x = self.acc_encoder(x)  # [B, embed_dim, seq_len]
            print(f"After encoder shape: {x.shape}")
            
            # Apply attention mechanisms
            # Channel attention
            chan_attn = self.spatial_attention(x.mean(dim=2))  # [B, embed_dim]
            x = x * chan_attn.unsqueeze(2)
            
            # Temporal attention
            temp_attn = self.channel_attention(x.mean(dim=1))  # [B, seq_len]
            x = x * temp_attn.unsqueeze(1)
            
            # Convert to sequence format [B, seq_len, embed_dim]
            x = x.transpose(1, 2)
            print(f"After attention shape: {x.shape}")
            
            # Add positional encoding
            x = x + self.pos_embedding[:, :x.size(1)]
            
            # Add CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            print(f"After adding CLS token shape: {x.shape}")
            
            # Process through transformer layers and collect features
            features = []
            for i, (transformer, fusion) in enumerate(zip(self.transformer_layers, self.fusion_layers)):
                x = transformer(x)
                features.append(fusion(x[:, 0]))  # Use CLS token
                print(f"Feature {i+1} shape: {features[-1].shape}")
            
            # Concatenate features for classification
            x = torch.cat(features, dim=-1)
            print(f"Concatenated features shape: {x.shape}")
            
            # Final classification
            x = self.classifier(x)
            print(f"Output shape: {x.shape}")
            
            return x
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Shapes during error:")
            if 'x' in locals():
                print(f"Current tensor shape: {x.shape}")
            raise

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, 
                 dropout: float = 0.1, norm_first: bool = True):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            attn_out = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
            out = attn_out + self.dropout(self.mlp(self.norm2(attn_out)))
        else:
            attn_out = self.norm1(x + self.dropout(self.attn(x, x, x)[0]))
            out = self.norm2(attn_out + self.dropout(self.mlp(attn_out)))
        return out
