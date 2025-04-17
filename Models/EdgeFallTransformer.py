import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ai_edge_torch.generative.layers.normalization import LayerNorm
from ai_edge_torch.generative.layers.attention import TransformerBlock
from ai_edge_torch.generative.layers.model_config import (
    ModelConfig, TransformerBlockConfig, AttentionConfig, FeedForwardConfig,
    NormalizationConfig, ActivationConfig, ActivationType, FeedForwardType, NormalizationType
)

class EdgeFallTransformer(nn.Module):
    def __init__(self,
                 acc_frames=128,
                 num_classes=1,
                 num_heads=4,
                 acc_coords=4,
                 num_layer=2,
                 embed_dim=32,
                 dropout=0.1):
        super().__init__()
        
        attn_config = AttentionConfig(
            num_heads=num_heads,
            head_dim=embed_dim//num_heads,
            num_query_groups=num_heads,
            qkv_use_bias=True,
            output_proj_use_bias=True
        )
        
        ff_config = FeedForwardConfig(
            type=FeedForwardType.SEQUENTIAL,
            activation=ActivationConfig(type=ActivationType.GELU),
            intermediate_size=embed_dim*2,
            use_bias=True
        )
        
        norm_config = NormalizationConfig(
            type=NormalizationType.LAYER_NORM,
            epsilon=1e-5
        )
        
        block_config = TransformerBlockConfig(
            attn_config=attn_config,
            ff_config=ff_config,
            pre_attention_norm_config=norm_config,
            post_attention_norm_config=norm_config
        )
        
        self.model_config = ModelConfig(
            vocab_size=1,
            num_layers=num_layer,
            max_seq_len=acc_frames,
            embedding_dim=embed_dim,
            block_configs=block_config,
            final_norm_config=norm_config
        )
        
        # Changed padding from 'same' to explicit integer to avoid warning and ensure stability
        self.input_proj = nn.Sequential(
            nn.LayerNorm([acc_coords, acc_frames]),  # Add normalization first
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()  # Use GELU instead of ReLU for better gradient flow
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.model_config.block_config(i), self.model_config)
            for i in range(num_layer)
        ])
        
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)
        
        # Proper initialization
        self._init_weights()
    
    def _init_weights(self):
        def _init_layer(m):
            if isinstance(m, nn.Linear):
                # Use smaller initialization for stability
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(_init_layer)
    
    def forward(self, acc_data, skl_data=None):
        # Apply layer norm to input data for stability
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x)
        x = rearrange(x, 'b c l -> b l c')
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.output(x)
        
        return x, x
