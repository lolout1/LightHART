# Models/transformer_edge.py
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from ai_edge_torch.generative.layers import model_config as cfg
from ai_edge_torch.generative.layers.normalization import LayerNorm
from ai_edge_torch.generative.layers.attention import TransformerBlock

class EdgeTransModel(nn.Module):
    def __init__(self,
                acc_frames=128,
                num_classes=1,
                num_heads=2,
                acc_coords=3,
                num_layer=2,
                norm_first=True,
                embed_dim=16,
                activation='relu',
                dropout=0.5,
                enable_hlfb=True,
                **kwargs):
        super().__init__()
        
        # Input projection (unchanged from original)
        self.input_proj = nn.Sequential(
            nn.Conv1d(4, embed_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(embed_dim)
        )
        
        # Configure normalization
        norm_config = cfg.NormalizationConfig(
            type=cfg.NormalizationType.LAYER_NORM,
            enable_hlfb=enable_hlfb,
            epsilon=1e-5
        )
        
        # Configure activation
        act_type = cfg.ActivationType.RELU if activation == 'relu' else cfg.ActivationType.GELU
        act_config = cfg.ActivationConfig(type=act_type)
        
        # Configure attention
        attn_config = cfg.AttentionConfig(
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
            num_query_groups=num_heads,
            qkv_use_bias=True,
            output_proj_use_bias=True,
            query_norm_config=norm_config,
            key_norm_config=norm_config,
            enable_kv_cache=False
        )
        
        # Configure feed-forward
        ff_config = cfg.FeedForwardConfig(
            type=cfg.FeedForwardType.SEQUENTIAL,
            activation=act_config,
            intermediate_size=embed_dim * 2,
            use_bias=True,
            pre_ff_norm_config=norm_config,
            post_ff_norm_config=norm_config
        )
        
        # Configure transformer block
        block_config = cfg.TransformerBlockConfig(
            attn_config=attn_config,
            ff_config=ff_config,
            pre_attention_norm_config=norm_config,
            post_attention_norm_config=norm_config,
            parallel_residual=False
        )
        
        # Create model config
        model_config = cfg.ModelConfig(
            vocab_size=1,  # Not used but required
            num_layers=num_layer,
            max_seq_len=acc_frames,
            embedding_dim=embed_dim,
            block_configs=block_config,
            enable_hlfb=enable_hlfb
        )
        
        # Create transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(block_config, model_config)
            for _ in range(num_layer)
        ])
        
        # Temporal normalization
        self.temporal_norm = LayerNorm(dim=embed_dim, eps=1e-5, enable_hlfb=enable_hlfb)
        
        # Output layer
        self.output = nn.Linear(embed_dim, num_classes)
    
    def forward(self, acc_data, skl_data=None):
        # Process input (unchanged from original)
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x)
        x = rearrange(x, 'b c l -> b l c')
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply normalization
        x = self.temporal_norm(x)
        feature = x
        
        # Global pooling and output (unchanged from original)
        x = rearrange(x, 'b f c -> b c f')
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)
        x = rearrange(x, 'b c f -> b (c f)')
        x = self.output(x)
        
        # Match the original model's return signature: (output, feature)
        return x, feature
