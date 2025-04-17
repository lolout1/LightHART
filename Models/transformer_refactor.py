# transformer_export.py
import torch
from torch import nn
import torch.nn.functional as F

# Import AI Edge Torch components
from ai_edge_torch.generative.layers import builder, normalization, attention, model_config
from ai_edge_torch.hlfb import StableHLOCompositeBuilder

class TransModelExport(nn.Module):
    def __init__(self,
                acc_frames=128,
                num_classes=1,
                num_heads=2,
                acc_coords=4,
                num_layer=2,
                embed_dim=16,
                activation='relu',
                **kwargs):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(embed_dim)
        )
        
        # Normalization config
        self.norm_config = model_config.NormalizationConfig(
            type=model_config.NormalizationType.LAYER_NORM,
            enable_hlfb=True,
            epsilon=1e-5
        )
        
        # Attention config
        self.attn_config = model_config.AttentionConfig(
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
            num_query_groups=num_heads,
            qkv_use_bias=True,
            output_proj_use_bias=True
        )
        
        # Feed Forward config
        activation_type = model_config.ActivationType.GELU if activation == 'gelu' else model_config.ActivationType.RELU
        self.ff_config = model_config.FeedForwardConfig(
            type=model_config.FeedForwardType.SEQUENTIAL,
            activation=model_config.ActivationConfig(
                type=activation_type
            ),
            intermediate_size=embed_dim*2,
            use_bias=True,
            pre_ff_norm_config=self.norm_config,
            post_ff_norm_config=self.norm_config
        )
        
        # Transformer block config
        self.block_config = model_config.TransformerBlockConfig(
            attn_config=self.attn_config,
            ff_config=self.ff_config,
            pre_attention_norm_config=self.norm_config,
            post_attention_norm_config=self.norm_config,
            parallel_residual=False
        )
        
        # Model config
        self.model_config = model_config.ModelConfig(
            vocab_size=1,  # Not used for this model
            num_layers=num_layer,
            max_seq_len=acc_frames,
            embedding_dim=embed_dim,
            block_configs=self.block_config,
            final_norm_config=self.norm_config,
            enable_hlfb=True
        )
        
        # Create transformer blocks
        self.blocks = nn.ModuleList([
            attention.TransformerBlock(self.block_config, self.model_config)
            for _ in range(num_layer)
        ])
        
        # Final normalization
        self.temporal_norm = normalization.LayerNorm(embed_dim, enable_hlfb=True)
        
        # Output projection
        self.output = nn.Linear(embed_dim, num_classes)
    
    def forward(self, acc_data):
        # Process accelerometer data
        x = torch.transpose(acc_data, 1, 2)  # [B, L, C] -> [B, C, L]
        x = self.input_proj(x)
        x = torch.transpose(x, 1, 2)  # [B, C, L] -> [B, L, C]
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final norm
        x = self.temporal_norm(x)
        
        # Global pooling and output projection
        x = torch.transpose(x, 1, 2)  # [B, L, C] -> [B, C, L]
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)
        x = torch.flatten(x, 1)
        x = self.output(x)
        
        return x
