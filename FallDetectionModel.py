import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import model_config as cfg
from ai_edge_torch.generative.layers.attention import TransformerBlock

class FallDetectionModel(nn.Module):
    def __init__(self, 
                 acc_frames=128,
                 acc_coords=4,  # x, y, z, smv
                 embed_dim=32,
                 num_heads=4,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()
        
        # Configure attention
        attn_config = cfg.AttentionConfig(
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
            num_query_groups=num_heads,
            qkv_use_bias=True,
            output_proj_use_bias=True
        )
        
        # Configure feed forward network
        ff_activation = cfg.ActivationConfig(
            type=cfg.ActivationType.GELU
        )
        
        ff_config = cfg.FeedForwardConfig(
            type=cfg.FeedForwardType.SEQUENTIAL,
            activation=ff_activation,
            intermediate_size=embed_dim * 2,
            use_bias=True
        )
        
        # Configure normalization
        norm_config = cfg.NormalizationConfig(
            type=cfg.NormalizationType.LAYER_NORM,
            epsilon=1e-5
        )
        
        # Create transformer block config
        transformer_config = cfg.TransformerBlockConfig(
            attn_config=attn_config,
            ff_config=ff_config,
            pre_attention_norm_config=norm_config,
            post_attention_norm_config=norm_config,
            parallel_residual=False
        )
        
        # Create full model config
        model_config = cfg.ModelConfig(
            vocab_size=1,  # Not used for this model but required
            num_layers=num_layers,
            max_seq_len=acc_frames,
            embedding_dim=embed_dim,
            block_configs=transformer_config,
            final_norm_config=norm_config,
            enable_hlfb=False  # Set to True for high-performance inference
        )
        
        # Input projection - convert accelerometer data to embeddings
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, acc_frames, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        # Create transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TransformerBlock(transformer_config, model_config)
            )
        
        # Output projection
        self.norm = builder.build_norm(embed_dim, norm_config)
        self.output = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Prepare input for 1D convolution
        x = x.transpose(1, 2)  # [batch_size, features, seq_len]
        
        # Apply input projection
        x = self.input_proj(x)  # [batch_size, embed_dim, seq_len]
        
        # Reshape for transformer
        x = x.transpose(1, 2)  # [batch_size, seq_len, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, embed_dim]
        
        # Final classification
        x = self.output(x)  # [batch_size, 1]
        
        return x
