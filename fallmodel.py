import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import model_config as cfg
from ai_edge_torch.generative.layers.attention import TransformerBlock

class FallDetectionModel(nn.Module):
    def __init__(self, acc_frames=128, acc_coords=4, embed_dim=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        attn_config = cfg.AttentionConfig(
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
            num_query_groups=num_heads,
            qkv_use_bias=True,
            output_proj_use_bias=True
        )
        
        ff_activation = cfg.ActivationConfig(
            type=cfg.ActivationType.RELU
        )
        
        ff_config = cfg.FeedForwardConfig(
            type=cfg.FeedForwardType.SEQUENTIAL,
            activation=ff_activation,
            intermediate_size=embed_dim * 2,
            use_bias=True
        )
        
        norm_config = cfg.NormalizationConfig(
            type=cfg.NormalizationType.LAYER_NORM,
            epsilon=1e-5
        )
        
        transformer_config = cfg.TransformerBlockConfig(
            attn_config=attn_config,
            ff_config=ff_config,
            pre_attention_norm_config=norm_config,
            post_attention_norm_config=norm_config,
            parallel_residual=False
        )
        
        self.model_config = cfg.ModelConfig(
            vocab_size=1,
            num_layers=num_layers,
            max_seq_len=acc_frames,
            embedding_dim=embed_dim,
            block_configs=transformer_config,
            final_norm_config=norm_config,
            enable_hlfb=False
        )
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        position = torch.arange(acc_frames).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_enc = torch.zeros(1, acc_frames, embed_dim)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_enc)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TransformerBlock(transformer_config, self.model_config)
            )
        
        self.norm = builder.build_norm(embed_dim, norm_config)
        self.output = nn.Linear(embed_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
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
        batch_size, seq_len, features = x.shape
        
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        
        x = x + self.pos_encoding
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        
        logits = self.output(x)
        features = x
        
        return logits, features
