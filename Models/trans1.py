import torch 
from torch import nn
from typing import Dict, Tuple
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer, ModuleList
import torch.nn.functional as F
from einops import rearrange
import itertools
import numpy as np
#from util.graph import Graph
import math

class TransformerEncoderWAttention(nn.TransformerEncoder):
    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src
        self.attention_weights = []
        for layer in self.layers :
            output, attn = layer.self_attn(output, output, output, attn_mask = mask,
                                            key_padding_mask = src_key_padding_mask, need_weights = True)
            self.attention_weights.append(attn)
            output = layer(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
        return output


class TransModel(nn.Module):
    def __init__(self, num_heads=4, num_layers=2, norm_first=True, embed_dim=64, activation='relu', input_dim=4, num_classes=2):
        super().__init__()
        # Input projection using Conv1D
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.ln1 = nn.Linear(embed_dim, embed_dim//2)
        self.ln2 = nn.Linear(embed_dim//2, embed_dim//4)
        self.output = nn.Linear(embed_dim//4, 1)  # Single output for binary classification
        
    def forward(self, acc_data, skl_data=None):
        # Calculate SMV from accelerometer data
        mean = torch.mean(acc_data, dim=1, keepdim=True)
        zero_mean = acc_data - mean
        sum_squared = torch.sum(torch.square(zero_mean), dim=-1, keepdim=True)
        smv = torch.sqrt(sum_squared)
        
        # Concatenate SMV with accelerometer data
        x = torch.cat([acc_data, smv], dim=-1)  # [B, T, 4]
        
        # Input projection
        x = x.transpose(1, 2)  # [B, 4, T]
        x = self.input_proj(x)  # [B, embed_dim, T]
        x = x.transpose(1, 2)  # [B, T, embed_dim]
        
        # Transformer encoding
        x = self.encoder(x)  # [B, T, embed_dim]
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # [B, embed_dim]
        
        # Classification
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        logits = self.output(x)
        probs = torch.sigmoid(logits)
        
        return probs

if __name__ == "__main__":
        data = torch.randn(size = (16,128,4))
        skl_data = torch.randn(size = (16,128,32,3))
        model = TransModel()
        output = model(data, skl_data)