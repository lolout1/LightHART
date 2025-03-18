# Models/transformer.py

import torch 
from torch import nn
from typing import Dict, Tuple
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math
import logging

logger = logging.getLogger("model")

class TransformerEncoderWAttention(nn.TransformerEncoder):
    """
    Transformer encoder that also returns attention weights
    
    This extends the standard TransformerEncoder to also return
    attention weights for visualization and analysis.
    """
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        self.attention_weights = []
        
        for layer in self.layers:
            output, attn = layer.self_attn(
                output, output, output, 
                attn_mask=mask,
                key_padding_mask=src_key_padding_mask, 
                need_weights=True
            )
            
            self.attention_weights.append(attn)
            output = layer(
                output, 
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask
            )
            
        return output

class TransModel(nn.Module):
    """
    Transformer model for inertial sensor data processing
    
    This model is optimized for accelerometer and gyroscope data
    to perform fall detection.
    """
    
    def __init__(self,
                 mocap_frames=64,
                 num_joints=32,
                 acc_frames=64,
                 num_classes=2, 
                 num_heads=4, 
                 acc_coords=3, 
                 av=False,
                 num_layers=2, 
                 norm_first=True, 
                 embed_dim=32, 
                 activation='relu',
                 **kwargs):
        """
        Initialize the transformer model
        
        Args:
            mocap_frames: Number of motion capture frames (unused)
            num_joints: Number of skeleton joints (unused)
            acc_frames: Number of accelerometer frames
            num_classes: Number of output classes (2 for fall detection)
            num_heads: Number of attention heads
            acc_coords: Number of accelerometer coordinates (3 for x,y,z)
            av: Whether to use audio-visual fusion (unused)
            num_layers: Number of transformer layers
            norm_first: Whether to apply normalization before attention
            embed_dim: Embedding dimension
            activation: Activation function
        """
        super().__init__()
        
        # Store data shape parameters
        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        self.num_classes = num_classes
        
        # Input projection layers - convert raw sensor data to embeddings
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim*2),
            nn.ReLU(),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=3, padding='same')
        )
        
        # Transformer encoder layers
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.3,
            activation=activation,
            batch_first=True,
            norm_first=norm_first
        )
        
        # Full transformer encoder
        self.encoder = TransformerEncoderWAttention(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(embed_dim)
        )
        
        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        # Initialize output layer weights for better convergence
        nn.init.xavier_normal_(self.classifier[-1].weight)
        
        logger.info(f"Initialized TransModel: embed_dim={embed_dim}, num_layers={num_layers}")
    
    def forward(self, data):
        """
        Forward pass through the model
        
        Args:
            data: Either accelerometer data tensor [batch, seq_len, 3]
                 or dictionary with 'accelerometer' key
        
        Returns:
            Class logits [batch, num_classes]
        """
        # Handle dictionary input
        if isinstance(data, dict):
            acc_data = data['accelerometer']
        else:
            acc_data = data
        
        # Get batch and sequence dimensions
        b, l, c = acc_data.shape
        
        # Apply input projection
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x)
        x = rearrange(x, 'b c l -> b l c')
        
        # Apply transformer encoder
        x = self.encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Apply classification head
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    # Test the model
    data = torch.randn(size=(16, 64, 3))
    model = TransModel()
    output = model(data)
    print(f"Output shape: {output.shape}")
