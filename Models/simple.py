# fall_detection_transformer.py
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

class FallDetectionTransformer(nn.Module):
    """
    FallDetectionTransformer optimized for TFLite conversion via AI Edge Torch.
    Takes accelerometer and gyroscope inputs only, with explicit tensor handling.
    """
    def __init__(self,
                seq_length=64,
                num_classes=2,
                num_heads=2,
                input_channels=3,
                num_layers=2,
                embed_dim=32,
                dropout=0.2,
                use_batch_norm=True):
        super().__init__()
        
        # Save parameters
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Accelerometer encoder (simple structure)
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        # Gyroscope encoder (simple structure)
        self.gyro_encoder = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        # Fusion dimension (concatenation)
        self.fusion_dim = embed_dim * 2
        
        # Position encoding
        self.register_buffer(
            "position_ids", 
            torch.arange(0, seq_length).expand((1, -1))
        )
        self.position_embeddings = nn.Embedding(seq_length, self.fusion_dim)
        
        # Transformer with explicit parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=num_heads,
            dim_feedforward=self.fusion_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.fusion_dim)
        )
        
        # Classification head (simple structure)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, acc_data, gyro_data):
        """
        Forward pass with explicit accelerometer and gyroscope tensor inputs.
        
        Args:
            acc_data: Accelerometer data [batch_size, seq_length, channels]
            gyro_data: Gyroscope data [batch_size, seq_length, channels]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Ensure 3D inputs with consistent shape
        batch_size = acc_data.shape[0]
        
        # Handle 2D inputs (add sequence dimension if needed)
        if len(acc_data.shape) == 2:
            acc_data = acc_data.unsqueeze(1)
        if len(gyro_data.shape) == 2:
            gyro_data = gyro_data.unsqueeze(1)
        
        # Process accelerometer data: [batch, seq, channels] -> [batch, channels, seq]
        acc_data = acc_data.transpose(1, 2)
        acc_features = self.acc_encoder(acc_data)
        # Back to [batch, seq, embed_dim]
        acc_features = acc_features.transpose(1, 2)
        
        # Process gyroscope data: [batch, seq, channels] -> [batch, channels, seq]
        gyro_data = gyro_data.transpose(1, 2)
        gyro_features = self.gyro_encoder(gyro_data)
        # Back to [batch, seq, embed_dim]
        gyro_features = gyro_features.transpose(1, 2)
        
        # Concatenate features along embedding dimension
        # [batch, seq, embed_dim*2]
        fused_features = torch.cat([acc_features, gyro_features], dim=2)
        
        # Add positional embeddings
        position_embeddings = self.position_embeddings(self.position_ids)
        fused_features = fused_features + position_embeddings
        
        # Apply transformer
        # [batch, seq, fusion_dim]
        transformer_output = self.transformer(fused_features)
        
        # Global average pooling over sequence dimension
        # [batch, fusion_dim]
        pooled_output = torch.mean(transformer_output, dim=1)
        
        # Classification
        # [batch, num_classes]
        logits = self.classifier(pooled_output)
        
        return logits

    def predict(self, acc_data, gyro_data):
        """Helper method for inference that applies sigmoid to get probabilities"""
        with torch.no_grad():
            logits = self.forward(acc_data, gyro_data)
            probabilities = torch.sigmoid(logits)
            return probabilities
