# Models/fusion_transformer.py
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import logging
import traceback

# Configure logging
logger = logging.getLogger("model")

class FusionTransModel(nn.Module):
    def __init__(self,
                acc_frames=64,
                mocap_frames=64,
                num_classes=2,
                num_heads=4,
                acc_coords=3,
                quat_coords=4,
                num_layers=2,
                embed_dim=32,
                fusion_type='concat',
                dropout=0.3,
                use_batch_norm=True,
                feature_dim=None,
                **kwargs):
        """
        Transformer model optimized for IMU fusion with robust handling of different sequence lengths.
        
        Args:
            acc_frames: Number of frames in acceleration data
            mocap_frames: Number of frames in mocap/skeleton data
            num_classes: Number of output classes
            num_heads: Number of attention heads
            acc_coords: Number of acceleration coordinates (3)
            quat_coords: Number of quaternion coordinates (4)
            num_layers: Number of transformer layers
            embed_dim: Embedding dimension for features
            fusion_type: How to combine different sensor data ('concat', 'attention', 'weighted')
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            feature_dim: Optional explicit feature dimension (computed automatically if None)
        """
        super().__init__()
        logger.info(f"Initializing FusionTransModel with fusion_type={fusion_type}")
        self.fusion_type = fusion_type
        self.acc_frames = acc_frames
        self.mocap_frames = mocap_frames
        self.embed_dim = embed_dim
        
        # Ensure sequence lengths are consistent
        self.seq_len = self.acc_frames  # Use accelerometer frames as reference
        
        # Accelerometer encoder
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        # Gyroscope encoder
        self.gyro_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        # Quaternion encoder
        self.quat_encoder = nn.Sequential(
            nn.Conv1d(quat_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        # Sequence length adapters for each modality
        self.acc_adapter = nn.Linear(acc_frames, self.seq_len)
        self.gyro_adapter = nn.Linear(acc_frames, self.seq_len)
        self.quat_adapter = nn.Linear(acc_frames, self.seq_len)

        # Feature dimension calculations based on fusion type
        if fusion_type == 'concat':
            # All three modalities concatenated
            self.feature_dim = embed_dim * 3
        elif fusion_type == 'attention':
            # Feature dimension stays the same with attention fusion
            self.feature_dim = embed_dim
            # Cross-attention for fusion
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        elif fusion_type == 'weighted':
            # Weighted combination of features
            self.feature_dim = embed_dim
            # Learned weights for each modality
            self.modality_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            # Default to concatenation
            logger.warning(f"Unknown fusion type '{fusion_type}', defaulting to 'concat'")
            self.fusion_type = 'concat'
            self.feature_dim = embed_dim * 3

        # Override feature_dim if explicitly provided
        if feature_dim is not None:
            self.feature_dim = feature_dim
            logger.info(f"Using provided feature_dim: {feature_dim}")

        # Feature projector for fusion features
        self.fusion_feature_projector = nn.Linear(43, embed_dim)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=self.feature_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.feature_dim)
        )

        # Classification head with regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def _adapt_sequence_length(self, x, adapter):
        """Adapt sequence length to the standard model sequence length"""
        # x shape: [batch, channels, seq_len]
        batch, channels, seq_len = x.shape
        
        if seq_len == self.seq_len:
            # Already the right length
            return x
        
        # Transpose for the linear adapter
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        # Apply adapter
        x = adapter(x)  # [batch, self.seq_len, channels]
        # Transpose back
        x = x.transpose(1, 2)  # [batch, channels, self.seq_len]
        
        return x

    def forward_all_modalities(self, acc_data, gyro_data, quat_data=None):
        """
        Forward pass with accelerometer, gyroscope, and optional quaternion data
        
        Args:
            acc_data: Accelerometer data [batch, seq_len, 3]
            gyro_data: Gyroscope data [batch, seq_len, 3]
            quat_data: Optional quaternion data [batch, seq_len, 4]
            
        Returns:
            Class logits
        """
        # Process accelerometer data
        acc_data = rearrange(acc_data, 'b l c -> b c l')
        acc_features = self.acc_encoder(acc_data)
        acc_features = self._adapt_sequence_length(acc_features, self.acc_adapter)
        
        # Process gyroscope data
        gyro_data = rearrange(gyro_data, 'b l c -> b c l')
        gyro_features = self.gyro_encoder(gyro_data)
        gyro_features = self._adapt_sequence_length(gyro_features, self.gyro_adapter)
        
        # Process quaternion data if available
        if quat_data is not None:
            quat_data = rearrange(quat_data, 'b l c -> b c l')
            quat_features = self.quat_encoder(quat_data)
            quat_features = self._adapt_sequence_length(quat_features, self.quat_adapter)
        else:
            # Create zero tensor if quaternion data is not available
            batch_size = acc_data.shape[0]
            quat_features = torch.zeros(batch_size, self.embed_dim, self.seq_len, 
                                         device=acc_data.device)
        
        # Perform fusion based on fusion type
        if self.fusion_type == 'concat':
            # Concatenate features along feature dimension
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            gyro_features = rearrange(gyro_features, 'b c l -> b l c')
            quat_features = rearrange(quat_features, 'b c l -> b l c')
            
            # Concatenate features
            fused_features = torch.cat([acc_features, gyro_features, quat_features], dim=2)
            
        elif self.fusion_type == 'attention':
            # Use cross-attention for fusion
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            gyro_features = rearrange(gyro_features, 'b c l -> b l c')
            quat_features = rearrange(quat_features, 'b c l -> b l c')
            
            # Use accelerometer as query, and concatenated gyro+quat as key/value
            kv_features = (gyro_features + quat_features) / 2
            fused_features, _ = self.cross_attention(
                query=acc_features,
                key=kv_features,
                value=kv_features
            )
            
        elif self.fusion_type == 'weighted':
            # Apply learned weights to each modality
            weights = F.softmax(self.modality_weights, dim=0)
            
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            gyro_features = rearrange(gyro_features, 'b c l -> b l c')
            quat_features = rearrange(quat_features, 'b c l -> b l c')
            
            # Weighted sum
            fused_features = (
                weights[0] * acc_features + 
                weights[1] * gyro_features + 
                weights[2] * quat_features
            )
        
        # Apply transformer
        transformer_output = self.transformer(fused_features)
        
        # Global average pooling
        pooled = torch.mean(transformer_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

    def forward_fusion_features(self, acc_data, fusion_features):
        """
        Forward pass using accelerometer data and pre-computed fusion features
        
        Args:
            acc_data: Accelerometer data [batch, seq_len, 3]
            fusion_features: Pre-computed fusion features [batch, feature_dim]
            
        Returns:
            Class logits
        """
        # Process accelerometer data
        acc_data = rearrange(acc_data, 'b l c -> b c l')
        acc_features = self.acc_encoder(acc_data)
        acc_features = self._adapt_sequence_length(acc_features, self.acc_adapter)
        acc_features = rearrange(acc_features, 'b c l -> b l c')
        
        # Process fusion features
        batch_size, seq_len, _ = acc_features.shape
        
        # Project fusion features to match embedding dimension
        fusion_embed = self.fusion_feature_projector(fusion_features)
        
        # Expand fusion features to match sequence length
        expanded_features = fusion_embed.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate features
        if self.fusion_type == 'concat':
            # Create zero tensor for missing modality
            dummy_features = torch.zeros(batch_size, seq_len, self.embed_dim, 
                                         device=acc_data.device)
            
            # Concatenate all features 
            fused_features = torch.cat([acc_features, dummy_features, expanded_features], dim=2)
            
            # Ensure correct feature dimension
            if fused_features.size(2) != self.feature_dim:
                fused_features = fused_features[:, :, :self.feature_dim]
        else:
            # For other fusion types, just use the accelerometer and fusion features
            fused_features = torch.cat([acc_features, expanded_features], dim=2)
            
            # Project to feature dimension if needed
            if fused_features.size(2) != self.feature_dim:
                fused_features = nn.functional.linear(
                    fused_features, 
                    nn.Parameter(torch.randn(self.feature_dim, fused_features.size(2))).to(fused_features.device)
                )
        
        # Apply transformer
        transformer_output = self.transformer(fused_features)
        
        # Global average pooling
        pooled = torch.mean(transformer_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def forward_accelerometer_only(self, acc_data):
        """
        Forward pass using only accelerometer data
        
        Args:
            acc_data: Accelerometer data [batch, seq_len, 3]
            
        Returns:
            Class logits
        """
        # Process accelerometer data
        acc_data = rearrange(acc_data, 'b l c -> b c l')
        acc_features = self.acc_encoder(acc_data)
        acc_features = self._adapt_sequence_length(acc_features, self.acc_adapter)
        acc_features = rearrange(acc_features, 'b c l -> b l c')
        
        # For accelerometer-only, create dummy features for other modalities
        batch_size, seq_len, _ = acc_features.shape
        dummy_features = torch.zeros(
            batch_size, 
            seq_len, 
            self.feature_dim - self.embed_dim,
            device=acc_data.device
        )
        
        # Concatenate with dummy features
        fused_features = torch.cat([acc_features, dummy_features], dim=2)
        
        # Apply transformer
        transformer_output = self.transformer(fused_features)
        
        # Global average pooling
        pooled = torch.mean(transformer_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

    def forward(self, data):
        """
        General forward pass with robust handling of different input formats
        
        Args:
            data: Dictionary with sensor data or direct tensor input
            
        Returns:
            Class logits
        """
        try:
            # Handle dictionary input
            if isinstance(data, dict):
                # Check what modalities are available
                has_acc = 'accelerometer' in data and data['accelerometer'] is not None
                has_gyro = 'gyroscope' in data and data['gyroscope'] is not None
                has_quaternion = 'quaternion' in data and data['quaternion'] is not None
                has_fusion_features = 'fusion_features' in data and data['fusion_features'] is not None
                
                # Select the appropriate forward method based on available data
                if has_acc and has_gyro and has_quaternion:
                    # Have all modalities
                    return self.forward_all_modalities(
                        data['accelerometer'],
                        data['gyroscope'],
                        data['quaternion']
                    )
                elif has_acc and has_gyro:
                    # Have accelerometer and gyroscope but no quaternion
                    return self.forward_all_modalities(
                        data['accelerometer'],
                        data['gyroscope']
                    )
                elif has_acc and has_fusion_features:
                    # Have accelerometer and fusion features
                    return self.forward_fusion_features(
                        data['accelerometer'],
                        data['fusion_features']
                    )
                elif has_acc:
                    # Accelerometer only
                    return self.forward_accelerometer_only(data['accelerometer'])
                else:
                    raise ValueError("Input must contain at least accelerometer data")
            
            # Handle direct tensor input (assumed to be accelerometer data)
            else:
                return self.forward_accelerometer_only(data)
                
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise
