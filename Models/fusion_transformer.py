# Models/fusion_transformer.py

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import logging
import traceback

logger = logging.getLogger("model")

class FusionTransModel(nn.Module):
    """
    Transformer-based model for sensor fusion and fall detection.
    """
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
        super().__init__()
        self.fusion_type = fusion_type
        self.acc_frames = acc_frames
        self.mocap_frames = mocap_frames
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        self.quat_coords = quat_coords
        self.num_classes = num_classes
        
        # Set the sequence length to match accelerometer frames
        self.seq_len = self.acc_frames
        
        # Create encoders for each modality
        # Linear accelerometer encoder
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

        # Quaternion encoder with special handling for orientation data
        self.quat_encoder = nn.Sequential(
            nn.Conv1d(quat_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        # Determine feature dimension based on fusion type
        if feature_dim is None:
            if fusion_type == 'concat':
                self.feature_dim = embed_dim * 3  # acc + gyro + quat
            elif fusion_type in ['attention', 'weighted']:
                self.feature_dim = embed_dim
            else:
                logger.warning(f"Unknown fusion type '{fusion_type}', defaulting to 'concat'")
                self.fusion_type = 'concat'
                self.feature_dim = embed_dim * 3
        else:
            self.feature_dim = feature_dim
            
        # Feature adapter to ensure correct dimensions
        self.feature_adapter = nn.Linear(self.feature_dim, self.feature_dim)
        
        # Ensure number of heads divides feature dimension evenly
        if self.feature_dim % num_heads != 0:
            adjusted_heads = max(1, self.feature_dim // (self.feature_dim // num_heads))
            if adjusted_heads != num_heads:
                logger.info(f"Adjusting number of heads from {num_heads} to {adjusted_heads} to match feature dimension")
            num_heads = adjusted_heads
            
        # Create transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=self.feature_dim * 4,
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

        # Classification head with improved architecture for fall detection
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.LayerNorm(128) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights properly for better convergence
        self._init_weights()
        
        logger.info(f"Initialized FusionTransModel with feature_dim={self.feature_dim}, num_heads={num_heads}")
        logger.info(f"Fusion type: {self.fusion_type}")
    
    def _init_weights(self):
        """Initialize weights with better values for convergence"""
        # Initialize transformer layers according to best practices
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize final classification layer with special care for better convergence
        nn.init.zeros_(self.classifier[-1].bias)
        fan_in = self.classifier[-1].weight.size(1)
        nn.init.normal_(self.classifier[-1].weight, 0, 1/math.sqrt(fan_in))

    def forward_all_modalities(self, acc_data, gyro_data, quat_data=None):
        """
        Process data with all available modalities: accelerometer, gyroscope, and quaternion.
        """
        try:
            # Process linear accelerometer data
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            # Process gyroscope data
            gyro_data = rearrange(gyro_data, 'b l c -> b c l')
            gyro_features = self.gyro_encoder(gyro_data)
            gyro_features = rearrange(gyro_features, 'b c l -> b l c')
            
            # Process quaternion data if available
            if quat_data is not None and not torch.all(quat_data == 0):
                quat_data = rearrange(quat_data, 'b l c -> b c l')
                quat_features = self.quat_encoder(quat_data)
                quat_features = rearrange(quat_features, 'b c l -> b l c')
            else:
                # Create zero tensor if quaternion data is not available
                batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
                quat_features = torch.zeros(batch_size, seq_len, self.embed_dim, 
                                         device=acc_data.device)
            
            # Fuse features based on fusion type
            if self.fusion_type == 'concat':
                fused_features = torch.cat([acc_features, gyro_features, quat_features], dim=2)
                
                # Ensure correct feature dimension
                if fused_features.shape[2] != self.feature_dim:
                    fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'attention':
                # Implement attention-based fusion
                q = self.attention_proj_q(acc_features)
                k = self.attention_proj_k(torch.cat([gyro_features, quat_features], dim=-1))
                v = self.attention_proj_v(torch.cat([gyro_features, quat_features], dim=-1))
                
                # Calculate attention scores
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                attn_weights = F.softmax(attn_scores, dim=-1)
                fused_features = torch.matmul(attn_weights, v)
                
                # Project to feature dimension
                fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'weighted':
                # Implement weighted fusion with learned weights
                acc_weight = torch.sigmoid(self.acc_weight)
                gyro_weight = torch.sigmoid(self.gyro_weight)
                quat_weight = torch.sigmoid(self.quat_weight)
                
                # Normalize weights to sum to 1
                total_weight = acc_weight + gyro_weight + quat_weight
                acc_weight = acc_weight / total_weight
                gyro_weight = gyro_weight / total_weight
                quat_weight = quat_weight / total_weight
                
                # Apply weighted fusion
                fused_features = (
                    acc_weight * acc_features +
                    gyro_weight * gyro_features +
                    quat_weight * quat_features
                )
                
                # Project to feature dimension
                fused_features = self.feature_adapter(fused_features)
            
            # Apply transformer encoder
            transformer_output = self.transformer(fused_features)
            
            # Apply global average pooling with attention weighting
            # This helps focus on the most important parts of the sequence
            if hasattr(self, 'attn_pool') and self.attn_pool:
                # Generate attention weights
                attn_weights = F.softmax(self.attn_pool(transformer_output), dim=1)
                # Apply attention weights
                pooled = torch.sum(transformer_output * attn_weights, dim=1)
            else:
                # Standard global average pooling
                pooled = torch.mean(transformer_output, dim=1)
            
            # Apply classifier
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward_all_modalities: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward_acc_gyro_only(self, acc_data, gyro_data):
        """
        Process data with accelerometer and gyroscope only (no quaternion).
        """
        try:
            # Process linear accelerometer data
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            # Process gyroscope data
            gyro_data = rearrange(gyro_data, 'b l c -> b c l')
            gyro_features = self.gyro_encoder(gyro_data)
            gyro_features = rearrange(gyro_features, 'b c l -> b l c')
            
            # Create zero tensor for quaternion data
            batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
            quat_features = torch.zeros(batch_size, seq_len, self.embed_dim, 
                                     device=acc_data.device)
            
            # Fuse features based on fusion type
            if self.fusion_type == 'concat':
                fused_features = torch.cat([acc_features, gyro_features, quat_features], dim=2)
                
                # Ensure correct feature dimension
                if fused_features.shape[2] != self.feature_dim:
                    fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'attention' or self.fusion_type == 'weighted':
                # For attention or weighted fusion without quaternion
                weights = torch.softmax(torch.tensor([1.2, 1.0], device=acc_data.device), dim=0)
                fused_features = weights[0] * acc_features + weights[1] * gyro_features
                fused_features = self.feature_adapter(fused_features)
            
            # Apply transformer encoder
            transformer_output = self.transformer(fused_features)
            
            # Apply global average pooling
            pooled = torch.mean(transformer_output, dim=1)
            
            # Apply classifier
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward_acc_gyro_only: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward_accelerometer_only(self, acc_data):
        """
        Process data with accelerometer only (fallback when other modalities unavailable).
        """
        try:
            # Process linear accelerometer data
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
            
            # Handle fusion for accelerometer-only data
            if self.fusion_type == 'concat':
                # Create zero tensors to maintain expected input dimensions
                dummy_features = torch.zeros(
                    batch_size, 
                    seq_len, 
                    self.feature_dim - self.embed_dim,
                    device=acc_data.device
                )
                
                fused_features = torch.cat([acc_features, dummy_features], dim=2)
            else:
                # For attention or weighted, just project to feature dimension
                fused_features = self.feature_adapter(acc_features)
            
            # Apply transformer encoder
            transformer_output = self.transformer(fused_features)
            
            # Apply global average pooling
            pooled = torch.mean(transformer_output, dim=1)
            
            # Apply classifier
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward_accelerometer_only: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward(self, data):
        """
        Main forward method that routes to appropriate processing method based on available modalities.
        """
        try:
            if isinstance(data, dict):
                # Check which modalities are available
                has_acc = 'accelerometer' in data and data['accelerometer'] is not None
                has_gyro = 'gyroscope' in data and data['gyroscope'] is not None
                has_quaternion = 'quaternion' in data and data['quaternion'] is not None
                
                # Check if quaternion data is missing but should be present
                if not has_quaternion and has_acc and has_gyro:
                    batch_size, seq_len = data['accelerometer'].shape[:2]
                    device = data['accelerometer'].device
                    data['quaternion'] = torch.zeros(batch_size, seq_len, 4, device=device)
                    has_quaternion = True
                
                # Verify we have non-empty tensors
                if has_acc and torch.all(torch.abs(data['accelerometer']) < 1e-6):
                    logger.warning("Accelerometer data contains only zeros or very small values")
                
                if has_gyro and torch.all(torch.abs(data['gyroscope']) < 1e-6):
                    logger.warning("Gyroscope data contains only zeros or very small values")
                
                # Choose appropriate forward method based on available modalities
                if has_acc and has_gyro and has_quaternion:
                    return self.forward_all_modalities(
                        data['accelerometer'],
                        data['gyroscope'],
                        data['quaternion']
                    )
                elif has_acc and has_gyro:
                    return self.forward_acc_gyro_only(
                        data['accelerometer'],
                        data['gyroscope']
                    )
                elif has_acc:
                    # Fallback to accelerometer-only if gyroscope is missing
                    return self.forward_accelerometer_only(data['accelerometer'])
                else:
                    raise ValueError("Input must contain at least accelerometer data")
            else:
                # Handle tensor input (assumed to be accelerometer only)
                return self.forward_accelerometer_only(data)
                
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            # Provide a graceful fallback in case of error
            if isinstance(data, dict) and 'accelerometer' in data:
                try:
                    return self.forward_accelerometer_only(data['accelerometer'])
                except:
                    pass
            
            # Re-raise the exception if we can't recover
            raise
