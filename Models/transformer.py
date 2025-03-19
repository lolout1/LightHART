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
    Optimized transformer-based model for sensor fusion and fall detection.
    
    This model handles the fact that accelerometer data is already linear acceleration,
    eliminating redundancy and focusing on the most effective modality combinations:
    - Accelerometer (already linear acceleration)
    - Accelerometer + Gyroscope
    - Accelerometer + Orientation (quaternion from fusion filters)
    
    Features:
    - Higher weight for accelerometer data (primary signal for fall detection)
    - Optional gyroscope usage (can be disabled to use just acc+quat)
    - Multi-head attention for temporal relationships
    - Adaptive modality fusion
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
                use_gyro=True,
                acc_weight=1.5,  # Increased weight for accelerometer
                gyro_weight=0.8, # Lower weight for gyroscope
                quat_weight=1.2, # Moderate weight for quaternion
                **kwargs):
        """
        Initialize the fusion transformer model.
        
        Args:
            acc_frames: Number of frames in accelerometer data
            mocap_frames: Number of frames in motion capture data (unused)
            num_classes: Number of output classes (2 for fall detection)
            num_heads: Number of attention heads in transformer
            acc_coords: Number of accelerometer coordinates (3 for x,y,z)
            quat_coords: Number of quaternion coordinates (4 for w,x,y,z)
            num_layers: Number of transformer layers
            embed_dim: Embedding dimension for each modality
            fusion_type: How to fuse modalities ('concat', 'attention', 'weighted')
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            feature_dim: Feature dimension after fusion (auto-calculated if None)
            use_gyro: Whether to use gyroscope data (can disable for acc+quat only)
            acc_weight: Weight for accelerometer data in fusion
            gyro_weight: Weight for gyroscope data in fusion
            quat_weight: Weight for quaternion data in fusion
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.acc_frames = acc_frames
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        self.quat_coords = quat_coords
        self.num_classes = num_classes
        self.use_gyro = use_gyro
        
        # Save modality weights for fusion
        self.modality_weights = {
            'acc': acc_weight,
            'gyro': gyro_weight if use_gyro else 0.0,
            'quat': quat_weight
        }
        
        # Set the sequence length to match accelerometer frames
        self.seq_len = self.acc_frames
        
        # Enhanced accelerometer encoder
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        # Gyroscope encoder (if enabled)
        if use_gyro:
            self.gyro_encoder = nn.Sequential(
                nn.Conv1d(acc_coords, embed_dim, kernel_size=5, padding='same'),
                nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout/2),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding='same'),
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

        # Determine feature dimension based on fusion type and enabled modalities
        if feature_dim is None:
            if fusion_type == 'concat':
                # Count enabled modalities
                num_modalities = 1 + int(use_gyro) + 1  # acc + (gyro) + quat
                self.feature_dim = embed_dim * num_modalities
            elif fusion_type in ['attention', 'weighted']:
                self.feature_dim = embed_dim
            else:
                logger.warning(f"Unknown fusion type '{fusion_type}', defaulting to 'concat'")
                self.fusion_type = 'concat'
                num_modalities = 1 + int(use_gyro) + 1
                self.feature_dim = embed_dim * num_modalities
        else:
            self.feature_dim = feature_dim
            
        # Feature adapter to ensure correct dimensions
        self.feature_adapter = nn.Linear(self.feature_dim, self.feature_dim)
        
        # Ensure number of heads divides feature dimension evenly
        if self.feature_dim % num_heads != 0:
            adjusted_heads = max(1, self.feature_dim // ((self.feature_dim // num_heads) or 1))
            if adjusted_heads != num_heads:
                logger.info(f"Adjusting number of heads from {num_heads} to {adjusted_heads} to match feature dimension {self.feature_dim}")
                num_heads = adjusted_heads
        
        # Create transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=num_heads,
            dim_feedforward=self.feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.feature_dim)
        )
        
        # Enhanced classification head with regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.LayerNorm(128) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(64, num_classes)
        )
        
        # Create modality attention for dynamic weighting
        self.modality_attention = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        logger.info(f"Initialized FusionTransModel with feature_dim={self.feature_dim}, num_heads={num_heads}")
        logger.info(f"Fusion type: {self.fusion_type}, Use gyro: {self.use_gyro}")
        logger.info(f"Modality weights: acc={acc_weight}, gyro={gyro_weight if use_gyro else 0.0}, quat={quat_weight}")
        logger.info("Note: Accelerometer data is already linear acceleration - ignoring redundant linear_acceleration")
        
        # Initialize weights for better convergence
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using modern practices for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_acc_quat(self, acc_data, quat_data):
        """
        Process data with accelerometer and quaternion only (no gyroscope).
        Optimized for orientation-filtered data without gyroscope.
        
        Args:
            acc_data: Accelerometer data [batch, seq_len, channels]
            quat_data: Quaternion data [batch, seq_len, channels]
            
        Returns:
            Class logits [batch, num_classes]
        """
        try:
            # Process accelerometer data
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
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
                # Apply modality weights during concatenation
                acc_weight = self.modality_weights['acc']
                quat_weight = self.modality_weights['quat']
                
                # Weight the features before concatenation
                weighted_acc = acc_features * acc_weight
                weighted_quat = quat_features * quat_weight
                
                # Concatenate weighted features
                fused_features = torch.cat([weighted_acc, weighted_quat], dim=2)
                
                # Ensure correct feature dimension
                if fused_features.shape[2] != self.feature_dim:
                    fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'attention':
                # Generate attention scores for each modality
                acc_scores = self.modality_attention(acc_features)
                quat_scores = self.modality_attention(quat_features)
                
                # Combine scores and apply softmax
                scores = torch.cat([acc_scores, quat_scores], dim=-1)
                attention_weights = F.softmax(scores, dim=-1)
                
                # Apply attention weights
                fused_features = (acc_features * attention_weights[:, :, 0:1] + 
                                 quat_features * attention_weights[:, :, 1:2])
                
                # Project to feature dimension
                fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'weighted':
                # Use predefined weights with normalization
                acc_weight = self.modality_weights['acc']
                quat_weight = self.modality_weights['quat']
                
                # Normalize weights
                total_weight = acc_weight + quat_weight
                acc_weight = acc_weight / total_weight
                quat_weight = quat_weight / total_weight
                
                # Apply weighted fusion
                fused_features = acc_weight * acc_features + quat_weight * quat_features
                fused_features = self.feature_adapter(fused_features)
            
            # Apply transformer encoder for temporal relationship modeling
            transformer_output = self.transformer(fused_features)
            
            # Apply global average pooling 
            pooled = torch.mean(transformer_output, dim=1)
            
            # Apply classifier
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward_acc_quat: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward_all_modalities(self, acc_data, gyro_data, quat_data=None):
        """
        Process data with all available modalities.
        Prioritizes accelerometer data for more reliable performance.
        
        Args:
            acc_data: Accelerometer data [batch, seq_len, channels]
            gyro_data: Gyroscope data [batch, seq_len, channels]
            quat_data: Quaternion data [batch, seq_len, channels]
            
        Returns:
            Class logits [batch, num_classes]
        """
        try:
            # Process accelerometer data (our most reliable signal)
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            # Process gyroscope data if enabled
            if self.use_gyro and gyro_data is not None:
                gyro_data = rearrange(gyro_data, 'b l c -> b c l')
                gyro_features = self.gyro_encoder(gyro_data)
                gyro_features = rearrange(gyro_features, 'b c l -> b l c')
            else:
                # Create zero tensor if gyroscope is disabled or missing
                batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
                gyro_features = torch.zeros(batch_size, seq_len, self.embed_dim,
                                         device=acc_data.device)
            
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
                # Apply concat fusion with weighted modalities
                modalities = []
                
                # Always include accelerometer (weighted)
                modalities.append(acc_features * self.modality_weights['acc'])
                
                # Add gyroscope if enabled
                if self.use_gyro:
                    modalities.append(gyro_features * self.modality_weights['gyro'])
                
                # Add quaternion
                modalities.append(quat_features * self.modality_weights['quat'])
                
                # Concatenate all modalities
                fused_features = torch.cat(modalities, dim=2)
                
                # Ensure correct feature dimension
                if fused_features.shape[2] != self.feature_dim:
                    fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'attention':
                # Implement attention-based fusion with learned feature interaction
                # Generate query from accelerometer (our anchor modality)
                q = acc_features
                
                # Generate key and value from all modalities
                if self.use_gyro:
                    k_modalities = [gyro_features, quat_features]
                    v_modalities = [gyro_features, quat_features]
                else:
                    k_modalities = [quat_features]
                    v_modalities = [quat_features]
                
                # Concatenate modalities for key and value
                k = torch.cat(k_modalities, dim=2) if k_modalities else q
                v = torch.cat(v_modalities, dim=2) if v_modalities else q
                
                # Create dynamic adapter modules
                k_adapter = nn.Linear(k.shape[2], q.shape[2], device=acc_data.device)
                v_adapter = nn.Linear(v.shape[2], q.shape[2], device=acc_data.device)
                
                # Apply adapters
                k = k_adapter(k)
                v = v_adapter(v)
                
                # Compute attention scores with scaling
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
                attn_weights = F.softmax(scores, dim=-1)
                
                # Apply attention
                fused_features = torch.matmul(attn_weights, v)
                
                # Project to feature dimension
                fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'weighted':
                # Extract weights
                acc_weight = self.modality_weights['acc']
                gyro_weight = self.modality_weights['gyro'] if self.use_gyro else 0.0
                quat_weight = self.modality_weights['quat']
                
                # Normalize weights to sum to 1
                total_weight = acc_weight + gyro_weight + quat_weight
                acc_weight = acc_weight / total_weight
                gyro_weight = gyro_weight / total_weight
                quat_weight = quat_weight / total_weight
                
                # Apply weighted fusion
                fused_features = (acc_weight * acc_features)
                
                if self.use_gyro:
                    fused_features += (gyro_weight * gyro_features)
                    
                fused_features += (quat_weight * quat_features)
                
                # Project to feature dimension
                fused_features = self.feature_adapter(fused_features)
            
            # Apply transformer encoder
            transformer_output = self.transformer(fused_features)
            
            # Apply global average pooling
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
        
        Args:
            acc_data: Accelerometer data [batch, seq_len, channels]
            gyro_data: Gyroscope data [batch, seq_len, channels]
            
        Returns:
            Class logits [batch, num_classes]
        """
        try:
            # Process accelerometer data
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            # Process gyroscope data if enabled
            if self.use_gyro:
                gyro_data = rearrange(gyro_data, 'b l c -> b c l')
                gyro_features = self.gyro_encoder(gyro_data)
                gyro_features = rearrange(gyro_features, 'b c l -> b l c')
            else:
                # If gyro is disabled, use accelerometer-only forward pass
                return self.forward_accelerometer_only(acc_data)
            
            # Fuse features based on fusion type
            if self.fusion_type == 'concat':
                # Apply modality weights
                acc_weight = self.modality_weights['acc']
                gyro_weight = self.modality_weights['gyro']
                
                # Weight the features
                weighted_acc = acc_features * acc_weight
                weighted_gyro = gyro_features * gyro_weight
                
                # Concatenate weighted features
                fused_features = torch.cat([weighted_acc, weighted_gyro], dim=2)
                
                # Ensure correct feature dimension
                if fused_features.shape[2] != self.feature_dim:
                    fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'attention' or self.fusion_type == 'weighted':
                # Extract weights
                acc_weight = self.modality_weights['acc']
                gyro_weight = self.modality_weights['gyro']
                
                # Normalize weights to sum to 1
                total_weight = acc_weight + gyro_weight
                acc_weight = acc_weight / total_weight
                gyro_weight = gyro_weight / total_weight
                
                # Apply weighted fusion
                fused_features = acc_weight * acc_features + gyro_weight * gyro_features
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
        Process data with accelerometer only (fallback mode).
        
        Args:
            acc_data: Accelerometer data [batch, seq_len, channels]
            
        Returns:
            Class logits [batch, num_classes]
        """
        try:
            # Process accelerometer data
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            # Handle fusion for accelerometer-only data
            if self.fusion_type == 'concat':
                batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
                
                # Determine padding size based on enabled modalities
                padding_dim = self.feature_dim - self.embed_dim
                
                if padding_dim > 0:
                    # Create padding features
                    dummy_features = torch.zeros(
                        batch_size, 
                        seq_len, 
                        padding_dim,
                        device=acc_data.device
                    )
                    
                    # Concatenate features
                    fused_features = torch.cat([acc_features, dummy_features], dim=2)
                else:
                    fused_features = acc_features
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
        Main forward method that intelligently routes to the appropriate processing method
        based on available modalities and configuration.
        
        Args:
            data: Dictionary with modality keys or raw tensor
            
        Returns:
            Class logits [batch, num_classes]
        """
        try:
            if isinstance(data, dict):
                # Check which modalities are available
                has_acc = 'accelerometer' in data and data['accelerometer'] is not None
                has_gyro = 'gyroscope' in data and data['gyroscope'] is not None and self.use_gyro
                has_quaternion = 'quaternion' in data and data['quaternion'] is not None
                
                # IMPORTANT: Handle redundant linear_acceleration
                if 'linear_acceleration' in data:
                    logger.debug("Ignoring redundant linear_acceleration - using accelerometer data only")
                    # We can simply ignore linear_acceleration as accelerometer data is already linear
                
                # We must have accelerometer data
                if not has_acc:
                    raise ValueError("Accelerometer data is required but not provided")
                
                # Check if quaternion data is missing but should be present
                if not has_quaternion and has_acc and has_gyro:
                    logger.debug("Quaternion data missing but gyro and acc present - creating empty quaternion tensor")
                    batch_size, seq_len = data['accelerometer'].shape[:2]
                    device = data['accelerometer'].device
                    data['quaternion'] = torch.zeros(batch_size, seq_len, 4, device=device)
                    has_quaternion = True
                
                # Route to the appropriate forward method based on available modalities
                if has_acc and has_gyro and has_quaternion:
                    # Full modality fusion
                    return self.forward_all_modalities(
                        data['accelerometer'],
                        data['gyroscope'],
                        data['quaternion']
                    )
                elif has_acc and has_quaternion:
                    # Accelerometer + quaternion (no gyroscope)
                    return self.forward_acc_quat(
                        data['accelerometer'],
                        data['quaternion']
                    )
                elif has_acc and has_gyro:
                    # Accelerometer + gyroscope (no quaternion)
                    return self.forward_acc_gyro_only(
                        data['accelerometer'],
                        data['gyroscope']
                    )
                elif has_acc:
                    # Accelerometer only (fallback)
                    return self.forward_accelerometer_only(data['accelerometer'])
                else:
                    # This should not happen due to the check above
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
                    logger.warning("Error in main forward path, falling back to accelerometer-only mode")
                    return self.forward_accelerometer_only(data['accelerometer'])
                except Exception as fallback_error:
                    logger.error(f"Fallback to accelerometer-only also failed: {str(fallback_error)}")
            
            # Re-raise the exception if we can't recover
            raise e
