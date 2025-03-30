import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import logging
import traceback

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
        super().__init__()
        self.fusion_type = fusion_type
        self.acc_frames = acc_frames
        self.mocap_frames = mocap_frames
        self.embed_dim = embed_dim
        self.acc_coords = acc_coords
        self.quat_coords = quat_coords
        self.num_classes = num_classes
        
        self.seq_len = self.acc_frames
        
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        self.gyro_encoder = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )

        self.quat_encoder = nn.Sequential(
            nn.Conv1d(quat_coords, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity(),
            nn.GELU()
        )
        
        self.pos_encoding = nn.Parameter(torch.zeros(1, acc_frames, embed_dim))
        nn.init.normal_(self.pos_encoding, 0, 0.02)

        if feature_dim is None:
            if fusion_type == 'concat':
                self.feature_dim = embed_dim * 3
            elif fusion_type in ['attention', 'weighted']:
                self.feature_dim = embed_dim
            else:
                self.fusion_type = 'concat'
                self.feature_dim = embed_dim * 3
        else:
            self.feature_dim = feature_dim
            
        self.feature_adapter = nn.Linear(self.feature_dim, self.feature_dim)
        
        if fusion_type == 'attention':
            self.attention_proj_q = nn.Linear(embed_dim, embed_dim)
            self.attention_proj_k = nn.Linear(embed_dim * 2, embed_dim)
            self.attention_proj_v = nn.Linear(embed_dim * 2, embed_dim)
        
        if fusion_type == 'weighted':
            self.acc_weight = nn.Parameter(torch.ones(1))
            self.gyro_weight = nn.Parameter(torch.ones(1))
            self.quat_weight = nn.Parameter(torch.ones(1))
        
        if self.feature_dim % num_heads != 0:
            adjusted_heads = max(1, self.feature_dim // (self.feature_dim // num_heads))
            num_heads = adjusted_heads
            
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

        self.attn_pool = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Softmax(dim=1)
        )
        
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
        
        self._init_weights()
        
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        nn.init.zeros_(self.classifier[-1].bias)
        fan_in = self.classifier[-1].weight.size(1)
        nn.init.normal_(self.classifier[-1].weight, 0, 1/math.sqrt(fan_in))

    def forward_all_modalities(self, acc_data, gyro_data, quat_data=None):
        try:
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            gyro_data = rearrange(gyro_data, 'b l c -> b c l')
            gyro_features = self.gyro_encoder(gyro_data)
            gyro_features = rearrange(gyro_features, 'b c l -> b l c')
            
            if quat_data is not None and not torch.all(quat_data == 0):
                quat_norm = torch.norm(quat_data, dim=2, keepdim=True)
                quat_norm = torch.where(quat_norm > 1e-8, quat_norm, torch.ones_like(quat_norm))
                quat_data = quat_data / quat_norm
                
                for b in range(quat_data.shape[0]):
                    for i in range(1, quat_data.shape[1]):
                        dot_product = torch.sum(quat_data[b, i-1] * quat_data[b, i])
                        if dot_product < 0:
                            quat_data[b, i] = -quat_data[b, i]
                
                quat_data = rearrange(quat_data, 'b l c -> b c l')
                quat_features = self.quat_encoder(quat_data)
                quat_features = rearrange(quat_features, 'b c l -> b l c')
            else:
                batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
                quat_features = torch.zeros(batch_size, seq_len, self.embed_dim, device=acc_data.device)
            
            if self.fusion_type == 'concat':
                fused_features = torch.cat([acc_features, gyro_features, quat_features], dim=2)
                
                if fused_features.shape[2] != self.feature_dim:
                    fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'attention':
                q = self.attention_proj_q(acc_features)
                k = self.attention_proj_k(torch.cat([gyro_features, quat_features], dim=-1))
                v = self.attention_proj_v(torch.cat([gyro_features, quat_features], dim=-1))
                
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                
                if quat_data is not None and not torch.all(quat_data == 0):
                    quat_boost = 0.2
                    bias = torch.zeros_like(attn_scores)
                    bias[:, :, attn_scores.size(-1)//2:] = quat_boost
                    attn_weights = F.softmax(attn_scores + bias, dim=-1)
                else:
                    attn_weights = F.softmax(attn_scores, dim=-1)
                    
                fused_features = torch.matmul(attn_weights, v)
                fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'weighted':
                acc_weight = torch.sigmoid(self.acc_weight)
                gyro_weight = torch.sigmoid(self.gyro_weight)
                quat_weight = torch.sigmoid(self.quat_weight)
                
                total_weight = acc_weight + gyro_weight + quat_weight
                acc_weight = acc_weight / total_weight
                gyro_weight = gyro_weight / total_weight
                quat_weight = quat_weight / total_weight
                
                fused_features = (
                    acc_weight * acc_features +
                    gyro_weight * gyro_features +
                    quat_weight * quat_features
                )
                
                fused_features = self.feature_adapter(fused_features)
            
            seq_len = fused_features.shape[1]
            fused_features = fused_features + self.pos_encoding[:, :seq_len, :]
            
            transformer_output = self.transformer(fused_features)
            attn_weights = self.attn_pool(transformer_output)
            pooled = torch.sum(transformer_output * attn_weights, dim=1)
            
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward_all_modalities: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward_acc_gyro_only(self, acc_data, gyro_data):
        try:
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            gyro_data = rearrange(gyro_data, 'b l c -> b c l')
            gyro_features = self.gyro_encoder(gyro_data)
            gyro_features = rearrange(gyro_features, 'b c l -> b l c')
            
            batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
            quat_features = torch.zeros(batch_size, seq_len, self.embed_dim, device=acc_data.device)
            
            if self.fusion_type == 'concat':
                fused_features = torch.cat([acc_features, gyro_features, quat_features], dim=2)
                
                if fused_features.shape[2] != self.feature_dim:
                    fused_features = self.feature_adapter(fused_features)
            elif self.fusion_type == 'attention' or self.fusion_type == 'weighted':
                weights = torch.softmax(torch.tensor([1.2, 1.0], device=acc_data.device), dim=0)
                fused_features = weights[0] * acc_features + weights[1] * gyro_features
                fused_features = self.feature_adapter(fused_features)
            
            seq_len = fused_features.shape[1]
            fused_features = fused_features + self.pos_encoding[:, :seq_len, :]
            
            transformer_output = self.transformer(fused_features)
            attn_weights = self.attn_pool(transformer_output)
            pooled = torch.sum(transformer_output * attn_weights, dim=1)
            
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward_acc_gyro_only: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward_accelerometer_only(self, acc_data):
        try:
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
            
            if self.fusion_type == 'concat':
                dummy_features = torch.zeros(
                    batch_size, 
                    seq_len, 
                    self.feature_dim - self.embed_dim,
                    device=acc_data.device
                )
                
                fused_features = torch.cat([acc_features, dummy_features], dim=2)
            else:
                fused_features = self.feature_adapter(acc_features)
            
            seq_len = fused_features.shape[1]
            fused_features = fused_features + self.pos_encoding[:, :seq_len, :]
            
            transformer_output = self.transformer(fused_features)
            attn_weights = self.attn_pool(transformer_output)
            pooled = torch.sum(transformer_output * attn_weights, dim=1)
            
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward_accelerometer_only: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward_quaternion(self, acc_data, quat_data):
        try:
            if quat_data is None or torch.all(torch.abs(quat_data) < 1e-6):
                return self.forward_accelerometer_only(acc_data)
                
            acc_data = rearrange(acc_data, 'b l c -> b c l')
            acc_features = self.acc_encoder(acc_data)
            acc_features = rearrange(acc_features, 'b c l -> b l c')
            
            quat_norm = torch.norm(quat_data, dim=2, keepdim=True)
            quat_norm = torch.where(quat_norm > 1e-8, quat_norm, torch.ones_like(quat_norm))
            quat_data = quat_data / quat_norm
            
            for b in range(quat_data.shape[0]):
                for i in range(1, quat_data.shape[1]):
                    dot_product = torch.sum(quat_data[b, i-1] * quat_data[b, i])
                    if dot_product < 0:
                        quat_data[b, i] = -quat_data[b, i]
            
            quat_data = rearrange(quat_data, 'b l c -> b c l')
            quat_features = self.quat_encoder(quat_data)
            quat_features = rearrange(quat_features, 'b c l -> b l c')
            
            batch_size, seq_len = acc_features.shape[0], acc_features.shape[1]
            gyro_features = torch.zeros(batch_size, seq_len, self.embed_dim, device=acc_data.device)
            
            if self.fusion_type == 'concat':
                fused_features = torch.cat([acc_features, gyro_features, quat_features], dim=2)
                if fused_features.shape[2] != self.feature_dim:
                    fused_features = self.feature_adapter(fused_features)
            else:
                weights = torch.softmax(torch.tensor([1.0, 0.0, 1.2], device=acc_data.device), dim=0)
                fused_features = weights[0] * acc_features + weights[2] * quat_features
                fused_features = self.feature_adapter(fused_features)
            
            seq_len = fused_features.shape[1]
            fused_features = fused_features + self.pos_encoding[:, :seq_len, :]
            
            transformer_output = self.transformer(fused_features)
            attn_weights = self.attn_pool(transformer_output)
            pooled = torch.sum(transformer_output * attn_weights, dim=1)
            
            logits = self.classifier(pooled)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward_quaternion: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward_multi_sensor(self, acc_data, gyro_data):
        try:
            return self.forward_acc_gyro_only(acc_data, gyro_data)
        except Exception as e:
            logger.error(f"Error in forward_multi_sensor: {e}")
            logger.error(traceback.format_exc())
            try:
                return self.forward_accelerometer_only(acc_data)
            except:
                raise

    def forward(self, data):
        try:
            if isinstance(data, dict):
                has_acc = 'accelerometer' in data and data['accelerometer'] is not None
                has_gyro = 'gyroscope' in data and data['gyroscope'] is not None
                has_quaternion = 'quaternion' in data and data['quaternion'] is not None
                
                if not has_acc:
                    raise ValueError("Input must contain accelerometer data")
                
                if not has_quaternion and has_acc and has_gyro:
                    batch_size, seq_len = data['accelerometer'].shape[:2]
                    device = data['accelerometer'].device
                    data['quaternion'] = torch.zeros(batch_size, seq_len, 4, device=device)
                    data['quaternion'][:, :, 0] = 1.0
                    has_quaternion = True
                
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
                    return self.forward_accelerometer_only(data['accelerometer'])
                else:
                    raise ValueError("Input must contain at least accelerometer data")
            else:
                return self.forward_accelerometer_only(data)
                
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            if isinstance(data, dict) and 'accelerometer' in data:
                try:
                    return self.forward_accelerometer_only(data['accelerometer'])
                except:
                    pass
            
            raise
