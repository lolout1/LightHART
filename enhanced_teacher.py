import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class EnhancedTeacherModel(nn.Module):
    def __init__(
        self,
        device='cuda',
        mocap_frames=128,
        acc_frames=128,
        num_joints=32,
        in_chans=3,
        num_patch=8,          # Added to match original
        acc_coords=4,         # Added to match original
        spatial_embed=128,
        sdepth=4,            # Added to match original
        adepth=4,            # Added to match original
        tdepth=6,            # Added to match original
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,       # Added to match original
        qk_scale=None,       # Added to match original
        op_type='all',       # Added to match original
        embed_type='lin',    # Added to match original
        drop_rate=0.3,
        attn_drop_rate=0.3,  # Added to match original
        drop_path_rate=0.3,  # Added to match original
        norm_layer=None,     # Added to match original
        num_classes=2
    ):
        super().__init__()
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # Save init parameters
        self.num_patch = num_patch
        self.acc_frames = acc_frames
        self.acc_coords = acc_coords
        self.mocap_frames = mocap_frames
        self.num_joints = num_joints
        self.embed_type = embed_type
        
        # Skeleton branch with enhanced spatial feature extraction
        self.skeleton_encoder = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Conv2d(128, spatial_embed, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(spatial_embed),
            nn.ReLU()
        )
        
        # Accelerometer branch
        if self.embed_type == 'lin':
            self.acc_encoder = nn.Sequential(
                nn.Linear(acc_coords, spatial_embed),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(spatial_embed, spatial_embed),
                nn.ReLU()
            )
        else:
            self.acc_encoder = nn.Sequential(
                nn.Conv1d(acc_coords, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Conv1d(64, spatial_embed, kernel_size=5, padding=2),
                nn.BatchNorm1d(spatial_embed),
                nn.ReLU()
            )
        
        # Cross-modal attention transformer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=spatial_embed,
                nhead=num_heads,
                dim_feedforward=spatial_embed * mlp_ratio,
                dropout=drop_rate,
                batch_first=True
            ),
            num_layers=tdepth
        )
        
        # Learnable modality tokens
        self.temp_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        
        # Classification head
        self.class_head = nn.Sequential(
            norm_layer(spatial_embed),
            nn.Linear(spatial_embed, num_classes)
        )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
    def forward(self, acc_data, skl_data):
        batch_size = skl_data.size(0)
        seq_length = skl_data.size(1)
        
        # Process skeleton data - [batch, seq, joints, channels] -> [batch, seq, embed]
        skl_data = skl_data.view(batch_size, seq_length, -1)
        if self.embed_type == 'lin':
            skl_embedded = self.skeleton_encoder(skl_data)
        else:
            skl_embedded = self.skeleton_encoder(skl_data.transpose(1, 2))
            skl_embedded = skl_embedded.transpose(1, 2)
            
        # Process accelerometer data
        if self.embed_type == 'lin':
            acc_embedded = self.acc_encoder(acc_data)
        else:
            acc_embedded = self.acc_encoder(acc_data.transpose(1, 2))
            acc_embedded = acc_embedded.transpose(1, 2)
        
        # Combine embeddings
        combined = skl_embedded + acc_embedded
        
        # Add classification token
        class_token = self.temp_token.expand(batch_size, -1, -1)
        combined = torch.cat((class_token, combined), dim=1)
        
        # Add positional embedding
        combined = combined + self.Temporal_pos_embed
        combined = self.pos_drop(combined)
        
        # Apply transformer
        for _ in range(self.tdepth):
            combined = self.transformer_encoder(combined)
        
        # Classification
        cls_token_final = combined[:, 0]
        logits = self.class_head(cls_token_final)
        
        return logits
