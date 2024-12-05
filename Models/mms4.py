import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiModalStudentModel(nn.Module):
    def __init__(
        self,
        acc_coords=4,          # x, y, z, SMV
        spatial_embed=256,
        num_heads=8,
        depth=8,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.3,
        use_skeleton=False     # Add flag for skeleton usage
    ):
        super(MultiModalStudentModel, self).__init__()
        self.use_skeleton = use_skeleton

        # =====================
        # Feature Extractor for Phone
        # =====================
        self.phone_conv = nn.Sequential(
            nn.Conv1d(acc_coords, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.phone_fc = nn.Sequential(
            nn.Linear(128 * 64, spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =====================
        # Feature Extractor for Watch
        # =====================
        self.watch_conv = nn.Sequential(
            nn.Conv1d(acc_coords, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.watch_fc = nn.Sequential(
            nn.Linear(128 * 64, spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =====================
        # Optional Skeleton Embedding
        # =====================
        if self.use_skeleton:
            self.skeleton_conv = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(dropout)
            )
            self.skeleton_fc = nn.Sequential(
                nn.Linear(128 * 16 * 16, spatial_embed),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # =====================
        # Fusion Layer with Attention
        # =====================
        fusion_input_size = 2 * spatial_embed
        if self.use_skeleton:
            fusion_input_size += spatial_embed

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=spatial_embed,
            num_heads=num_heads // 2,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =====================
        # Temporal Transformer Encoder
        # =====================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spatial_embed,
            nhead=num_heads,
            dim_feedforward=spatial_embed * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(spatial_embed)
        )

        # =====================
        # Classification Head
        # =====================
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed),
            nn.Linear(spatial_embed, spatial_embed // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_embed // 2, 1)  # Single output for binary classification
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize weights for convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        Args:
            data (dict): Dictionary containing:
                - accelerometer_phone: [B, T, 4] tensor (x, y, z, SMV)
                - accelerometer_watch: [B, T, 4] tensor (x, y, z, SMV)
                - skeleton (optional): [B, T, J, C] tensor
        Returns:
            logits (torch.Tensor): [B, 1]
        """
        # Ensure data types are float32 and handle the 4-channel input
        phone_data = data['accelerometer_phone'].float()      # [B, T, 4]
        watch_data = data['accelerometer_watch'].float()      # [B, T, 4]

        # Rearrange for convolution while preserving all channels
        phone_feat = rearrange(phone_data, 'b t c -> b c t')  # [B, 4, T]
        watch_feat = rearrange(watch_data, 'b t c -> b c t')  # [B, 4, T]

        # Process through convolutional networks
        phone_feat = self.phone_conv(phone_feat)             # [B, 128, T/2]
        watch_feat = self.watch_conv(watch_feat)             # [B, 128, T/2]

        # Flatten and project to embedding space
        phone_feat = phone_feat.view(phone_feat.size(0), -1) # [B, 128 * T/2]
        watch_feat = watch_feat.view(watch_feat.size(0), -1) # [B, 128 * T/2]
        
        phone_feat = self.phone_fc(phone_feat)               # [B, spatial_embed]
        watch_feat = self.watch_fc(watch_feat)               # [B, spatial_embed]

        # Rest of the processing remains the same...
        if self.use_skeleton and 'skeleton' in data:
            skeleton_data = data['skeleton'].float()
            skeleton_feat = rearrange(skeleton_data, 'b t j c -> b c j t')
            skeleton_feat = self.skeleton_conv(skeleton_feat)
            skeleton_feat = skeleton_feat.view(skeleton_feat.size(0), -1)
            skeleton_feat = self.skeleton_fc(skeleton_feat)
            fused = torch.cat([phone_feat, watch_feat, skeleton_feat], dim=1)
        else:
            fused = torch.cat([phone_feat, watch_feat], dim=1)

        fused = self.fusion_layer(fused)
        fused = fused.unsqueeze(1)  # Add sequence dimension for transformer
        temporal_features = self.transformer_encoder(fused)
        pooled_features = temporal_features.mean(dim=1)
        logits = self.classifier(pooled_features)
        
        return logits
