import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import random

class RandomApplyCustom(nn.Module):
    def __init__(self, module, p=0.5):
        super(RandomApplyCustom, self).__init__()
        self.module = module
        self.p = p

    def forward(self, x):
        if self.training and random.random() < self.p:
            return self.module(x)
        return x


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers.
    """
    def __init__(self, in_channels, out_channels, dropout):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        return out


class MultiModalStudentModel(nn.Module):
    def __init__(
        self,
        acc_coords=4,          # x, y, z, SMV
        spatial_embed=256,
        num_heads=8,
        depth=8,
        mlp_ratio=4,
        num_classes=2,
        dropout=0.5,            # Increased dropout rate
        use_skeleton=False     # Flag for skeleton usage
    ):
        super(MultiModalStudentModel, self).__init__()
        self.use_skeleton = use_skeleton

        # =====================
        # Feature Extractor for Phone with Residual Blocks
        # =====================
        self.phone_conv = nn.Sequential(
            ResidualBlock(acc_coords, 64, dropout),
            ResidualBlock(64, 128, dropout),
            ResidualBlock(128, 256, dropout),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.phone_fc = nn.Sequential(
            nn.Linear(256 * 64, spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =====================
        # Feature Extractor for Watch with Residual Blocks
        # =====================
        self.watch_conv = nn.Sequential(
            ResidualBlock(acc_coords, 64, dropout),
            ResidualBlock(64, 128, dropout),
            ResidualBlock(128, 256, dropout),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )
        self.watch_fc = nn.Sequential(
            nn.Linear(256 * 64, spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =====================
        # Optional Skeleton Embedding with Residual Blocks
        # =====================
        if self.use_skeleton:
            self.skeleton_conv = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(dropout)
            )
            self.skeleton_fc = nn.Sequential(
                nn.Linear(256 * 16 * 16, spatial_embed),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # =====================
        # Weighted Fusion Layers
        # =====================
        fusion_input_size = 2 * spatial_embed
        if self.use_skeleton:
            fusion_input_size += spatial_embed

        # Separate fusion layers for initial and weighted fusion
        self.fusion_layer_initial = nn.Sequential(
            nn.Linear(fusion_input_size, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fusion_layer_weighted = nn.Sequential(
            nn.Linear(fusion_input_size, spatial_embed),
            nn.LayerNorm(spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # =====================
        # Learnable Weights
        # =====================
        self.weight_phone = nn.Parameter(torch.ones(1, 1, spatial_embed))
        self.weight_watch = nn.Parameter(torch.ones(1, 1, spatial_embed))
        if self.use_skeleton:
            self.weight_skeleton = nn.Parameter(torch.ones(1, 1, spatial_embed))

        # =====================
        # Data Augmentation Layers
        # =====================
        self.augmentation = nn.Sequential(
            RandomApplyCustom(nn.Dropout(p=0.1), p=0.5),
            RandomApplyCustom(nn.Conv1d(spatial_embed, spatial_embed, kernel_size=3, padding=1), p=0.3)
        )

        # =====================
        # Temporal Transformer Encoder with Ensemble
        # =====================
        encoder_layers = []
        for _ in range(2):  # Two parallel transformer encoders
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=spatial_embed,
                nhead=num_heads,
                dim_feedforward=spatial_embed * mlp_ratio,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            encoder_layers.append(encoder_layer)
        self.transformer_encoder1 = nn.TransformerEncoder(
            encoder_layers[0],
            num_layers=depth // 2,
            norm=nn.LayerNorm(spatial_embed)
        )
        self.transformer_encoder2 = nn.TransformerEncoder(
            encoder_layers[1],
            num_layers=depth // 2,
            norm=nn.LayerNorm(spatial_embed)
        )

        # =====================
        # Classification Head
        # =====================
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_embed * 2),  # Concatenated transformer outputs
            nn.Linear(spatial_embed * 2, spatial_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_embed, num_classes)
        )

        # =====================
        # Weight Initialization
        # =====================
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
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        Args:
            data (dict): Dictionary containing:
                - accelerometer_phone: [B, T, C] tensor
                - accelerometer_watch: [B, T, C] tensor
                - skeleton (optional): [B, T, J, C] tensor
        Returns:
            logits (torch.Tensor): [B, num_classes]
        """
        # Ensure data types are float32
        phone_data = data['accelerometer_phone'].float()      # [B, T, C]
        watch_data = data['accelerometer_watch'].float()      # [B, T, C]

        # =====================
        # Feature Extraction
        # =====================
        # Phone
        phone_feat = rearrange(phone_data, 'b t c -> b c t')  # [B, C, T]
        phone_feat = self.phone_conv(phone_feat)             # [B, 256, T/2]
        phone_feat = phone_feat.view(phone_feat.size(0), -1) # Flatten
        phone_feat = self.phone_fc(phone_feat)               # [B, 256]

        # Watch
        watch_feat = rearrange(watch_data, 'b t c -> b c t')  # [B, C, T]
        watch_feat = self.watch_conv(watch_feat)              # [B, 256, T/2]
        watch_feat = watch_feat.view(watch_feat.size(0), -1) # Flatten
        watch_feat = self.watch_fc(watch_feat)               # [B, 256]

        # Skeleton (if used)
        if self.use_skeleton and 'skeleton' in data:
            skeleton_data = data['skeleton'].float()          # [B, T, J, C]
            skeleton_feat = rearrange(skeleton_data, 'b t j c -> b c j t')  # [B, C, J, T]
            skeleton_feat = self.skeleton_conv(skeleton_feat)             # [B, 256, J/2, T/2]
            skeleton_feat = skeleton_feat.view(skeleton_feat.size(0), -1)  # Flatten
            skeleton_feat = self.skeleton_fc(skeleton_feat)               # [B, 256]
        else:
            skeleton_feat = None

        # =====================
        # Initial Fusion
        # =====================
        if self.use_skeleton and skeleton_feat is not None:
            fused = torch.cat([phone_feat, watch_feat, skeleton_feat], dim=1)  # [B, 768]
        else:
            fused = torch.cat([phone_feat, watch_feat], dim=1)                # [B, 512]

        print(f"Shape before initial fusion_layer: {fused.shape}")  # Debug
        fused_initial = self.fusion_layer_initial(fused)                     # [B, 256]
        print(f"Shape after initial fusion_layer: {fused_initial.shape}")   # Debug

        # =====================
        # Apply Learnable Weights
        # =====================
        phone_weighted = phone_feat * self.weight_phone  # [B, 256]
        watch_weighted = watch_feat * self.weight_watch  # [B, 256]
        if self.use_skeleton and skeleton_feat is not None:
            skeleton_weighted = skeleton_feat * self.weight_skeleton  # [B, 256]
            fused_weighted = torch.cat([phone_weighted, watch_weighted, skeleton_weighted], dim=1)  # [B, 768]
        else:
            fused_weighted = torch.cat([phone_weighted, watch_weighted], dim=1)                # [B, 512]
        
        print(f"Shape before weighted fusion_layer: {fused_weighted.shape}")  # Debug
        fused_weighted = self.fusion_layer_weighted(fused_weighted)                    # [B, 256]
        print(f"Shape after weighted fusion_layer: {fused_weighted.shape}")   # Debug

        # =====================
        # Data Augmentation
        # =====================
        fused_augmented = self.augmentation(fused_weighted.unsqueeze(1))      # [B, 1, 256]
        fused_augmented = fused_augmented.squeeze(1)                         # [B, 256]

        # =====================
        # Temporal Modeling with Ensemble Transformers
        # =====================
        fused_augmented = fused_augmented.unsqueeze(1)                       # [B, 1, 256]

        transformer_out1 = self.transformer_encoder1(fused_augmented)        # [B, 1, 256]
        transformer_out2 = self.transformer_encoder2(fused_augmented)        # [B, 1, 256]

        # Concatenate outputs from both transformers
        transformer_concat = torch.cat([transformer_out1, transformer_out2], dim=-1)  # [B, 1, 512]

        # =====================
        # Classification
        # =====================
        pooled_features = transformer_concat.mean(dim=1)                     # [B, 512]
        logits = self.classifier(pooled_features)                           # [B, num_classes]

        # =====================
        # Threshold-Based Decision Logic
        # =====================
        # Example: Adjust predictions based on SMV thresholds
        # This is a simple post-processing step; more sophisticated methods can be applied
        smv_phone = phone_data[:, :, 3]  # [B, T]
        smv_watch = watch_data[:, :, 3]  # [B, T]
        smv_phone_avg = smv_phone.mean(dim=1)  # [B]
        smv_watch_avg = smv_watch.mean(dim=1)  # [B]

        # Define fall thresholds
        fall_threshold_phone = 2.0  # Example threshold; adjust based on data
        fall_threshold_watch = 1.5  # Example threshold; adjust based on data

        # Apply thresholds to influence predictions
        fall_indicators = ((smv_phone_avg > fall_threshold_phone) | 
                           (smv_watch_avg > fall_threshold_watch)).float().unsqueeze(1)  # [B, 1]

        # Modify logits to enforce fall predictions when thresholds are exceeded
        # Assuming class 1 is 'fall' and class 0 is 'no fall'
        logits = logits + fall_indicators * 10.0  # Add a large value to the 'fall' class logits

        return logits
