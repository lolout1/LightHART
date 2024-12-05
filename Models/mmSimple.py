import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SimpleFallDetector(nn.Module):
    def __init__(self, input_channels=3, hidden_size=64, dropout=0.3):
        super(SimpleFallDetector, self).__init__()
        
        # Feature extractor with adaptive pooling
        self.feature_extractor = nn.Sequential(
            # Conv1d layer for temporal feature extraction
            nn.Conv1d(input_channels, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            # Adaptive pooling to handle variable sequence lengths
            nn.AdaptiveMaxPool1d(32),  # Fixed output size regardless of input length
            nn.Dropout(dropout)
        )
        
        # Calculate feature dimensions
        # After adaptive pooling, sequence length is always 32
        self.flattened_size = hidden_size * 32
        
        # Classifier network
        self.classifier = nn.Sequential(
            # Combine features from both sensors
            nn.Linear(self.flattened_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, data):
        """
        Forward pass of the model
        Args:
            data (dict): Contains sensor data
                - accelerometer_phone: [batch, time, channels]
                - accelerometer_watch: [batch, time, channels]
        Returns:
            torch.Tensor: Fall detection logits [batch, 1]
        """
        # Process phone data
        phone_data = data['accelerometer_phone'].float()
        phone_feat = rearrange(phone_data, 'b t c -> b c t')
        phone_feat = self.feature_extractor(phone_feat)
        phone_feat = phone_feat.flatten(1)  # Flatten to [batch, features]
        
        # Process watch data
        watch_data = data['accelerometer_watch'].float()
        watch_feat = rearrange(watch_data, 'b t c -> b c t')
        watch_feat = self.feature_extractor(watch_feat)
        watch_feat = watch_feat.flatten(1)  # Flatten to [batch, features]
        
        # Combine features from both sensors
        combined = torch.cat([phone_feat, watch_feat], dim=1)
        
        return self.classifier(combined)