# watchStudentFixed.py

import torch
import torch.nn as nn

class watchStudentFixed(nn.Module):
    """
    Student => watch-only, shape (B, 128, feat_dim), no mask => suitable for TFLite.
    We'll do a small 1D-CNN for demonstration.
    """
    def __init__(self,
                 hidden_dim=48,
                 num_layers=2,
                 dropout=0.1,
                 orientation_rep="quat",
                 num_classes=2):
        super().__init__()
        in_dim= 8 if orientation_rep=="quat" else 7
        self.conv1= nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2= nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act= nn.ReLU()
        self.dropout= nn.Dropout(dropout)
        self.pool= nn.AdaptiveAvgPool1d(1)
        self.fc= nn.Linear(hidden_dim, num_classes)

    def forward(self, x_fixed):
        """
        x_fixed => shape (B, 128, in_dim)
        """
        B, T, F= x_fixed.shape
        x= x_fixed.permute(0,2,1)  # => (B, in_dim, T)
        x= self.conv1(x)
        x= self.act(x)
        x= self.conv2(x)
        x= self.act(x)
        x= self.pool(x)  # => (B, hidden_dim, 1)
        x= x.squeeze(-1) # => (B, hidden_dim)
        x= self.dropout(x)
        logits= self.fc(x)
        # debug prints
        # print(f"DEBUG: Student forward => x_fixed={x_fixed.shape}, feats={x.shape}")
        return {
            "logits": logits,
            "feat": x
        }

