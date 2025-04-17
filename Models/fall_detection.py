import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        nn.init.xavier_uniform_(self.query.weight, gain=0.01)
        nn.init.xavier_uniform_(self.key.weight, gain=0.01)
        nn.init.xavier_uniform_(self.value.weight, gain=0.01)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, v)

class FallDetectionTransformer(nn.Module):
    def __init__(self, acc_frames=128, num_classes=1, num_heads=4, 
                acc_coords=3, num_layer=2, embed_dim=32):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(4)
        self.embedding = nn.Sequential(
            nn.Conv1d(4, embed_dim, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(0.1)
        )
        
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layer):
            layer = nn.Sequential(
                nn.LayerNorm(embed_dim),
                SimpleAttention(embed_dim),
                nn.Dropout(0.1),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim*4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim*4, embed_dim),
                nn.Dropout(0.1)
            )
            self.transformer_layers.append(layer)
            
        self.final_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, acc_data, *args):
        # Normalize and embed input
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_norm(x)
        x = self.embedding(x)
        x = rearrange(x, 'b c l -> b l c')
        
        # Apply transformer layers with residual connections
        for layer in self.transformer_layers:
            residual = x
            x = residual + layer(x)
        
        # Global average pooling and classification
        x = self.final_norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        
        return logits, x
