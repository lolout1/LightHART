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
        num_patch=8,
        acc_coords=4,
        spatial_embed=128,
        sdepth=4,
        adepth=4,
        tdepth=6,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        op_type='all',
        embed_type='lin',
        drop_rate=0.3,
        attn_drop_rate=0.3,
        drop_path_rate=0.3,
        norm_layer=None,
        num_classes=2
    ):
        super().__init__()
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # Save parameters
        self.spatial_embed = spatial_embed
        self.num_patch = num_patch
        self.embed_type = embed_type
        self.tdepth = tdepth
        
        # Enhanced skeleton encoder
        if self.embed_type == 'lin':
            self.Skeleton_embedding = nn.Sequential(
                nn.Linear(num_joints * in_chans, spatial_embed),
                nn.LayerNorm(spatial_embed),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(spatial_embed, spatial_embed),
                nn.LayerNorm(spatial_embed),
                nn.GELU()
            )
        else:
            self.Skeleton_embedding = nn.Sequential(
                nn.Conv1d(in_chans, spatial_embed, kernel_size=1),
                nn.BatchNorm1d(spatial_embed),
                nn.GELU()
            )
            
        # Enhanced accelerometer encoder
        if self.embed_type == 'lin':
            self.Accelerometer_embedding = nn.Sequential(
                nn.Linear(acc_coords, spatial_embed),
                nn.LayerNorm(spatial_embed),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(spatial_embed, spatial_embed),
                nn.LayerNorm(spatial_embed),
                nn.GELU()
            )
        else:
            self.Accelerometer_embedding = nn.Sequential(
                nn.Conv1d(acc_coords, spatial_embed, kernel_size=1),
                nn.BatchNorm1d(spatial_embed),
                nn.GELU()
            )

        # Tokens and embeddings
        self.temp_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]
        self.Temporal_blocks = nn.ModuleList([
            TransformerBlock(
                dim=spatial_embed,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(tdepth)
        ])
        
        # Classification head
        self.class_head = nn.Sequential(
            norm_layer(spatial_embed),
            nn.Linear(spatial_embed, num_classes)
        )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        def _init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
        self.apply(_init_layer)
        nn.init.trunc_normal_(self.temp_token, std=0.02)
        nn.init.trunc_normal_(self.Temporal_pos_embed, std=0.02)
    
    def forward(self, acc_data, skl_data):
        batch_size = skl_data.size(0)
        seq_length = skl_data.size(1)
        
        # Process skeleton data
        skl_data = skl_data.view(batch_size, seq_length, -1)
        skl_embedded = self.Skeleton_embedding(skl_data)
        
        # Process accelerometer data
        if acc_data.dim() == 2:
            acc_data = acc_data.unsqueeze(-1)
        elif acc_data.dim() > 3:
            acc_data = acc_data.view(batch_size, seq_length, -1)
            
        acc_embedded = self.Accelerometer_embedding(acc_data)
        
        # Combine embeddings
        combined = skl_embedded + acc_embedded
        
        # Add classification token
        cls_token = self.temp_token.expand(batch_size, -1, -1)
        combined = torch.cat((cls_token, combined), dim=1)
        combined = combined + self.Temporal_pos_embed
        combined = self.pos_drop(combined)
        
        # Apply transformer blocks
        for blk in self.Temporal_blocks:
            combined = blk(combined)
            
        # Get classification token and predict
        cls_token_final = combined[:, 0]
        logits = self.class_head(cls_token_final)
        
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
