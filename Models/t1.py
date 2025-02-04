class TransformerTeacher(nn.Module):
    def __init__(self,
                 num_joints=32,
                 joint_dim=3,
                 hidden_skel=128,
                 accel_dim=3,
                 time2vec_dim=8,
                 hidden_accel=64,
                 accel_heads=4,
                 accel_layers=2,
                 skeleton_heads=4,  # New param
                 skeleton_layers=2, # New param
                 fusion_hidden=128,
                 num_classes=2,
                 dropout=0.2,
                 dim_feedforward=128,
                 **kwargs):
        super().__init__()

        # 1) Skeleton Transformer Branch
        self.skel_embed = nn.Linear(num_joints*joint_dim, hidden_skel)
        self.skel_pos = nn.Parameter(torch.randn(1, 64, hidden_skel)) # Max seq len=64
        skel_layer = nn.TransformerEncoderLayer(
            d_model=hidden_skel,
            nhead=skeleton_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.skel_transformer = nn.TransformerEncoder(skel_layer, num_layers=skeleton_layers)

        # 2) Accelerometer Branch (same as before but enhanced)
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        self.accel_proj = nn.Linear(accel_dim + time2vec_dim, hidden_accel)
        accel_layer = nn.TransformerEncoderLayer(
            d_model=hidden_accel,
            nhead=accel_heads,
            dim_feedforward=dim_feedforward*2, # More capacity
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.accel_transformer = nn.TransformerEncoder(accel_layer, num_layers=accel_layers)

        # 3) Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_skel + hidden_accel, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, skel_seq, accel_seq, accel_time, accel_mask=None):
        # Skeleton processing
        B, Ts, _ = skel_seq.shape
        skel_emb = self.skel_embed(skel_seq) + self.skel_pos[:,:Ts,:]
        skel_feat = self.skel_transformer(skel_emb).mean(dim=1)

        # Accelerometer processing (optimized)
        B, Ta, _ = accel_seq.shape
        t_emb = self.time2vec(accel_time.view(B*Ta, 1)).view(B, Ta, -1)
        accel_in = F.gelu(self.accel_proj(torch.cat([accel_seq, t_emb], -1)))
        accel_feat = self.accel_transformer(accel_in, src_key_padding_mask=accel_mask)
        accel_feat = masked_mean(accel_feat, accel_mask)  # Reusable function

        # Fusion
        fused = self.fusion(torch.cat([skel_feat, accel_feat], -1))
        return self.classifier(fused)

def masked_mean(features, mask):
    if mask is not None:
        features = features * (~mask).unsqueeze(-1).float()
        return features.sum(dim=1) / (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
    return features.mean(dim=1)
