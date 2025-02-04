# Models/teach.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processor.base import Time2Vec

class TeacherModel(nn.Module):
    def __init__(self,
                 num_joints=32,
                 joint_dim=3,
                 hidden_skel=128,
                 accel_dim=3,
                 time2vec_dim=8,
                 hidden_accel=64,
                 accel_heads=4,
                 accel_layers=2,
                 fusion_hidden=128,
                 num_classes=2,
                 dropout=0.2,
                 dim_feedforward=128,
                 **kwargs):
        super().__init__()

        # 1) Skeleton Branch
        self.skel_input_size = num_joints * joint_dim
        self.hidden_skel = hidden_skel  # Store as instance variable
        self.skel_lstm = nn.LSTM(
            input_size=self.skel_input_size,
            hidden_size=hidden_skel,
            num_layers=1,  # Revert to original 1 layer
            batch_first=True,
            bidirectional=True
        )

        # 2) Accelerometer Branch (original config)
        self.time2vec = Time2Vec(out_channels=time2vec_dim)
        self.accel_input_proj = nn.Linear(accel_dim + time2vec_dim, hidden_accel)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_accel,
            nhead=accel_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.accel_transformer = nn.TransformerEncoder(enc_layer, num_layers=accel_layers)

        # 3) Fusion + Classifier (original architecture)
        self.fusion_fc = nn.Linear(2*hidden_skel + hidden_accel, fusion_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, skel_seq, accel_seq, accel_time, accel_mask=None):
        B = skel_seq.shape[0]

        # (A) Original Skeleton Processing
        h0 = torch.zeros(2, B, self.skel_lstm.hidden_size, device=skel_seq.device)
        c0 = torch.zeros(2, B, self.skel_lstm.hidden_size, device=skel_seq.device)
        _, (hn, _) = self.skel_lstm(skel_seq, (h0, c0))
        skel_feat = torch.cat([hn[0], hn[1]], dim=-1)  # Original implementation

        # (B) Original Accelerometer Processing
        time_embeds = []
        for i in range(B):
            tvals = accel_time[i].unsqueeze(-1)
            emb = self.time2vec(tvals)
            time_embeds.append(emb)
        time_emb = torch.stack(time_embeds, dim=0)
        
        accel_input = torch.cat([accel_seq, time_emb], dim=-1)
        accel_proj = F.relu(self.accel_input_proj(accel_input))
        
        feat_seq = self.accel_transformer(accel_proj, src_key_padding_mask=accel_mask)
        
        if accel_mask is not None:
            valid = ~accel_mask
            feat_seq = feat_seq * valid.unsqueeze(-1).float()
            accel_feat = feat_seq.sum(dim=1) / valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
        else:
            accel_feat = feat_seq.mean(dim=1)

        # (C) Original Fusion
        fused = torch.cat([skel_feat, accel_feat], dim=-1)
        fused = F.relu(self.fusion_fc(fused))
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits
