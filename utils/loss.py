# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtendedDistillationLoss(nn.Module):
    """
    Enhanced distillation loss with feature alignment for Kalman filtered data.
    Combines:
    1. Soft-target KL between teacher & student logits
    2. Hard-label CE for student predictions vs. ground truth
    3. Feature alignment MSE between teacher and student features
    """
    
    def __init__(
        self,
        temperature=3.0,
        alpha=0.5,           # weight on the KL vs. CE
        beta=1.0,            # weight on feature alignment
        teacher_feat_dim=128,
        student_feat_dim=64
    ):
        """
        Args:
          temperature: Temperature for logit distillation.
          alpha: weight for (KL) vs. (CE). If alpha=0.5 => 50% KD + 50% CE.
          beta: weight for MSE feature alignment.
          teacher_feat_dim: dimension of teacher's internal feature.
          student_feat_dim: dimension of student's feature.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

        self.ce_loss = nn.CrossEntropyLoss()
        # A small linear layer to project teacher's feature to student_feat_dim
        self.teacher_proj = nn.Linear(teacher_feat_dim, student_feat_dim)

    def forward(
        self,
        *,
        student_logits,     # (B, num_classes)
        teacher_logits,     # (B, num_classes)
        labels,             # (B,) ground-truth
        student_feat=None,  # (B, student_feat_dim), optional
        teacher_feat=None   # (B, teacher_feat_dim), optional
    ):
        """Forward pass for the extended distillation loss"""
        # -----------------------------
        # (1) Logit-based KD
        # -----------------------------
        T = self.temperature
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)  # teacher distribution
        student_log = F.log_softmax(student_logits / T, dim=-1)
        kl_div = F.kl_div(student_log, teacher_probs, reduction='batchmean') * (T * T)

        # -----------------------------
        # (2) Hard-label cross-entropy
        # -----------------------------
        ce = self.ce_loss(student_logits, labels)

        # -----------------------------
        # (3) Feature alignment (MSE)
        # -----------------------------
        feat_mse = 0.0
        if (student_feat is not None) and (teacher_feat is not None):
            # Project teacher => student's dimension
            teacher_feat_proj = self.teacher_proj(teacher_feat)
            feat_mse = F.mse_loss(student_feat, teacher_feat_proj)

        # Combine them
        total = self.alpha * kl_div + (1.0 - self.alpha) * ce + self.beta * feat_mse
        return total
