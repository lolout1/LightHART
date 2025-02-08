# File: utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtendedDistillationLoss(nn.Module):
    """
    Distillation loss that combines:
      1) Soft-target KL between teacher & student logits (at temperature T),
      2) Hard-label CE for the student's predictions vs. ground-truth,
      3) Feature alignment MSE between teacher's `teacher_feat` and student's `student_feat`
         with a learned linear projection to match dimensions.
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
          teacher_feat_dim: dimension of teacher's internal feature (e.g. hidden_accel=128).
          student_feat_dim: dimension of student's feature (e.g. d_model=64).
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
        """
        Named arguments (starred) so we can accept calls like:
            kd_loss_fn(
                student_logits=...,
                teacher_logits=...,
                labels=...,
                student_feat=...,
                teacher_feat=...
            )
        without errors.

        You can omit student_feat/teacher_feat if you only want logit-based KD + CE.
        """
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


# Optionally, you could also keep a simpler DistillationLoss if you want:
class DistillationLoss(nn.Module):
    """
    Simple Distillation Loss:
      alpha * KL + (1-alpha)*CE
    """
    def __init__(self, temperature=2.0, alpha=0.3):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # ...
        pass  # omitted for brevity

