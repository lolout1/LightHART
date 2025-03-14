"""
Enhanced distillation losses for cross-modal knowledge transfer.

Provides:
1. EnhancedDistillationLoss - Comprehensive distillation with multiple knowledge sources
2. AdversarialDistillationLoss - Adversarial distillation with discriminator
3. OrientationDistillationLoss - Specialized for quaternion orientation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

class EnhancedDistillationLoss(nn.Module):
    """
    Enhanced distillation loss that combines multiple knowledge transfer methods:
    1. KL divergence between teacher and student logits
    2. Cross-entropy with ground truth labels
    3. Feature alignment between teacher's fused features and student's features
    4. Attention map alignment
    5. Intermediate layer alignment
    6. Optional contrastive learning for better feature space alignment
    """
    
    def __init__(
        self,
        temperature=3.0,            # Temperature for softening distributions
        alpha=0.5,                  # Weight between KL and CE: alpha*KL + (1-alpha)*CE
        beta=1.0,                   # Weight for final feature alignment
        gamma=0.3,                  # Weight for attention map alignment
        delta=0.2,                  # Weight for intermediate layer alignment
        epsilon=0.1,                # Weight for quaternion orientation alignment
        teacher_feat_dim=64,        # Teacher's feature dimension
        student_feat_dim=48,        # Student's feature dimension
        teacher_layers=3,           # Number of teacher layers
        student_layers=2,           # Number of student layers
        use_contrastive=False,      # Whether to use contrastive loss
        contrastive_temp=0.1,       # Temperature for contrastive loss
        contrastive_weight=0.5,     # Weight for contrastive loss
        quat_indices=None           # Indices of quaternion components in feature vector
    ):
        """
        Initialize enhanced distillation loss.
        
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight between KL and CE
            beta: Weight for feature alignment
            gamma: Weight for attention map alignment
            delta: Weight for intermediate layer alignment
            epsilon: Weight for quaternion orientation alignment
            teacher_feat_dim: Teacher's feature dimension
            student_feat_dim: Student's feature dimension
            teacher_layers: Number of teacher transformer layers
            student_layers: Number of student transformer layers
            use_contrastive: Whether to use contrastive loss
            contrastive_temp: Temperature for contrastive loss
            contrastive_weight: Weight for contrastive loss
            quat_indices: Indices of quaternion components in feature vector
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.use_contrastive = use_contrastive
        self.contrastive_temp = contrastive_temp
        self.contrastive_weight = contrastive_weight
        self.quat_indices = quat_indices or []
        
        # Standard cross-entropy for hard targets
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Projection layer to align teacher and student feature spaces
        self.teacher_proj = nn.Linear(teacher_feat_dim, student_feat_dim)
        
        # Projection layers for intermediate features
        # We match student layers to teacher layers (potentially different counts)
        teacher_indices = torch.linspace(0, teacher_layers-1, student_layers).long()
        self.teacher_indices = teacher_indices
        
        self.intermediate_projections = nn.ModuleList([
            nn.Linear(teacher_feat_dim, student_feat_dim)
            for _ in range(student_layers)
        ])
        
        # Projection layers for attention maps
        self.attention_projections = nn.ModuleList([
            nn.Linear(teacher_feat_dim, student_feat_dim)
            for _ in range(student_layers)
        ])
    
    def forward(self, student_outputs, teacher_outputs, labels):
        """
        Calculate enhanced distillation loss.
        
        Args:
            student_outputs: Dictionary with student model outputs
            teacher_outputs: Dictionary with teacher model outputs
            labels: Ground truth labels
            
        Returns:
            Dictionary with loss components
        """
        # Extract outputs from dictionaries
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]
        student_feat = student_outputs["feat"]
        teacher_feat = teacher_outputs["fused_feat"]  # Use fused features from teacher
        
        # 1. KL divergence for soft targets with temperature scaling
        T = self.temperature
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        kl_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (T * T)
        
        # 2. Cross-entropy with ground truth
        ce_loss = self.ce_loss(student_logits, labels)
        
        # 3. Feature alignment with projection
        with torch.no_grad():
            projected_teacher = self.teacher_proj(teacher_feat)
        
        feat_loss = F.mse_loss(student_feat, projected_teacher)
        
        # 4. Intermediate layer alignment (if available)
        intermediate_loss = 0.0
        if "intermediates" in student_outputs and "imu_intermediates" in teacher_outputs:
            student_intermediates = student_outputs["intermediates"]
            teacher_intermediates = teacher_outputs["imu_intermediates"]
            
            # Match selected teacher layers to student layers
            for i, proj in enumerate(self.intermediate_projections):
                teacher_idx = self.teacher_indices[i]
                student_inter = torch.mean(student_intermediates[i], dim=1)  # Global pooling
                with torch.no_grad():
                    teacher_inter = torch.mean(teacher_intermediates[teacher_idx], dim=1)
                    projected_teacher_inter = proj(teacher_inter)
                
                intermediate_loss += F.mse_loss(student_inter, projected_teacher_inter)
            
            # Normalize by number of layers
            intermediate_loss /= len(self.intermediate_projections)
        
        # 5. Attention map alignment (if available)
        attention_loss = 0.0
        if "attentions" in student_outputs and "imu_attentions" in teacher_outputs:
            student_attentions = student_outputs["attentions"]
            teacher_attentions = teacher_outputs["imu_attentions"]
            
            # Match selected teacher attention maps to student attention maps
            for i, proj in enumerate(self.attention_projections):
                teacher_idx = self.teacher_indices[i]
                
                # Average across attention heads
                student_attn = torch.mean(student_attentions[i], dim=1)  # (B, seq_len, seq_len)
                with torch.no_grad():
                    teacher_attn = torch.mean(teacher_attentions[teacher_idx], dim=1)
                
                # Match dimensions via interpolation if necessary
                if student_attn.shape != teacher_attn.shape:
                    teacher_attn = F.interpolate(
                        teacher_attn.unsqueeze(1),
                        size=student_attn.shape[1:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                
                attention_loss += F.mse_loss(student_attn, teacher_attn)
            
            # Normalize by number of layers
            attention_loss /= len(self.attention_projections)
        
        # 6. Quaternion orientation alignment (if indices provided)
        orientation_loss = 0.0
        if self.quat_indices and len(self.quat_indices) >= 4:
            # Extract quaternion components from features
            student_quat = student_feat[:, self.quat_indices[:4]]
            teacher_quat = projected_teacher[:, self.quat_indices[:4]]
            
            # Normalize quaternions
            student_quat_norm = F.normalize(student_quat, p=2, dim=1)
            teacher_quat_norm = F.normalize(teacher_quat, p=2, dim=1)
            
            # Quaternion distance (1 - dot product)
            # This measures the angle between quaternion orientations
            quat_dot = torch.sum(student_quat_norm * teacher_quat_norm, dim=1)
            orientation_loss = torch.mean(1.0 - torch.abs(quat_dot))
        
        # 7. Contrastive loss (if enabled)
        contrastive_loss = 0.0
        if self.use_contrastive:
            contrastive_loss = self.contrastive_feature_loss(
                student_feat, projected_teacher, self.contrastive_temp
            )
        
        # Combine all losses with weights
        total_loss = (
            self.alpha * kl_loss + 
            (1.0 - self.alpha) * ce_loss + 
            self.beta * feat_loss +
            self.gamma * attention_loss + 
            self.delta * intermediate_loss +
            self.epsilon * orientation_loss
        )
        
        if self.use_contrastive:
            total_loss += self.contrastive_weight * contrastive_loss
        
        # Return individual loss components for logging
        return {
            "total": total_loss,
            "kl": kl_loss,
            "ce": ce_loss,
            "feat": feat_loss,
            "attn": attention_loss,
            "inter": intermediate_loss,
            "orient": orientation_loss,
            "contrastive": contrastive_loss if self.use_contrastive else 0.0
        }
    
    def contrastive_feature_loss(self, student_feat, teacher_feat, temperature=0.1):
        """
        Contrastive loss to align feature spaces.
        
        Args:
            student_feat: Student features (B, dim)
            teacher_feat: Teacher features (B, dim)
            temperature: Temperature scaling factor
            
        Returns:
            Contrastive loss value
        """
        # Normalize features
        student_norm = F.normalize(student_feat, dim=1)
        teacher_norm = F.normalize(teacher_feat, dim=1)
        
        # Compute similarity matrix
        batch_size = student_norm.shape[0]
        similarity = torch.matmul(student_norm, teacher_norm.t()) / temperature
        
        # Labels are the diagonal elements (matching pairs)
        labels = torch.arange(batch_size).to(similarity.device)
        
        # Compute NT-Xent loss (normalized temperature-scaled cross entropy)
        loss = F.cross_entropy(similarity, labels)
        
        return loss

class AdversarialDistillationLoss(nn.Module):
    """
    Adversarial knowledge distillation loss.
    Uses a discriminator to distinguish between teacher and student features.
    """
    
    def __init__(
        self,
        teacher_feat_dim=64,  # Teacher's feature dimension
        student_feat_dim=48,  # Student's feature dimension
        hidden_dim=32,        # Discriminator hidden dimension
        lambda_adv=0.1,       # Weight for adversarial loss
        quat_indices=None     # Indices of quaternion components in feature vector
    ):
        """
        Initialize adversarial distillation loss.
        
        Args:
            teacher_feat_dim: Teacher's feature dimension
            student_feat_dim: Student's feature dimension
            hidden_dim: Discriminator hidden dimension
            lambda_adv: Weight for adversarial loss
            quat_indices: Indices of quaternion components in feature vector
        """
        super().__init__()
        self.lambda_adv = lambda_adv
        self.quat_indices = quat_indices or []
        
        # Projection for teacher features to match student dimension
        self.teacher_proj = nn.Linear(teacher_feat_dim, student_feat_dim)
        
        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(student_feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Base distillation loss
        self.base_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs, teacher_outputs, labels):
        """
        Calculate adversarial distillation loss.
        
        Args:
            student_outputs: Dictionary with student model outputs
            teacher_outputs: Dictionary with teacher model outputs
            labels: Ground truth labels
            
        Returns:
            Dictionary with loss components
        """
        # Extract outputs
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]
        student_feat = student_outputs["feat"]
        teacher_feat = teacher_outputs["fused_feat"]
        
        # Project teacher features to student space
        with torch.no_grad():
            teacher_feat_proj = self.teacher_proj(teacher_feat)
        
        # Base distillation loss (KL + CE)
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        kl_loss = self.base_loss(student_log_probs, teacher_probs)
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Adversarial loss
        # Train discriminator to distinguish teacher from student
        real_labels = torch.ones(teacher_feat.size(0), 1).to(teacher_feat.device)
        fake_labels = torch.zeros(student_feat.size(0), 1).to(student_feat.device)
        
        # Discriminator predictions
        d_real = self.discriminator(teacher_feat_proj.detach())
        d_fake = self.discriminator(student_feat.detach())
        
        # Discriminator loss
        d_loss_real = F.binary_cross_entropy(d_real, real_labels)
        d_loss_fake = F.binary_cross_entropy(d_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        # Generator (student) loss
        g_fake = self.discriminator(student_feat)
        g_loss = F.binary_cross_entropy(g_fake, real_labels)
        
        # Quaternion orientation loss (if indices provided)
        orientation_loss = 0.0
        if self.quat_indices and len(self.quat_indices) >= 4:
            # Extract quaternion components from features
            student_quat = student_feat[:, self.quat_indices[:4]]
            teacher_quat = teacher_feat_proj[:, self.quat_indices[:4]]
            
            # Normalize quaternions
            student_quat_norm = F.normalize(student_quat, p=2, dim=1)
            teacher_quat_norm = F.normalize(teacher_quat, p=2, dim=1)
            
            # Quaternion distance (1 - dot product)
            quat_dot = torch.sum(student_quat_norm * teacher_quat_norm, dim=1)
            orientation_loss = torch.mean(1.0 - torch.abs(quat_dot))
        
        # Combined loss
        total_loss = kl_loss + ce_loss + self.lambda_adv * g_loss + orientation_loss
        
        return {
            "total": total_loss,
            "kl": kl_loss,
            "ce": ce_loss,
            "adv": g_loss,
            "d_loss": d_loss,
            "orient": orientation_loss
        }

class OrientationDistillationLoss(nn.Module):
    """
    Specialized distillation loss focusing on quaternion orientation transfer.
    Ensures that the student model learns to accurately represent 3D orientation.
    """
    
    def __init__(
        self,
        temperature=3.0,            # Temperature for softening distributions
        alpha=0.3,                  # Weight for classification losses
        beta=0.4,                   # Weight for quaternion orientation loss
        gamma=0.3,                  # Weight for general feature alignment
        teacher_feat_dim=64,        # Teacher's feature dimension
        student_feat_dim=48,        # Student's feature dimension
        quat_indices=[7, 8, 9, 10]  # Indices of quaternion components in feature vector
    ):
        """
        Initialize orientation distillation loss.
        
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for classification losses
            beta: Weight for quaternion orientation loss
            gamma: Weight for general feature alignment
            teacher_feat_dim: Teacher's feature dimension
            student_feat_dim: Student's feature dimension
            quat_indices: Indices of quaternion components in feature vector
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.quat_indices = quat_indices
        
        # Base distillation losses
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Projection layer to align teacher and student feature spaces
        self.teacher_proj = nn.Linear(teacher_feat_dim, student_feat_dim)
    
    def forward(self, student_outputs, teacher_outputs, labels):
        """
        Calculate orientation-focused distillation loss.
        
        Args:
            student_outputs: Dictionary with student model outputs
            teacher_outputs: Dictionary with teacher model outputs
            labels: Ground truth labels
            
        Returns:
            Dictionary with loss components
        """
        # Extract outputs
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]
        student_feat = student_outputs["feat"]
        teacher_feat = teacher_outputs["fused_feat"]
        
        # Project teacher features to student space
        with torch.no_grad():
            teacher_feat_proj = self.teacher_proj(teacher_feat)
        
        # Classification losses
        # KL divergence for soft targets
        T = self.temperature
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        kl_loss = self.kl_loss(student_log_probs, teacher_probs) * (T * T)
        
        # Cross-entropy with ground truth
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Quaternion orientation loss
        orientation_loss = 0.0
        if len(self.quat_indices) >= 4:
            # Extract quaternion components
            student_quat = student_feat[:, self.quat_indices[:4]]
            teacher_quat = teacher_feat_proj[:, self.quat_indices[:4]]
            
            # Normalize quaternions
            student_quat_norm = F.normalize(student_quat, p=2, dim=1)
            teacher_quat_norm = F.normalize(teacher_quat, p=2, dim=1)
            
            # Quaternion angle loss
            # dot(q1, q2) = cos(theta/2), where theta is the rotation angle between quaternions
            cos_half_angle = torch.sum(student_quat_norm * teacher_quat_norm, dim=1)
            
            # Account for q and -q representing the same rotation
            cos_half_angle = torch.abs(cos_half_angle)
            
            # Convert to angle and compute loss
            half_angle = torch.acos(torch.clamp(cos_half_angle, -1.0, 1.0))
            orientation_loss = torch.mean(half_angle ** 2)  # Squared angle loss
        
        # General feature alignment loss (excluding quaternion components)
        if len(self.quat_indices) >= 4:
            # Create mask to exclude quaternion indices
            mask = torch.ones(student_feat.shape[1], dtype=torch.bool, device=student_feat.device)
            mask[self.quat_indices[:4]] = False
            
            # Select non-quaternion features
            student_other = student_feat[:, mask]
            teacher_other = teacher_feat_proj[:, mask]
            
            # Feature alignment loss
            feature_loss = F.mse_loss(student_other, teacher_other)
        else:
            # If no quaternion indices specified, use all features
            feature_loss = F.mse_loss(student_feat, teacher_feat_proj)
        
        # Combined loss
        total_loss = (
            self.alpha * (0.5 * kl_loss + 0.5 * ce_loss) +
            self.beta * orientation_loss +
            self.gamma * feature_loss
        )
        
        return {
            "total": total_loss,
            "kl": kl_loss,
            "ce": ce_loss,
            "orient": orientation_loss,
            "feat": feature_loss
        }

def get_distillation_loss(loss_type="enhanced", **kwargs):
    """
    Factory function to create the requested distillation loss.
    
    Args:
        loss_type: Type of distillation loss ('enhanced', 'adversarial', 'orientation')
        **kwargs: Additional arguments for the loss
        
    Returns:
        Distillation loss module
    """
    if loss_type == "enhanced":
        return EnhancedDistillationLoss(**kwargs)
    elif loss_type == "adversarial":
        return AdversarialDistillationLoss(**kwargs)
    elif loss_type == "orientation":
        return OrientationDistillationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
