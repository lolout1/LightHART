"""
Cross-modal distillation losses for SmartFallMM.
Enables knowledge transfer from skeleton to IMU-only models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedDistillationLoss(nn.Module):
    """
    Enhanced distillation loss with orientation alignment.
    
    Includes:
    1. KL divergence between teacher and student logits
    2. Hard-label CE with ground truth labels 
    3. Feature alignment between teacher and student features
    4. Special handling for quaternion orientation features
    """
    
    def __init__(
        self,
        temperature=3.0,
        alpha=0.5,           # Weight for KL vs CE
        beta=1.0,            # Weight for feature alignment
        gamma=0.2,           # Weight for quaternion orientation alignment
        teacher_feat_dim=128,
        student_feat_dim=64,
        quat_indices=None    # Indices of quaternion components in feature vector
    ):
        """
        Initialize enhanced distillation loss.
        
        Args:
            temperature: Temperature for softening logits
            alpha: Weight for KL vs CE
            beta: Weight for feature alignment
            gamma: Weight for quaternion orientation alignment
            teacher_feat_dim: Teacher feature dimension
            student_feat_dim: Student feature dimension
            quat_indices: Indices of quaternion components in features
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.quat_indices = quat_indices or []
        
        # Standard cross-entropy for hard targets
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Projection for teacher features to student dimension
        self.teacher_proj = nn.Linear(teacher_feat_dim, student_feat_dim)
    
    def forward(
        self,
        student_outputs,     # Dictionary from student model
        teacher_outputs,     # Dictionary from teacher model
        labels               # Ground truth labels
    ):
        """
        Compute the distillation loss.
        
        Args:
            student_outputs: Dictionary with student model outputs
            teacher_outputs: Dictionary with teacher model outputs
            labels: Ground truth labels
            
        Returns:
            Total loss value
        """
        # Extract outputs
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]
        
        # Get features if available
        student_feat = student_outputs.get("feat", None)
        teacher_feat = teacher_outputs.get("fused_feat", None)
        
        # 1. KL divergence for soft targets
        T = self.temperature
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (T * T)
        
        # 2. Cross-entropy with ground truth
        ce_loss = self.ce_loss(student_logits, labels)
        
        # 3. Feature alignment if features are available
        feat_loss = 0.0
        if student_feat is not None and teacher_feat is not None:
            with torch.no_grad():
                projected_teacher = self.teacher_proj(teacher_feat)
            
            # Basic feature alignment
            feat_loss = F.mse_loss(student_feat, projected_teacher)
            
            # 4. Special handling for quaternion orientation if indices provided
            quat_loss = 0.0
            if len(self.quat_indices) >= 4:
                # Extract quaternion components
                student_quat = student_feat[:, self.quat_indices[:4]]
                teacher_quat = projected_teacher[:, self.quat_indices[:4]]
                
                # Normalize quaternions
                student_quat_norm = F.normalize(student_quat, p=2, dim=1)
                teacher_quat_norm = F.normalize(teacher_quat, p=2, dim=1)
                
                # Quaternion distance (1 - dot product)
                # This handles the fact that q and -q represent the same rotation
                quat_dot = torch.abs(torch.sum(student_quat_norm * teacher_quat_norm, dim=1))
                quat_loss = torch.mean(1.0 - quat_dot)
                
                # Add to total loss
                feat_loss = feat_loss + self.gamma * quat_loss
        
        # Combine all losses with weights
        total_loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss + self.beta * feat_loss
        
        return total_loss

class CrossModalLoss(nn.Module):
    """
    Cross-modal distillation loss for transferring knowledge from 
    skeleton+IMU teacher to IMU-only student.
    """
    
    def __init__(
        self,
        temperature=2.0,
        alpha=0.7,           # Weight for soft targets
        beta=0.5,            # Weight for feature alignment
        teacher_feat_dim=128,
        student_feat_dim=64
    ):
        """
        Initialize cross-modal loss.
        
        Args:
            temperature: Temperature for softening logits
            alpha: Weight for soft vs hard targets
            beta: Weight for feature alignment
            teacher_feat_dim: Teacher feature dimension
            student_feat_dim: Student feature dimension
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # Standard cross-entropy for hard targets
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Projection for teacher features
        self.teacher_proj = nn.Linear(teacher_feat_dim, student_feat_dim)
    
    def forward(
        self,
        student_logits,      # Student model logits
        teacher_logits,      # Teacher model logits
        labels,              # Ground truth labels
        student_feat=None,   # Student features (optional)
        teacher_feat=None    # Teacher features (optional)
    ):
        """
        Compute the cross-modal distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: Ground truth labels
            student_feat: Features from student model (optional)
            teacher_feat: Features from teacher model (optional)
            
        Returns:
            Total loss value
        """
        # 1. KL divergence for soft targets
        T = self.temperature
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (T * T)
        
        # 2. Cross-entropy with ground truth
        ce_loss = self.ce_loss(student_logits, labels)
        
        # 3. Feature alignment if features are available
        feat_loss = 0.0
        if student_feat is not None and teacher_feat is not None:
            with torch.no_grad():
                projected_teacher = self.teacher_proj(teacher_feat)
            
            feat_loss = F.mse_loss(student_feat, projected_teacher)
        
        # Combine all losses with weights
        soft_target_loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
        total_loss = soft_target_loss + self.beta * feat_loss
        
        return total_loss
