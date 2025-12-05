"""
Knowledge Distillation Losses - MLX Implementation
"""

import mlx.core as mx


def kd_loss_kl(
    student_logits: mx.array,
    teacher_logits: mx.array,
    temperature: float = 4.0,
) -> mx.array:
    """
    Standard knowledge distillation loss using KL divergence.
    
    L_KD = T² * KL(softmax(teacher/T) || softmax(student/T))
    
    Args:
        student_logits: (batch, seq, vocab) student output logits
        teacher_logits: (batch, seq, vocab) teacher output logits
        temperature: softmax temperature (higher = softer)
        
    Returns:
        Scalar loss tensor
    """
    # Scale by temperature
    student_soft = mx.log(mx.softmax(student_logits / temperature, axis=-1) + 1e-10)
    teacher_soft = mx.softmax(teacher_logits / temperature, axis=-1)
    
    # KL divergence: sum P * (log P - log Q)
    kl = teacher_soft * (mx.log(teacher_soft + 1e-10) - student_soft)
    kl = mx.sum(kl, axis=-1)  # Sum over vocab
    
    # Scale by T^2 (standard in KD)
    loss = (temperature ** 2) * mx.mean(kl)
    
    return loss


def kd_loss_mse(
    student_logits: mx.array,
    teacher_logits: mx.array,
    temperature: float = 4.0,
) -> mx.array:
    """
    Knowledge distillation using MSE on softened logits.
    
    Args:
        student_logits: (batch, seq, vocab) student output logits
        teacher_logits: (batch, seq, vocab) teacher output logits
        temperature: softmax temperature
        
    Returns:
        Scalar loss tensor
    """
    student_soft = mx.softmax(student_logits / temperature, axis=-1)
    teacher_soft = mx.softmax(teacher_logits / temperature, axis=-1)
    
    mse = mx.mean((student_soft - teacher_soft) ** 2)
    return (temperature ** 2) * mse


def kd_loss_jsd(
    student_logits: mx.array,
    teacher_logits: mx.array,
    temperature: float = 4.0,
) -> mx.array:
    """
    Knowledge distillation using Jensen-Shannon Divergence.
    
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = (P + Q) / 2
    
    Args:
        student_logits: (batch, seq, vocab) student output logits
        teacher_logits: (batch, seq, vocab) teacher output logits
        temperature: softmax temperature
        
    Returns:
        Scalar loss tensor
    """
    student_soft = mx.softmax(student_logits / temperature, axis=-1)
    teacher_soft = mx.softmax(teacher_logits / temperature, axis=-1)
    
    # Mixture distribution
    m = 0.5 * (student_soft + teacher_soft)
    
    # KL divergences
    eps = 1e-10
    kl_student = mx.sum(student_soft * (mx.log(student_soft + eps) - mx.log(m + eps)), axis=-1)
    kl_teacher = mx.sum(teacher_soft * (mx.log(teacher_soft + eps) - mx.log(m + eps)), axis=-1)
    
    jsd = 0.5 * (kl_student + kl_teacher)
    
    return (temperature ** 2) * mx.mean(jsd)

from enum import Enum, auto

class KDLossType(Enum):
    KL = auto()
    MSE = auto()
    JSD = auto()

class DistillationConfig:
    def __init__(self, temperature=4.0, alpha=0.5, kd_loss_type=KDLossType.KL):
        self.temperature = temperature
        self.alpha = alpha
        self.kd_loss_type = kd_loss_type

def compute_distillation_loss(student_logits, teacher_logits, config: DistillationConfig):
    if config.kd_loss_type == KDLossType.KL:
        return kd_loss_kl(student_logits, teacher_logits, config.temperature)
    elif config.kd_loss_type == KDLossType.MSE:
        return kd_loss_mse(student_logits, teacher_logits, config.temperature)
    elif config.kd_loss_type == KDLossType.JSD:
        return kd_loss_jsd(student_logits, teacher_logits, config.temperature)
    else:
        raise ValueError("Unknown KD loss type")
