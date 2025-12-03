"""
Knowledge Distillation for DeepSeek
====================================

This module implements:
- Standard knowledge distillation (KD)
- Sequence-level knowledge distillation (SeqKD)
- Feature/Hidden state distillation
- Progressive distillation

Based on: DeepSeek distillation techniques and Hinton et al.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================

class KDLossType(Enum):
    """Types of knowledge distillation loss."""
    KL_DIVERGENCE = "kl"
    MSE = "mse"
    COSINE = "cosine"
    JSD = "jsd"  # Jensen-Shannon Divergence


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 4.0
    alpha: float = 0.5  # Weight for KD loss vs CE loss
    kd_loss_type: KDLossType = KDLossType.KL_DIVERGENCE
    use_hard_labels: bool = True
    
    # Feature distillation
    hidden_distill: bool = False
    hidden_weight: float = 0.1
    hidden_layers: Optional[List[int]] = None  # Which layers to distill
    
    # Attention distillation
    attention_distill: bool = False
    attention_weight: float = 0.1
    
    # Training
    learning_rate: float = 2e-5
    batch_size: int = 4
    max_length: int = 2048
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0


@dataclass
class ProgressiveConfig:
    """Configuration for progressive distillation."""
    num_stages: int = 3
    intermediate_sizes: List[int] = field(default_factory=lambda: [7168, 4096, 2048])
    
    # Temperature schedule
    temperature_start: float = 6.0
    temperature_end: float = 2.0
    temperature_schedule: str = "linear"  # linear, cosine, constant


# =============================================================================
# Core Distillation Losses
# =============================================================================

def kd_loss_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 4.0,
    reduction: str = "batchmean",
) -> Tensor:
    """
    Standard knowledge distillation loss using KL divergence.
    
    L_KD = T² * KL(softmax(teacher/T) || softmax(student/T))
    
    Args:
        student_logits: (batch, seq, vocab) student output logits
        teacher_logits: (batch, seq, vocab) teacher output logits
        temperature: softmax temperature (higher = softer)
        reduction: reduction method
    
    Returns:
        Scalar loss tensor
    """
    # Scale by temperature
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL divergence
    kl = F.kl_div(student_soft, teacher_soft, reduction=reduction)
    
    # Scale by T² (Hinton et al.)
    return kl * (temperature ** 2)


def kd_loss_jsd(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 4.0,
) -> Tensor:
    """
    Jensen-Shannon Divergence for distillation (symmetric).
    
    JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P + Q) / 2
    """
    student_soft = F.softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    # M = (P + Q) / 2
    m = (student_soft + teacher_soft) / 2
    
    # JSD
    kl_pm = F.kl_div(m.log(), student_soft, reduction="batchmean")
    kl_qm = F.kl_div(m.log(), teacher_soft, reduction="batchmean")
    
    return 0.5 * (kl_pm + kl_qm)


def kd_loss_mse(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 4.0,
) -> Tensor:
    """MSE loss on softmax probabilities."""
    student_soft = F.softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    return F.mse_loss(student_soft, teacher_soft)


def kd_loss_cosine(
    student_logits: Tensor,
    teacher_logits: Tensor,
) -> Tensor:
    """Cosine similarity loss on logits."""
    # Normalize
    student_norm = F.normalize(student_logits, dim=-1)
    teacher_norm = F.normalize(teacher_logits, dim=-1)
    
    # Cosine similarity
    cos_sim = (student_norm * teacher_norm).sum(dim=-1)
    
    # Loss = 1 - similarity
    return (1 - cos_sim).mean()


def compute_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    config: DistillationConfig,
) -> Tensor:
    """Compute distillation loss based on configuration."""
    if config.kd_loss_type == KDLossType.KL_DIVERGENCE:
        return kd_loss_kl(student_logits, teacher_logits, config.temperature)
    elif config.kd_loss_type == KDLossType.MSE:
        return kd_loss_mse(student_logits, teacher_logits, config.temperature)
    elif config.kd_loss_type == KDLossType.COSINE:
        return kd_loss_cosine(student_logits, teacher_logits)
    elif config.kd_loss_type == KDLossType.JSD:
        return kd_loss_jsd(student_logits, teacher_logits, config.temperature)
    else:
        raise ValueError(f"Unknown KD loss type: {config.kd_loss_type}")


# =============================================================================
# Hidden State Distillation
# =============================================================================

class HiddenDistillation(nn.Module):
    """
    Distill hidden state representations.
    
    L_hidden = MSE(proj(student_hidden), teacher_hidden)
    """
    
    def __init__(
        self,
        student_hidden_size: int,
        teacher_hidden_size: int,
        use_projection: bool = True,
    ):
        super().__init__()
        
        self.use_projection = use_projection and (student_hidden_size != teacher_hidden_size)
        
        if self.use_projection:
            self.projection = nn.Linear(student_hidden_size, teacher_hidden_size, bias=False)
    
    def forward(
        self,
        student_hidden: Tensor,
        teacher_hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute hidden state distillation loss.
        
        Args:
            student_hidden: (batch, seq, d_student)
            teacher_hidden: (batch, seq, d_teacher)
            attention_mask: (batch, seq) optional mask
        """
        if self.use_projection:
            student_proj = self.projection(student_hidden)
        else:
            student_proj = student_hidden
        
        # MSE loss
        loss = F.mse_loss(student_proj, teacher_hidden, reduction='none')
        
        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss


def compute_layer_mapping(
    student_layers: int,
    teacher_layers: int,
    mapping_type: str = "uniform",
) -> List[Tuple[int, int]]:
    """
    Compute mapping from student layers to teacher layers.
    
    Args:
        student_layers: number of student layers
        teacher_layers: number of teacher layers
        mapping_type: "uniform" or "top_heavy"
    
    Returns:
        List of (student_layer, teacher_layer) tuples
    """
    if mapping_type == "uniform":
        ratio = teacher_layers / student_layers
        return [
            (s, int((s + 0.5) * ratio))
            for s in range(student_layers)
        ]
    
    elif mapping_type == "top_heavy":
        # More student layers map to top teacher layers
        mapping = []
        bottom_student = student_layers // 2
        bottom_teacher = teacher_layers // 3
        
        # Bottom half → bottom third
        for s in range(bottom_student):
            t = int(s / bottom_student * bottom_teacher)
            mapping.append((s, t))
        
        # Top half → top two-thirds
        remaining_student = student_layers - bottom_student
        remaining_teacher = teacher_layers - bottom_teacher
        
        for s in range(bottom_student, student_layers):
            local_s = s - bottom_student
            t = bottom_teacher + int(local_s / remaining_student * remaining_teacher)
            mapping.append((s, min(t, teacher_layers - 1)))
        
        return mapping
    
    else:
        raise ValueError(f"Unknown mapping type: {mapping_type}")


# =============================================================================
# Attention Distillation
# =============================================================================

def attention_distillation_loss(
    student_attention: Tensor,  # (batch, heads, seq, seq)
    teacher_attention: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Distill attention patterns.
    
    L_attn = KL(teacher_attn || student_attn)
    """
    # Average over heads if different
    if student_attention.size(1) != teacher_attention.size(1):
        student_attention = student_attention.mean(dim=1, keepdim=True)
        teacher_attention = teacher_attention.mean(dim=1, keepdim=True)
    
    # KL divergence over attention distribution
    student_log = (student_attention + 1e-10).log()
    
    kl = F.kl_div(student_log, teacher_attention, reduction='none')
    
    # Sum over attention dimension, mean over rest
    loss = kl.sum(dim=-1).mean()
    
    return loss


# =============================================================================
# Combined Distillation Loss
# =============================================================================

@dataclass
class DistillationOutput:
    """Output from distillation loss computation."""
    total_loss: Tensor
    kd_loss: Tensor
    ce_loss: Optional[Tensor] = None
    hidden_loss: Optional[Tensor] = None
    attention_loss: Optional[Tensor] = None


def combined_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    config: DistillationConfig,
    student_hidden: Optional[List[Tensor]] = None,
    teacher_hidden: Optional[List[Tensor]] = None,
    hidden_distiller: Optional[HiddenDistillation] = None,
    layer_mapping: Optional[List[Tuple[int, int]]] = None,
) -> DistillationOutput:
    """
    Compute combined distillation loss.
    
    L = α * L_KD + (1-α) * L_CE + β * L_hidden
    """
    # Distillation loss
    kd_loss = compute_distillation_loss(student_logits, teacher_logits, config)
    
    # Hard label loss
    ce_loss = None
    if config.use_hard_labels:
        # Shift for next-token prediction
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    
    # Hidden state distillation
    hidden_loss = None
    if config.hidden_distill and student_hidden is not None and teacher_hidden is not None:
        hidden_losses = []
        
        if layer_mapping is None:
            # Default: match layer indices
            layer_mapping = [(i, i) for i in range(min(len(student_hidden), len(teacher_hidden)))]
        
        for s_idx, t_idx in layer_mapping:
            if s_idx < len(student_hidden) and t_idx < len(teacher_hidden):
                loss = hidden_distiller(student_hidden[s_idx], teacher_hidden[t_idx])
                hidden_losses.append(loss)
        
        if hidden_losses:
            hidden_loss = torch.stack(hidden_losses).mean()
    
    # Combine losses
    total_loss = config.alpha * kd_loss
    
    if ce_loss is not None:
        total_loss = total_loss + (1 - config.alpha) * ce_loss
    
    if hidden_loss is not None:
        total_loss = total_loss + config.hidden_weight * hidden_loss
    
    return DistillationOutput(
        total_loss=total_loss,
        kd_loss=kd_loss,
        ce_loss=ce_loss,
        hidden_loss=hidden_loss,
    )


# =============================================================================
# Sequence-Level Knowledge Distillation
# =============================================================================

@dataclass
class SeqKDConfig:
    """Configuration for sequence-level KD."""
    num_samples: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    max_length: int = 512
    mix_ratio: float = 0.5  # Ratio of teacher sequences


class SeqKDDataset:
    """
    Dataset for sequence-level KD.
    
    Contains both teacher-generated and ground-truth sequences.
    """
    
    def __init__(
        self,
        prompts: List[str],
        ground_truth: List[str],
        teacher_generations: List[str],
        tokenizer,
        max_length: int = 512,
        mix_ratio: float = 0.5,
    ):
        self.prompts = prompts
        self.ground_truth = ground_truth
        self.teacher_generations = teacher_generations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mix_ratio = mix_ratio
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        prompt = self.prompts[idx]
        
        # Randomly select teacher or ground truth
        if torch.rand(1).item() < self.mix_ratio:
            response = self.teacher_generations[idx]
        else:
            response = self.ground_truth[idx]
        
        # Tokenize
        text = prompt + response
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


def generate_teacher_sequences(
    model: nn.Module,
    prompts: List[str],
    tokenizer,
    config: SeqKDConfig,
    device: torch.device,
) -> List[str]:
    """
    Generate sequences from teacher model for SeqKD.
    """
    model.eval()
    generations = []
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            outputs = model.generate(
                **inputs,
                max_length=config.max_length,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                num_return_sequences=1,
                do_sample=True,
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append(generated[len(prompt):])  # Remove prompt
    
    return generations


# =============================================================================
# Progressive Distillation
# =============================================================================

class TemperatureScheduler:
    """Schedule temperature during distillation."""
    
    def __init__(
        self,
        start: float,
        end: float,
        total_steps: int,
        schedule_type: str = "linear",
    ):
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.schedule_type = schedule_type
    
    def get_temperature(self, step: int) -> float:
        """Get temperature at current step."""
        progress = min(step / max(self.total_steps, 1), 1.0)
        
        if self.schedule_type == "constant":
            return self.start
        
        elif self.schedule_type == "linear":
            return self.start + (self.end - self.start) * progress
        
        elif self.schedule_type == "cosine":
            cosine = (1 + math.cos(math.pi * progress)) / 2
            return self.end + (self.start - self.end) * cosine
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class ProgressiveDistiller:
    """
    Manages progressive distillation across multiple stages.
    """
    
    def __init__(
        self,
        config: ProgressiveConfig,
        total_steps: int,
    ):
        self.config = config
        self.total_steps = total_steps
        self.steps_per_stage = total_steps // config.num_stages
        
        self.current_stage = 0
        self.current_step = 0
        
        # Temperature scheduler
        self.temp_scheduler = TemperatureScheduler(
            start=config.temperature_start,
            end=config.temperature_end,
            total_steps=total_steps,
            schedule_type=config.temperature_schedule,
        )
    
    @property
    def temperature(self) -> float:
        """Get current temperature."""
        return self.temp_scheduler.get_temperature(self.current_step)
    
    @property
    def intermediate_size(self) -> int:
        """Get current intermediate size for student."""
        if self.current_stage < len(self.config.intermediate_sizes):
            return self.config.intermediate_sizes[self.current_stage]
        return self.config.intermediate_sizes[-1]
    
    def step(self):
        """Advance one training step."""
        self.current_step += 1
        
        # Check if should advance stage
        if (self.current_step > 0 and 
            self.current_step % self.steps_per_stage == 0 and
            self.current_stage < self.config.num_stages - 1):
            self.current_stage += 1
    
    def should_save_checkpoint(self) -> bool:
        """Check if should save checkpoint (end of stage)."""
        return (
            self.current_step > 0 and
            self.current_step % self.steps_per_stage == 0
        )
    
    def get_state(self) -> Dict:
        """Get current state for logging."""
        return {
            'step': self.current_step,
            'stage': self.current_stage,
            'temperature': self.temperature,
            'intermediate_size': self.intermediate_size,
        }


# =============================================================================
# Distillation Trainer
# =============================================================================

class DistillationTrainer:
    """
    Full distillation trainer with support for various KD techniques.
    """
    
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: DistillationConfig,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        hidden_distiller: Optional[HiddenDistillation] = None,
        layer_mapping: Optional[List[Tuple[int, int]]] = None,
    ):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.hidden_distiller = hidden_distiller
        self.layer_mapping = layer_mapping
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Track metrics
        self.step = 0
        self.metrics_history: List[Dict] = []
    
    def train_step(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> Dict[str, float]:
        """Execute one training step."""
        self.student.train()
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.config.hidden_distill,
            )
            teacher_logits = teacher_outputs.logits
            teacher_hidden = getattr(teacher_outputs, 'hidden_states', None)
        
        # Student forward
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.config.hidden_distill,
        )
        student_logits = student_outputs.logits
        student_hidden = getattr(student_outputs, 'hidden_states', None)
        
        # Compute loss
        loss_output = combined_distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            config=self.config,
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden,
            hidden_distiller=self.hidden_distiller,
            layer_mapping=self.layer_mapping,
        )
        
        # Backward
        self.optimizer.zero_grad()
        loss_output.total_loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config.max_grad_norm,
            )
        
        self.optimizer.step()
        self.step += 1
        
        # Record metrics
        metrics = {
            'loss': loss_output.total_loss.item(),
            'kd_loss': loss_output.kd_loss.item(),
        }
        if loss_output.ce_loss is not None:
            metrics['ce_loss'] = loss_output.ce_loss.item()
        if loss_output.hidden_loss is not None:
            metrics['hidden_loss'] = loss_output.hidden_loss.item()
        
        self.metrics_history.append(metrics)
        
        return metrics



