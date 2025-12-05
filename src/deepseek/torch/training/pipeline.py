"""
DeepSeek Training Pipeline

This module implements the complete pre-training pipeline used by DeepSeek,
including:
- Scaling law calculations for compute-optimal training
- Data pipeline with domain mixing
- WSD (Warmup-Stable-Decay) learning rate scheduler
- Curriculum learning for sequence lengths
- Distributed training configuration
- Checkpointing and monitoring

Based on: DeepSeek LLM paper (https://arxiv.org/abs/2401.02954)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Iterator, Any, Callable
import numpy as np
import os
import glob
import json
import time
import math
from collections import defaultdict
from threading import Thread
from queue import Queue


# =============================================================================
# Scaling Laws
# =============================================================================

@dataclass
class ScalingLawParams:
    """Parameters for DeepSeek scaling laws."""
    alpha: float = 0.34      # Parameter scaling exponent
    beta: float = 0.28       # Data scaling exponent
    A: float = 406.4         # Parameter coefficient
    B: float = 410.7         # Data coefficient
    L_inf: float = 1.69      # Irreducible loss


class ScalingLaws:
    """
    Scaling law calculator based on DeepSeek's findings.
    
    L(N, D) = A/N^α + B/D^β + L∞
    
    Where:
    - N = number of parameters
    - D = number of training tokens
    - α ≈ 0.34, β ≈ 0.28 (from DeepSeek experiments)
    """
    
    def __init__(self, params: Optional[ScalingLawParams] = None):
        self.params = params or ScalingLawParams()
    
    def predict_loss(self, num_params: float, num_tokens: float) -> float:
        """
        Predict final training loss.
        
        Args:
            num_params: Number of model parameters
            num_tokens: Number of training tokens
        
        Returns:
            Predicted loss
        """
        p = self.params
        return (p.A / (num_params ** p.alpha) + 
                p.B / (num_tokens ** p.beta) + 
                p.L_inf)
    
    def optimal_config(self, compute_flops: float) -> Dict[str, int]:
        """
        Compute optimal model size and data amount for given compute budget.
        
        Based on C ≈ 6ND (compute ≈ 6 * params * tokens)
        
        Args:
            compute_flops: Total compute budget in FLOPs
        
        Returns:
            Dictionary with optimal_params and optimal_tokens
        """
        # For Chinchilla-optimal: N ≈ D
        optimal = (compute_flops / 6) ** 0.5
        
        return {
            "optimal_params": int(optimal),
            "optimal_tokens": int(optimal),
            "compute_flops": compute_flops,
            "tokens_per_param": 1.0,
        }
    
    def recommended_lr(self, num_params: float) -> float:
        """
        Get recommended learning rate based on model size.
        
        η ∝ N^(-0.05)
        
        Args:
            num_params: Number of model parameters
        
        Returns:
            Recommended peak learning rate
        """
        # Base: 3e-4 for 1B model
        base_lr = 3e-4
        base_size = 1e9
        return base_lr * (base_size / num_params) ** 0.05
    
    def tokens_for_target_loss(
        self, 
        num_params: float, 
        target_loss: float
    ) -> float:
        """
        Calculate tokens needed to achieve target loss.
        
        Args:
            num_params: Number of model parameters
            target_loss: Target training loss
        
        Returns:
            Number of tokens needed
        """
        p = self.params
        # Rearrange: D = (B / (L - A/N^α - L∞))^(1/β)
        param_contrib = p.A / (num_params ** p.alpha)
        remaining = target_loss - param_contrib - p.L_inf
        
        if remaining <= 0:
            return float('inf')  # Cannot achieve with any data
        
        return (p.B / remaining) ** (1 / p.beta)


# =============================================================================
# Data Pipeline
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    # Paths
    data_paths: Dict[str, str] = field(default_factory=dict)
    tokenizer_path: Optional[str] = None
    
    # Mixing
    mixing_weights: Dict[str, float] = field(default_factory=lambda: {
        "web": 0.60,
        "code": 0.20,
        "math": 0.10,
        "books": 0.05,
        "scientific": 0.05,
    })
    mixing_temperature: float = 1.0
    
    # Sequence
    seq_length: int = 4096
    
    # Loading
    batch_size: int = 8
    num_workers: int = 4
    shuffle_buffer_size: int = 10000
    prefetch_factor: int = 2
    
    # Misc
    seed: int = 42


class DataMixer:
    """
    Handles domain-based data mixing with temperature scaling.
    
    P(domain_i) = w_i^τ / Σ_j w_j^τ
    """
    
    def __init__(self, weights: Dict[str, float], temperature: float = 1.0):
        self.weights = weights
        self.temperature = temperature
        self.domains = list(weights.keys())
        self._compute_probs()
    
    def _compute_probs(self):
        """Compute sampling probabilities."""
        scaled = np.array([
            self.weights[d] ** self.temperature 
            for d in self.domains
        ])
        self.probs = scaled / scaled.sum()
    
    def sample_domain(self) -> str:
        """Sample a domain according to mixing probabilities."""
        idx = np.random.choice(len(self.domains), p=self.probs)
        return self.domains[idx]
    
    def get_probs(self) -> Dict[str, float]:
        """Get current sampling probabilities."""
        return {d: float(p) for d, p in zip(self.domains, self.probs)}


class StreamingDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large-scale training.
    
    Features:
    - Domain-based mixing
    - Streaming from disk
    - Dynamic sequence length
    """
    
    def __init__(
        self,
        config: DataConfig,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.mixer = DataMixer(
            config.mixing_weights, 
            config.mixing_temperature
        )
        self.rng = np.random.default_rng(config.seed)
        self._current_seq_length = config.seq_length
    
    def set_seq_length(self, seq_length: int):
        """Update sequence length (for curriculum learning)."""
        self._current_seq_length = seq_length
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield tokenized sequences."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # Adjust seed for this worker
            seed = self.config.seed + worker_info.id
            self.rng = np.random.default_rng(seed)
        
        # Create buffer for efficient chunking
        token_buffer = []
        
        while True:
            # Sample domain
            domain = self.mixer.sample_domain()
            
            # Get text from domain (simplified - in practice use file iterators)
            text = self._get_text_from_domain(domain)
            
            if text is None:
                continue
            
            # Tokenize
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
            else:
                # Dummy tokenization for demo
                tokens = [ord(c) % 1000 for c in text[:self._current_seq_length * 2]]
            
            token_buffer.extend(tokens)
            
            # Yield chunks of seq_length
            while len(token_buffer) >= self._current_seq_length:
                chunk = token_buffer[:self._current_seq_length]
                token_buffer = token_buffer[self._current_seq_length:]
                yield torch.tensor(chunk, dtype=torch.long)
    
    def _get_text_from_domain(self, domain: str) -> Optional[str]:
        """Get text sample from a domain. Override for real data loading."""
        # Placeholder - returns synthetic data
        return f"Sample text from {domain} domain. " * 100


class BatchCollator:
    """Collate function for creating training batches."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stack sequences into a batch."""
        input_ids = torch.stack(batch)
        
        # For causal LM, labels = input_ids (shifted internally)
        labels = input_ids.clone()
        
        # Create attention mask (all ones for packed sequences)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def create_dataloader(
    config: DataConfig,
    tokenizer: Optional[Any] = None,
) -> DataLoader:
    """Create a streaming DataLoader."""
    dataset = StreamingDataset(config, tokenizer)
    collator = BatchCollator()
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        pin_memory=True,
    )


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

class WSDScheduler:
    """
    Warmup-Stable-Decay learning rate scheduler.
    
    Three phases:
    1. Warmup: Linear increase from 0 to peak_lr
    2. Stable: Constant at peak_lr
    3. Decay: Linear decrease to min_lr
    
    Used in DeepSeek training.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        peak_lr: float,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0
        
        # Set initial LR
        self._set_lr(0.0)
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate for a given step."""
        if step is None:
            step = self.current_step
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * step / max(1, self.warmup_steps)
        
        step -= self.warmup_steps
        
        if step < self.stable_steps:
            # Stable phase
            return self.peak_lr
        
        step -= self.stable_steps
        
        if step < self.decay_steps:
            # Linear decay
            progress = step / max(1, self.decay_steps)
            return self.min_lr + (self.peak_lr - self.min_lr) * (1 - progress)
        
        return self.min_lr
    
    def step(self) -> float:
        """Advance scheduler and update optimizer LR."""
        lr = self.get_lr()
        self._set_lr(lr)
        self.current_step += 1
        return lr
    
    def _set_lr(self, lr: float):
        """Set learning rate in optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self) -> Dict:
        return {"current_step": self.current_step}
    
    def load_state_dict(self, state_dict: Dict):
        self.current_step = state_dict["current_step"]


class CosineWarmupScheduler:
    """
    Cosine annealing with warmup.
    Alternative scheduler used in some training runs.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = peak_lr * min_lr_ratio
        self.current_step = 0
    
    def get_lr(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step
        
        if step < self.warmup_steps:
            return self.peak_lr * step / max(1, self.warmup_steps)
        
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
    
    def step(self) -> float:
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr


# =============================================================================
# Curriculum Learning
# =============================================================================

class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive training.
    
    Features:
    - Sequence length curriculum
    - Difficulty-based sampling
    """
    
    def __init__(
        self,
        start_seq_len: int = 512,
        end_seq_len: int = 4096,
        seq_curriculum_steps: int = 10000,
        difficulty_warmup_steps: int = 5000,
    ):
        self.start_seq_len = start_seq_len
        self.end_seq_len = end_seq_len
        self.seq_curriculum_steps = seq_curriculum_steps
        self.difficulty_warmup_steps = difficulty_warmup_steps
    
    def get_seq_length(self, step: int) -> int:
        """Get sequence length for current step."""
        if step >= self.seq_curriculum_steps:
            return self.end_seq_len
        
        progress = step / self.seq_curriculum_steps
        seq_len = self.start_seq_len + progress * (self.end_seq_len - self.start_seq_len)
        
        # Round to nearest power of 2 for efficiency
        log2 = int(np.log2(seq_len) + 0.5)
        return 2 ** log2
    
    def get_difficulty_weight(self, step: int) -> float:
        """
        Get weight for difficult examples.
        0 = easy only, 1 = include all difficulties
        """
        if step >= self.difficulty_warmup_steps:
            return 1.0
        return step / self.difficulty_warmup_steps
    
    def get_config(self, step: int) -> Dict[str, Any]:
        """Get all curriculum settings for current step."""
        return {
            "seq_length": self.get_seq_length(step),
            "difficulty_weight": self.get_difficulty_weight(step),
        }


# =============================================================================
# DualPipe: Bidirectional Pipeline Parallelism
# =============================================================================

from enum import Enum


class DualPipePhase(Enum):
    """Phases in DualPipe scheduling."""
    WARMUP = "warmup"
    STEADY = "steady"
    COOLDOWN = "cooldown"


@dataclass
class DualPipeSlot:
    """A single execution slot in DualPipe schedule."""
    micro_batch_id: int
    direction: str  # "forward" or "backward"
    stage_id: int
    phase: DualPipePhase


class DualPipeScheduler:
    """
    DualPipe: Bidirectional Pipeline Parallelism Scheduler.
    
    From DeepSeek-V3 paper - reduces pipeline bubble ratio from
    O((P-1)/M) in 1F1B to O((P-1)/(2M)) by utilizing bidirectional
    data flow.
    
    Key insight: While waiting for gradients to flow back, we can
    start processing a new batch in the reverse direction.
    
    Features:
    - Bidirectional scheduling (forward and backward waves)
    - Reduced bubble ratio compared to 1F1B
    - Compatible with ZeRO and tensor parallelism
    """
    
    def __init__(
        self,
        num_stages: int = 4,
        num_micro_batches: int = 8,
    ):
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        
        # Track which micro-batches are in flight
        self.forward_wave: List[int] = []   # Forward direction
        self.backward_wave: List[int] = []  # Backward direction
        
        # Current phase
        self.current_phase = DualPipePhase.WARMUP
        self.current_step = 0
        
        # Schedule
        self.schedule: List[List[DualPipeSlot]] = []
    
    def build_schedule(self) -> List[List[DualPipeSlot]]:
        """
        Build the DualPipe execution schedule.
        
        Returns:
            List of time steps, each containing list of slots to execute
        """
        schedule: List[List[DualPipeSlot]] = []
        P = self.num_stages
        M = self.num_micro_batches
        
        # Phase 1: Warmup - fill the pipeline
        for t in range(P - 1):
            step_slots = []
            for stage in range(min(t + 1, P)):
                micro_batch = t - stage
                if 0 <= micro_batch < M:
                    step_slots.append(DualPipeSlot(
                        micro_batch_id=micro_batch,
                        direction="forward",
                        stage_id=stage,
                        phase=DualPipePhase.WARMUP,
                    ))
            schedule.append(step_slots)
        
        # Phase 2: Steady state - bidirectional execution
        for t in range(M - P + 1):
            step_slots = []
            
            # Forward wave
            for stage in range(P):
                micro_batch_fwd = t + P - 1 - stage
                if 0 <= micro_batch_fwd < M:
                    step_slots.append(DualPipeSlot(
                        micro_batch_id=micro_batch_fwd,
                        direction="forward",
                        stage_id=stage,
                        phase=DualPipePhase.STEADY,
                    ))
            
            # Backward wave (interleaved)
            for stage in range(P - 1, -1, -1):
                micro_batch_bwd = t + stage
                if 0 <= micro_batch_bwd < M:
                    step_slots.append(DualPipeSlot(
                        micro_batch_id=micro_batch_bwd,
                        direction="backward",
                        stage_id=stage,
                        phase=DualPipePhase.STEADY,
                    ))
            
            schedule.append(step_slots)
        
        # Phase 3: Cooldown - drain the pipeline
        for t in range(P - 1):
            step_slots = []
            for stage in range(P - 1, -1, -1):
                micro_batch = M - P + t + stage
                if 0 <= micro_batch < M:
                    step_slots.append(DualPipeSlot(
                        micro_batch_id=micro_batch,
                        direction="backward",
                        stage_id=stage,
                        phase=DualPipePhase.COOLDOWN,
                    ))
            schedule.append(step_slots)
        
        self.schedule = schedule
        return schedule
    
    def get_bubble_ratio(self) -> float:
        """
        Calculate theoretical bubble ratio.
        
        For DualPipe: bubble_ratio ≈ (P-1)/(2M)
        For 1F1B:     bubble_ratio ≈ (P-1)/M
        """
        return (self.num_stages - 1) / (2 * self.num_micro_batches)
    
    def get_schedule_stats(self) -> Dict[str, Any]:
        """Get statistics about the schedule."""
        if not self.schedule:
            self.build_schedule()
        
        total_slots = sum(len(step) for step in self.schedule)
        forward_slots = sum(
            1 for step in self.schedule 
            for slot in step 
            if slot.direction == "forward"
        )
        backward_slots = total_slots - forward_slots
        
        return {
            "num_stages": self.num_stages,
            "num_micro_batches": self.num_micro_batches,
            "total_time_steps": len(self.schedule),
            "total_slots": total_slots,
            "forward_slots": forward_slots,
            "backward_slots": backward_slots,
            "bubble_ratio": self.get_bubble_ratio(),
        }
    
    def reset(self) -> None:
        """Reset scheduler state for new iteration."""
        self.forward_wave = []
        self.backward_wave = []
        self.current_phase = DualPipePhase.WARMUP
        self.current_step = 0
    
    def step(self) -> Optional[List[DualPipeSlot]]:
        """
        Get next execution step.
        
        Returns:
            List of slots to execute, or None if complete
        """
        if not self.schedule:
            self.build_schedule()
        
        if self.current_step >= len(self.schedule):
            return None
        
        slots = self.schedule[self.current_step]
        self.current_step += 1
        
        # Update phase
        if slots:
            self.current_phase = slots[0].phase
        
        return slots


# =============================================================================
# Distributed Training

# =============================================================================

@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # Parallelism dimensions
    world_size: int = 1
    dp_size: int = 1       # Data parallelism
    tp_size: int = 1       # Tensor parallelism
    pp_size: int = 1       # Pipeline parallelism
    
    # ZeRO optimization stage (0-3)
    zero_stage: int = 0
    
    # Communication
    backend: str = "nccl"
    
    # Memory optimization
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    
    def __post_init__(self):
        computed = self.dp_size * self.tp_size * self.pp_size
        if computed != self.world_size:
            raise ValueError(
                f"DP({self.dp_size}) x TP({self.tp_size}) x PP({self.pp_size}) = "
                f"{computed} != world_size({self.world_size})"
            )
    
    @classmethod
    def single_gpu(cls) -> "DistributedConfig":
        """Configuration for single GPU training."""
        return cls(world_size=1, dp_size=1, tp_size=1, pp_size=1)
    
    @classmethod
    def multi_gpu(cls, num_gpus: int, zero_stage: int = 2) -> "DistributedConfig":
        """Configuration for multi-GPU data parallel training."""
        return cls(
            world_size=num_gpus,
            dp_size=num_gpus,
            tp_size=1,
            pp_size=1,
            zero_stage=zero_stage,
        )


# =============================================================================
# Activation Checkpointing (Phase 3)
# =============================================================================

class CheckpointingStrategy(Enum):
    """Strategy for activation checkpointing."""
    NONE = "none"                    # No checkpointing (fastest, highest memory)
    EVERY_N_LAYERS = "every_n"       # Checkpoint every N layers
    SELECTIVE = "selective"          # Checkpoint specific layers
    FULL = "full"                    # Checkpoint all layers (slowest, lowest memory)


@dataclass
class ActivationCheckpointing:
    """
    Activation checkpoint manager for gradient checkpointing.
    
    Trades compute for memory by recomputing activations during backward pass
    instead of storing them. Critical for training large models.
    """
    strategy: CheckpointingStrategy = CheckpointingStrategy.NONE
    num_layers: int = 32
    checkpoint_interval: int = 1  # For EVERY_N_LAYERS strategy
    checkpoint_layers: List[int] = field(default_factory=list)  # For SELECTIVE
    
    def __post_init__(self):
        self._compute_checkpoint_mask()
    
    def _compute_checkpoint_mask(self):
        """Compute which layers should be checkpointed."""
        if self.strategy == CheckpointingStrategy.NONE:
            self._mask = [False] * self.num_layers
        elif self.strategy == CheckpointingStrategy.EVERY_N_LAYERS:
            self._mask = [i % self.checkpoint_interval == 0 for i in range(self.num_layers)]
        elif self.strategy == CheckpointingStrategy.SELECTIVE:
            self._mask = [i in self.checkpoint_layers for i in range(self.num_layers)]
        elif self.strategy == CheckpointingStrategy.FULL:
            self._mask = [True] * self.num_layers
    
    def should_checkpoint(self, layer_idx: int) -> bool:
        """Check if a specific layer should be checkpointed."""
        if layer_idx >= len(self._mask):
            return False
        return self._mask[layer_idx]
    
    def num_checkpointed_layers(self) -> int:
        """Get count of checkpointed layers."""
        return sum(self._mask)
    
    def estimate_memory_savings(
        self, 
        hidden_dim: int, 
        seq_len: int, 
        batch_size: int,
        dtype_bytes: int = 4
    ) -> int:
        """
        Estimate memory savings (rough estimate based on layer count).
        
        Args:
            hidden_dim: Model hidden dimension
            seq_len: Sequence length
            batch_size: Batch size
            dtype_bytes: Bytes per element (4 for float32, 2 for float16)
        
        Returns:
            Estimated bytes saved
        """
        # Each checkpointed layer saves approximately:
        # 2 * hidden_dim * seq_len * batch_size * dtype_bytes for activations
        activation_size = 2 * hidden_dim * seq_len * batch_size * dtype_bytes
        return self.num_checkpointed_layers() * activation_size
    
    @classmethod
    def optimal(cls, num_layers: int) -> "ActivationCheckpointing":
        """
        Create strategy for optimal memory/compute tradeoff.
        Checkpoints every sqrt(num_layers) layers for optimal balance.
        """
        n = int(math.ceil(math.sqrt(num_layers)))
        return cls(
            strategy=CheckpointingStrategy.EVERY_N_LAYERS,
            num_layers=num_layers,
            checkpoint_interval=n,
        )


# =============================================================================
# ZeRO Optimizer Sharding (Phase 3)
# =============================================================================

class ZeROStage(Enum):
    """
    ZeRO (Zero Redundancy Optimizer) stage configuration.
    
    - Stage 0: No sharding (baseline)
    - Stage 1: Optimizer state sharding (Adam moments)
    - Stage 2: Stage 1 + Gradient sharding
    - Stage 3: Stage 2 + Parameter sharding
    """
    STAGE_0 = 0
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3


@dataclass
class ZeROOptimizer:
    """
    ZeRO optimizer state sharding manager.
    
    Implements memory-efficient optimizer state distribution across data-parallel ranks.
    """
    stage: ZeROStage = ZeROStage.STAGE_0
    dp_size: int = 1
    dp_rank: int = 0
    total_params: int = 0
    
    def __post_init__(self):
        self.partition_size = (self.total_params + self.dp_size - 1) // self.dp_size
    
    def get_param_partition(self) -> Tuple[int, int]:
        """Get parameter indices this rank is responsible for."""
        start = self.dp_rank * self.partition_size
        end = min(start + self.partition_size, self.total_params)
        return start, end
    
    def owns_param(self, param_idx: int) -> bool:
        """Check if this rank owns a specific parameter index."""
        start, end = self.get_param_partition()
        return start <= param_idx < end
    
    def memory_per_param(self) -> float:
        """
        Calculate memory per parameter for this ZeRO stage.
        
        For Adam optimizer with fp32 params and fp32 moments:
        - Stage 0: 16 bytes per param (param + grad + m + v)
        - Stage 1: 4 + 4 + 8/dp_size bytes per param
        - Stage 2: 4 + 4/dp_size + 8/dp_size bytes per param
        - Stage 3: (4 + 4 + 8)/dp_size bytes per param
        """
        dp = float(self.dp_size)
        if self.stage == ZeROStage.STAGE_0:
            return 16.0
        elif self.stage == ZeROStage.STAGE_1:
            return 8.0 + 8.0 / dp
        elif self.stage == ZeROStage.STAGE_2:
            return 4.0 + 12.0 / dp
        else:  # STAGE_3
            return 16.0 / dp
    
    def estimate_memory(self) -> int:
        """Estimate total memory for optimizer state."""
        return int(self.total_params * self.memory_per_param())
    
    def needs_grad_reduce(self) -> bool:
        """Check if gradients need to be reduced across ranks."""
        return self.stage in (ZeROStage.STAGE_0, ZeROStage.STAGE_1)
    
    def grads_sharded(self) -> bool:
        """Check if gradients are sharded (Stage 2+)."""
        return self.stage in (ZeROStage.STAGE_2, ZeROStage.STAGE_3)
    
    def params_sharded(self) -> bool:
        """Check if parameters are sharded (Stage 3)."""
        return self.stage == ZeROStage.STAGE_3


# =============================================================================
# Hierarchical All-to-All for Cross-Node Expert Parallelism (Phase 3)
# =============================================================================

@dataclass
class HierarchicalAllToAll:
    """
    Hierarchical communication topology for cross-node expert dispatch.
    
    Uses two-level hierarchy:
    1. Intra-node: Fast NVLink/NVSwitch communication
    2. Inter-node: InfiniBand/Ethernet communication
    """
    world_size: int = 1
    ranks_per_node: int = 8
    global_rank: int = 0
    
    def __post_init__(self):
        self.num_nodes = (self.world_size + self.ranks_per_node - 1) // self.ranks_per_node
        self.node_id = self.global_rank // self.ranks_per_node
        self.local_rank = self.global_rank % self.ranks_per_node
    
    def intra_node_ranks(self) -> List[int]:
        """Get ranks in the same node (for intra-node communication)."""
        start = self.node_id * self.ranks_per_node
        end = min(start + self.ranks_per_node, self.world_size)
        return list(range(start, end))
    
    def inter_node_leader_ranks(self) -> List[int]:
        """Get one rank from each node (for inter-node communication)."""
        return [
            n * self.ranks_per_node 
            for n in range(self.num_nodes) 
            if n * self.ranks_per_node < self.world_size
        ]
    
    def is_node_leader(self) -> bool:
        """Check if this rank is the leader for its node."""
        return self.local_rank == 0
    
    def compute_send_order(self, dest_rank: int) -> Tuple[bool, bool]:
        """
        Compute optimal send order for hierarchical all-to-all.
        
        Returns:
            (intra_node, inter_node) - whether to use each communication path
        """
        dest_node = dest_rank // self.ranks_per_node
        same_node = dest_node == self.node_id
        return (same_node, not same_node)
    
    def efficiency_ratio(self, intra_bw_gbps: float, inter_bw_gbps: float) -> float:
        """
        Estimate bandwidth utilization for hierarchical vs flat all-to-all.
        
        Args:
            intra_bw_gbps: Intra-node bandwidth (NVLink)
            inter_bw_gbps: Inter-node bandwidth (InfiniBand)
        
        Returns:
            Efficiency ratio (>1 means hierarchical is better)
        """
        n = float(self.world_size)
        r = float(self.ranks_per_node)
        nodes = float(self.num_nodes)
        
        # Simplified model
        flat_time = n / inter_bw_gbps
        hier_time = 2.0 * r / intra_bw_gbps + nodes / inter_bw_gbps
        
        return flat_time / hier_time


# =============================================================================
# Checkpointing
# =============================================================================

@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    save_dir: str = "./checkpoints"
    save_interval: int = 1000
    keep_last_n: int = 5
    save_optimizer: bool = True
    save_scheduler: bool = True
    async_save: bool = True


class CheckpointManager:
    """
    Manages model checkpoints with fault tolerance.
    
    Features:
    - Automatic cleanup of old checkpoints
    - Async saving (optional)
    - Resume from latest
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        os.makedirs(config.save_dir, exist_ok=True)
        self._save_queue: Queue = Queue()
        self._save_thread: Optional[Thread] = None
        
        if config.async_save:
            self._start_save_thread()
    
    def _start_save_thread(self):
        """Start background thread for async saves."""
        def save_worker():
            while True:
                item = self._save_queue.get()
                if item is None:
                    break
                checkpoint, path = item
                torch.save(checkpoint, path)
                self._save_queue.task_done()
        
        self._save_thread = Thread(target=save_worker, daemon=True)
        self._save_thread.start()
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        metrics: Dict[str, float],
    ):
        """Save a checkpoint."""
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        
        if self.config.save_optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if self.config.save_scheduler and hasattr(scheduler, 'state_dict'):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        path = os.path.join(self.config.save_dir, f"checkpoint_{step}.pt")
        
        if self.config.async_save:
            self._save_queue.put((checkpoint, path))
        else:
            torch.save(checkpoint, path)
            print(f"Checkpoint saved: {path}")
        
        self._cleanup_old_checkpoints()
    
    def load_latest(self) -> Optional[Dict]:
        """Load most recent checkpoint."""
        checkpoints = glob.glob(
            os.path.join(self.config.save_dir, "checkpoint_*.pt")
        )
        
        if not checkpoints:
            return None
        
        # Sort by step number
        def get_step(path):
            return int(path.split("_")[-1].split(".")[0])
        
        checkpoints.sort(key=get_step)
        latest = checkpoints[-1]
        
        print(f"Loading checkpoint: {latest}")
        return torch.load(latest, weights_only=False)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = glob.glob(
            os.path.join(self.config.save_dir, "checkpoint_*.pt")
        )
        
        def get_step(path):
            return int(path.split("_")[-1].split(".")[0])
        
        checkpoints.sort(key=get_step)
        
        while len(checkpoints) > self.config.keep_last_n:
            old = checkpoints.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass
    
    def close(self):
        """Cleanup background thread."""
        if self._save_thread is not None:
            self._save_queue.put(None)
            self._save_thread.join()


# =============================================================================
# Monitoring
# =============================================================================

class TrainingMonitor:
    """
    Monitor training progress and detect issues.
    
    Features:
    - Metric aggregation
    - Health checks
    - Optional wandb integration
    """
    
    def __init__(
        self,
        log_interval: int = 10,
        use_wandb: bool = False,
        project_name: str = "deepseek-pretrain",
    ):
        self.log_interval = log_interval
        self.metrics_buffer: Dict[str, List[float]] = defaultdict(list)
        self.start_time = time.time()
        self.wandb = None
        
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project_name)
                self.wandb = wandb
            except ImportError:
                print("wandb not installed, skipping")
    
    def log(self, step: int, metrics: Dict[str, float]):
        """Log metrics for this step."""
        for k, v in metrics.items():
            self.metrics_buffer[k].append(v)
        
        if step % self.log_interval == 0 and step > 0:
            self._flush(step)
    
    def _flush(self, step: int):
        """Aggregate and output metrics."""
        avg_metrics = {
            k: np.mean(v) for k, v in self.metrics_buffer.items()
        }
        
        # Add timing
        elapsed = time.time() - self.start_time
        avg_metrics["elapsed_time"] = elapsed
        
        # Log to wandb
        if self.wandb:
            self.wandb.log(avg_metrics, step=step)
        
        # Print summary
        self._print_summary(step, avg_metrics)
        
        # Check health
        warnings = self.check_health(avg_metrics)
        for w in warnings:
            print(f"  ⚠️ {w}")
        
        # Clear buffer
        self.metrics_buffer.clear()
    
    def _print_summary(self, step: int, metrics: Dict[str, float]):
        """Print training summary."""
        loss = metrics.get("loss", 0)
        lr = metrics.get("lr", 0)
        grad_norm = metrics.get("grad_norm", 0)
        tokens_per_sec = metrics.get("tokens_per_sec", 0)
        
        print(
            f"Step {step:6d} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Grad: {grad_norm:.2f} | "
            f"{tokens_per_sec:.0f} tok/s"
        )
    
    def check_health(self, metrics: Dict[str, float]) -> List[str]:
        """Check for training issues."""
        warnings = []
        
        loss = metrics.get("loss", 0)
        grad_norm = metrics.get("grad_norm", 0)
        
        if np.isnan(loss):
            warnings.append("NaN loss detected!")
        elif loss > 10:
            warnings.append("Loss is very high - possible instability")
        
        if grad_norm > 10:
            warnings.append("High gradient norm - consider lower LR")
        
        return warnings


# =============================================================================
# Training Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Complete configuration for training pipeline."""
    # Model
    model_size: int = 7_000_000_000  # 7B
    
    # Data
    data: DataConfig = field(default_factory=DataConfig)
    
    # Training
    max_steps: int = 100000
    warmup_steps: int = 2000
    stable_ratio: float = 0.8  # Portion in stable phase
    learning_rate: float = 2e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    
    # Distributed
    distributed: DistributedConfig = field(
        default_factory=DistributedConfig.single_gpu
    )
    
    # Checkpointing
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Curriculum
    use_curriculum: bool = True
    curriculum_seq_steps: int = 10000
    
    # Monitoring
    log_interval: int = 10
    use_wandb: bool = False
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> WSDScheduler:
        """Create WSD scheduler from config."""
        stable_steps = int(self.max_steps * self.stable_ratio)
        decay_steps = self.max_steps - self.warmup_steps - stable_steps
        
        return WSDScheduler(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            peak_lr=self.learning_rate,
            min_lr=self.learning_rate * self.min_lr_ratio,
        )


# =============================================================================
# Pre-Trainer
# =============================================================================

class PreTrainer:
    """
    Complete pre-training pipeline for DeepSeek-style training.
    
    Features:
    - WSD learning rate schedule
    - Curriculum learning
    - Gradient accumulation
    - Checkpointing with auto-resume
    - Monitoring and health checks
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PipelineConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        
        # Device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        
        # Model
        self.model = model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        self.scheduler = config.create_scheduler(self.optimizer)
        
        # Curriculum
        self.curriculum = CurriculumScheduler(
            start_seq_len=512,
            end_seq_len=config.data.seq_length,
            seq_curriculum_steps=config.curriculum_seq_steps,
        ) if config.use_curriculum else None
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager(config.checkpoint)
        
        # Monitoring
        self.monitor = TrainingMonitor(
            log_interval=config.log_interval,
            use_wandb=config.use_wandb,
        )
        
        # State
        self.global_step = 0
        
        print(f"PreTrainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute language modeling loss."""
        logits = self.model(input_ids)
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        return loss
    
    def train(
        self,
        dataloader: DataLoader,
        resume: bool = True,
    ):
        """
        Main training loop.
        
        Args:
            dataloader: DataLoader yielding batches
            resume: Whether to resume from checkpoint
        """
        # Resume if checkpoint exists
        if resume:
            checkpoint = self.checkpoint_manager.load_latest()
            if checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.global_step = checkpoint["step"]
                print(f"Resumed from step {self.global_step}")
        
        self.model.train()
        data_iter = iter(dataloader)
        
        accum_step = 0
        accum_loss = 0.0
        
        while self.global_step < self.config.max_steps:
            # Get curriculum settings
            if self.curriculum:
                curriculum_config = self.curriculum.get_config(self.global_step)
                # Could update dataloader seq_length here
            
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            loss = self.compute_loss(input_ids, labels)
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            accum_loss += loss.item()
            accum_step += 1
            
            # Optimizer step
            if accum_step >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                
                # Step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Scheduler
                lr = self.scheduler.step()
                
                # Logging
                metrics = {
                    "loss": accum_loss,
                    "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }
                
                if self.curriculum:
                    metrics["seq_length"] = curriculum_config["seq_length"]
                
                self.monitor.log(self.global_step, metrics)
                
                # Checkpointing
                if self.global_step % self.config.checkpoint.save_interval == 0:
                    self.checkpoint_manager.save(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.global_step,
                        metrics,
                    )
                
                # Reset accumulation
                accum_loss = 0.0
                accum_step = 0
                self.global_step += 1
        
        # Final checkpoint
        self.checkpoint_manager.save(
            self.model, self.optimizer, self.scheduler,
            self.global_step, {"loss": accum_loss},
        )
        
        print(f"Training complete at step {self.global_step}")
        self.checkpoint_manager.close()


        self.checkpoint_manager.close()
