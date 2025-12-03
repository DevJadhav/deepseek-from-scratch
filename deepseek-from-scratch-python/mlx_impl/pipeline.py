import mlx.core as mx
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# DualPipe: Bidirectional Pipeline Parallelism (DeepSeek-V3)
# ============================================================================

class DualPipePhase(Enum):
    """Phases of DualPipe schedule."""
    WARMUP = "warmup"
    STEADY = "steady"
    COOLDOWN = "cooldown"
    DONE = "done"


@dataclass
class DualPipeAction:
    """Action for DualPipe scheduler."""
    regular_fwd: Optional[int] = None  # Forward on regular stream
    regular_bwd: Optional[int] = None  # Backward on regular stream
    reverse_fwd: Optional[int] = None  # Forward on reverse stream
    reverse_bwd: Optional[int] = None  # Backward on reverse stream
    
    def is_done(self) -> bool:
        return (self.regular_fwd is None and 
                self.regular_bwd is None and 
                self.reverse_fwd is None and 
                self.reverse_bwd is None)


class DualPipeScheduler:
    """
    DualPipe scheduler for bidirectional pipeline parallelism.
    
    DeepSeek-V3 innovation achieving ~2x throughput by running
    two streams through the pipeline in opposite directions:
    - Regular stream: stage 0 → N-1 (forward), N-1 → 0 (backward)
    - Reverse stream: stage N-1 → 0 (forward), 0 → N-1 (backward)
    
    This keeps both ends of the pipeline busy, minimizing bubble time.
    """
    
    def __init__(
        self, 
        num_micro_batches: int,
        num_stages: int = 1,
        stage_rank: int = 0
    ):
        assert num_micro_batches >= 2, "DualPipe requires at least 2 micro-batches"
        
        self.num_micro_batches = num_micro_batches
        self.num_stages = num_stages
        self.stage_rank = stage_rank
        self.micro_batches_per_stream = num_micro_batches // 2
        
        self.phase = DualPipePhase.WARMUP
        self.regular_fwd_idx = 0
        self.regular_bwd_idx = 0
        self.reverse_fwd_idx = 0
        self.reverse_bwd_idx = 0
        self.steps_in_phase = 0
    
    @property
    def is_first_stage(self) -> bool:
        return self.stage_rank == 0
    
    @property
    def is_last_stage(self) -> bool:
        return self.stage_rank == self.num_stages - 1
    
    def warmup_steps(self) -> int:
        """Number of warmup steps for this stage."""
        return max(
            self.num_stages - self.stage_rank - 1,
            self.stage_rank
        )
    
    def next_action(self) -> DualPipeAction:
        """Get next action in DualPipe schedule."""
        if self.phase == DualPipePhase.WARMUP:
            return self._warmup_action()
        elif self.phase == DualPipePhase.STEADY:
            return self._steady_action()
        elif self.phase == DualPipePhase.COOLDOWN:
            return self._cooldown_action()
        else:
            return DualPipeAction()  # Done
    
    def _warmup_action(self) -> DualPipeAction:
        """Warmup: fill both directions with forward passes."""
        warmup_needed = self.warmup_steps()
        
        if self.steps_in_phase >= warmup_needed:
            self.phase = DualPipePhase.STEADY
            self.steps_in_phase = 0
            return self._steady_action()
        
        action = DualPipeAction()
        
        # Regular stream forward
        if self.regular_fwd_idx < self.micro_batches_per_stream:
            action.regular_fwd = self.regular_fwd_idx
            self.regular_fwd_idx += 1
        
        # Reverse stream forward (offset by micro_batches_per_stream)
        if self.reverse_fwd_idx < self.micro_batches_per_stream:
            action.reverse_fwd = self.micro_batches_per_stream + self.reverse_fwd_idx
            self.reverse_fwd_idx += 1
        
        self.steps_in_phase += 1
        return action
    
    def _steady_action(self) -> DualPipeAction:
        """Steady state: 1 forward, 1 backward for each stream."""
        total_steady = self.micro_batches_per_stream - self.warmup_steps()
        total_steady = max(0, total_steady)
        
        if self.steps_in_phase >= total_steady:
            self.phase = DualPipePhase.COOLDOWN
            self.steps_in_phase = 0
            return self._cooldown_action()
        
        action = DualPipeAction()
        
        # Regular stream
        if self.regular_fwd_idx < self.micro_batches_per_stream:
            action.regular_fwd = self.regular_fwd_idx
            self.regular_fwd_idx += 1
        if self.regular_bwd_idx < self.micro_batches_per_stream:
            action.regular_bwd = self.regular_bwd_idx
            self.regular_bwd_idx += 1
        
        # Reverse stream
        if self.reverse_fwd_idx < self.micro_batches_per_stream:
            action.reverse_fwd = self.micro_batches_per_stream + self.reverse_fwd_idx
            self.reverse_fwd_idx += 1
        if self.reverse_bwd_idx < self.micro_batches_per_stream:
            action.reverse_bwd = self.micro_batches_per_stream + self.reverse_bwd_idx
            self.reverse_bwd_idx += 1
        
        self.steps_in_phase += 1
        return action
    
    def _cooldown_action(self) -> DualPipeAction:
        """Cooldown: drain remaining backward passes."""
        remaining_regular = self.micro_batches_per_stream - self.regular_bwd_idx
        remaining_reverse = self.micro_batches_per_stream - self.reverse_bwd_idx
        
        if remaining_regular <= 0 and remaining_reverse <= 0:
            self.phase = DualPipePhase.DONE
            return DualPipeAction()
        
        action = DualPipeAction()
        
        if self.regular_bwd_idx < self.micro_batches_per_stream:
            action.regular_bwd = self.regular_bwd_idx
            self.regular_bwd_idx += 1
        
        if self.reverse_bwd_idx < self.micro_batches_per_stream:
            action.reverse_bwd = self.micro_batches_per_stream + self.reverse_bwd_idx
            self.reverse_bwd_idx += 1
        
        self.steps_in_phase += 1
        return action
    
    def reset(self):
        """Reset scheduler for next iteration."""
        self.phase = DualPipePhase.WARMUP
        self.regular_fwd_idx = 0
        self.regular_bwd_idx = 0
        self.reverse_fwd_idx = 0
        self.reverse_bwd_idx = 0
        self.steps_in_phase = 0
    
    def is_done(self) -> bool:
        """Check if schedule is complete."""
        return self.phase == DualPipePhase.DONE
    
    def bubble_ratio(self) -> float:
        """
        Theoretical bubble ratio improvement over 1F1B.
        
        DualPipe reduces bubble from (pp - 1) / total
        to approximately (pp - 1) / (2 * total)
        """
        pp = self.num_stages
        mb = self.num_micro_batches
        
        # 1F1B bubble for comparison
        baseline = (pp - 1) / (mb + pp - 1) if (mb + pp - 1) > 0 else 0
        
        # DualPipe halves bubble by using both directions
        dualpipe = (pp - 1) / (2 * (mb / 2) + pp - 1) if (mb / 2 + pp - 1) > 0 else 0
        
        return dualpipe


# ============================================================================
# DSA Alignment Loss for Continued Pre-training
# ============================================================================

class DSAAlignmentLossType(Enum):
    """Types of alignment loss for DSA training."""
    MSE = "mse"
    COSINE = "cosine"
    KL = "kl"
    COMBINED = "combined"


@dataclass
class DSAAlignmentConfig:
    """Configuration for DSA alignment loss."""
    loss_type: DSAAlignmentLossType = DSAAlignmentLossType.COMBINED
    mse_weight: float = 1.0
    cosine_weight: float = 0.5
    kl_weight: float = 0.1
    temperature: float = 1.0
    sample_fraction: float = 1.0  # Fraction of positions to sample


class DSAAlignmentLoss:
    """
    Alignment loss for DSA continued pre-training.
    
    Ensures sparse attention outputs match dense attention
    during the transition from dense to sparse attention.
    """
    
    def __init__(self, config: DSAAlignmentConfig):
        self.config = config
    
    def compute(
        self, 
        sparse_output: mx.array, 
        dense_output: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Compute alignment loss between sparse and dense outputs.
        
        Args:
            sparse_output: Output from sparse attention (B, T, D)
            dense_output: Output from dense attention (B, T, D)
            mask: Optional mask for valid positions
        
        Returns:
            Scalar loss value
        """
        # Sample positions if fraction < 1
        if self.config.sample_fraction < 1.0:
            sparse_output, dense_output = self._sample_positions(
                sparse_output, dense_output
            )
        
        if self.config.loss_type == DSAAlignmentLossType.MSE:
            return self._mse_loss(sparse_output, dense_output, mask)
        elif self.config.loss_type == DSAAlignmentLossType.COSINE:
            return self._cosine_loss(sparse_output, dense_output, mask)
        elif self.config.loss_type == DSAAlignmentLossType.KL:
            return self._kl_loss(sparse_output, dense_output, mask)
        else:  # COMBINED
            mse = self._mse_loss(sparse_output, dense_output, mask)
            cosine = self._cosine_loss(sparse_output, dense_output, mask)
            kl = self._kl_loss(sparse_output, dense_output, mask)
            return (
                self.config.mse_weight * mse +
                self.config.cosine_weight * cosine +
                self.config.kl_weight * kl
            )
    
    def _sample_positions(
        self, 
        sparse: mx.array, 
        dense: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Sample a fraction of sequence positions."""
        B, T, D = sparse.shape
        n_samples = max(1, int(T * self.config.sample_fraction))
        
        # Random indices (uniform across sequence)
        indices = mx.random.permutation(T)[:n_samples]
        
        sparse_sampled = sparse[:, indices, :]
        dense_sampled = dense[:, indices, :]
        
        return sparse_sampled, dense_sampled
    
    def _mse_loss(
        self, 
        pred: mx.array, 
        target: mx.array, 
        mask: Optional[mx.array]
    ) -> mx.array:
        """Mean squared error loss."""
        diff = pred - target
        sq = diff * diff
        if mask is not None:
            sq = sq * mask.reshape(mask.shape + (1,))
        return mx.mean(sq)
    
    def _cosine_loss(
        self, 
        pred: mx.array, 
        target: mx.array, 
        mask: Optional[mx.array]
    ) -> mx.array:
        """Cosine similarity loss (1 - cos_sim)."""
        pred_norm = pred / (mx.linalg.norm(pred, axis=-1, keepdims=True) + 1e-10)
        target_norm = target / (mx.linalg.norm(target, axis=-1, keepdims=True) + 1e-10)
        
        cos_sim = mx.sum(pred_norm * target_norm, axis=-1)
        if mask is not None:
            cos_sim = cos_sim * mask
        
        return 1.0 - mx.mean(cos_sim)
    
    def _kl_loss(
        self, 
        pred: mx.array, 
        target: mx.array, 
        mask: Optional[mx.array]
    ) -> mx.array:
        """KL divergence-style loss on softmax outputs."""
        temp = self.config.temperature
        pred_soft = mx.softmax(pred / temp, axis=-1)
        target_soft = mx.softmax(target / temp, axis=-1)
        
        # KL = sum(target * log(target / pred))
        kl = target_soft * (mx.log(target_soft + 1e-10) - mx.log(pred_soft + 1e-10))
        kl = mx.sum(kl, axis=-1)
        
        if mask is not None:
            kl = kl * mask
        
        return mx.mean(kl)


@dataclass
class DSATrainingConfig:
    """Configuration for DSA continued pre-training."""
    alignment: DSAAlignmentConfig = None
    dense_alignment_every: int = 100  # Compute dense attention every N steps
    alignment_warmup_steps: int = 1000
    alignment_ramp: bool = True
    
    def __post_init__(self):
        if self.alignment is None:
            self.alignment = DSAAlignmentConfig()
    
    def get_alignment_weight(self, step: int) -> float:
        """Get effective alignment weight at given step."""
        if step < self.alignment_warmup_steps:
            if self.alignment_ramp:
                return step / self.alignment_warmup_steps
            return 0.0
        return 1.0


# ============================================================================
# Original Pipeline Utilities
# ============================================================================

class ScalingLaws:
    """
    Implements Kaplan/Chinchilla scaling laws.
    """
    def __init__(self):
        # Constants from Chinchilla paper (approximate)
        self.A = 406.4
        self.B = 410.7
        self.alpha = 0.34
        self.beta = 0.28
        
    def predict_loss(self, N: float, D: float) -> float:
        """
        Predict loss given parameter count N and dataset size D.
        L(N, D) = E + A/N^alpha + B/D^beta
        """
        # E is irreducible loss (entropy of natural language), approx 1.69
        E = 1.69
        return E + (self.A / (N ** self.alpha)) + (self.B / (D ** self.beta))
        
    def recommended_lr(self, N: float) -> float:
        """
        Heuristic for learning rate based on model size.
        """
        # Smaller models -> larger LR
        # GPT-3 rule of thumb: 0.003239 - 0.0001395 * log(N)
        # Simplified:
        return 0.001 * (1e9 / N) ** 0.5

    def optimal_config(self, C: float) -> Dict[str, float]:
        """
        Calculate optimal N and D for compute budget C (FLOPs).
        C approx 6 * N * D
        """
        # Chinchilla: N_opt = k * C^a, D_opt = k * C^b
        # a approx 0.5, b approx 0.5
        # N approx C / 20 (tokens per param = 20)
        # Actually Chinchilla says 20 tokens per param.
        
        # C = 6 * N * 20N = 120 N^2
        # N = sqrt(C/120)
        
        N_opt = math.sqrt(C / 120)
        D_opt = 20 * N_opt
        
        return {
            "optimal_params": N_opt,
            "optimal_tokens": D_opt
        }

class DataMixer:
    """
    Manages data mixing ratios.
    """
    def __init__(self, domain_weights: Dict[str, float]):
        self.weights = domain_weights
        self.total = sum(domain_weights.values())
        
    def get_probs(self) -> Dict[str, float]:
        return {k: v / self.total for k, v in self.weights.items()}

class PipelineConfig:
    def __init__(self):
        self.model_size = 7e9 # 7B
        self.max_steps = 100000
        self.learning_rate = 3e-4
        
        # Nested configs
        self.data = type('DataConfig', (), {'batch_size': 32, 'seq_length': 2048})
        self.distributed = type('DistConfig', (), {'world_size': 8})
        self.gradient_accumulation_steps = 4

    def create_scheduler(self, optimizer):
        # Mock scheduler
        return type('Scheduler', (), {'get_lr': lambda step: self.learning_rate})()

class CurriculumScheduler:
    """
    Manages curriculum learning (seq len and difficulty).
    """
    def __init__(self, min_seq_len, max_seq_len, total_steps, warmup_steps):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        
    def get_seq_length(self, step: int) -> int:
        if step < self.warmup_steps:
            # Linear increase
            progress = step / self.warmup_steps
            return int(self.min_seq_len + (self.max_seq_len - self.min_seq_len) * progress)
        return self.max_seq_len
        
    def get_difficulty_weight(self, step: int) -> float:
        # Increase difficulty
        return min(1.0, step / self.total_steps)

class DistributedConfig:
    def __init__(self, world_size=1, dp_size=1):
        self.world_size = world_size
        self.dp_size = dp_size
        self.zero_stage = 1


# ============================================================================
# Activation Checkpointing (Phase 3)
# ============================================================================

class CheckpointingStrategy(Enum):
    """Strategy for activation checkpointing."""
    NONE = "none"
    EVERY_N_LAYERS = "every_n"
    SELECTIVE = "selective"
    FULL = "full"


@dataclass
class ActivationCheckpointing:
    """
    Activation checkpoint manager for gradient checkpointing.
    
    Trades compute for memory by recomputing activations during backward pass.
    """
    strategy: CheckpointingStrategy = CheckpointingStrategy.NONE
    num_layers: int = 32
    checkpoint_interval: int = 1
    checkpoint_layers: List[int] = None
    
    def __post_init__(self):
        if self.checkpoint_layers is None:
            self.checkpoint_layers = []
        self._compute_mask()
    
    def _compute_mask(self):
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
        """Estimate memory savings in bytes."""
        activation_size = 2 * hidden_dim * seq_len * batch_size * dtype_bytes
        return self.num_checkpointed_layers() * activation_size
    
    @classmethod
    def optimal(cls, num_layers: int) -> "ActivationCheckpointing":
        """Create optimal strategy: checkpoint every sqrt(num_layers) layers."""
        n = int(math.ceil(math.sqrt(num_layers)))
        return cls(
            strategy=CheckpointingStrategy.EVERY_N_LAYERS,
            num_layers=num_layers,
            checkpoint_interval=n,
        )


# ============================================================================
# ZeRO Optimizer Sharding (Phase 3)
# ============================================================================

class ZeROStage(Enum):
    """ZeRO optimization stage."""
    STAGE_0 = 0
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3


@dataclass
class ZeROOptimizer:
    """ZeRO optimizer state sharding manager."""
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
        """Calculate memory per parameter for this ZeRO stage."""
        dp = float(self.dp_size)
        if self.stage == ZeROStage.STAGE_0:
            return 16.0
        elif self.stage == ZeROStage.STAGE_1:
            return 8.0 + 8.0 / dp
        elif self.stage == ZeROStage.STAGE_2:
            return 4.0 + 12.0 / dp
        else:
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


# ============================================================================
# Hierarchical All-to-All for Cross-Node Expert Parallelism (Phase 3)
# ============================================================================

@dataclass
class HierarchicalAllToAll:
    """Hierarchical communication topology for cross-node expert dispatch."""
    world_size: int = 1
    ranks_per_node: int = 8
    global_rank: int = 0
    
    def __post_init__(self):
        self.num_nodes = (self.world_size + self.ranks_per_node - 1) // self.ranks_per_node
        self.node_id = self.global_rank // self.ranks_per_node
        self.local_rank = self.global_rank % self.ranks_per_node
    
    def intra_node_ranks(self) -> List[int]:
        """Get ranks in the same node."""
        start = self.node_id * self.ranks_per_node
        end = min(start + self.ranks_per_node, self.world_size)
        return list(range(start, end))
    
    def inter_node_leader_ranks(self) -> List[int]:
        """Get one rank from each node."""
        return [
            n * self.ranks_per_node 
            for n in range(self.num_nodes) 
            if n * self.ranks_per_node < self.world_size
        ]
    
    def is_node_leader(self) -> bool:
        """Check if this rank is the leader for its node."""
        return self.local_rank == 0
    
    def compute_send_order(self, dest_rank: int) -> Tuple[bool, bool]:
        """Compute optimal send order."""
        dest_node = dest_rank // self.ranks_per_node
        same_node = dest_node == self.node_id
        return (same_node, not same_node)
    
    def efficiency_ratio(self, intra_bw_gbps: float, inter_bw_gbps: float) -> float:
        """Estimate hierarchical vs flat all-to-all efficiency."""
        n = float(self.world_size)
        r = float(self.ranks_per_node)
        nodes = float(self.num_nodes)
        
        flat_time = n / inter_bw_gbps
        hier_time = 2.0 * r / intra_bw_gbps + nodes / inter_bw_gbps
        
        return flat_time / hier_time
